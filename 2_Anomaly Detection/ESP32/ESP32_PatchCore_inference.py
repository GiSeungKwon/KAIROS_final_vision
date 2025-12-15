import os
import cv2
import numpy as np
from PIL import Image
from glob import glob

import torch
import torch.nn as nn
from torchvision import transforms
import timm
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter

# --- 설정 (Hyperparameters and Paths) ---
# 기존 학습 설정과 동일하게 맞춰야 함
MODULE = "ESP32"
MEMORY_BANK_NAME = MODULE + "_patchcore_memory_bank.npy"
OUTPUT_DIR = "patchcore_results"

# 학습 시 사용했던 설정
BACKBONE_MODEL = "resnet18"
FEATURE_LAYER_NAMES = ["layer2", "layer3"]
IMAGE_SIZE = 256
PATCH_SIZE = 3
NEIGHBOR_COUNT = 9 # 이상 스코어 계산 시 사용할 최근접 이웃 개수
PATCH_STRIDE = 8

# 저장된 메모리 뱅크 경로
MEMORY_BANK_PATH = os.path.join(OUTPUT_DIR, MEMORY_BANK_NAME)

# --- 이상 탐지 임계값 (Threshold) ---
# 이 값은 실제 테스트를 통해 결정해야 합니다. (AUC/ROC curve 분석 등)
# 초기 테스트를 위해 임의의 값을 설정합니다.
ANOMALY_THRESHOLD = 25.0 

# --- 기타 설정 ---
CAMERA_INDEX = 1 # 사용할 카메라 인덱스 (0번이 기본 카메라, 1번이 두 번째 등)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --- 1. 특징 추출기 (Feature Extractor) 클래스 ---
# 학습 시 사용했던 FeatureExtractor와 동일해야 함
class FeatureExtractor(nn.Module):
    def __init__(self, backbone_name, feature_layer_names):
        super(FeatureExtractor, self).__init__()
        self.model = timm.create_model(
            backbone_name, 
            pretrained=True, 
            features_only=True
        )
        self.feature_layer_indices = []
        for i, info in enumerate(self.model.feature_info):
            if info["module"] in feature_layer_names:
                self.feature_layer_indices.append(i)

    def forward(self, x):
        features = self.model(x)
        return [features[i] for i in self.feature_layer_indices]

# --- 2. 특징 패치화 함수 (extract_patches) ---
# 학습 시 사용했던 함수와 동일해야 함
def extract_patches(features, patch_size):
    """
    PatchCore 방식의 특징 패치화: 
    - 특징 맵 업샘플링 후, 같은 위치 패치 concat
    """
    H_max = max([f.shape[2] for f in features])
    W_max = max([f.shape[3] for f in features])

    aligned_features = []
    for feat in features:
        if feat.shape[2] != H_max or feat.shape[3] != W_max:
            feat = F.interpolate(
                feat,
                size=(H_max, W_max),
                mode="bilinear",
                align_corners=False
            )
        aligned_features.append(feat)

    all_patches = []
    # 패치 크기/스트라이드를 이용하여 패치 추출
    for feat in aligned_features:
        B, C, H, W = feat.shape
        # unfold를 이용해 패치 추출
        patches = feat.unfold(2, patch_size, PATCH_STRIDE).unfold(3, patch_size, PATCH_STRIDE)
        patches = patches.permute(0, 2, 3, 1, 4, 5)
        patches = patches.contiguous().view(
            B, -1, C * patch_size * patch_size
        )
        all_patches.append(patches)

    combined_patches = torch.cat(all_patches, dim=-1)

    # (B, N_patches, D) 형태로 반환
    return combined_patches


# --- 3. 모델 로드 및 메모리 뱅크 준비 ---

print("--- 1. 모델 및 메모리 뱅크 로드 시작 ---")

# 특징 추출기 초기화
extractor = FeatureExtractor(BACKBONE_MODEL, FEATURE_LAYER_NAMES).to(device)
extractor.eval()

# 데이터 전처리 파이프라인 (학습과 동일)
data_transforms = transforms.Compose([
    transforms.ToPILImage(), # OpenCV 이미지를 PIL 이미지로 변환
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 메모리 뱅크 로드 (NumPy 파일)
try:
    memory_bank_np = np.load(MEMORY_BANK_PATH)
    # NumPy 배열을 PyTorch 텐서로 변환하고 GPU에 올림
    memory_bank = torch.from_numpy(memory_bank_np).to(device).float()
    print(f"메모리 뱅크 로드 완료. 크기: {memory_bank.shape}")
except FileNotFoundError:
    print(f"오류: 메모리 뱅크 파일이 없습니다. 경로: {MEMORY_BANK_PATH}")
    exit()

# --- 4. 이상 탐지 추론 함수 ---

def predict_anomaly(image_tensor, extractor, memory_bank, neighbor_count):
    """
    주어진 이미지 텐서에 대해 PatchCore 이상 탐지 추론을 수행합니다.
    - image_score: 이미지 레벨의 이상 스코어 (가장 큰 패치 스코어)
    - heatmap: 패치 레벨의 이상 스코어로 생성된 히트맵
    """
    with torch.no_grad():
        # (1, 3, H, W) 텐서 입력
        
        # 1. 특징 추출 및 패치화
        features = extractor(image_tensor) 
        # (1, N_patches, D)
        batch_patches = extract_patches(features, PATCH_SIZE) 
        
        # 2. PatchCore 이상 스코어 계산
        # N_patches, D
        query_patches = batch_patches.squeeze(0)
        
        # Query 패치와 메모리 뱅크 간의 거리 계산 (L2-norm)
        # N_patches, M_coresets
        distances = torch.cdist(query_patches, memory_bank, p=2.0)

        # 각 Query 패치에 대해 K개의 최근접 이웃 거리와 인덱스 찾기
        # distances.shape: (N_patches, NEIGHBOR_COUNT)
        min_distances, min_indices = torch.topk(
            distances, 
            neighbor_count, 
            largest=False, # 가장 작은 거리 (가장 비슷한 정상 패치)
            dim=1
        )
        
        # PatchCore 스코어 계산: K개의 최근접 이웃 거리의 제곱합 / K
        # PatchCore 논문: '최근접 이웃의 거리를 사용하여 이상 스코어 보정'
        # 여기서는 간단히 가장 가까운 거리(최소 거리)를 이상 스코어로 사용
        patch_scores = min_distances[:, 0] # 가장 가까운 이웃과의 거리 (최소 거리)
        
        # 3. 이미지 스코어 계산
        # 이미지의 이상 스코어는 가장 높은 패치 스코어
        image_score = torch.max(patch_scores)
        
        # 4. 히트맵 생성
        # 패치 스코어는 (N_patches,) 크기이므로, 추출된 패치의 그리드 크기로 재구성해야 함
        
        # 특징 맵 크기 계산 (H_max, W_max)
        H_max = max([f.shape[2] for f in features])
        W_max = max([f.shape[3] for f in features])
        
        # Patch 개수 (H_grid, W_grid) 계산
        H_grid = int((H_max - PATCH_SIZE) / PATCH_STRIDE) + 1
        W_grid = int((W_max - PATCH_SIZE) / PATCH_STRIDE) + 1
        
        # (H_grid * W_grid,) 크기의 스코어를 (H_grid, W_grid)로 reshape
        try:
            score_map = patch_scores.reshape(H_grid, W_grid)
        except RuntimeError:
            # 패치가 하나만 추출되는 등의 예외 처리 (거의 발생하지 않음)
            score_map = patch_scores.unsqueeze(0).unsqueeze(0)
        
        # 히트맵을 원본 이미지 크기(IMAGE_SIZE)로 업샘플링
        # (1, 1, H_grid, W_grid)
        score_map = score_map.unsqueeze(0).unsqueeze(0) 
        # (1, 1, IMAGE_SIZE, IMAGE_SIZE)
        heatmap = F.interpolate(
            score_map,
            size=(IMAGE_SIZE, IMAGE_SIZE),
            mode='bilinear',
            align_corners=False
        ).squeeze().cpu().numpy() # (IMAGE_SIZE, IMAGE_SIZE) NumPy 배열
        
        # Gaussian Smoothing (결과 개선에 도움을 줌)
        heatmap = gaussian_filter(heatmap, sigma=4) # 시그마 값 조정 가능
        
        return image_score.item(), heatmap

# --- 5. 실시간 스트리밍 및 시각화 루프 ---

print("--- 2. 카메라 스트리밍 시작 (종료: 'q' 키) ---")

cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    print(f"오류: 카메라 인덱스 {CAMERA_INDEX}를 열 수 없습니다.")
    exit()

cap.set(cv2.CAP_PROP_AUTOFOCUS, 1.0)

# 원본 카메라 해상도 저장 (오버레이 시 필요)
ORIG_W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
ORIG_H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

while True:
    ret, frame = cap.read()
    if not ret:
        print("프레임을 받을 수 없습니다. (스트림 종료?)")
        break

    # 1. 전처리 (추론용)
    # OpenCV (BGR) -> PIL (RGB) 변환은 data_transforms의 ToPILImage가 처리
    # (H, W, 3) BGR -> (3, H, W) 정규화된 텐서
    input_tensor = data_transforms(frame).unsqueeze(0).to(device) # (1, 3, 256, 256)
    
    # 2. 이상 탐지 추론
    anomaly_score, heatmap = predict_anomaly(
        input_tensor, 
        extractor, 
        memory_bank, 
        NEIGHBOR_COUNT
    )
    
    # 3. 시각화 준비 (OpenCV를 이용한 시각화)
    
    # --- 스트리밍 창 시각화 ---
    display_frame = frame.copy()
    
    # 이상 탐지 결과 텍스트 표시
    is_anomaly = anomaly_score > ANOMALY_THRESHOLD
    color = (0, 0, 255) if is_anomaly else (0, 255, 0) # 빨강: 이상, 초록: 정상
    status_text = f"Score: {anomaly_score:.2f}"
    status_text += " / ANOMALY!" if is_anomaly else " / NORMAL"

    cv2.putText(display_frame, status_text, (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    
    # 원본 프레임 표시
    cv2.imshow("Streaming Window (Anomaly Status)", display_frame)


    # --- 히트맵 창 시각화 ---
    
    # 히트맵을 0-255 범위로 정규화 및 크기 조정
    # 히트맵의 최댓값으로 정규화하여 색상 범위를 설정
    if heatmap.max() > 0:
        heatmap_norm = (heatmap / heatmap.max()) * 255 
    else:
        heatmap_norm = np.zeros_like(heatmap) # 전부 0이면 검은색

    # 8-bit, 흑백으로 변환 후, 컬러맵 적용 (e.g., MAGMA)
    heatmap_norm = heatmap_norm.astype(np.uint8)
    heatmap_colored = cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_MAGMA)
    
    # 히트맵을 원본 프레임 크기로 조정 (ORIG_W, ORIG_H)
    heatmap_colored_resized = cv2.resize(
        heatmap_colored, 
        (ORIG_W, ORIG_H), 
        interpolation=cv2.INTER_LINEAR
    )
    
    # 원본 프레임과 히트맵 오버레이 (투명도 조절)
    alpha = 0.5 # 투명도
    # addWeighted 함수로 두 이미지를 합성 (배경 이미지, 배경 가중치, 전경 이미지, 전경 가중치, 감마)
    overlay_frame = cv2.addWeighted(
        frame, 
        1 - alpha, 
        heatmap_colored_resized, 
        alpha, 
        0
    )
    
    # 히트맵 오버레이 결과 표시
    cv2.imshow("Anomaly Heatmap Window", overlay_frame)

    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 종료 시 리소스 해제
cap.release()
cv2.destroyAllWindows()
print("--- 스트리밍 종료 ---")