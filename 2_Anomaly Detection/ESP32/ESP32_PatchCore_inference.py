import os
import sys
import numpy as np
import cv2
from PIL import Image
import time

import torch
import torch.nn as nn
from torchvision import transforms
import timm
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors
from scipy.ndimage import gaussian_filter

# --- 1. 설정 (Settings - 학습 코드와 동일해야 함) ---
MODULE = "ESP32"
MEMORY_BANK_PATH = f"patchcore_results/{MODULE}_patchcore_memory_bank.npy"

BACKBONE_MODEL = "resnet18" 
FEATURE_LAYER_NAMES = ["layer2", "layer3"]
IMAGE_SIZE = 256 # 모델 입력 크기
PATCH_SIZE = 3 
NEIGHBOR_COUNT = 9 # 이상 스코어 계산 시 사용할 최근접 이웃 개수
PATCH_STRIDE = 8

# 이상 판단 기준 스코어 (데이터셋에 따라 조정 필요)
ANOMALY_THRESHOLD = 0.5 
# 가우시안 필터링 시그마 값 (히트맵 부드러움 정도)
GAUSSIAN_SIGMA = 4 

# --- 2. 특징 추출기 (Feature Extractor) - 학습 코드와 동일하게 재정의 ---

class FeatureExtractor(nn.Module):
    """
    사전 학습된 모델을 사용하여 중간 특징 맵을 추출하는 클래스
    """
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

# --- 3. 특징 패치화 함수 - 학습 코드와 동일하게 재정의 ---

def extract_patches(features, patch_size, patch_stride):
    """
    학습 코드와 동일한 방식으로 특징 맵을 업샘플링하고 패치화
    """
    # 기준 해상도 (가장 큰 H, W)
    H_max = max([f.shape[2] for f in features])
    W_max = max([f.shape[3] for f in features])

    aligned_features = []

    for feat in features:
        if feat.shape[2] != H_max or feat.shape[3] != W_max:
            # Bilinear Interpolation으로 해상도 맞추기
            feat = F.interpolate(
                feat,
                size=(H_max, W_max),
                mode="bilinear",
                align_corners=False
            )
        aligned_features.append(feat)

    all_patches = []
    for feat in aligned_features:
        B, C, H, W = feat.shape
        # PATCH_STRIDE를 사용하여 패치 추출 간격 조정
        patches = feat.unfold(2, patch_size, patch_stride).unfold(3, patch_size, patch_stride)
        patches = patches.permute(0, 2, 3, 1, 4, 5)
        # 패치 위치 정보를 위해 H', W' 정보를 유지하여 뷰 변환
        patches = patches.contiguous().view(
            B, -1, C * patch_size * patch_size
        )
        all_patches.append(patches)

    # 이제 patch 개수가 동일 → concat 가능
    combined_patches = torch.cat(all_patches, dim=-1)

    # 패치 위치 정보(H', W') 복원 및 반환 (B=1이므로 [0] 인덱싱)
    B, N_patches_total, C_total = combined_patches.shape
    H_prime = patches.shape[1] # (H'/stride) * (W'/stride)
    
    # 맵 사이즈 계산:
    # 256 이미지, ResNet18 layer2(32x32), layer3(16x16) -> H_max=32, W_max=32
    # 패치 스트라이드 8, 패치 사이즈 3 -> (32-3)/8 + 1 = 4.625 -> 4
    # 따라서 H' x W' = 4x4 = 16
    map_size = int(np.sqrt(N_patches_total)) # H_map = W_map
    
    # [ (1, H_map * W_map, C_total) ] 형태와 맵 사이즈 반환
    return combined_patches, (map_size, map_size)


# --- 4. PatchCore 추론 클래스 ---

class PatchCoreInference:
    def __init__(self, memory_bank_path, device):
        print(f"메모리 뱅크 로드 중: {memory_bank_path}")
        try:
            # 학습된 코어셋 메모리 뱅크 로드 (NumPy 배열)
            self.memory_bank = np.load(memory_bank_path)
        except FileNotFoundError:
            print(f"오류: 메모리 뱅크 파일({memory_bank_path})을 찾을 수 없습니다.")
            sys.exit(1)

        # KNN 탐색을 위한 NearestNeighbors 초기화 및 학습
        self.knn = NearestNeighbors(n_neighbors=NEIGHBOR_COUNT, metric='euclidean', n_jobs=-1)
        self.knn.fit(self.memory_bank)
        print(f"KNN 모델 초기화 완료. 메모리 크기: {self.memory_bank.shape}")
        
        self.device = device
        # 특징 추출기 초기화
        self.extractor = FeatureExtractor(BACKBONE_MODEL, FEATURE_LAYER_NAMES).to(device).eval()

        # 데이터 전처리 파이프라인
        self.transform = transforms.Compose([
            transforms.ToPILImage(), # OpenCV BGR 배열을 PIL 이미지로 변환
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    @torch.no_grad()
    def infer(self, image_bgr):
        
        # 1. 이미지 전처리 및 특징 추출
        
        # OpenCV BGR을 입력으로 받음. transform 내부에서 PIL 이미지로 변환 후 처리
        input_tensor = self.transform(image_bgr).unsqueeze(0).to(self.device)
        
        # 특징 추출
        features = self.extractor(input_tensor)
        
        # 2. 특징 패치화 및 KNN 탐색
        
        # 패치 추출: (1, N_patches_total, C_total), (H_map, W_map)
        test_patches_tensor, map_size = extract_patches(features, PATCH_SIZE, PATCH_STRIDE)
        test_patches = test_patches_tensor.squeeze(0).cpu().numpy() # (N_patches, C_total)
        
        # KNN 탐색: 가장 가까운 NEIGHBOR_COUNT개의 정상 패치 찾기
        # distances: (N_patches, NEIGHBOR_COUNT)
        distances, _ = self.knn.kneighbors(test_patches, n_neighbors=NEIGHBOR_COUNT)
        
        # PatchCore Anomaly Score: 가장 가까운 (k=1) 패치와의 거리 사용
        patch_scores = distances[:, 0]
        
        # 3. Anomaly Map 생성
        
        # 스코어를 특징 맵 그리드에 다시 배치 (예: 16 -> 4x4)
        map_height, map_width = map_size
        anomaly_map = patch_scores.reshape(map_height, map_width)
        
        # 가우시안 필터링 (히트맵 부드럽게)
        anomaly_map_smooth = gaussian_filter(anomaly_map, sigma=GAUSSIAN_SIGMA) 
        
        # 4. 최종 이상 스코어 계산
        # 이미지의 최종 스코어는 맵에서 가장 높은 스코어입니다.
        image_anomaly_score = np.max(anomaly_map_smooth)
        
        # 5. Anomaly Map 시각화
        
        # 맵 정규화 (0~1)
        min_score = anomaly_map_smooth.min()
        max_score = anomaly_map_smooth.max()
        if max_score > min_score:
             anomaly_map_norm = (anomaly_map_smooth - min_score) / (max_score - min_score)
        else:
             anomaly_map_norm = np.zeros_like(anomaly_map_smooth)

        # 원본 이미지 크기(H, W)로 업스케일 
        anomaly_map_resized = cv2.resize(anomaly_map_norm, 
                                         (image_bgr.shape[1], image_bgr.shape[0]), 
                                         interpolation=cv2.INTER_LINEAR)
        
        # 히트맵 색상 적용 (JET Colormap 사용)
        anomaly_heatmap = (anomaly_map_resized * 255).astype(np.uint8)
        anomaly_heatmap_color = cv2.applyColorMap(anomaly_heatmap, cv2.COLORMAP_JET)

        # 원본 이미지와 히트맵 블렌딩
        # 가중치 (0.6: 히트맵, 0.4: 원본 이미지)
        blended_image = cv2.addWeighted(image_bgr, 0.4, anomaly_heatmap_color, 0.6, 0)

        return image_anomaly_score, blended_image, anomaly_heatmap_color


# --- 5. 실시간 카메라 메인 루프 ---

def main():
    # GPU 사용 가능 여부 확인
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"사용 장치: {device}")

    # PatchCore 추론 객체 초기화
    try:
        pc_infer = PatchCoreInference(MEMORY_BANK_PATH, device)
    except SystemExit:
        return

    # 카메라 초기화 (0은 일반적으로 기본 웹캠)
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("오류: 카메라를 열 수 없습니다. 카메라 연결 상태를 확인하세요.")
        return

    # 스트리밍 및 이상 탐지 루프
    while True:
        start_time = time.time() # FPS 측정을 위한 시작 시간

        # 프레임 읽기
        ret, frame = cap.read()
        if not ret:
            print("오류: 카메라에서 프레임을 읽을 수 없습니다.")
            break

        original_frame = frame.copy() 

        # PatchCore 이상 탐지 수행
        score, blended_image, heatmap_image = pc_infer.infer(original_frame)
        
        # --- 시각화: 스트리밍 창 ---
        
        # 이상 감지 결과 텍스트 표시
        status = "이상 감지됨 (Anomaly)" if score > ANOMALY_THRESHOLD else "정상 (Normal)"
        color = (0, 0, 255) if score > ANOMALY_THRESHOLD else (0, 255, 0) # 빨강/초록
        
        # 블렌딩 이미지에 스코어 및 상태 표시
        cv2.putText(blended_image, f"Score: {score:.4f} ({status})", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # FPS 계산 및 표시
        end_time = time.time()
        fps = 1 / (end_time - start_time)
        cv2.putText(blended_image, f"FPS: {fps:.2f}", (10, blended_image.shape[0] - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)


        # "스트리밍 창" 표시
        cv2.imshow("Streaming & Blended Anomaly Map", blended_image)
        
        # --- 시각화: 히트맵 창 ---
        
        # "Hit Map 창" 표시
        cv2.putText(heatmap_image, f"Score: {score:.4f}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.imshow("Anomaly Hit Map", heatmap_image)

        # 'q' 또는 ESC 키를 누르면 종료
        if cv2.waitKey(1) & 0xFF in [ord('q'), 27]:
            break

    # 자원 해제
    cap.release()
    cv2.destroyAllWindows()
    print("실시간 이상 탐지 종료.")

if __name__ == "__main__":
    main()