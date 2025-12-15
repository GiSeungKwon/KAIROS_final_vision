import os
import sys
import numpy as np
import cv2
from PIL import Image

import torch
import torch.nn as nn
from torchvision import transforms
import timm
from sklearn.neighbors import NearestNeighbors
from scipy.ndimage import gaussian_filter

# --- 1. 설정 (Settings - 학습 코드와 동일해야 함) ---
BACKBONE_MODEL = "resnet18" 
FEATURE_LAYER_NAMES = ["layer2", "layer3"]
IMAGE_SIZE = 256 # 모델 입력 크기
PATCH_SIZE = 3 
NEIGHBOR_COUNT = 9 # 이상 스코어 계산 시 사용할 최근접 이웃 개수
MEMORY_BANK_PATH = "patchcore_results/patchcore_memory_bank.npy"
ANOMALY_THRESHOLD = 0.5 # 이상 판단 기준 스코어 (필요에 따라 조정 필요)


# --- 2. 특징 추출기 (Feature Extractor) - 학습 코드 재사용 ---

class FeatureExtractor(nn.Module):
    def __init__(self, backbone_name, feature_layer_names):
        super(FeatureExtractor, self).__init__()
        self.model = timm.create_model(
            backbone_name, 
            pretrained=True, 
            features_only=True
        )
        self.feature_layer_indices = [
            list(self.model.feature_info.module_names).index(name) 
            for name in feature_layer_names
        ]

    def forward(self, x):
        features = self.model(x)
        return [features[i] for i in self.feature_layer_indices]

# --- 3. 특징 패치화 함수 - 학습 코드 재사용 ---

def extract_patches(features, patch_size):
    all_patches = []
    for feat in features: 
        B, C, H, W = feat.shape
        patches = feat.unfold(2, patch_size, 1).unfold(3, patch_size, 1)
        patches = patches.permute(0, 2, 3, 1, 4, 5)
        patches = patches.contiguous().view(B, -1, C * patch_size * patch_size) 
        all_patches.append(patches)
    
    combined_patches = torch.cat(all_patches, dim=-1)
    # 패치 위치 정보(H', W')를 위해 shape 유지: (B, H'*W', C_total)
    return combined_patches 


# --- 4. PatchCore 추론 클래스 ---

class PatchCoreInference:
    def __init__(self, memory_bank_path, device):
        print(f"메모리 뱅크 로드 중: {memory_bank_path}")
        try:
            # 학습된 코어셋 메모리 뱅크 로드
            self.memory_bank = np.load(memory_bank_path)
        except FileNotFoundError:
            print(f"오류: 메모리 뱅크 파일({memory_bank_path})을 찾을 수 없습니다.")
            sys.exit(1)

        # KNN 탐색을 위한 NearestNeighbors 초기화
        # Metric: Euclidean distance (p=2)를 사용합니다.
        # 알고리즘은 'auto'로 설정하여 데이터에 따라 최적화
        self.knn = NearestNeighbors(n_neighbors=NEIGHBOR_COUNT, metric='euclidean', n_jobs=-1)
        self.knn.fit(self.memory_bank)
        print(f"KNN 모델 초기화 완료. 메모리 크기: {self.memory_bank.shape}")
        
        self.device = device
        self.extractor = FeatureExtractor(BACKBONE_MODEL, FEATURE_LAYER_NAMES).to(device).eval()

        # 데이터 전처리 파이프라인
        self.transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    @torch.no_grad()
    def infer(self, image_bgr):
        # 1. 이미지 전처리 및 특징 추출
        
        # BGR -> RGB 변환 (OpenCV 기본값 BGR, PyTorch/PIL은 RGB)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image_rgb)
        
        # 이미지 전처리 및 PyTorch 텐서 변환
        input_tensor = self.transform(image_pil).unsqueeze(0).to(self.device)
        
        # 특징 추출
        features = self.extractor(input_tensor)
        
        # 특징 맵의 크기 (Anomaly Map을 만들 때 사용)
        map_height = features[0].shape[2] # Layer2의 H (예: 63)
        map_width = features[0].shape[3] # Layer2의 W (예: 63)
        
        # 2. 특징 패치화 및 KNN 탐색
        
        # 패치 추출: (1, N_patches_total, C_total)
        test_patches_tensor = extract_patches(features, PATCH_SIZE)
        test_patches = test_patches_tensor.squeeze(0).cpu().numpy() # (N_patches, C_total)
        
        # KNN 탐색: 각 테스트 패치에 가장 가까운 NEIGHBOR_COUNT개의 정상 패치 찾기
        # distances: (N_patches, NEIGHBOR_COUNT), indices: (N_patches, NEIGHBOR_COUNT)
        distances, _ = self.knn.kneighbors(test_patches, n_neighbors=NEIGHBOR_COUNT)
        
        # PatchCore Anomaly Score: 가장 가까운 (k=1) 패치와의 거리
        # (실제 PatchCore 구현에서는 k=9를 사용하여 이웃 간의 가중치를 계산하지만, 
        # 여기서는 단순화하여 가장 가까운 1개 이웃과의 거리를 기본 스코어로 사용합니다.)
        patch_scores = distances[:, 0]
        
        # 3. Anomaly Map 생성
        
        # 스코어를 특징 맵 그리드에 다시 배치
        num_patches = patch_scores.shape[0]
        anomaly_map = patch_scores.reshape(map_height, map_width)
        
        # 가우시안 필터링 (이상 위치의 경계를 부드럽게)
        # PatchCore의 공식 구현에서 사용되는 핵심 단계
        anomaly_map_smooth = gaussian_filter(anomaly_map, sigma=4) # sigma 값은 조정 가능
        
        # 4. 최종 이상 스코어 계산
        # 이미지의 최종 스코어는 맵에서 가장 높은 스코어입니다.
        image_anomaly_score = np.max(anomaly_map_smooth)
        
        # 5. Anomaly Map 정규화 및 시각화
        
        # 맵 정규화 (0~1)
        min_score = anomaly_map_smooth.min()
        max_score = anomaly_map_smooth.max()
        if max_score > min_score:
            anomaly_map_norm = (anomaly_map_smooth - min_score) / (max_score - min_score)
        else:
            anomaly_map_norm = np.zeros_like(anomaly_map_smooth)

        # 이미지 크기로 업스케일 (Interpolation: Linear)
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
        # 파일 로드 실패 시 종료
        return

    # 카메라 초기화 (0은 일반적으로 기본 웹캠)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("오류: 카메라를 열 수 없습니다. 카메라 연결 상태를 확인하세요.")
        return

    # 스트리밍 및 이상 탐지 루프
    while True:
        # 프레임 읽기
        ret, frame = cap.read()
        if not ret:
            print("오류: 카메라에서 프레임을 읽을 수 없습니다.")
            break

        # 원본 프레임의 크기 유지
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

        # "스트리밍 창" 표시 (이상 감지 위치가 블렌딩된 이미지)
        cv2.imshow("Streaming & Blended Anomaly Map", blended_image)
        
        # --- 시각화: 히트맵 창 ---
        
        # "Hit Map 창" 표시 (순수한 이상 스코어 히트맵)
        # 히트맵에 스코어 표시
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