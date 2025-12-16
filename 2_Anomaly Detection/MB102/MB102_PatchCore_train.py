import os
from glob import glob
from tqdm import tqdm
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import timm
import torch.nn.functional as F

# --- 설정 (Hyperparameters and Paths) ---
# 정상 이미지 경로 (사용자 지정 경로)
MODULE = "MB102"
NORMAL_IMAGE_DIR = r"C:\Dev\KAIROS_Project\data\aug\aug_Anomaly_" + MODULE
MEMORY_BANK_NAME = MODULE + "_patchcore_memory_bank.npy"
# 모델 설정
BACKBONE_MODEL = "resnet18" # 특징 추출기 모델 (예: WideResNet, ResNet18 등)
FEATURE_LAYER_NAMES = ["layer2", "layer3"] # PatchCore에서 사용할 특징 맵 레이어
IMAGE_SIZE = 256
BATCH_SIZE = 32
PATCH_SIZE = 3 # 패치 크기 (일반적으로 3)
NEIGHBOR_COUNT = 9 # 이상 스코어 계산 시 사용할 최근접 이웃 개수
SUBSAMPLING_RATIO = 0.1 # 메모리 뱅크 구축 시 코어셋 서브샘플링 비율 (0.01~0.2)

PATCH_STRIDE = 8

# 출력 경로
OUTPUT_DIR = "patchcore_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# --- 1. 데이터셋 및 데이터로더 ---

class ESP32Dataset(Dataset):
    """
    ESP32 정상 이미지를 로드하는 Dataset 클래스
    """
    def __init__(self, image_dir, transform=None):
        # 지정된 경로에서 모든 이미지 파일 경로를 찾음
        self.image_paths = sorted(glob(os.path.join(image_dir, '*.*')))
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # 이미지 로드 및 RGB 변환
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert('RGB')
        
        # 변환 적용
        if self.transform:
            img = self.transform(img)
        
        # 파일 경로도 반환하여 나중에 디버깅에 활용 가능
        return img, img_path

# 데이터 전처리 파이프라인
data_transforms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    # ImageNet 평균/표준편차로 정규화 (사전 학습 모델 사용 시 일반적)
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 데이터셋 및 데이터로더 생성
train_dataset = ESP32Dataset(NORMAL_IMAGE_DIR, transform=data_transforms)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)


# --- 2. 특징 추출기 (Feature Extractor) ---

class FeatureExtractor(nn.Module):
    """
    사전 학습된 모델을 사용하여 중간 특징 맵을 추출하는 클래스
    """
    def __init__(self, backbone_name, feature_layer_names):
        super(FeatureExtractor, self).__init__()
        # timm 라이브러리를 사용하여 사전 학습된 모델 로드 (pre=True)
        self.model = timm.create_model(
            backbone_name, 
            pretrained=True, 
            features_only=True # 특징 맵만 추출하도록 설정
        )
        # 사용할 특징 레이어의 인덱스를 찾음
        self.feature_layer_indices = []
        for i, info in enumerate(self.model.feature_info):
            if info["module"] in feature_layer_names:
                self.feature_layer_indices.append(i)

    def forward(self, x):
        # 특징 맵 추출
        features = self.model(x)
        # 지정된 레이어의 특징만 반환
        return [features[i] for i in self.feature_layer_indices]

# 특징 추출기 초기화
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
extractor = FeatureExtractor(BACKBONE_MODEL, FEATURE_LAYER_NAMES).to(device)
extractor.eval() # 특징 추출기는 학습 모드가 아닌 평가 모드(가중치 고정)로 사용


# --- 3. 특징 패치화 및 메모리 뱅크 구축 (학습) ---

def extract_patches(features, patch_size):
    """
    PatchCore 방식:
    - 서로 다른 해상도의 feature map을
    - 가장 큰 해상도 기준으로 upsample
    - 같은 위치의 패치끼리 concat
    """
    # 기준 해상도 (가장 큰 H, W)
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
    for feat in aligned_features:
        B, C, H, W = feat.shape
        patches = feat.unfold(2, patch_size, PATCH_STRIDE).unfold(3, patch_size, PATCH_STRIDE)
        patches = patches.permute(0, 2, 3, 1, 4, 5)
        patches = patches.contiguous().view(
            B, -1, C * patch_size * patch_size
        )
        all_patches.append(patches)

    # 이제 patch 개수가 동일 → concat 가능
    combined_patches = torch.cat(all_patches, dim=-1)

    return combined_patches.view(-1, combined_patches.shape[-1])


print("--- 1. 특징 추출 및 메모리 뱅크 초기 구축 시작 ---")
memory_bank = []

with torch.no_grad(): # 특징 추출은 학습이 아님 (기울기 계산 불필요)
    for images, _ in tqdm(train_loader, desc="특징 추출"):
        images = images.to(device)
        
        # 1단계: 특징 추출
        # (B, C_l2, H_l2, W_l2), (B, C_l3, H_l3, W_l3)
        features = extractor(images) 
        
        # 2단계: 특징 패치화
        # (Total_patches_in_batch, C_total)
        batch_patches = extract_patches(features, PATCH_SIZE)
        
        # CPU로 옮기고 NumPy 배열로 변환 후 메모리 뱅크에 추가
        # memory_bank.append(batch_patches.cpu().numpy())
        memory_bank.append(batch_patches.detach())

# 전체 메모리 뱅크 결합
# memory_bank = np.concatenate(memory_bank, axis=0)
memory_bank = torch.cat(memory_bank, dim=0)
print(f"초기 메모리 뱅크 크기: {memory_bank.shape}") # (총 패치 개수, 특징 차원)


# --- 4. 코어셋 서브샘플링 (Core-Set Subsampling) ---
# 대규모 메모리 뱅크를 효율적으로 줄여 추론 속도와 메모리 사용량을 최적화

def get_coreset_subsampling_gpu(features, M, device):
    N, D = features.shape
    if M >= N:
        return features

    center_idx = torch.randint(0, N, (1,), device=device)
    centers = features[center_idx]
    min_distances = torch.cdist(features, centers).squeeze(1)
    selected_indices = [center_idx.item()]

    for _ in tqdm(range(1, M), desc="GPU 코어셋 선택"):
        new_center_idx = torch.argmax(min_distances)
        selected_indices.append(new_center_idx.item())
        new_center = features[new_center_idx].unsqueeze(0)
        new_distances = torch.cdist(features, new_center).squeeze(1)
        min_distances = torch.minimum(min_distances, new_distances)

    return features[selected_indices]

# 코어셋으로 줄일 개수 계산
M = int(len(memory_bank) * SUBSAMPLING_RATIO)
print(f"코어셋 서브샘플링 목표 개수: {M}")

# 코어셋 서브샘플링 실행
coreset_memory_bank = get_coreset_subsampling_gpu(memory_bank, M, device=device)
print(f"최종 코어셋 메모리 뱅크 크기: {coreset_memory_bank.shape}")

# 학습된 메모리 뱅크 저장
memory_bank_path = os.path.join(OUTPUT_DIR, MEMORY_BANK_NAME)
coreset_memory_bank_np = coreset_memory_bank.cpu().numpy()
np.save(memory_bank_path, coreset_memory_bank_np)
print(f"\n--- 학습 완료: 메모리 뱅크가 다음 경로에 저장되었습니다: {memory_bank_path} ---")