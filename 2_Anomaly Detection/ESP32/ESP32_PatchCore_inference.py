import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from torchvision import transforms
from PIL import Image

# ==============================
# 설정
# ==============================
MODULE = "ESP32"
MEMORY_BANK_PATH = "patchcore_results/ESP32_patchcore_memory_bank.npy"

BACKBONE_MODEL = "resnet18"
FEATURE_LAYER_NAMES = ["layer2", "layer3"]

IMAGE_SIZE = 256
PATCH_SIZE = 3
PATCH_STRIDE = 8
NEIGHBOR_COUNT = 9

CAMERA_INDEX = 1
ANOMALY_THRESHOLD = 15.0  # ⚠️ 필요 시 조정

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==============================
# Feature Extractor
# ==============================
class FeatureExtractor(nn.Module):
    def __init__(self, backbone_name, feature_layer_names):
        super().__init__()
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
        feats = self.model(x)
        return [feats[i] for i in self.feature_layer_indices]


# ==============================
# Patch Extraction (학습 코드와 동일)
# ==============================
def extract_patches(features):
    H_max = max([f.shape[2] for f in features])
    W_max = max([f.shape[3] for f in features])

    aligned = []
    for feat in features:
        if feat.shape[2] != H_max or feat.shape[3] != W_max:
            feat = F.interpolate(
                feat,
                size=(H_max, W_max),
                mode="bilinear",
                align_corners=False
            )
        aligned.append(feat)

    patches_all = []
    for feat in aligned:
        B, C, H, W = feat.shape
        patches = feat.unfold(2, PATCH_SIZE, PATCH_STRIDE)\
                      .unfold(3, PATCH_SIZE, PATCH_STRIDE)
        patches = patches.permute(0, 2, 3, 1, 4, 5)
        patches = patches.contiguous().view(
            B, -1, C * PATCH_SIZE * PATCH_SIZE
        )
        patches_all.append(patches)

    combined = torch.cat(patches_all, dim=-1)
    return combined.view(-1, combined.shape[-1])


# ==============================
# kNN 기반 Anomaly Score
# ==============================
def compute_anomaly_score(patches, memory_bank):
    """
    patches: (N_patches, D)
    memory_bank: (N_memory, D)
    """
    distances = torch.cdist(patches, memory_bank)  # (P, M)
    knn_distances, _ = torch.topk(
        distances, k=NEIGHBOR_COUNT, largest=False, dim=1
    )
    patch_scores = knn_distances.mean(dim=1)
    image_score = patch_scores.max()  # PatchCore 방식
    return image_score.item()


# ==============================
# 전처리
# ==============================
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ==============================
# 모델 및 메모리 뱅크 로드
# ==============================
print("[INFO] Feature Extractor 로딩...")
extractor = FeatureExtractor(BACKBONE_MODEL, FEATURE_LAYER_NAMES).to(device)
extractor.eval()

print("[INFO] 메모리 뱅크 로딩...")
memory_bank = np.load(MEMORY_BANK_PATH)
memory_bank = torch.from_numpy(memory_bank).to(device)
print(f"[INFO] Memory Bank Shape: {memory_bank.shape}")

# ==============================
# 카메라 스트리밍
# ==============================
cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    raise RuntimeError("카메라를 열 수 없습니다.")

print("[INFO] 실시간 이상 탐지 시작 (ESC 종료)")

with torch.no_grad():
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # OpenCV → PIL
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        img_tensor = transform(img).unsqueeze(0).to(device)

        # Feature → Patch
        features = extractor(img_tensor)
        patches = extract_patches(features)

        # Anomaly Score
        score = compute_anomaly_score(patches, memory_bank)

        # 결과 표시
        status = "ANOMALY" if score > ANOMALY_THRESHOLD else "NORMAL"
        color = (0, 0, 255) if status == "ANOMALY" else (0, 255, 0)

        cv2.putText(
            frame,
            f"{status} | Score: {score:.2f}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            color,
            2
        )

        cv2.imshow("PatchCore ESP32 Anomaly Detection", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
