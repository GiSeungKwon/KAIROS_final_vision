import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import numpy as np
import cv2
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import roc_auc_score

MODULE = "ESP32"

class ModuleDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.image_paths = list(Path(root_dir).glob("*.jpg")) + list(Path(root_dir).glob("*.png"))
        self.transform = transform
        # HSV 설정값 (HSV_Crop.py에서 검증된 값)
        self.lower = np.array([0, 0, 70])
        self.upper = np.array([179, 255, 255])
        self.kernel = np.ones((5, 5), np.uint8)

    def apply_hsv_rotated_crop(self, img_bgr):
        """이미지를 읽어 보드를 찾고 수평으로 정렬하여 Crop 합니다."""
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower, self.upper)
        mask_cleaned = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel, iterations=2)
        contours, _ = cv2.findContours(mask_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            largest_cnt = max(contours, key=cv2.contourArea)
            rect = cv2.minAreaRect(largest_cnt)
            box = cv2.boxPoints(rect)
            box = np.int64(box)

            width, height = int(rect[1][0]), int(rect[1][1])
            src_pts = box.astype("float32")
            dst_pts = np.array([[0, height-1], [0, 0], [width-1, 0], [width-1, height-1]], dtype="float32")
            
            M = cv2.getPerspectiveTransform(src_pts, dst_pts)
            warped = cv2.warpPerspective(img_bgr, M, (width, height))

            if width < height:
                warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)
            return cv2.resize(warped, (224, 224))
        
        # 보드를 못 찾으면 중앙 Crop 등으로 대체
        return cv2.resize(img_bgr, (224, 224))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # OpenCV로 읽기
        img_bgr = cv2.imread(str(self.image_paths[idx]))
        # HSV Rotated Crop 적용
        cropped_img = self.apply_hsv_rotated_crop(img_bgr)
        # BGR -> RGB 변환 후 PIL 이미지로 변경 (Transform 적용을 위해)
        img_rgb = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        
        if self.transform:
            img_pil = self.transform(img_pil)
        return img_pil

class PatchCoreDetector:
    def __init__(self, device="cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.model.to(self.device).eval()
        
        self.features = []
        def hook(m, i, o): self.features.append(o)
        
        # layer1, 2 출력을 위주로 사용 (Anomaly Detection에서 가장 효과적)
        self.model.layer1[-1].register_forward_hook(hook)
        self.model.layer2[-1].register_forward_hook(hook)
        self.memory_bank = None

    def embed(self, x):
        self.features = []
        with torch.no_grad(): _ = self.model(x)
        
        f1, f2 = self.features
        target_size = (f1.shape[2], f1.shape[3])
        
        # Gaussian mask 대신 정렬된 보드이므로 더 균일한 패치 분석 가능
        f1 = F.avg_pool2d(f1, 3, 1, 1)
        f2 = F.interpolate(f2, size=target_size, mode="bilinear", align_corners=False)
        f2 = F.avg_pool2d(f2, 3, 1, 1)
        
        combined = torch.cat([f1, f2], dim=1)
        return combined.permute(0, 2, 3, 1).reshape(-1, combined.shape[1])

    def build_memory_bank_safe(self, dataloader, coreset_size=8000):
        kmeans = MiniBatchKMeans(n_clusters=coreset_size, batch_size=2048, n_init=3, random_state=42)
        for batch in tqdm(dataloader, desc="Extracting Features"):
            batch = batch.to(self.device)
            emb = self.embed(batch).cpu().numpy()
            kmeans.partial_fit(emb)
        self.memory_bank = torch.from_numpy(kmeans.cluster_centers_).to(self.device)

    def predict(self, dataloader):
        """이상 점수를 계산합니다."""
        scores = []
        for batch in tqdm(dataloader, desc="Scoring"):
            batch = batch.to(self.device)
            emb = self.embed(batch)
            # Memory Bank와의 최단 거리 계산 (Anomaly Score)
            dist = torch.cdist(emb, self.memory_bank, p=2)
            score = dist.min(dim=1)[0].max().item() # 가장 큰 패치 점수를 이미지 점수로 사용
            scores.append(score)
        return scores

if __name__ == "__main__":
    # 경로 설정
    NORMAL_PATH = f"C:/Dev/KAIROS_Project/data/Anomaly_augmented/aug_Anomaly_{MODULE}"
    ANOMALY_PATH = f"C:/Dev/KAIROS_Project/data/Anomaly_augmented/Anomaly_img_{MODULE}"
    SAVE_DIR = Path(f"../../../../models/memoryBank_{MODULE}")
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 1. 학습
    train_loader = DataLoader(ModuleDataset(NORMAL_PATH, transform), batch_size=8, shuffle=False)
    detector = PatchCoreDetector()
    detector.build_memory_bank_safe(train_loader, coreset_size=8000)
    torch.save(detector.memory_bank, SAVE_DIR / f"{MODULE}_memory_bank.pt")

    # 2. 검증 (Validation)
    print("--- 합성 이상 데이터를 이용한 성능 검증 시작 ---")
    # 정상을 정상으로 분류하는지 확인용
    test_normal_loader = DataLoader(ModuleDataset(NORMAL_PATH, transform), batch_size=1)
    # 합성이상을 이상으로 분류하는지 확인용
    test_anomaly_loader = DataLoader(ModuleDataset(ANOMALY_PATH, transform), batch_size=1)

    normal_scores = detector.predict(test_normal_loader)
    anomaly_scores = detector.predict(test_anomaly_loader)

    # AUC 계산 (성능 지표)
    labels = [0] * len(normal_scores) + [1] * len(anomaly_scores)
    preds = normal_scores + anomaly_scores
    auc = roc_auc_score(labels, preds)
    print(f"📊 Validation AUC: {auc:.4f}")