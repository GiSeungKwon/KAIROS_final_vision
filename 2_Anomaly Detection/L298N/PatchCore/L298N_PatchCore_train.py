import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import numpy as np
from sklearn.cluster import MiniBatchKMeans

MODULE = "L298N"

class ModuleDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.image_paths = list(Path(root_dir).glob("*.jpg")) + list(Path(root_dir).glob("*.png"))
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img

class PatchCoreDetector:
    def __init__(self, device="cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.model.to(self.device)
        self.model.eval()
        
        self.features = []
        def hook(module, input, output):
            self.features.append(output)
        
        # layer1, layer2, layer3의 마지막 출력을 모두 가져옵니다.
        self.model.layer1[-1].register_forward_hook(hook)
        self.model.layer2[-1].register_forward_hook(hook)
        self.model.layer3[-1].register_forward_hook(hook)
        
        self.memory_bank = None

    def embed(self, x):
        self.features = []
        with torch.no_grad():
            _ = self.model(x)
        
        # layer1, 2, 3 피처 맵 추출
        f1, f2, f3 = self.features
        
        # 가장 해상도가 높은 layer1의 크기에 맞춰 보간(Interpolation)
        target_size = (f1.shape[2], f1.shape[3])
        
        # Local smoothing (Average Pooling) 및 크기 맞추기
        f1 = F.avg_pool2d(f1, 3, 1, 1)
        f2 = F.interpolate(f2, size=target_size, mode="bilinear", align_corners=False)
        f2 = F.avg_pool2d(f2, 3, 1, 1)
        f3 = F.interpolate(f3, size=target_size, mode="bilinear", align_corners=False)
        f3 = F.avg_pool2d(f3, 3, 1, 1)
        
        # 특징 결합 (Concat)
        combined = torch.cat([f1, f2, f3], dim=1)
        # (B, C, H, W) -> (B*H*W, C)
        combined = combined.permute(0, 2, 3, 1).reshape(-1, combined.shape[1])
        return combined

    def build_memory_bank_safe(self, dataloader, coreset_size=5000):
        # MiniBatchKMeans를 미리 선언
        kmeans = MiniBatchKMeans(n_clusters=coreset_size, batch_size=2048, n_init=3, random_state=42)
        
        print(f"--- {MODULE} 특징 추출 및 점진적 학습 시작 ---")
        
        # 데이터를 한꺼번에 합치지 않고, 배치마다 partial_fit을 수행합니다.
        for batch in tqdm(dataloader, desc="Training K-Means"):
            batch = batch.to(self.device)
            emb = self.embed(batch).cpu().numpy()
            
            # 여기서 핵심: partial_fit을 사용해 메모리 누적을 방지합니다.
            kmeans.partial_fit(emb)
        
        # 학습된 중심점을 메모리 뱅크로 저장
        self.memory_bank = torch.from_numpy(kmeans.cluster_centers_).to(self.device)
        print(f"메모리 뱅크 구축 완료: {self.memory_bank.shape}")

if __name__ == "__main__":
    NORMAL_PATH = f"C:/Dev/KAIROS_Project/data/Anomaly_augmented/aug_Anomaly_{MODULE}"
    
    # 모델 저장 경로 확인 및 생성
    SAVE_DIR = Path(f"../../../../models/memoryBank_{MODULE}")
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = ModuleDataset(NORMAL_PATH, transform=transform)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False)

    detector = PatchCoreDetector()
    
    # coreset_size: 메모리 뱅크에 담을 대표 점의 개수 (5,000 ~ 10,000 사이 추천)
    detector.build_memory_bank_safe(dataloader, coreset_size=5000)

    # 저장
    torch.save(detector.memory_bank, SAVE_DIR / f"{MODULE}_memory_bank.pt")
    print(f"K-Means 기반 {MODULE} 모델 저장 완료!")