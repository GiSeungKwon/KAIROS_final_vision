import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from tqdm import tqdm
import cv2
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

# --- 1. 경로 및 하이퍼파라미터 설정 ---
# 사용자가 제공한 경로
BASE_DIR = r"C:\Dev\KAIROS_Project\data\coordinate_tracking"
MODEL_PATH = r"C:\Dev\KAIROS_Project\Vision\3_Cordinate Detection\models"
CSV_PATH = os.path.join(BASE_DIR, "labeled_coordinates.csv") 
IMAGE_DIR = os.path.join(BASE_DIR, "coordinate_tracking_data")

# 학습 하이퍼파라미터
NUM_EPOCHS = 200
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
NUM_CLASSES = 17 # Class 0 ~ 16
RESIDUAL_WEIGHT = 5.0 # 잔차 회귀 손실 가중치 (lambda), 초기 1.0에서 조정 권장
SAVE_INTERVAL = 10 # 10 에포크마다 모델 저장

# 디바이스 설정
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


# --- 2. 데이터셋 정의 (Custom Dataset) ---

class RzMultiTaskDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_name = row['Image_Filename']
        img_path = os.path.join(self.img_dir, img_name)
        
        # 이미지 로드 (OpenCV 사용)
        img = cv2.imread(img_path)
        if img is None:
            # 파일이 없을 경우 예외 처리
            print(f"Warning: Image not found at {img_path}. Skipping.")
            return None 

        # BGR을 RGB로 변환
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 데이터 증강/전처리
        if self.transform:
            img = self.transform(img)
        
        # 레이블 준비
        # 분류 레이블 (Class Index C): Long Tensor (CrossEntropyLoss 요구 형식)
        class_label = torch.tensor(row['Class_Index_C'], dtype=torch.long)
        
        # 잔차 회귀 레이블 (Delta_Rz): Float Tensor
        residual_label = torch.tensor([row['Delta_Rz']], dtype=torch.float32)
        
        return img, class_label, residual_label


# --- 3. 모델 정의 (ResNet-50 기반) ---

class ResNetMultiTask(nn.Module):
    def __init__(self, num_classes):
        super(ResNetMultiTask, self).__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        
        # 마지막 Fully Connected 레이어 제거
        self.features = nn.Sequential(*(list(resnet.children())[:-2]))
        
        # Global Average Pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 분류 헤드 (Classification Head)
        self.cls_head = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes) # Softmax는 Loss 함수에 포함됨
        )
        
        # 잔차 회귀 헤드 (Residual Regression Head)
        self.res_head = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 1)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
        cls_output = self.cls_head(x)
        res_output = self.res_head(x)
        
        return cls_output, res_output


# --- 4. 데이터 전처리 및 증강 정의 ---

# PyTorch의 ResNet 표준 전처리 사용
data_transforms = {
    'train': transforms.Compose([
        # 데이터 증강 (로봇 제어 문제의 특성을 고려하여 제한적으로 적용)
        transforms.ToPILImage(),
        transforms.Resize(256),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # ImageNet 표준 정규화
    ]),
    'val': transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(256),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}


# --- 5. 학습 함수 정의 ---

def train_model():
    # 1. 데이터 로드 및 전처리
    try:
        # 이전에 생성한 labeled_coordinates.csv 파일을 로드합니다.
        data_df = pd.read_csv(CSV_PATH) 
    except FileNotFoundError:
        print(f"Error: Labeled CSV file not found at {CSV_PATH}. Please ensure it is created.")
        return

    # 데이터 분할 (80% 훈련, 20% 검증)
    train_df, val_df = train_test_split(data_df, test_size=0.2, random_state=42)
    
    # 2. 클래스 가중치 계산 (분류 손실에 사용)
    # Class_Index_C 열의 모든 고유 값을 사용하여 클래스 라벨 정의
    class_labels = data_df['Class_Index_C'].unique()
    class_labels.sort()
    
    # 클래스 가중치 계산 (scikit-learn 사용)
    class_weights = compute_class_weight(
        class_weight='balanced', 
        classes=class_labels, 
        y=train_df['Class_Index_C']
    )
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(DEVICE)
    print(f"Computed Class Weights: {class_weights}")

    # 3. DataLoader 준비
    train_dataset = RzMultiTaskDataset(train_df, IMAGE_DIR, data_transforms['train'])
    val_dataset = RzMultiTaskDataset(val_df, IMAGE_DIR, data_transforms['val'])
    
    # Collate fn: 데이터셋에서 None이 반환되는 경우를 처리합니다 (예: 이미지를 찾지 못한 경우)
    def custom_collate_fn(batch):
        batch = [item for item in batch if item is not None]
        if not batch:
            return None, None, None
        return torch.utils.data.dataloader.default_collate(batch)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
                              collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, 
                            collate_fn=custom_collate_fn)

    # 4. 모델, 손실 함수, 최적화 도구 설정
    model = ResNetMultiTask(NUM_CLASSES).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # 분류 손실: 가중치가 적용된 CrossEntropyLoss
    cls_criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    
    # 잔차 회귀 손실: Huber Loss (L1/L2의 장점을 합쳐 아웃라이어에 덜 민감함)
    res_criterion = nn.HuberLoss() 
    
    best_val_loss = float('inf')

    # 5. 학습 루프
    print("\n--- Starting Training ---")
    for epoch in range(1, NUM_EPOCHS + 1):
        # 훈련 단계
        model.train()
        train_bar = tqdm(train_loader, desc=f'Epoch {epoch}/{NUM_EPOCHS} [Train]', leave=False)
        total_train_loss = 0
        
        for imgs, classes, residuals in train_bar:
            if imgs is None: continue # Collate fn에서 None이 반환된 경우 스킵
            
            imgs, classes, residuals = imgs.to(DEVICE), classes.to(DEVICE), residuals.to(DEVICE)
            
            optimizer.zero_grad()
            cls_preds, res_preds = model(imgs)
            
            # 손실 계산
            loss_cls = cls_criterion(cls_preds, classes)
            loss_res = res_criterion(res_preds, residuals)
            
            # 결합 손실: 분류 손실 + 람다 * 잔차 손실
            total_loss = loss_cls + RESIDUAL_WEIGHT * loss_res
            
            total_loss.backward()
            optimizer.step()
            
            total_train_loss += total_loss.item()
            train_bar.set_postfix(
                cls_loss=f'{loss_cls.item():.4f}', 
                res_loss=f'{loss_res.item():.4f}'
            )

        avg_train_loss = total_train_loss / len(train_loader)
        
        # 검증 단계
        model.eval()
        total_val_loss = 0
        total_val_cls_loss = 0
        total_val_res_loss = 0
        
        with torch.no_grad():
            val_bar = tqdm(val_loader, desc=f'Epoch {epoch}/{NUM_EPOCHS} [Validation]', leave=False)
            for imgs, classes, residuals in val_bar:
                if imgs is None: continue

                imgs, classes, residuals = imgs.to(DEVICE), classes.to(DEVICE), residuals.to(DEVICE)
                
                cls_preds, res_preds = model(imgs)
                
                loss_cls = cls_criterion(cls_preds, classes)
                loss_res = res_criterion(res_preds, residuals)
                total_loss = loss_cls + RESIDUAL_WEIGHT * loss_res
                
                total_val_loss += total_loss.item()
                total_val_cls_loss += loss_cls.item()
                total_val_res_loss += loss_res.item()

            avg_val_loss = total_val_loss / len(val_loader)
            avg_val_cls_loss = total_val_cls_loss / len(val_loader)
            avg_val_res_loss = total_val_res_loss / len(val_loader)
            
            print(f"Epoch {epoch} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} "
                  f"(CLS: {avg_val_cls_loss:.4f}, RES: {avg_val_res_loss:.4f})")

        # 6. 모델 저장 로직

        # A. 최고 성능 모델 저장
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model_path = os.path.join(MODEL_PATH, 'best_multitask_model.pth')
            torch.save(model.state_dict(), model_path)
            print(f"  -> Model saved: New best validation loss {best_val_loss:.4f}")

        # B. 10 에포크마다 모델 저장
        if epoch % SAVE_INTERVAL == 0:
            model_path_interval = os.path.join(MODEL_PATH, f'multitask_model_epoch_{epoch}.pth')
            torch.save(model.state_dict(), model_path_interval)
            print(f"  -> Model saved at epoch {epoch}")

if __name__ == '__main__':
    train_model()