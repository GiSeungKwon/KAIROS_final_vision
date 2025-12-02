import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import glob
import time
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

# =================================================================
# 1. 하이퍼파라미터 및 경로 설정
# =================================================================
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
NUM_EPOCHS = 20
IMAGE_SIZE = 128
DATA_DIR = "../../../data/ObjectClassification/Aug_MB102"
# 최적 모델 저장 경로 (기존 유지)
MODEL_SAVE_PATH_BEST = 'MB102_anomaly_detector_best_loss.pth' 
# 주기적 백업 모델 저장 경로 (새로 추가, 포맷 문자열로 사용)
MODEL_SAVE_PATH_PERIODIC = 'MB102_anomaly_detector_epoch_{:03d}.pth' 
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# ⭐⭐ 주기적 저장 간격 설정 ⭐⭐
SAVE_INTERVAL_EPOCH = 5

print(f"사용 장치: {DEVICE}")

# =================================================================
# 2. 커스텀 데이터셋 정의 (변경 없음)
# =================================================================
class CustomImageDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.image_paths = glob.glob(os.path.join(data_dir, '*.jpg')) 
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
        
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image

# =================================================================
# 3. 데이터 변환 및 DataLoader 설정 (변경 없음)
# =================================================================
data_transforms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
])

dataset = CustomImageDataset(DATA_DIR, transform=data_transforms)

if len(dataset) == 0:
    print(f"\n[오류 발생] 데이터셋에 이미지가 0개입니다. 경로({DATA_DIR})를 확인해주세요.")
    exit() 

dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)

print(f"로드된 MB102 정상 이미지 수: {len(dataset)}")

# =================================================================
# 4. 모델 정의 (Convolutional Autoencoder - 변경 없음)
# =================================================================
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),   nn.ReLU(True), # 128 -> 64
            nn.Conv2d(16, 32, 3, stride=2, padding=1),  nn.ReLU(True), # 64 -> 32
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  nn.ReLU(True), # 32 -> 16
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(True)  # 16 -> 8 (Latent Space: 128 x 8 x 8)
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1), nn.ReLU(True), # 8 -> 16
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1), nn.ReLU(True),  # 16 -> 32
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1), nn.ReLU(True),  # 32 -> 64
            nn.ConvTranspose2d(16, 3, 3, stride=2, padding=1, output_padding=1),                  # 64 -> 128 (Output: 3 x 128 x 128)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# 모델 인스턴스화, 손실 함수 및 옵티마이저 정의 (변경 없음)
model = Autoencoder().to(DEVICE)
criterion = nn.MSELoss() 
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# =================================================================
# 5. 모델 학습 루프 (수정됨: 5 에포크 주기 저장 로직 추가)
# =================================================================
def train_model(model, dataloader, criterion, optimizer, num_epochs, best_save_path, periodic_save_path_format, save_interval):
    print("--- 모델 학습 시작 ---")
    start_time = time.time()
    
    epoch_losses = []
    best_loss = np.inf 
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        train_bar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{num_epochs}', unit='batch')
        
        for inputs in train_bar:
            inputs = inputs.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            train_bar.set_postfix({'Loss': f'{loss.item():.6f}'})

        epoch_loss = running_loss / len(dataset)
        epoch_losses.append(epoch_loss)
        
        print(f"\n[Epoch {epoch+1}/{num_epochs}] 평균 Loss: {epoch_loss:.6f}")
        
        # 1. 최적 모델 저장 로직 (Loss 개선 시)
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), best_save_path)
            print(f"-> ⭐ 최적 모델 업데이트 완료! ({best_save_path}에 저장됨, Loss: {best_loss:.6f}로 감소)")

        # 2. ⭐⭐ 5 에포크 주기적 백업 저장 로직 추가 ⭐⭐
        if (epoch + 1) % save_interval == 0:
            periodic_path = periodic_save_path_format.format(epoch + 1)
            torch.save(model.state_dict(), periodic_path)
            print(f"-> 💾 주기적 백업 완료! ({epoch+1} 에포크, {periodic_path}에 저장됨)")

    end_time = time.time()
    print(f"\n--- 모델 학습 완료. 총 소요 시간: {end_time - start_time:.2f}초 ---")
    print(f"최종 저장된 최적 모델: {best_save_path} (최소 Loss: {best_loss:.6f})")

    # Loss 그래프 시각화 함수 호출
    plot_loss_curve(epoch_losses, num_epochs, best_loss)
    
# =================================================================
# 6. Loss 그래프 시각화 함수 (변경 없음)
# =================================================================
def plot_loss_curve(losses, num_epochs, best_loss):
    """
    학습 과정의 손실 변화를 그래프로 시각화합니다.
    """
    epochs = range(1, num_epochs + 1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, losses, 'b-', label='Training Loss (MSE)')
    
    # 최적 손실 라인 추가
    min_loss_index = np.argmin(losses)
    min_epoch = min_loss_index + 1
    
    # 최적 모델 저장 라인 (min_epoch 위치에 표시)
    plt.axvline(x=min_epoch, color='r', linestyle='--', label=f'Best Model Saved (Epoch {min_epoch})')
    # 텍스트가 그래프 위에 겹치지 않도록 조정
    text_y_position = max(losses) - (max(losses) - min(losses)) * 0.1 
    plt.text(min_epoch, text_y_position, f'Min Loss: {best_loss:.6f}', color='r', rotation=0, ha='center', va='bottom', fontsize=9)
    
    plt.title('Autoencoder Training Loss Convergence')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Squared Error (MSE) Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()

# =================================================================
# 7. 메인 실행 블록
# =================================================================
if __name__ == '__main__':
    # Matplotlib이 설치되어 있는지 확인 (변경 없음)
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Matplotlib 라이브러리가 설치되어 있지 않습니다. pip install matplotlib 명령으로 설치하세요.")
        exit()
        
    train_model(
        model, 
        dataloader, 
        criterion, 
        optimizer, 
        NUM_EPOCHS,
        MODEL_SAVE_PATH_BEST,         # 최적 손실 모델 저장 경로
        MODEL_SAVE_PATH_PERIODIC,     # 주기적 백업 모델 파일명 포맷
        SAVE_INTERVAL_EPOCH           # 주기적 저장 간격 (5)
    )