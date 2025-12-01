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
import matplotlib.pyplot as plt # Matplotlib 임포트

# =================================================================
# 1. 하이퍼파라미터 및 경로 설정
# =================================================================
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
NUM_EPOCHS = 200
IMAGE_SIZE = 128
DATA_DIR = "../../../data/ObjectClassification/Aug_ESP32"
MODEL_SAVE_PATH = 'ESP32_anomaly_detector_best_loss.pth' 
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

print(f"로드된 ESP32 정상 이미지 수: {len(dataset)}")

# =================================================================
# 4. 모델 정의 (Convolutional Autoencoder - Sigmoid 없음)
# =================================================================
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),   nn.ReLU(True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),  nn.ReLU(True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  nn.ReLU(True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(True)
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1), nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1), nn.ReLU(True),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1), nn.ReLU(True),
            nn.ConvTranspose2d(16, 3, 3, stride=2, padding=1, output_padding=1), 
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# 모델 인스턴스화, 손실 함수 및 옵티마이저 정의
model = Autoencoder().to(DEVICE)
criterion = nn.MSELoss() 
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# =================================================================
# 5. 모델 학습 루프 (손실 기록 및 저장 로직 추가)
# =================================================================
def train_model(model, dataloader, criterion, optimizer, num_epochs):
    print("--- 모델 학습 시작 ---")
    start_time = time.time()
    
    # ⭐⭐ 에포크별 손실 기록을 위한 리스트 초기화 ⭐⭐
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
        # ⭐⭐ 에포크 평균 손실 기록 ⭐⭐
        epoch_losses.append(epoch_loss)
        
        print(f"\n[Epoch {epoch+1}/{num_epochs}] 평균 Loss: {epoch_loss:.6f}")
        
        # 최적 모델 저장 로직
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"-> 최적 모델 업데이트 완료! (Loss: {best_loss:.6f}로 감소)")

    end_time = time.time()
    print(f"\n--- 모델 학습 완료. 총 소요 시간: {end_time - start_time:.2f}초 ---")
    print(f"최종 저장된 모델: {MODEL_SAVE_PATH} (최소 Loss: {best_loss:.6f})")

    # ⭐⭐ Loss 그래프 시각화 함수 호출 ⭐⭐
    plot_loss_curve(epoch_losses, NUM_EPOCHS, best_loss)
    
# =================================================================
# 6. Loss 그래프 시각화 함수
# =================================================================
def plot_loss_curve(losses, num_epochs, best_loss):
    """
    학습 과정의 손실 변화를 그래프로 시각화합니다.
    """
    epochs = range(1, num_epochs + 1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, losses, 'b-', label='Training Loss (MSE)')
    
    # 최적 손실 라인 추가
    min_epoch = losses.index(best_loss) + 1
    plt.axvline(x=min_epoch, color='r', linestyle='--', label=f'Best Model Saved (Epoch {min_epoch})')
    plt.text(min_epoch, max(losses), f'Min Loss: {best_loss:.6f}', color='r', rotation=0, ha='center', va='bottom')
    
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
    # Matplotlib이 설치되어 있는지 확인
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Matplotlib 라이브러리가 설치되어 있지 않습니다. pip install matplotlib 명령으로 설치하세요.")
        exit()
        
    train_model(model, dataloader, criterion, optimizer, NUM_EPOCHS)