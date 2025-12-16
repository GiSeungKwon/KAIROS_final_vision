import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, random_split
import os
import copy
from tqdm import tqdm

# --- 1. 환경 설정 ---
BASE_DIR = 'C:/Dev/KAIROS_Project/data/tracking_ObjCls'
NUM_CLASSES = 3  # ESP32, L298N, MB102
NUM_EPOCHS = 50
SAVE_INTERVAL = 10  # 10 epoch마다 모델 저장
MODEL_SAVE_PATH = 'C:/Dev/KAIROS_Project/models/trck_ObjectClassification_models'
# ResNet은 일반적으로 224x224 이미지를 입력으로 사용합니다.
INPUT_SIZE = 224 

# 모델 저장 디렉터리 생성
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

# --- 2. 데이터셋 경로 및 변환 정의 ---

# 훈련에 사용할 이미지 변환 (Resize만 적용)
# Note: Augmentation은 이미 데이터 전처리 단계에서 완료되었으므로,
# 여기서는 Tensor 변환 및 Normalization만 수행합니다.
data_transforms = {
    # 훈련 및 검증 데이터에 동일하게 적용
    'all': transforms.Compose([
        # 이미지를 ResNet 입력 크기에 맞게 조정 (Resize)
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)), 
        # PyTorch Tensor로 변환
        transforms.ToTensor(),
        # ImageNet 표준 정규화 적용 (전이 학습에 필수)
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# 3개의 클래스가 포함된 루트 폴더
full_dataset = datasets.ImageFolder(
    root=BASE_DIR, 
    transform=data_transforms['all'] # 훈련/검증 모두 동일한 변환 적용
)

# 클래스 이름 확인 (순서대로 0, 1, 2 인덱스에 매핑됩니다)
print("클래스 매핑:", full_dataset.class_to_idx)
# 예: {'aug_Anomaly_ESP32': 0, 'aug_Anomaly_L298N': 1, 'aug_Anomaly_MB102': 2}


# --- 3. 데이터 분할 (Train/Validation) ---

# 전체 데이터셋 크기
dataset_size = len(full_dataset)

# 분할 비율 정의 (예: Train 80%, Validation 20%)
# 사용자 요청에 따라 비율은 조정 가능합니다.
train_ratio = 0.8
val_ratio = 1.0 - train_ratio

# 크기 계산
train_size = int(train_ratio * dataset_size)
val_size = dataset_size - train_size

# random_split을 사용하여 데이터셋 분할
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

print(f"총 데이터 수: {dataset_size}")
print(f"훈련 데이터 수: {len(train_dataset)}")
print(f"검증 데이터 수: {len(val_dataset)}")


# DataLoader 설정
BATCH_SIZE = 32 # 시스템 환경에 따라 조정 가능
dataloaders = {
    'train': DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4),
    'val': DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
}
dataset_sizes = {'train': len(train_dataset), 'val': len(val_dataset)}


# --- 4. 모델 설정 (ResNet-50) ---

# CUDA 사용 가능 여부 확인
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"사용 장치: {device}")

# ImageNet으로 사전 학습된 ResNet-50 모델 로드
model_ft = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

# 전이 학습을 위해 마지막 fully connected (fc) 레이어를 재정의
num_ftrs = model_ft.fc.in_features
# 최종 출력 클래스 수 (3개: ESP32, L298N, MB102)에 맞게 변경
model_ft.fc = nn.Linear(num_ftrs, NUM_CLASSES)

# 모델을 GPU로 이동
model_ft = model_ft.to(device)

# 손실 함수 및 옵티마이저 정의
criterion = nn.CrossEntropyLoss()
# AdamW는 Adam보다 가중치 감쇠(Weight Decay) 처리가 좋아 성능이 더 안정적입니다.
optimizer = optim.AdamW(model_ft.parameters(), lr=0.0001)

# 학습률 스케줄러 (성능 개선을 위해 10 에포크마다 학습률 10% 감소)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)


# --- 5. 학습 함수 정의 ---

def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    # 가장 좋은 성능의 모델 저장을 위한 변수 초기화
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        # 각 에포크는 훈련(train) 단계와 검증(val) 단계를 가집니다.
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # 모델을 훈련 모드로 설정
            else:
                model.eval()   # 모델을 평가 모드로 설정

            running_loss = 0.0
            running_corrects = 0

            # 데이터를 순회하며 학습/평가
            # tqdm으로 진행률 표시
            for inputs, labels in tqdm(dataloaders[phase], desc=f'{phase.upper()}'):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # 매개변수 기울기 0으로 초기화
                optimizer.zero_grad()

                # 순전파
                # 훈련 단계에서만 기울기 계산을 활성화 (torch.set_grad_enabled)
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # 역전파 + 옵티마이즈 (훈련 단계에서만)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # 통계 계산
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            # 에포크가 끝날 때 학습률 업데이트
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # 모델 저장 조건 확인
            if phase == 'val':
                # 1. 최고 성능 모델 저장
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    best_model_path = os.path.join(MODEL_SAVE_PATH, 'best_model.pth')
                    torch.save(model.state_dict(), best_model_path)
                    print(f"-> 🏆 최고 검증 정확도 {best_acc:.4f} 달성. 모델 저장됨: {best_model_path}")

                # 2. 10 에포크마다 모델 저장
                if (epoch + 1) % SAVE_INTERVAL == 0:
                    checkpoint_path = os.path.join(MODEL_SAVE_PATH, f'model_epoch_{epoch+1}.pth')
                    torch.save(model.state_dict(), checkpoint_path)
                    print(f"-> 💾 Checkpoint 저장됨 (Epoch {epoch+1}): {checkpoint_path}")

    print(f'\n--- 학습 완료 ---')
    print(f'최고 검증 정확도: {best_acc:.4f}')

    # 최종적으로 가장 좋은 가중치를 모델에 로드
    model.load_state_dict(best_model_wts)
    return model


# --- 6. 학습 실행 ---
if __name__ == '__main__':
    # 모델 학습 시작
    final_model = train_model(model_ft, criterion, optimizer, scheduler, num_epochs=NUM_EPOCHS)

    # 최종 모델 저장 (가장 좋은 성능의 가중치)
    final_save_path = os.path.join(MODEL_SAVE_PATH, 'final_best_model.pth')
    torch.save(final_model.state_dict(), final_save_path)
    print(f"\n✅ 최종 베스트 모델 저장 완료: {final_save_path}")