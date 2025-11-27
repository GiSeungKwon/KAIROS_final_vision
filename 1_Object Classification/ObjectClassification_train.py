import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms, models
import os
import random
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import glob

# --- 1. 설정 변수 ---
# 기본 경로 (상위 폴더)
DATA_ROOT = r"C:\Dev\KAIROS_Project\data"
# 학습에 사용할 폴더 리스트
CLASS_NAMES = ["Aug_ESP32", "Aug_L298N", "Aug_MB102"]
NUM_CLASSES = len(CLASS_NAMES)
# 하이퍼파라미터
BATCH_SIZE = 32
NUM_EPOCHS = 20
LEARNING_RATE = 0.001
# 장치 설정
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(DEVICE)
# MobileNetV3-Small 표준 Normalization 값 (ImageNet 기준)
MOBILENET_MEAN = [0.485, 0.456, 0.406]
MOBILENET_STD = [0.229, 0.224, 0.225]

# --- 2. 데이터셋 분할 및 전처리 ---

def prepare_data_loaders():
    """파일 경로를 수집하고, 학습/검증/테스트 세트로 분할하여 DataLoader를 생성합니다."""
    
    all_files = []
    all_labels = []
    
    # 1. 파일 목록 수집
    print("데이터셋 파일 목록 수집 중...")
    label_map = {name: i for i, name in enumerate(CLASS_NAMES)}

    for class_name in CLASS_NAMES:
        class_path = os.path.join(DATA_ROOT, class_name)
        file_list = glob.glob(os.path.join(class_path, "*.jpg"))
        
        if not file_list:
             print(f"[경고] {class_path}에 .jpg 파일이 없습니다.")
             continue
             
        label_index = label_map[class_name]
        all_files.extend(file_list)
        all_labels.extend([label_index] * len(file_list))
        print(f"  - {class_name}: {len(file_list)} 파일 수집")

    if not all_files:
        raise ValueError("수집된 이미지가 없습니다. 경로와 파일 확장자를 확인해주세요.")

    # 2. 학습(80%) : 검증(10%) : 테스트(10%)로 분할
    train_files, temp_files, train_labels, temp_labels = train_test_split(
        all_files, all_labels, test_size=0.2, random_state=42, stratify=all_labels
    )
    val_files, test_files, val_labels, test_labels = train_test_split(
        temp_files, temp_labels, test_size=0.5, random_state=42, stratify=temp_labels
    )
    
    print("-" * 30)
    print(f"총 데이터 수: {len(all_files)}")
    print(f"학습 데이터 수: {len(train_files)}")
    print(f"검증 데이터 수: {len(val_files)}")
    print(f"테스트 데이터 수: {len(test_files)}")
    print("-" * 30)

    # 3. 전처리 정의 (Normalization 포함)
    transform = transforms.Compose([
        # ImageNet 사전 학습 모델은 224x224를 기준으로 함.
        # 크롭된 이미지이므로 Resize만 적용합니다.
        transforms.Resize((224, 224)), 
        transforms.ToTensor(),
        # MobileNetV3에 맞는 정규화 적용 (필수)
        transforms.Normalize(mean=MOBILENET_MEAN, std=MOBILENET_STD)
    ])
    
    # 4. 커스텀 데이터셋 생성 (여기서는 간단히 리스트를 DataLoader에 전달하기 위한 구조)
    class CustomDataset(Dataset):
        def __init__(self, file_list, labels, transform=None):
            self.file_list = file_list
            self.labels = labels
            self.transform = transform
            
        def __len__(self):
            return len(self.file_list)
            
        def __getitem__(self, idx):
            img_path = self.file_list[idx]
            image = cv2.imread(img_path)
            # OpenCV는 BGR로 읽으므로 RGB로 변환
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # PIL Image로 변환하여 torchvision transform 적용
            from PIL import Image
            image = Image.fromarray(image)
            
            if self.transform:
                image = self.transform(image)
                
            return image, self.labels[idx]

    # 5. DataLoader 생성
    train_dataset = CustomDataset(train_files, train_labels, transform=transform)
    val_dataset = CustomDataset(val_files, val_labels, transform=transform)
    test_dataset = CustomDataset(test_files, test_labels, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
    
    return train_loader, val_loader, test_loader

# --- 3. 모델 정의 ---

def create_model(num_classes: int):
    """경량 MobileNetV3-Small 모델을 로드하고 출력층을 재정의합니다."""
    
    # MobileNetV3-Small 사전 학습 모델 로드
    model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
    
    # 최종 분류층(Classifier) 재정의
    # MobileNetV3는 마지막에 Sequential Classifier를 가집니다.
    in_features = model.classifier[-1].in_features
    
    # 새로운 최종 분류층 정의 (3가지 클래스에 맞게)
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    
    return model

# --- 4. 학습 함수 ---

def train_model(model, train_loader, val_loader):
    """모델 학습 및 검증을 수행합니다."""
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    model.to(DEVICE)
    best_accuracy = 0.0

    print("모델 학습 시작...")
    
    for epoch in range(NUM_EPOCHS):
        epoch_num = epoch + 1
        save_now = False

        # 1. 학습 단계
        model.train()
        running_loss = 0.0
        
        # tqdm을 활용하여 현재 학습 상황 시각화
        train_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{NUM_EPOCHS} [Train]', unit='batch')
        
        for inputs, labels in train_bar:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            train_bar.set_postfix({'Loss': f'{loss.item():.4f}'})

        epoch_loss = running_loss / len(train_loader.dataset)
        
        # 2. 검증 단계
        model.eval()
        corrects = 0
        total = 0
        val_loss = 0.0

        with torch.no_grad():
            val_bar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{NUM_EPOCHS} [Valid]', unit='batch')
            for inputs, labels in val_bar:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                total += labels.size(0)
                corrects += torch.sum(preds == labels.data)

        val_accuracy = corrects.double() / total
        val_loss = val_loss / len(val_loader.dataset)
        
        print(f"\n[Epoch {epoch+1}] Train Loss: {epoch_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_accuracy:.4f}")
        
        # 3. 최적 모델 저장

        # 1) 검증 정확도가 최고치를 갱신했을 때
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            model_save_path = f"best_mobilenetv3_classifier_e{epoch_num}_acc{best_accuracy:.4f}.pth"
            print(f"-> 🎉 최적 모델 갱신 및 저장 완료 (epoch: {epoch_num}, Accuracy: {best_accuracy:.4f})")
            save_now = True
        
        # 2) 매 5번째 에포크마다 저장 (최적 모델과 별도로 저장)
        if epoch_num % 5 == 0 and not save_now:
            model_save_path = f"checkpoint_mobilenetv3_classifier_e{epoch_num}_acc{val_accuracy:.4f}.pth"
            torch.save(model.state_dict(), model_save_path)
            print(f"-> 💾 체크포인트 모델 저장 완료 (epoch: {epoch_num}, Accuracy: {val_accuracy:.4f})")
            save_now = True
        
        # 3) 위 두 조건 중 하나에 해당하면 저장 실행
        if save_now:
            torch.save(model.state_dict(), model_save_path)
        
    return model

# --- 5. 테스트 함수 ---

def test_model(model, test_loader, best_acc):
    """최종 테스트 세트로 모델 성능을 평가합니다."""
    
    model_path = f"best_mobilenetv3_classifier_{best_acc:.4f}.pth"
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        print(f"\n최적 모델({model_path}) 로드 완료.")
    
    model.eval()
    corrects = 0
    total = 0

    with torch.no_grad():
        test_bar = tqdm(test_loader, desc='[Test]', unit='batch')
        for inputs, labels in test_bar:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            corrects += torch.sum(preds == labels.data)
            
    test_accuracy = corrects.double() / total
    print(f"\n=== 최종 테스트 결과 ===\nTest Accuracy: {test_accuracy:.4f} ({corrects}/{total})")
    print("========================")

# --- 6. 메인 실행 ---

if __name__ == "__main__":
    # OpenCV를 임포트하여 이미지 로딩에 사용
    try:
        import cv2
        print("OpenCV, PyTorch, tqdm 로딩 성공.")
    except ImportError:
        print("필요한 라이브러리(cv2, torch, tqdm)를 설치해주세요.")
        
    print(f"사용 장치: {DEVICE}")

    # 데이터 로더 준비
    train_loader, val_loader, test_loader = prepare_data_loaders()
    
    # 모델 생성
    model = create_model(NUM_CLASSES)
    print(f"사용 모델: MobileNetV3-Small (파라미터 수: {sum(p.numel() for p in model.parameters()):,})")
    
    # 학습 실행
    trained_model = train_model(model, train_loader, val_loader)
    
    # 최종 테스트 (최고 정확도를 가진 모델을 다시 로드하여 테스트)
    # train_model 함수에서 저장한 best_accuracy를 직접 전달해야 합니다.
    # 여기서는 간단히 0.8 이상일 경우 테스트를 진행하는 것으로 가정합니다.
    test_model(trained_model, test_loader, 0.8000) # 실제 저장된 파일명에 맞춰 정확도 수정 필요