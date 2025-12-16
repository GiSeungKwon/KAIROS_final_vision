import os
import sys
import numpy as np
import cv2
from PIL import Image
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms

# --- 1. 설정 (학습 코드와 동일하게 유지) ---
NUM_CLASSES = 3  # ESP32, L298N, MB102
INPUT_SIZE = 224

# anomaly sensor에서 object classification
# MODEL_SAVE_PATH = 'C:/Dev/KAIROS_Project/models/ano_ObjectClassification_models'

# tracking sensor에서 object classification
MODEL_SAVE_PATH = 'C:/Dev/KAIROS_Project/models/trck_ObjectClassification_models'

MODEL_FILE_NAME = 'best_model.pth'
MODEL_PATH = os.path.join(MODEL_SAVE_PATH, MODEL_FILE_NAME)

# 클래스 인덱스와 이름 매핑 (학습 시의 순서를 따름)
# 데이터셋 폴더 이름을 기준으로 순서가 결정됩니다.
# 예시: 'aug_Anomaly_ESP32' -> 0, 'aug_Anomaly_L298N' -> 1, 'aug_Anomaly_MB102' -> 2
CLASS_NAMES = ['ESP32', 'L298N', 'MB102'] # 실제 데이터셋의 알파벳 순서에 맞게 조정하세요.
# 참고: 학습 코드의 출력 `print("클래스 매핑:", full_dataset.class_to_idx)`을 확인하여 정확한 순서를 적용해야 합니다.


# --- 2. 모델 로드 및 구조 재정의 ---

def load_classification_model(model_path, num_classes, device):
    print(f"모델 로드 중: {model_path}")
    
    # 1. ImageNet으로 사전 학습된 ResNet-50 구조를 로드
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    
    # 2. 마지막 Fully Connected (FC) 레이어를 재정의 (클래스 수 맞추기)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    # 3. 저장된 가중치(State Dictionary) 로드
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
    except FileNotFoundError:
        print(f"오류: 모델 가중치 파일({model_path})을 찾을 수 없습니다.")
        sys.exit(1)
    except Exception as e:
        print(f"오류: 모델 가중치 로드 중 문제가 발생했습니다: {e}")
        sys.exit(1)

    # 4. 모델 설정 및 장치 이동
    model = model.to(device)
    model.eval()  # 추론 모드 설정
    
    return model


# --- 3. 데이터 전처리 (추론용) ---

# 학습 코드의 'all' 변환과 동일해야 함
data_transform = transforms.Compose([
    transforms.ToPILImage(), # OpenCV BGR 배열을 PIL 이미지로 변환
    transforms.Resize((INPUT_SIZE, INPUT_SIZE)), 
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


# --- 4. 실시간 카메라 메인 루프 ---

def main():
    # CUDA 사용 가능 여부 확인
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"사용 장치: {device}")

    # 모델 로드
    model = load_classification_model(MODEL_PATH, NUM_CLASSES, device)

    # 카메라 초기화 (0은 일반적으로 기본 웹캠)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("오류: 카메라를 열 수 없습니다. 카메라 연결 상태를 확인하세요.")
        return

    # 스트리밍 및 분류 루프
    with torch.no_grad(): # 추론 시에는 기울기 계산 불필요
        while True:
            start_time = time.time() # FPS 측정을 위한 시작 시간

            # 프레임 읽기 (OpenCV는 BGR 포맷으로 읽음)
            ret, frame = cap.read()
            if not ret:
                print("오류: 카메라에서 프레임을 읽을 수 없습니다.")
                break

            # 1. 이미지 전처리
            original_frame = frame.copy() 
            input_tensor = data_transform(original_frame).unsqueeze(0).to(device)

            # 2. 모델 추론
            outputs = model(input_tensor)
            
            # 3. 결과 해석 (Softmax 및 예측 클래스)
            probabilities = F.softmax(outputs, dim=1).squeeze().cpu().numpy()
            
            # 가장 높은 확률의 클래스 인덱스
            predicted_index = np.argmax(probabilities)
            predicted_class = CLASS_NAMES[predicted_index]
            confidence = probabilities[predicted_index]

            # 4. 시각화 (스트리밍 창)
            
            # 예측 결과 텍스트 표시
            text_result = f"Prediction: {predicted_class} ({confidence:.2f})"
            color = (0, 255, 0) # 초록색
            
            # 결과 텍스트와 신뢰도 표시
            cv2.putText(frame, text_result, (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            # 모든 클래스의 확률을 막대 그래프 형태로 표시 (선택 사항)
            y_offset = 60
            for i, (name, prob) in enumerate(zip(CLASS_NAMES, probabilities)):
                prob_text = f"{name}: {prob:.2f}"
                cv2.putText(frame, prob_text, (10, y_offset + i * 25), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            # FPS 계산 및 표시
            end_time = time.time()
            fps = 1 / (end_time - start_time)
            cv2.putText(frame, f"FPS: {fps:.2f}", (10, frame.shape[0] - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # "스트리밍 창" 표시
            cv2.imshow("Real-Time Object Classification Test", frame)

            # 'q' 또는 ESC 키를 누르면 종료
            if cv2.waitKey(1) & 0xFF in [ord('q'), 27]:
                break

    # 자원 해제
    cap.release()
    cv2.destroyAllWindows()
    print("실시간 객체 분류 테스트 종료.")

if __name__ == '__main__':
    main()