import torch
import cv2
import numpy as np
from torchvision import transforms
from PIL import Image

# --- 1. 설정 변수 (학습 스크립트와 동일하게 설정) ---
CLASS_NAMES = ["Aug_ESP32", "Aug_L298N", "Aug_MB102"]
NUM_CLASSES = len(CLASS_NAMES)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MOBILENET_MEAN = [0.485, 0.456, 0.406]
MOBILENET_STD = [0.229, 0.224, 0.225]

# --- 2. 모델 및 전처리 로드 ---

def load_model(model_path, num_classes):
    """학습된 모델 가중치를 로드하고 평가 모드로 설정합니다."""
    # models.mobilenet_v3_small 함수를 재사용 (학습 코드의 create_model 함수 내용)
    model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v3_small', weights=None)
    
    # 최종 분류층(Classifier) 재정의
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = torch.nn.Linear(in_features, num_classes)
    
    # 저장된 가중치 로드
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval() # 평가 모드 설정
    print(f"모델 로드 완료: {model_path} ({DEVICE})")
    return model

# 이미지 전처리 (학습 시 사용한 정규화/리사이즈와 동일해야 함)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=MOBILENET_MEAN, std=MOBILENET_STD)
])

# --- 3. 실시간 추론 함수 ---

def inference_stream(model, transform):
    """실시간 웹캠 스트리밍에서 객체 분류를 수행하고 신뢰도를 표시합니다."""
    
    # 0번 카메라 (웹캠) 캡처 시작
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("🔴 오류: 웹캠을 열 수 없습니다. 카메라 인덱스를 확인하세요.")
        return

    print("🟢 실시간 스트리밍 시작. 'q'를 눌러 종료하세요.")
    
    with torch.no_grad(): # 추론 시에는 기울기 계산을 비활성화
        while True:
            # 1. 프레임 읽기
            ret, frame = cap.read()
            if not ret:
                break
                
            # 2. 전처리
            # OpenCV (BGR) -> RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # NumPy 배열 -> PIL Image -> Tensor로 변환 및 정규화
            pil_image = Image.fromarray(rgb_frame)
            input_tensor = transform(pil_image).unsqueeze(0).to(DEVICE) # (1, C, H, W) 형태로 변환
            
            # 3. 모델 추론
            outputs = model(input_tensor)
            
            # 4. 결과 해석
            # 로짓(Logits)을 확률(Probabilities)로 변환
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            
            # 가장 높은 신뢰도와 해당 클래스 인덱스 찾기
            conf_score, predicted_idx = torch.max(probabilities, 1)
            
            # 결과 값 추출 (Tensor -> Python Value)
            predicted_class = CLASS_NAMES[predicted_idx.item()]
            confidence = conf_score.item()
            
            # 5. 결과 시각화 (OpenCV)
            text = f"Class: {predicted_class}"
            confidence_text = f"Confidence: {confidence*100:.2f}%"
            
            # 결과 텍스트 표시
            cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, confidence_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)

            # 프레임 표시
            cv2.imshow('Real-time Object Classification', frame)
            
            # 'q' 또는 ESC 키를 누르면 종료
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # 캡처 및 창 해제
    cap.release()
    cv2.destroyAllWindows()

# --- 4. 메인 실행 ---
if __name__ == "__main__":
    # ⚠️ 테스트할 모델 파일 경로를 여기에 정확히 지정해야 합니다!
    # 예시: 학습 중 저장된 'best' 모델 파일
    MODEL_WEIGHTS_PATH = "best_mobilenetv3_classifier_eX_accY.pth" 
    
    # ⚠️ 학습 코드에서 최고 정확도로 저장된 실제 파일 이름으로 변경해주세요.
    # 예: "best_mobilenetv3_classifier_e18_acc0.9870.pth"
    
    try:
        # 모델 로드
        loaded_model = load_model(MODEL_WEIGHTS_PATH, NUM_CLASSES)
        
        # 스트리밍 시작
        inference_stream(loaded_model, transform)
        
    except FileNotFoundError:
        print(f"\n❌ 오류: 모델 가중치 파일({MODEL_WEIGHTS_PATH})을 찾을 수 없습니다.")
        print("   -> 파일 경로와 이름을 확인하고, 학습이 완료되었는지 확인하세요.")
    except Exception as e:
        print(f"\n❌ 예기치 않은 오류 발생: {e}")