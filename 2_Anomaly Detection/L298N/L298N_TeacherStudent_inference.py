import os
import yaml
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image

# --------------------------------------------------
# Config (훈련 코드와 동일)
# --------------------------------------------------
CONFIG_YAML_PATH = "L298N_config.yaml"
# 학습된 모델의 경로 (최고 성능 모델을 사용한다고 가정)
STUDENT_MODEL_PATH = "output/student_kd_best.pth" 


def load_config(path):
    """YAML 설정 파일을 로드합니다."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# --------------------------------------------------
# Teacher (ResNet18, layer3) (훈련 코드와 동일)
# --------------------------------------------------
class TeacherNet(nn.Module):
    def __init__(self):
        super().__init__()
        # NOTE: pretrained=True는 ImageNet weights를 사용함을 의미합니다.
        self.model = models.resnet18(pretrained=True)
        self.features = {}

        def hook_fn(name):
            def fn(_, __, output):
                # .clone()을 사용하여 features 딕셔너리에 복사본을 저장합니다.
                # (일반적으로 추론 시에는 불필요하지만, 안전을 위해 유지)
                self.features[name] = output.clone() 
            return fn

        # 훈련 시와 동일하게 layer3에만 hook을 걸어 피처를 추출합니다.
        self.model.layer3.register_forward_hook(hook_fn("layer3"))

    def forward(self, x):
        _ = self.model(x)
        return self.features["layer3"]  # [B, 256, 16, 16]


# --------------------------------------------------
# Student (MobileNetV2 기반) (훈련 코드와 동일)
# --------------------------------------------------
class StudentNet(nn.Module):
    def __init__(self, width_mult=0.5):
        super().__init__()
        backbone = models.mobilenet_v2(
            pretrained=False,
            width_mult=width_mult
        )

        # 훈련 코드에서 :14까지 사용했으므로 동일하게 설정
        self.features = backbone.features[:14]

        # 훈련 코드에서 수행했던 채널 수 계산 로직은 로드 시 불필요하나,
        # 모델의 __init__ 구조를 훈련과 동일하게 유지해야 정확한 weight 로드가 가능합니다.
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 256, 256)
            # MobileNetV2의 기본 입력 크기가 224x224이지만, 여기서는 256x256에 맞춰 진행
            c = self.features(dummy).shape[1] 

        # 훈련 코드와 동일하게 채널 수를 256으로 맞추는 1x1 Conv 레이어
        self.proj = nn.Conv2d(c, 256, kernel_size=1)

    def forward(self, x):
        feat = self.features(x)
        feat = self.proj(feat)
        return feat


def run_anomaly_detection_stream():
    """웹캠 스트리밍을 통해 KD 기반 이상 탐지를 수행합니다."""
    
    # 1. 설정 로드
    config = load_config(CONFIG_YAML_PATH)
    device = torch.device(
        "cuda" if torch.cuda.is_available() and config["device"] == "cuda" else "cpu"
    )
    
    # 2. 이미지 변환 (훈련 시와 동일한 정규화, 크기 조정)
    # PIL Image를 입력으로 받도록 합니다.
    transform = transforms.Compose([
        transforms.Resize((config["image_size"], config["image_size"])),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    # 3. 모델 로드 및 준비
    print("\n--- KD Anomaly Detection 모델 로드 중 ---")
    teacher = TeacherNet().to(device)
    student = StudentNet(config["student_width_mult"]).to(device)
    
    # 학습된 Student 모델 가중치 로드
    if not os.path.exists(STUDENT_MODEL_PATH):
        print(f"ERROR: 학습된 모델 파일이 없습니다! 경로: {STUDENT_MODEL_PATH}")
        return

    student.load_state_dict(torch.load(STUDENT_MODEL_PATH, map_location=device))
    
    # Teacher와 Student 모두 추론 모드 (eval)로 설정
    teacher.eval()
    student.eval()
    
    print(f"모델 로드 완료: {STUDENT_MODEL_PATH}")
    print(f"이상 탐지 임계값 (예상): {config.get('anomaly_threshold', '설정되지 않음')} (yaml 파일에서 확인)")
    
    # 4. 웹캠 설정
    cap = cv2.VideoCapture(1) # 0은 보통 기본 웹캠을 의미합니다.
    if not cap.isOpened():
        print("ERROR: 웹캠을 열 수 없습니다.")
        return

    print("\n--- 실시간 이상 탐지 스트리밍 시작 (Q 키를 눌러 종료) ---")

    try:
        while True:
            # 프레임 읽기
            ret, frame = cap.read()
            if not ret:
                print("스트림에서 프레임을 읽을 수 없습니다. 종료합니다.")
                break

            # BGR to RGB (OpenCV 기본)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # NumPy 배열을 PIL Image로 변환 (transforms.Compose를 사용하기 위함)
            pil_image = Image.fromarray(rgb_frame)
            
            # 이미지 전처리 (훈련 시와 동일)
            # [C, H, W] 텐서로 변환
            input_tensor = transform(pil_image).unsqueeze(0).to(device) 

            # 5. 이상 탐지 로직 (KD Loss 계산)
            with torch.no_grad():
                # Teacher/Student 피처 추출
                teacher_feat = teacher(input_tensor)
                student_feat = student(input_tensor)
                
                # Resizing (훈련 코드와 동일한 보간법)
                student_feat = F.interpolate(
                    student_feat,
                    size=teacher_feat.shape[2:],
                    mode="bilinear",
                    align_corners=False
                )
                
                # MSE Loss 계산 (지식 증류 Loss == 이상 점수)
                # kd_loss는 배치 차원 [1, ] 스칼라 텐서입니다.
                kd_loss = F.mse_loss(student_feat, teacher_feat, reduction='mean') 
                
                # 이상 점수 (Anomaly Score)
                # 훈련 시 적용된 kd_loss_weight를 곱하여 최종 점수로 사용
                anomaly_score = kd_loss.item() * config.get("kd_loss_weight", 1.0)
                
                # 이상 판단 (yaml 파일에 threshold가 설정되어 있어야 합니다)
                threshold = config.get("anomaly_threshold", 0.08) # 기본값 0.08 사용
                is_anomaly = anomaly_score > threshold
                
                # 6. 결과 시각화
                display_text = f"Anomaly Score: {anomaly_score:.6f}"
                color = (0, 255, 0) # Green (정상)
                
                if is_anomaly:
                    display_text += " | ANOMALY DETECTED!"
                    color = (0, 0, 255) # Red (이상)
                else:
                    display_text += " | Normal"
                    
                # 화면에 텍스트 표시 (OpenCV 이미지 사용)
                cv2.putText(frame, display_text, (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
                cv2.putText(frame, f"Threshold: {threshold:.4f}", (10, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA) # Yellow
                
                # 결과 창에 표시
                cv2.imshow("Real-time Anomaly Detection (KD) L298N", frame)

            # 'q' 키를 누르면 종료
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        # 종료 시 웹캠 및 창 해제
        cap.release()
        cv2.destroyAllWindows()
        print("\n--- 스트리밍 종료 ---")


if __name__ == "__main__":
    run_anomaly_detection_stream()