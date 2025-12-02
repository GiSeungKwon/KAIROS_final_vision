import torch
import torch.nn as nn
from torchvision import transforms
import cv2
import numpy as np
import os
import time

# =================================================================
# 1. 설정 변수 (학습 파라미터 및 ROI 설정)
# =================================================================
MODEL_PATH = 'ESP32_anomaly_detector_best_loss.pth' 
IMAGE_SIZE = 128 # 모델 학습 시 사용한 이미지 크기
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Autoencoder 모델 학습 시 사용한 정규화 파라미터
MOBILENET_MEAN = [0.485, 0.456, 0.406]
MOBILENET_STD = [0.229, 0.224, 0.225]

# ⭐⭐ 관심 영역 (Region of Interest, ROI) 설정 ⭐⭐
# 이 값들은 카메라 해상도 및 제품 위치에 맞게 조정해야 합니다.
# (X, Y)는 좌상단 좌표, W, H는 가로/세로 길이
ROI_X, ROI_Y = 100, 50 
ROI_W, ROI_H = 500, 400 
# -----------------------------------------------------------------

# =================================================================
# 2. 모델 정의 (학습 스크립트와 완전히 동일해야 함)
# =================================================================
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),   nn.ReLU(True), # 128 -> 64
            nn.Conv2d(16, 32, 3, stride=2, padding=1),  nn.ReLU(True), # 64 -> 32
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  nn.ReLU(True), # 32 -> 16
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(True)  # 16 -> 8
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

# =================================================================
# 3. 데이터 전처리 및 후처리 파이프라인
# =================================================================

preprocess = transforms.Compose([
    transforms.ToPILImage(),
    # ROI를 IMAGE_SIZE에 맞춰 리사이즈
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), 
    transforms.ToTensor(), 
    transforms.Normalize(mean=MOBILENET_MEAN, std=MOBILENET_STD) 
])

postprocess = transforms.Compose([
    transforms.Normalize(mean=[0.0, 0.0, 0.0], std=[1/s for s in MOBILENET_STD]),
    transforms.Normalize(mean=[-m for m in MOBILENET_MEAN], std=[1.0, 1.0, 1.0]),
    transforms.ToPILImage() 
])

# =================================================================
# 4. 모델 로드 함수 (이전과 동일)
# =================================================================
def load_model(model_path):
    model = Autoencoder().to(DEVICE)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model.eval()
        print(f"✅ 모델 로드 성공: {model_path}")
        return model
    else:
        print(f"❌ 오류: 모델 파일({model_path})을 찾을 수 없습니다. 학습을 먼저 실행하세요.")
        return None

# =================================================================
# 5. 실시간 ROI 기반 복원 테스트 함수
# =================================================================
def real_time_reconstruction_test(detector, cap):
    """
    카메라 스트리밍을 통해 ROI만 모델에 적용하고 결과를 시각화합니다.
    """
    print("\n--- 실시간 ROI 복원 테스트 시작 (Ctrl+C 또는 'q' 키로 종료) ---")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("프레임을 가져올 수 없습니다. 카메라 연결 상태를 확인하세요.")
            break

        # 원본 프레임 복사본 생성 (시각화용)
        display_frame = frame.copy()
        
        # 1. 원본 프레임에서 ROI 추출
        x1, y1 = ROI_X, ROI_Y
        x2, y2 = ROI_X + ROI_W, ROI_Y + ROI_H
        
        # ROI 영역이 화면을 벗어나지 않도록 클리핑 (방어 코드)
        H, W, _ = frame.shape
        x1 = max(0, min(x1, W))
        y1 = max(0, min(y1, H))
        x2 = max(0, min(x2, W))
        y2 = max(0, min(y2, H))

        # 실제 ROI 영역 자르기
        roi = frame[y1:y2, x1:x2]

        if roi.size == 0:
            cv2.putText(display_frame, "Error: Invalid ROI (Check coordinates)", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow("ESP32 Anomaly Detection", display_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break
            continue

        # 2. ROI 전처리 및 모델 추론
        frame_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        input_tensor = preprocess(frame_rgb).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            # 3. Autoencoder를 통한 복원 이미지 생성 (Normalized 텐서)
            output_tensor = detector(input_tensor)
            
            # 4. 재구성 오류 (MSE) 계산
            reconstruction_error = torch.mean((input_tensor - output_tensor) ** 2).item()

            # 5. 복원된 이미지 후처리 (역정규화 및 OpenCV BGR 변환)
            reconstructed_pil = postprocess(output_tensor.squeeze(0).cpu())
            
            # PIL Image (0-1) -> NumPy array (0-255) -> BGR
            reconstructed_roi = np.array(reconstructed_pil) * 255
            reconstructed_roi = reconstructed_roi.astype(np.uint8)
            reconstructed_roi = cv2.cvtColor(reconstructed_roi, cv2.COLOR_RGB2BGR)
            
            # 6. 복원된 ROI를 원본 크기(ROI_W, ROI_H)로 리사이즈
            reconstructed_roi_resized = cv2.resize(reconstructed_roi, (ROI_W, ROI_H))
            
            # 7. 복원된 ROI를 원본 프레임의 해당 위치에 삽입
            display_frame[y1:y2, x1:x2] = reconstructed_roi_resized

        # 8. 시각화 및 정보 표시
        
        # 원본 ROI 영역을 표시하기 위한 사각형 (빨간색)
        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(display_frame, "ROI (Processed)", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # 전체 화면에 결과 정보 표시
        cv2.putText(display_frame, f"MSE Loss (ROI): {reconstruction_error:.6f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(display_frame, "Press 'q' to quit", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.imshow("Real-time ROI Reconstruction Test", display_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# =================================================================
# 6. 메인 실행 블록
# =================================================================
if __name__ == '__main__':
    # 모델 로드
    detector = load_model(MODEL_PATH)
    if detector is None:
        exit()

    # 카메라 초기화 (1은 두 번째 카메라, 사용자 설정에 따름)
    cap = cv2.VideoCapture(1) 
    if not cap.isOpened():
        print("❌ 카메라를 열 수 없습니다. 카메라 인덱스를 확인하거나 연결을 점검하세요.")
        exit()
    
    # 테스트 실행
    real_time_reconstruction_test(detector, cap)