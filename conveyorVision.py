import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import cv2
import numpy as np
import os

# =================================================================
# 1. 시스템 설정 및 하이퍼파라미터
# =================================================================

# 공통 설정
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MOBILENET_MEAN = [0.485, 0.456, 0.406]
MOBILENET_STD = [0.229, 0.224, 0.225]
CAMERA_INDEX = 1 # 웹캠 인덱스 (필요에 따라 0으로 변경)

# Object Classification 설정
CLASS_NAMES = ["ESP32", "L298N", "MB102"] # L298N(Motor), MB102(Power)에서 이름 단순화
NUM_CLASSES = len(CLASS_NAMES)
# ⚠️ Classification 모델 경로 설정 (사용자 파일 경로에 맞게 변경)
<<<<<<< HEAD
CLASSIFIER_WEIGHTS_PATH = "../models/ano_ObjectClassification_models/best_model.pthg"
=======
CLASSIFIER_WEIGHTS_PATH = "1_Object Classification/checkpoint_mobilenetv3_classifier_e5_acc1.0000.pth"
>>>>>>> 3a42c12ca734b2603a24907d2402bf37765c68e7
CLASSIFIER_IMAGE_SIZE = 224

# Anomaly Detection 설정
AD_IMAGE_SIZE = 128 # AD 모델 학습 시 사용한 이미지 크기
# ⚠️ AD 모델 경로 매핑 (보드 이름과 파일 이름 매핑)
AD_MODEL_PATHS = {
    "ESP32": "2_Anomaly Detection/ESP32/ESP32_anomaly_detector_best_loss.pth",
    "L298N": "2_Anomaly Detection/L298N/L298N_anomaly_detector_best_loss.pth",
    "MB102": "2_Anomaly Detection/MB102/MB102_anomaly_detector_best_loss.pth",
}
# ⚠️ 임계값 설정 (각 보드별 통계 분석 후 설정된 값 사용)
# 임시로 낮은 값을 사용하며, 실제 사용 시에는 통계적으로 재설정해야 합니다.
AD_THRESHOLDS = {
    "ESP32": 0.045,
    "L298N": 0.045,
    "MB102": 0.060,
}

# ROI 설정 (Anomaly Detection 시 사용할 관심 영역 - 모든 보드에 공통 적용 가정)
# 이 값들은 카메라 해상도 및 제품 위치에 맞게 조정해야 합니다.
ROI_X, ROI_Y = 100, 50 
ROI_W, ROI_H = 500, 400 

# =================================================================
# 2. 모델 아키텍처 정의
# =================================================================

# 2.1. Object Classification 모델 아키텍처 (MobileNetV3 Small)
def create_classifier_model(num_classes):
    """Classification 모델 아키텍처를 정의합니다."""
    # 로드 방식은 사용자님의 원본 코드와 동일하게 유지
    model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v3_small', weights=None)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = torch.nn.Linear(in_features, num_classes)
    return model

# 2.2. Anomaly Detection 모델 아키텍처 (Autoencoder)
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        # 인코더 및 디코더 정의 (학습 코드와 완전히 동일해야 함)
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
        return self.decoder(self.encoder(x))

# =================================================================
# 3. 모델 로드 및 전처리 정의
# =================================================================

# 3.1. Classification 모델 로드
def load_classifier(model_path, num_classes):
    model = create_classifier_model(num_classes)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model.to(DEVICE)
        model.eval()
        print(f"✅ Classifier 로드 완료: {model_path}")
        return model
    else:
        print(f"❌ 오류: Classification 모델({model_path})을 찾을 수 없습니다.")
        return None

# 3.2. AD 모델 로드 (Anomaly Detection 모델은 필요할 때 동적으로 로드)
def load_ad_model(class_name):
    model_path = AD_MODEL_PATHS.get(class_name)
    if not model_path:
        print(f"⚠️ {class_name}에 대한 Anomaly Detection 모델 경로가 정의되지 않았습니다.")
        return None
    
    model = Autoencoder().to(DEVICE)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model.to(DEVICE)
        model.eval()
        return model
    else:
        print(f"❌ 오류: AD 모델({model_path})을 찾을 수 없습니다.")
        return None

# 3.3. 전처리 파이프라인
# Classification 전처리
classifier_transform = transforms.Compose([
    transforms.Resize((CLASSIFIER_IMAGE_SIZE, CLASSIFIER_IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=MOBILENET_MEAN, std=MOBILENET_STD)
])

# Anomaly Detection 전처리
ad_preprocess = transforms.Compose([
    transforms.Resize((AD_IMAGE_SIZE, AD_IMAGE_SIZE)),
    transforms.ToTensor(), 
    transforms.Normalize(mean=MOBILENET_MEAN, std=MOBILENET_STD) 
])

# Anomaly Detection 후처리 (역정규화)
ad_postprocess = transforms.Compose([
    transforms.Normalize(mean=[0.0, 0.0, 0.0], std=[1/s for s in MOBILENET_STD]),
    transforms.Normalize(mean=[-m for m in MOBILENET_MEAN], std=[1.0, 1.0, 1.0]),
    transforms.ToPILImage() 
])

# =================================================================
# 4. 검사 실행 메인 함수
# =================================================================
def run_inspection_pipeline(classifier):
    """통합 검사 파이프라인을 실행합니다."""    
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("🔴 오류: 웹캠을 열 수 없습니다. 카메라 인덱스를 확인하세요.")
        return

    print(f"🟢 통합 시스템 시작. 'c'를 눌러 검사, 'q'를 눌러 종료하세요.")
    
    # 이전에 로드된 AD 모델을 저장하여 반복 로딩 방지 (캐싱)
    ad_model_cache = {}
	
    # 디스플레이 크기 설정 (원본 프레임 기준)
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if W == 0: W, H = 640, 480
		
    inspection_result = {"status": "Waiting...", "class": "", "conf": 0.0, "ad_loss": 0.0}

    while True:
        ret, frame = cap.read()
        if not ret: break
		
        display_frame = frame.copy()
		
		# ROI 좌표 설정 (c 키 입력과 상관없이 사용)
        x1, y1, x2, y2 = ROI_X, ROI_Y, ROI_X + ROI_W, ROI_Y + ROI_H
		
		# -----------------------------------------------------------
		# 💡 [수정] 검사 전에도 ROI를 항상 표시: 회색/파란색 테두리 사용
		# -----------------------------------------------------------
		
		# 검사 중이 아닐 때 또는 결과 대기 중일 때 표시할 기본 색상 (BGR: 밝은 회색 또는 파란색)
        default_color = (150, 150, 150) # 회색
		
		# ROI 영역을 기본 색상으로 먼저 그립니다.
        cv2.rectangle(display_frame, (x1, y1), (x2, y2), default_color, 2) # 두께 2
		
		# 'c' 키 입력 시 검사 수행
        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):
			
			# --- 단계 1: Object Classification ---
			# ... (분류 로직은 동일)
			
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)
            input_tensor = classifier_transform(pil_image).unsqueeze(0).to(DEVICE)
			
            with torch.no_grad():
                outputs = classifier(input_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                conf_score, predicted_idx = torch.max(probabilities, 1)
				
                predicted_class = CLASS_NAMES[predicted_idx.item()]
                confidence = conf_score.item()
				
                print(f"\n[Classification] Class: {predicted_class}, Confidence: {confidence*100:.2f}%")
				
				# 결과 업데이트
                inspection_result["class"] = predicted_class
                inspection_result["conf"] = confidence

				# --- 단계 2: Anomaly Detection ---
				
				# 해당 클래스의 AD 모델 로드 (캐시 사용)
                if predicted_class not in ad_model_cache:
                    ad_model_cache[predicted_class] = load_ad_model(predicted_class)
				
                ad_detector = ad_model_cache.get(predicted_class)
				
                if ad_detector:
					# ROI 추출
					# x1, y1, x2, y2 = ROI_X, ROI_Y, ROI_X + ROI_W, ROI_Y + ROI_H # 👆 이미 루프 시작에서 정의됨
                    roi = frame[y1:y2, x1:x2]
					
                    if roi.size > 0:
						# AD 전처리
                        roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
						# PIL 변환 추가 (이전 오류 해결에 따라)
                        ad_pil_image = Image.fromarray(roi_rgb) 
                        ad_input_tensor = ad_preprocess(ad_pil_image).unsqueeze(0).to(DEVICE)
						
						# AD 추론
                        ad_output_tensor = ad_detector(ad_input_tensor)
                        ad_loss = torch.mean((ad_input_tensor - ad_output_tensor) ** 2).item()
						
                        threshold = AD_THRESHOLDS[predicted_class]
						
						# 결과 판별
                        if ad_loss > threshold:
                            ad_status = "ANOMALY"
                            result_color = (0, 0, 255) # 빨강
                        else:
                            ad_status = "NORMAL"
                            result_color = (0, 255, 0) # 초록
							
						# 복원 이미지 시각화 (선택적)
                        reconstructed_pil = ad_postprocess(ad_output_tensor.squeeze(0).cpu())
                        reconstructed_roi = np.array(reconstructed_pil) * 255
                        reconstructed_roi = reconstructed_roi.astype(np.uint8)
                        reconstructed_roi = cv2.cvtColor(reconstructed_roi, cv2.COLOR_RGB2BGR)
                        reconstructed_roi_resized = cv2.resize(reconstructed_roi, (ROI_W, ROI_H))
						
						# 원본 프레임에 복원된 ROI 삽입
                        display_frame[y1:y2, x1:x2] = reconstructed_roi_resized
						
						# -----------------------------------------------------------
						# 💡 [수정] 검사 결과에 따른 색상으로 ROI를 덮어씁니다.
						# -----------------------------------------------------------
                        cv2.rectangle(display_frame, (x1, y1), (x2, y2), result_color, 4) # 두께 4

						# 결과 업데이트
                        inspection_result["status"] = ad_status
                        inspection_result["ad_loss"] = ad_loss
						
                    else:
                        inspection_result["status"] = "ERROR (ROI 추출 실패)"
                        inspection_result["ad_loss"] = 0.0
						
                else:
                    inspection_result["status"] = "ERROR (AD 모델 로드 실패)"
                    inspection_result["ad_loss"] = 0.0

		# --- 단계 3: 최종 결과 시각화 ---
		# ... (텍스트 오버레이 로직은 동일)
        cv2.putText(display_frame, f"Class: {inspection_result['class']} ({inspection_result['conf']*100:.1f}%)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2, cv2.LINE_AA)
		
        status_text = f"Status: {inspection_result['status']} (Loss: {inspection_result['ad_loss']:.5f})"
		
		# 상태에 따른 색상 설정
        if "ANOMALY" in inspection_result['status']:
            status_color = (0, 0, 255) # 빨강
        elif "NORMAL" in inspection_result['status']:
            status_color = (0, 255, 0) # 초록
        else:
            status_color = (255, 255, 255) # 흰색
			
        cv2.putText(display_frame, status_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2, cv2.LINE_AA)
        cv2.putText(display_frame, "Press 'c' to inspect, 'q' to quit", (10, H - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.imshow("Integrated Inspection System", display_frame)

        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# =================================================================
# 5. 메인 실행 블록
# =================================================================
if __name__ == "__main__":
    classifier = load_classifier(CLASSIFIER_WEIGHTS_PATH, NUM_CLASSES)
    if classifier:
        try:
            run_inspection_pipeline(classifier)
        except Exception as e:
            print(f"시스템 실행 중 예기치 않은 오류 발생: {e}")