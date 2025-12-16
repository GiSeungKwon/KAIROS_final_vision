import os
import torch
import torch.nn as nn
from torchvision import models, transforms
import numpy as np
import cv2
import time
from pymycobot import MyCobot320 

CAMERA_INDEX = 0

MODEL_PATH = "C:/Dev/KAIROS_Project/models/Coordinate_Detection_models"
WEIGHTS_FILE = os.path.join(MODEL_PATH, 'multitask_model_epoch_60.pth')
# multitask_model_epoch_10
# best_multitask_model - 비스듬하면 못잡음

NUM_CLASSES = 17 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 224

MYCOBOT_PORT = "COM3" 
MYCOBOT_BAUDRATE = 115200

PICK_COORDS = [-237.90, 20, 183.6, -174.98, 0, -90]
MOVEMENT_SPEED = 70       # ⚙️ 관절/좌표 이동 속도 (퍼센트 단위, 1-100)
GRIPPER_SPEED = 50        # ⚙️ 그리퍼 작동 속도 (20 -> 50으로 상향 조정)
SEQUENTIAL_MOVE_DELAY = 1.5 # ⏱️ 자세 이동 명령 간 대기 시간 (안정성 확보를 위해 1.5초로 조정)
GRIPPER_ACTION_DELAY = 1  # ⏱️ 그리퍼 작동 후 대기 시간
GRIPPER_OPEN_VALUE = 85   # 👐 그리퍼 완전 열림 위치 값 (max 100)
GRIPPER_CLOSED_VALUE = 25 # ✊ 그리퍼 완전 닫힘 위치 값 (min 0)

INTERMEDIATE_POSE_ANGLES = [-17.2, 30.49, 4.48, 53.08, -90.87, -85.86] # 충돌 방지 경유 자세
ZERO_POSE_ANGLES = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

CONVEYOR_CAPTURE_POSE = [0, 0, 90, 0, -90, -90] # 컨베이어벨트 캡처를 위한 시야 확보 자세
ROBOTARM_CAPTURE_POSE = [0, 0, 10, 80, -90, 90] # 로봇 팔 위 물체 캡처 자세

TEST_PICK_POSE_WIDTH = [-229.30, 20, 183.6, -174.98, 0, 0]
TEST_PICK_POSE_HEIGHT = [-229.30, 7.80, 183.6, -174.98, 0, 90]

TMP_PICK_POSE = [-229.30, 20, 300.6, -174.98, 0, 0]
TEST_PICK_POSE = [-229.30, 20, 183.6, -174.98, 0, 0]

# --- 2. 클래스 중심값 (Center Rz) 정의 ---
RZ_CENTERS = np.arange(-90 + 5, 70 + 5 + 1e-6, 10, dtype=np.float32)


# --- 3. 모델 정의 (학습 시 사용한 것과 동일해야 함) ---

class ResNetMultiTask(nn.Module):
    def __init__(self, num_classes):
        super(ResNetMultiTask, self).__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        
        self.features = nn.Sequential(*(list(resnet.children())[:-2]))
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.cls_head = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        
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


# --- 4. 이미지 전처리 정의 (Validation/Test와 동일) ---

test_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(256),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


# --- 5. Rz 각도 추론 함수 (AI) ---

def predict_rz_angle(model, img, device):
    """
    이미지를 모델에 입력하여 Rz 각도를 추론합니다.
    """
    # OpenCV BGR -> RGB 변환
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 전처리 및 Tensor 변환
    img_tensor = test_transform(img_rgb).unsqueeze(0).to(device) # [1, 3, 224, 224]

    model.eval()
    with torch.no_grad():
        cls_preds, res_preds = model(img_tensor)
        
        # 1. 분류 결과 (Class C)
        predicted_class = torch.argmax(cls_preds, dim=1).item()
        
        # 2. 잔차 회귀 결과 (Delta Rz)
        predicted_residual = res_preds.squeeze().item()
        
    # 3. 최종 Rz 각도 복원: Rz = Rz_center[C] + Delta_Rz
    rz_center = RZ_CENTERS[predicted_class]
    final_rz = rz_center + predicted_residual
    
    # 각도 범위 [-90, 90] 제한 (MyCobot 가동 범위에 맞게)
    final_rz = np.clip(final_rz, -90, 90)
    
    return final_rz, predicted_class, predicted_residual


# --- 6. MyCobot 제어 함수 ---

def move_robot_to_rz(mc, rz_angle):
    """
    MyCobot의 현재 좌표를 가져와 Rz 값만 업데이트하여 로봇을 이동시킵니다.
    """
    try:
        # PICK_COORDS의 원본 값을 보존하기 위해 '깊은 복사' 사용
        # (1) tmp_pick_coords: Z=300의 중간 지점 좌표
        tmp_pick_coords = list(PICK_COORDS) # 💡 PICK_COORDS를 복사본으로 만듦
        tmp_pick_coords[2] = 300 
        
        # (2) target_coords: 최종 목표 좌표 (Z는 원래 PICK_COORDS의 Z)
        rz_float = float(rz_angle + 5)
        # 💡 PICK_COORDS의 원본 값을 사용해야 함
        target_coords = PICK_COORDS[:5] + [round(rz_float, 2)]
        
        # 1. Z=300 (중간 지점)으로 이동
        mc.send_coords(tmp_pick_coords, speed=50)
        time.sleep(5) 
        
        # 2. 최종 목표 지점으로 이동
        mc.send_coords(target_coords, speed=50)
        time.sleep(5) 

        print(f"✅ 로봇 이동 요청: Rz={rz_angle:.2f}° (중간: {tmp_pick_coords}, 최종: {target_coords})")
        time.sleep(1)
        
    except Exception as e:
        print(f"❌ 로봇 제어 중 오류 발생: {e}")


# --- 5-1. HSV 기반 Vision Rz 추론 함수 (ROI 적용) ---

def get_vision_rz(img):
    """
    HSV 마스킹 및 minAreaRect를 사용하여 물체의 중심(Cx, Cy)과 Rz 각도를 추론합니다.
    """
    # 💡 1. ROI 영역 설정 (요청 반영)
    x_start, y_start = 90, 70
    x_end, y_end = 390, 330
    
    # ROI 추출
    # 경계를 벗어나지 않도록 클리핑 (선택 사항이지만 안전을 위해 필요)
    H, W, _ = img.shape
    x_start = max(0, x_start)
    y_start = max(0, y_start)
    x_end = min(W, x_end)
    y_end = min(H, y_end)

    if x_start >= x_end or y_start >= y_end:
        print("경고: ROI 영역이 유효하지 않습니다.")
        return None, None, None, None

    roi = img[y_start:y_end, x_start:x_end]
    
    # 2. BGR -> HSV 변환 (ROI에 대해서만 수행)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    
    # 3. HSV 임계값 설정 (어두운 기판 영역 추출한다고 가정)
    # 현재 설정된 임계값 (V: 0~170)은 어두운 부분을 잘 추출할 것으로 예상
    lower_bound = np.array([0, 0, 210])
    upper_bound = np.array([180, 255, 255])
    
    # 마스크 생성 및 노이즈 제거
    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    kernel = np.ones((10, 10), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel) # 열기(Open) 연산
    # mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    # 4. 외곽선 찾기
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 5. 가장 큰 외곽선 선택 (가장 큰 객체가 물체라고 가정)
    if not contours:
        return None, None, None, None # 찾지 못함
    
    main_contour = max(contours, key=cv2.contourArea)
    
    # 6. 최소 면적 직사각형 찾기 (Rz 각도 계산의 핵심)
    rect = cv2.minAreaRect(main_contour)
    
    # 7. 결과 추출
    (center_x_rel, center_y_rel), (width, height), angle = rect
    
    # 8. Rz 각도 계산 및 보정
    # minAreaRect의 각도는 긴 변이 수평축과 이루는 각도(-90 ~ 0) 또는 (0 ~ 90)으로 반환됨.
    # 긴 변이 세로일 경우, 각도에 90을 더해 긴 변 기준 각도로 변환.
    if width < height:
        angle = angle + 90
        
    # 물체의 긴 변이 수평(0도)일 때 0이 되도록 angle을 보정 (실제 환경에 맞게 튜닝 필요)
    # 임시 Rz 계산: 시계 반대 방향(CCW)이 양수(+)라고 가정하고, angle을 반전
    vision_rz = -angle + 90
    
    # 각도 범위 [-90, 90] 제한
    vision_rz = np.clip(vision_rz, -90, 90)

    # 9. 중심 좌표를 원본 이미지 기준으로 변환
    center_x_abs = center_x_rel + x_start
    center_y_abs = center_y_rel + y_start

    # 10. 시각화 (원본 이미지에 그리기)
    box = cv2.boxPoints(rect)
    box = np.intp(box)
    
    # ROI 오프셋을 더하여 절대 좌표로 변환
    box_abs = box + (x_start, y_start) 
    cv2.drawContours(img, [box_abs], 0, (255, 0, 0), 2) # 파란색으로 외곽선 표시

    # 물체 면적 (앙상블 가중치 계산용)
    area = cv2.contourArea(main_contour)
    
    return vision_rz, (center_x_abs, center_y_abs), mask, area # area 값 추가 반환


def ensemble_rz(rz_vision, rz_ai, area):
    """
    Vision Rz와 AI Rz를 결합하여 최종 Rz를 결정합니다.
    Vision 결과의 신뢰도(area)에 따라 가중치를 조절합니다.
    """
    # 1. Vision 결과가 유효한지 확인 (물체를 찾지 못했을 경우)
    if rz_vision is None:
        print("➡️ Vision Rz 실패. AI Rz만 사용.")
        return rz_ai 

    # 2. Vision 신뢰도(area) 기반 가중치 설정
    # 💡 면적을 정규화하지 않고, 픽셀 면적 임계값으로 Vision 신뢰도를 판단. (환경에 따라 튜닝 필요)
    VISION_MIN_AREA_THRESHOLD = 500  # 최소 500 픽셀 이상이어야 신뢰도 높음 가정
    
    if area >= VISION_MIN_AREA_THRESHOLD:
        w_vis = 0.8  # Vision 신뢰도 높음
        w_ai = 0.2
    else:
        w_vis = 0.4  # Vision 신뢰도 낮음 (노이즈, 작은 객체 등)
        w_ai = 0.6

    # 3. 가중 평균으로 최종 Rz 계산
    final_rz = w_vis * rz_vision + w_ai * rz_ai
    
    # 각도 범위 [-90, 90] 제한
    final_rz = np.clip(final_rz, -90, 90)
    
    return final_rz


# --- 7. 메인 실행 루프 (카메라 및 테스트) ---

def main_test_loop():
    # 1. 모델 로드
    try:
        model = ResNetMultiTask(NUM_CLASSES).to(DEVICE)
        model.load_state_dict(torch.load(WEIGHTS_FILE, map_location=DEVICE))
        model.eval()
        print(f"✅ 모델 로드 성공: {WEIGHTS_FILE}")
    except Exception as e:
        print(f"❌ 모델 로드 실패: {e}")
        return

    # 2. 카메라 설정
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("❌ 카메라(인덱스 0)를 열 수 없습니다. 장치 연결을 확인하세요.")
        return
    
    # 3. MyCobot 연결 (실제 로봇이 연결되어 있어야 함)
    mc = None
    try:
        mc = MyCobot320(MYCOBOT_PORT, MYCOBOT_BAUDRATE)
        # 로봇 준비 (옵션)
        # mc.set_free_mode(0) # 로봇이 프리 모드가 아닌지 확인
        print(f"✅ MyCobot 연결 성공: {MYCOBOT_PORT}")
    except Exception as e:
        print(f"⚠️ MyCobot 연결 실패: {e}. 로봇 제어 없이 Vision 추론만 진행합니다.")


    # 4. 루프 시작
    print("\n--- Rz 앙상블 추론 테스트 시작 ---")
    print(" 'c' 키를 눌러 추론 및 로봇을 제어하세요.")
    print(" 'q' 키를 눌러 종료하세요.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("프레임 캡처 실패.")
            break

        # 추론 수행 (프레임 캡처 후 Rz 추론)
        # (1) AI Rz 추론 (기존 코드)
        final_rz_ai, cls_idx, res_val = predict_rz_angle(model, frame.copy(), DEVICE)

        # (2) Vision Rz 추론 (새로운 로직)
        rz_vision, center_pt, mask, area = get_vision_rz(frame.copy())
        
        # (3) 앙상블 로직
        if rz_vision is not None:
            # 면적(area)을 기반으로 Vision 신뢰도 판단 및 앙상블 수행
            final_rz = ensemble_rz(rz_vision, final_rz_ai, area) 
            
            # 디버깅을 위해 Vision 마스크 표시
            mask_resized = cv2.resize(mask, (int(frame.shape[1]/3), int(frame.shape[0]/3)))
            cv2.imshow('HSV Mask (ROI)', mask_resized)
            
            # Vision 중심점 시각화 
            cv2.circle(frame, (int(center_pt[0]), int(center_pt[1])), 5, (255, 0, 255), -1) 
            
            # 화면에 Vision Rz 정보 추가 표시
            cv2.putText(frame, f"Vision Rz: {rz_vision:.2f} deg (Area={area:.0f})", 
                        (10, frame.shape[0] - 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)

        else:
            final_rz = final_rz_ai # Vision 실패 시 AI 결과만 사용
            # Vision 실패 시 텍스트 피드백
            cv2.putText(frame, "Vision Failed. Using AI Rz only.", 
                        (10, frame.shape[0] - 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
            # area가 None이므로 임시 값 할당 (로그용)
            area = 0 
            print("Vision 처리 실패. Rz_AI 사용.")

        # 화면에 정보 표시 (최종 앙상블 결과 사용)
        H, W, _ = frame.shape
        text_predict = f"Final Rz: {final_rz:.2f} deg"
        text_details = f"AI CLS={cls_idx}, Delta_Rz={res_val:.2f}"
        
        cv2.putText(frame, text_predict, (10, H - 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, text_details, (10, H - 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        
        # 💡 ROI 영역 시각화 (옵션)
        cv2.rectangle(frame, (110, 70), (390, 350), (0, 165, 255), 1) 
        
        cv2.imshow('Camera Feed (Press "c" to capture and move)', frame)

        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('c'):
            # C 키 입력 시 추론 및 로봇 제어
            print("-" * 30)
            print(f"📸 C 키 입력: Rz 추론 시작")
            print(f"   -> Vision Rz: {rz_vision:.2f}° (Area: {area:.0f})")
            print(f"   -> AI Rz: {final_rz_ai:.2f}°")
            print(f"   -> 최종 Rz: {final_rz:.2f}°")
            
            if mc:
                # if rz_vision 
                move_robot_to_rz(mc, final_rz)
            else:
                print("   -> 로봇 연결 실패로 제어 생략.")
            print("-" * 30)

        elif key == ord('q'):
            break

        elif key == ord('0'): 
            print(f"\n🔄 로봇을 0도 자세 이동 시작...")
            if mc:
                mc.set_gripper_value(GRIPPER_OPEN_VALUE, GRIPPER_SPEED) 
                mc.send_angles(INTERMEDIATE_POSE_ANGLES, MOVEMENT_SPEED)
                time.sleep(SEQUENTIAL_MOVE_DELAY)
                mc.send_angles(ZERO_POSE_ANGLES, MOVEMENT_SPEED)
                print("✅ 0도 자세 이동 완료.")
            else:
                print("로봇 연결 실패.")
        
        elif key == ord('1'):
            print(f"\n🚀 컨베이어 캡처 자세 ({CONVEYOR_CAPTURE_POSE})로 이동 시작...")
            if mc:
                mc.set_gripper_value(GRIPPER_OPEN_VALUE, GRIPPER_SPEED) 
                mc.send_angles(INTERMEDIATE_POSE_ANGLES, MOVEMENT_SPEED)
                time.sleep(SEQUENTIAL_MOVE_DELAY)
                mc.send_angles(CONVEYOR_CAPTURE_POSE, MOVEMENT_SPEED)
                time.sleep(SEQUENTIAL_MOVE_DELAY)
                print("✅ CONVEYOR_CAPTURE_POSE 이동 완료.")
            else:
                print("로봇 연결 실패.")
        
        elif key == ord('2'):
            print(f"\n⬇️ 테스트 픽업 자세 ({TEST_PICK_POSE})로 이동 시작...")
            if mc:
                mc.send_angles(INTERMEDIATE_POSE_ANGLES, MOVEMENT_SPEED)
                time.sleep(SEQUENTIAL_MOVE_DELAY)
                mc.send_coords(TMP_PICK_POSE, MOVEMENT_SPEED - 30) 
                time.sleep(SEQUENTIAL_MOVE_DELAY)
                mc.send_coords(TEST_PICK_POSE, MOVEMENT_SPEED) 
                time.sleep(SEQUENTIAL_MOVE_DELAY)
                mc.set_gripper_value(GRIPPER_CLOSED_VALUE, GRIPPER_SPEED)
                print("✅ TEST_PICK_POSE 이동 완료.")
            else:
                print("로봇 연결 실패.")
        
        elif key == ord('3'):
            print(f"\n🚀 로봇팔 위 캡처 자세 ({ROBOTARM_CAPTURE_POSE})로 이동 시작...")
            if mc:
                mc.send_angles(INTERMEDIATE_POSE_ANGLES, MOVEMENT_SPEED)
                time.sleep(SEQUENTIAL_MOVE_DELAY)
                mc.send_angles(ROBOTARM_CAPTURE_POSE, MOVEMENT_SPEED)
                time.sleep(SEQUENTIAL_MOVE_DELAY)
                print("✅ ROBOTARM_CAPTURE_POSE 이동 완료.")
            else:
                print("로봇 연결 실패.")


    # 5. 종료
    cap.release()
    cv2.destroyAllWindows()
    if mc:
        # 로봇 연결 해제 (옵션)
        # mc.set_free_mode(1) 
        pass 

if __name__ == '__main__':
    main_test_loop()