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
WEIGHTS_FILE = os.path.join(MODEL_PATH, 'best_multitask_model.pth')

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
# 분류 결과를 잔차 회귀와 결합하여 최종 Rz 값을 복원하는 데 사용됩니다.
# Rz_center[C] = Class C의 Rz 구간 중심값
# Class 0: [-90, -80) -> -85
# Class 16: [70, 80] -> 75
RZ_CENTERS = np.arange(-90 + 5, 70 + 5 + 1e-6, 10, dtype=np.float32)
# RZ_CENTERS: [-85., -75., -65., ..., 55., 65., 75.]


# --- 3. 모델 정의 (학습 시 사용한 것과 동일해야 함) ---

class ResNetMultiTask(nn.Module):
    def __init__(self, num_classes):
        super(ResNetMultiTask, self).__init__()
        # PyTorch 모델 로드 시, weights 파라미터는 제거해야 합니다.
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


# --- 5. Rz 각도 추론 함수 ---

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
        # argmax로 가장 높은 확률의 클래스 인덱스를 가져옴
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
        # 현재 좌표(x, y, z, Rx, Ry, Rz)를 가져옵니다.
        default_pick_coords = PICK_COORDS
        rz_float = float(rz_angle)
        target_coords = default_pick_coords[:5] + [round(rz_float+90, 2)]
        tmp_pick_coords = default_pick_coords
        tmp_pick_coords[2] = 300
        mc.send_coords(tmp_pick_coords, speed=50)
        time.sleep(5)

        mc.send_coords(target_coords, speed=50)
        time.sleep(2)
        print(f"✅ 로봇 이동 요청: Rz={rz_angle:.2f}° (전체 좌표: {target_coords})")
        time.sleep(1) # 이동 대기 시간
        
    except Exception as e:
        print(f"❌ 로봇 제어 중 오류 발생: {e}")


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
    print("\n--- Rz 추론 테스트 시작 ---")
    print(" 'c' 키를 눌러 추론 및 로봇을 제어하세요.")
    print(" 'q' 키를 눌러 종료하세요.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("프레임 캡처 실패.")
            break

        # 추론 수행 (프레임 캡처 후 Rz 추론)
        final_rz, cls_idx, res_val = predict_rz_angle(model, frame, DEVICE)

        # 화면에 정보 표시
        H, W, _ = frame.shape
        # 추론 결과
        text_predict = f"Predicted Rz: {final_rz:.2f} deg"
        text_details = f"CLS={cls_idx}, Delta_Rz={res_val:.2f}"
        
        cv2.putText(frame, text_predict, (10, H - 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, text_details, (10, H - 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
        
        cv2.imshow('Camera Feed (Press "c" to capture and move)', frame)

        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('c'):
            # C 키 입력 시 추론 및 로봇 제어
            print("-" * 30)
            print(f"📸 C 키 입력: Rz 추론 시작")
            print(f"   -> 추론 결과: Rz={final_rz:.2f}° (Class {cls_idx} + Residual {res_val:.2f})")
            
            if mc:
                move_robot_to_rz(mc, final_rz)
            else:
                print("   -> 로봇 연결 실패로 제어 생략.")
            print("-" * 30)

        elif key == ord('q'):
            break

        elif key == ord('0'): # 0도 자세
            print(f"\n🔄 로봇을 0도 자세 이동 시작...")
            mc.set_gripper_value(GRIPPER_OPEN_VALUE, GRIPPER_SPEED) 
            mc.send_angles(INTERMEDIATE_POSE_ANGLES, MOVEMENT_SPEED)
            time.sleep(SEQUENTIAL_MOVE_DELAY)
            mc.send_angles(ZERO_POSE_ANGLES, MOVEMENT_SPEED)
            print("✅ 0도 자세 이동 완료.")
        
        elif key == ord('1'): # 컨베이어 캡처 자세
            print(f"\n🚀 컨베이어 캡처 자세 ({CONVEYOR_CAPTURE_POSE})로 이동 시작...")
            mc.set_gripper_value(GRIPPER_OPEN_VALUE, GRIPPER_SPEED) 
            mc.send_angles(INTERMEDIATE_POSE_ANGLES, MOVEMENT_SPEED)
            time.sleep(SEQUENTIAL_MOVE_DELAY)
            mc.send_angles(CONVEYOR_CAPTURE_POSE, MOVEMENT_SPEED)
            time.sleep(SEQUENTIAL_MOVE_DELAY)
            print("✅ CONVEYOR_CAPTURE_POSE 이동 완료.")
            
        elif key == ord('2'): # 테스트 픽업 자세 (관절 각도)
            print(f"\n⬇️ 테스트 픽업 자세 ({TEST_PICK_POSE})로 이동 시작...")
            mc.send_angles(INTERMEDIATE_POSE_ANGLES, MOVEMENT_SPEED)
            time.sleep(SEQUENTIAL_MOVE_DELAY)
            mc.send_coords(TMP_PICK_POSE, MOVEMENT_SPEED - 30) 
            time.sleep(SEQUENTIAL_MOVE_DELAY)
            mc.send_coords(TEST_PICK_POSE, MOVEMENT_SPEED) 
            time.sleep(SEQUENTIAL_MOVE_DELAY)
            mc.set_gripper_value(GRIPPER_CLOSED_VALUE, GRIPPER_SPEED)
            print("✅ TEST_PICK_POSE 이동 완료.")
        
        elif key == ord('3'): # 로봇팔 위 캡처 자세
            print(f"\n🚀 로봇팔 위 캡처 자세 ({ROBOTARM_CAPTURE_POSE})로 이동 시작...")
            mc.send_angles(INTERMEDIATE_POSE_ANGLES, MOVEMENT_SPEED)
            time.sleep(SEQUENTIAL_MOVE_DELAY)
            mc.send_angles(ROBOTARM_CAPTURE_POSE, MOVEMENT_SPEED)
            time.sleep(SEQUENTIAL_MOVE_DELAY)
            print("✅ ROBOTARM_CAPTURE_POSE 이동 완료.")

    # 5. 종료
    cap.release()
    cv2.destroyAllWindows()
    if mc:
        # 로봇 연결 해제 (옵션)
        # mc.set_free_mode(1) 
        pass 

if __name__ == '__main__':
    main_test_loop()