import cv2
import os
import time
import pandas as pd

# --- 설정값 (Configuration) ---
CAMERA_INDEX = 1
BASE_DIR = os.path.join("..", "..", "data")
CSV_FILE_PATH = os.path.join(BASE_DIR, "classification_labels.csv")

# 캡처 키와 레이블/폴더 매핑
LABEL_MAP = {
    ord('1'): {"label": "ESP32", "folder": "ESP32"},
    ord('2'): {"label": "L298N", "folder": "L298N"},
    ord('3'): {"label": "MB102", "folder": "MB102"},
}

# --- 제어 변수 초기화 및 상수 정의 ---
# 🚨 사용자 설정값 반영 (FOCUS: 33% -> 84, EXPOSURE: -12)
CURRENT_FOCUS = 84
CURRENT_EXPOSURE = -12
CURRENT_WB_TEMPERATURE = 4000 # WB는 이전 값 유지

# 제어 변수 업데이트 함수 (캡슐화하여 사용)
def update_camera_property(cap, prop_id, value, min_val=-1000, max_val=10000):
    """카메라 속성 값을 설정하고, 최소/최대 범위 내에서 값을 반환합니다."""
    # 노출이 마이너스 값을 가질 수 있도록 min_val을 변경했습니다.
    value = max(min_val, min(max_val, value))
    cap.set(prop_id, value)
    return value

# --- 기존 코드 (setup_directories_and_csv 함수는 변경 없음) ---
def setup_directories_and_csv():
    """필요한 데이터 폴더를 생성하고, CSV 파일의 헤더를 설정합니다."""
    if not os.path.exists(BASE_DIR):
        os.makedirs(BASE_DIR)
        print(f"기본 데이터 디렉토리 생성: {BASE_DIR}")
        
    for key in LABEL_MAP:
        folder_name = LABEL_MAP[key]["folder"]
        path = os.path.join(BASE_DIR, folder_name)
        if not os.path.exists(path):
            os.makedirs(path)
            print(f"데이터 폴더 생성: {path}")

    if not os.path.exists(CSV_FILE_PATH):
        df = pd.DataFrame(columns=['filename', 'label'])
        df.to_csv(CSV_FILE_PATH, index=False)
        print(f"새로운 레이블 CSV 파일 생성: {CSV_FILE_PATH}")
    else:
        print(f"기존 레이블 CSV 파일 사용: {CSV_FILE_PATH}")

def run_data_collection():
    global CURRENT_FOCUS, CURRENT_EXPOSURE, CURRENT_WB_TEMPERATURE
    
    setup_directories_and_csv()
    
    # 💡 백엔드 자동 감지 모드 유지
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_MSMF) 
    
    # --- 웹캠 초기 설정 ---
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cap.set(cv2.CAP_PROP_FPS, 60)
    
    # 자동 설정 비활성화
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25) # 수동 모드 설정
    cap.set(cv2.CAP_PROP_AUTO_WB, 0)
    
    # 🚨 수동 초기값 적용
    cap.set(cv2.CAP_PROP_FOCUS, CURRENT_FOCUS)
    cap.set(cv2.CAP_PROP_EXPOSURE, CURRENT_EXPOSURE)
    cap.set(cv2.CAP_PROP_WB_TEMPERATURE, CURRENT_WB_TEMPERATURE)
    
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)

    print(f"실제 설정된 해상도: {actual_width}x{actual_height}, FPS: {actual_fps}")
    print(f"초기 FOCUS: {CURRENT_FOCUS}, EXPOSURE: {CURRENT_EXPOSURE}, WB: {CURRENT_WB_TEMPERATURE}")

    if not cap.isOpened():
        print(f"오류: 웹캠 인덱스 {CAMERA_INDEX}를 열 수 없습니다. 인덱스를 확인하세요.")
        return

    try:
        df = pd.read_csv(CSV_FILE_PATH)
    except:
        df = pd.DataFrame(columns=['filename', 'label'])

    print("\n--- 데이터 수집 시작 ---")
    print("-----------------------------------------------------------------")
    print(" [수집] '1', '2', '3': 캡처 | 'Q': 종료")
    print(" [초점] 'A': +5 증가 | 'Z': -5 감소 (FOCUS)")
    print(" [노출] 'D': +5 증가 | 'C': -5 감소 (EXPOSURE)")
    print(" [WB]   'G': +100 증가 | 'B': -100 감소 (WB)")
    print("-----------------------------------------------------------------")
    
    initial_count = len(df)
    print(f"현재까지 수집된 데이터 수: {initial_count}개")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("프레임을 읽을 수 없습니다 (카메라 연결 문제).")
                break

            # --- 화면 표시를 위한 축소 및 처리 ---
            DISPLAY_WIDTH = 640
            DISPLAY_HEIGHT = 360
            display_frame_resized = cv2.resize(frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT), interpolation=cv2.INTER_LINEAR)
            
            font_scale = 0.5
            y_offset = 20
            
            cv2.putText(display_frame_resized, "Press 1:ESP32, 2:L298N, 3:MB102 | Q:Quit", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), 1)
            y_offset += 25
            cv2.putText(display_frame_resized, f"Total Samples: {len(df)}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 0, 0), 1)
            y_offset += 25

            # 3. 현재 카메라 속성 값 표시
            cv2.putText(display_frame_resized, f"FOCUS (A/Z, +5): {CURRENT_FOCUS}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1)
            y_offset += 25
            cv2.putText(display_frame_resized, f"EXPOSURE (D/C, +5): {CURRENT_EXPOSURE}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1)
            y_offset += 25
            cv2.putText(display_frame_resized, f"WB (G/B, +100): {CURRENT_WB_TEMPERATURE}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1)

            cv2.imshow("Module Data Collector (Capturing 1920x1080)", display_frame_resized)

            # --- 키 입력 처리 ---
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q') or key == ord('Q'):
                print("사용자 요청으로 데이터 수집을 종료합니다.")
                break
            
            # --------------------------------------------------------
            # 카메라 속성 제어 키 처리 (조정 단위 5/100 적용)
            # --------------------------------------------------------
            ADJUST_STEP_FOCUS = 5
            ADJUST_STEP_EXPOSURE = 5
            ADJUST_STEP_WB = 100

            # A/Z: FOCUS 제어 (0~255 범위 유지)
            if key == ord('a') or key == ord('A'):
                CURRENT_FOCUS = update_camera_property(cap, cv2.CAP_PROP_FOCUS, CURRENT_FOCUS + ADJUST_STEP_FOCUS, min_val=0, max_val=255)
            elif key == ord('z') or key == ord('Z'):
                CURRENT_FOCUS = update_camera_property(cap, cv2.CAP_PROP_FOCUS, CURRENT_FOCUS - ADJUST_STEP_FOCUS, min_val=0, max_val=255)
            
            # D/C: EXPOSURE 제어 (낮은 마이너스 값 포함 가능하도록 설정)
            elif key == ord('d') or key == ord('D'):
                CURRENT_EXPOSURE = update_camera_property(cap, cv2.CAP_PROP_EXPOSURE, CURRENT_EXPOSURE + ADJUST_STEP_EXPOSURE, min_val=-13, max_val=10) # 흔히 사용되는 범위로 조정
            elif key == ord('c') or key == ord('C'):
                CURRENT_EXPOSURE = update_camera_property(cap, cv2.CAP_PROP_EXPOSURE, CURRENT_EXPOSURE - ADJUST_STEP_EXPOSURE, min_val=-13, max_val=10) # 흔히 사용되는 범위로 조정
            
            # G/B: WB_TEMPERATURE 제어 (2000K ~ 6500K 범위 유지)
            elif key == ord('g') or key == ord('G'):
                CURRENT_WB_TEMPERATURE = update_camera_property(cap, cv2.CAP_PROP_WB_TEMPERATURE, CURRENT_WB_TEMPERATURE + ADJUST_STEP_WB, min_val=2000, max_val=6500)
            elif key == ord('b') or key == ord('B'):
                CURRENT_WB_TEMPERATURE = update_camera_property(cap, cv2.CAP_PROP_WB_TEMPERATURE, CURRENT_WB_TEMPERATURE - ADJUST_STEP_WB, min_val=2000, max_val=6500)

            
            # --------------------------------------------------------
            # 이미지 캡처 키 처리
            # --------------------------------------------------------
            elif key in LABEL_MAP:
                info = LABEL_MAP[key]
                label = info["label"]
                folder = info["folder"]
                
                timestamp = int(time.time() * 1000)
                filename = f"{timestamp}_{label}.png"
                save_path = os.path.join(BASE_DIR, folder, filename)
                
                cv2.imwrite(save_path, frame)
                
                new_row = pd.DataFrame([{'filename': os.path.join(folder, filename), 'label': label}])
                df = pd.concat([df, new_row], ignore_index=True)
                
                df.to_csv(CSV_FILE_PATH, index=False)
                
                print(f"  [CAP] {label} 모듈 캡처: {os.path.join(folder, filename)}")
                print(f"  > 현재 데이터 수: {len(df)}개 (F:{CURRENT_FOCUS}, E:{CURRENT_EXPOSURE}, WB:{CURRENT_WB_TEMPERATURE})")
                
                time.sleep(0.3)

    finally:
        cap.release()
        cv2.destroyAllWindows()
        
        final_count = len(df)
        captured_count = final_count - initial_count
        print(f"\n--- 수집 완료 ---")
        print(f"새로 수집된 이미지: {captured_count}개")
        print(f"총 데이터셋 크기: {final_count}개")
        print(f"레이블 CSV 파일 위치: {CSV_FILE_PATH}")


if __name__ == "__main__":
    run_data_collection()