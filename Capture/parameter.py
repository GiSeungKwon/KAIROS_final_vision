import cv2
import time
import os
import numpy as np

# --- 💡 전역 변수 설정 (트랙바의 현재 값 저장) ---
# 웹캠 드라이버마다 지원 범위가 다릅니다. 이 값들은 테스트 후 조정이 필요합니다.
INITIAL_EXPOSURE = 250   # 노출 시간 (양수, 1000이 1초라고 가정. 실제는 드라이버마다 다름)
INITIAL_BRIGHTNESS = 128 # 밝기 (일반적으로 0~255)
INITIAL_CONTRAST = 100   # 대비 (일반적으로 0~100)
INITIAL_GAIN = 100       # 이득/증폭 (노이즈에 영향, 일반적으로 0~255)
MAX_VAL = 255            # 트랙바 최대값

def on_trackbar_change(val):
    """트랙바 값이 변경될 때 호출되지만, 실제 설정은 메인 루프에서 처리합니다."""
    pass

def stream_and_capture_with_controls():
    # 1. 카메라 열기 (캡처 인덱스: 1 사용)
    cap = cv2.VideoCapture(1)

    if not cap.isOpened():
        print("🚨 오류: 카메라를 열 수 없습니다. 인덱스를 0으로 변경해 보세요.")
        return

    # 2. 제어 창 및 트랙바 생성
    control_window_name = 'Camera Controls'
    cv2.namedWindow(control_window_name, cv2.WINDOW_AUTOSIZE)

    # Exposure: 트랙바는 양수 정수만 지원합니다. 
    # 따라서, 음수 노출 값(-1~-13 등)을 사용하는 드라이버를 위해 'EXP_OFFSET'이라는 
    # 가상의 트랙바를 만들고 메인 루프에서 실제 노출 값으로 변환합니다.
    cv2.createTrackbar('Exposure (EXP)', control_window_name, INITIAL_EXPOSURE, 1000, on_trackbar_change)
    cv2.createTrackbar('Brightness (BRT)', control_window_name, INITIAL_BRIGHTNESS, MAX_VAL, on_trackbar_change)
    cv2.createTrackbar('Contrast (CON)', control_window_name, INITIAL_CONTRAST, MAX_VAL, on_trackbar_change)
    cv2.createTrackbar('Gain (GAIN)', control_window_name, INITIAL_GAIN, MAX_VAL, on_trackbar_change)
    
    # 트랙바를 통해 값을 설정하기 위해, 자동 모드를 꺼줍니다.
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1) # 수동 노출 모드

    print("--- 📸 카메라 스트리밍 시작 ---")
    print("  - 'Camera Controls' 창에서 트랙바로 값을 조절하세요.")
    print("  - 스트리밍 창에서 **'c'** 키를 누르면 **캡처**됩니다.")
    print("  - 스트리밍 창에서 **'q'** 키를 누르면 **종료**됩니다.")
    print("----------------------------")

    # 캡처 파일을 저장할 폴더 생성
    save_dir = "captured_images"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    frame_count = 0

    while True:
        # 3. 트랙바 값 읽기
        exp_val = cv2.getTrackbarPos('Exposure (EXP)', control_window_name)
        brt_val = cv2.getTrackbarPos('Brightness (BRT)', control_window_name)
        con_val = cv2.getTrackbarPos('Contrast (CON)', control_window_name)
        gain_val = cv2.getTrackbarPos('Gain (GAIN)', control_window_name)

        # 4. 카메라 속성 실시간 적용
        
        # 노출 설정 (Exposure)
        # 트랙바 값이 0이면 카메라 자동 노출(AE)로 설정. 
        # 값이 1 이상이면 수동 값으로 설정합니다. 
        # 대부분의 웹캠은 음수 값(예: -7.0)으로 노출 단계를 설정합니다.
        # 여기서는 트랙바 값(1~1000)을 노출 시간(ms) 또는 노출 단계로 변환하여 시도합니다.
        if exp_val > 0:
            # 노출 시간이 양수(ms)로 설정되는 카메라의 경우 이 방법을 사용합니다.
            cap.set(cv2.CAP_PROP_EXPOSURE, exp_val / 1000.0) # 0.001초 단위로 가정
            # 음수 노출 단계가 필요한 경우, 이 부분을 활성화하고 위 코드를 주석 처리하세요.
            # exposure_level = -(1000 - exp_val) / 100.0 
            # cap.set(cv2.CAP_PROP_EXPOSURE, exposure_level)
        else:
            # 0이면 자동 노출로 전환
            cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3) 

        cap.set(cv2.CAP_PROP_BRIGHTNESS, brt_val)
        cap.set(cv2.CAP_PROP_CONTRAST, con_val)
        cap.set(cv2.CAP_PROP_GAIN, gain_val)
        
        # 플리커 방지 모드 (60Hz 고정)
        # cap.set(cv2.CAP_PROP_SETTINGS, 2) # 매 루프마다 설정하면 성능 저하를 일으킬 수 있으므로 주석 처리

        # 5. 프레임 읽기 및 표시
        ret, frame = cap.read()

        if not ret:
            print("🚨 오류: 프레임을 읽을 수 없습니다. 스트림 종료.")
            break

        cv2.imshow('Live Stream - Press C to Capture, Q to Quit', frame)

        # 6. 키 입력 처리
        key = cv2.waitKey(1) & 0xFF

        if key == ord('c'):
            frame_count += 1
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(save_dir, f"capture_{timestamp}_{frame_count}.png")
            if cv2.imwrite(filename, frame):
                print(f"✅ 캡처 완료: {filename}에 저장되었습니다. (EXP={exp_val}, BRT={brt_val}, CON={con_val})")
            else:
                print(f"❌ 오류: {filename} 저장 실패.")

        elif key == ord('q'):
            print("👋 스트리밍을 종료합니다.")
            break

    # 7. 리소스 해제
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    stream_and_capture_with_controls()