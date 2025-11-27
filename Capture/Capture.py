import cv2
import time
import os

def stream_and_capture():
    cap = cv2.VideoCapture(1)

    # --- 이미지 품질 개선을 위한 속성 설정 시작 ---
    
    # 1. 자동 노출/초점 끄기: 수동 설정을 위해 필수적입니다.
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0) # 자동 초점 끄기 (물리적 초점 조절 후)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1) # 수동 노출 모드 설정 (대부분의 웹캠에서 1)
    
    # 2. 노출 시간 설정 (Exposure Time)
    # 현재 이미지가 너무 밝으므로 노출 값을 더 낮춰보세요.
    # -6.0에서 시작해서 -7.0, -8.0, -9.0 등으로 점차 낮춰보면서
    # 이미지가 선명해지고 적절한 밝기가 될 때까지 테스트합니다.
    # 또는 양수 값(ms 단위)을 지원하는 카메라의 경우 16 (1/60s), 8 (1/120s) 등으로 설정해 볼 수 있습니다.
    cap.set(cv2.CAP_PROP_EXPOSURE, -7.6) # 더 낮은 노출 값 시도

    # 3. 밝기(Brightness) 조절
    # CAP_PROP_EXPOSURE와 별개로 밝기를 조절할 수 있습니다.
    # 0.0 (어둡게) ~ 1.0 (밝게) 사이의 값을 사용하거나, 카메라 드라이버에 따라 다른 범위일 수 있습니다.
    # 노출을 줄여도 너무 밝다면 이 값을 낮춰보세요 (예: 64, 32 등).
    cap.set(cv2.CAP_PROP_BRIGHTNESS, 64) # 주석 처리: 노출로 먼저 조절해보고 필요시 활성화
    
    # 4. 대비(Contrast) 조절
    # 이미지의 선명도를 높이는 데 도움이 될 수 있습니다.
    # cap.set(cv2.CAP_PROP_CONTRAST, 64) # 주석 처리: 필요시 활성화

    # 5. 플리커 방지 모드 (Anti-Banding)
    # 0: Off, 1: 50Hz, 2: 60Hz, 3: Auto
    cap.set(cv2.CAP_PROP_SETTINGS, 2) # 한국 표준 60Hz 설정

    # --- 이미지 품질 개선을 위한 속성 설정 끝 ---

    # 카메라가 제대로 열렸는지 확인
    if not cap.isOpened():
        print("🚨 오류: 카메라를 열 수 없습니다.")
        return

    print("--- 📸 카메라 스트리밍 시작 ---")
    print("  - 스트리밍 창에서 **'c'** 키를 누르면 **캡처**됩니다.")
    print("  - 스트리밍 창에서 **'q'** 키를 누르면 **종료**됩니다.")
    print("----------------------------")

    # 캡처 파일을 저장할 폴더 생성 (선택 사항)
    save_dir = "captured_images"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    frame_count = 0

    while True:
        # 프레임 읽기: ret은 성공/실패 여부, frame은 실제 이미지 데이터
        ret, frame = cap.read()

        if not ret:
            print("🚨 오류: 프레임을 읽을 수 없습니다. 스트림 종료.")
            break

        # 화면에 프레임 표시
        cv2.imshow('Live Stream - Press C to Capture, Q to Quit', frame)

        # 1ms 동안 키 입력을 대기하고 입력된 키의 ASCII 값을 key 변수에 저장
        key = cv2.waitKey(1) & 0xFF

        # 1. 'c' 키를 누르면 캡처
        if key == ord('c'):
            frame_count += 1
            # 파일 이름: capture_YYYYMMDD_HHMMSS_N.png
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(save_dir, f"capture_{timestamp}_{frame_count}.png")

            # 이미지 파일 저장
            if cv2.imwrite(filename, frame):
                print(f"✅ 캡처 완료: {filename}에 저장되었습니다.")
            else:
                print(f"❌ 오류: {filename} 저장 실패.")

        # 2. 'q' 키를 누르면 루프 종료
        elif key == ord('q'):
            print("👋 스트리밍을 종료합니다.")
            break

    # 스트림이 끝나면 리소스 해제
    cap.release()
    # 열려 있는 모든 OpenCV 창 닫기
    cv2.destroyAllWindows()

if __name__ == "__main__":
    stream_and_capture()