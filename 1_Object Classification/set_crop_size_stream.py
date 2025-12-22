import cv2
import numpy as np

# --- 설정 변수 ---
CAMERA_INDEX = 3  # 사용자가 지정한 3번 카메라
WINDOW_NAME = "Real-time ROI Selector"
# ----------------

# Global 변수 설정
cropping = False
x_start, y_start, x_end, y_end = 0, 0, 0, 0
crop_roi = None  # 최종 크롭 영역 (x_min, y_min, x_max, y_max)

def mouse_callback(event, x, y, flags, param):
    """마우스 이벤트 콜백 함수: 실시간 드래그 및 ROI 좌표 계산"""
    global x_start, y_start, x_end, y_end, cropping, crop_roi

    if event == cv2.EVENT_LBUTTONDOWN:
        # 마우스 왼쪽 버튼 클릭 시 시작점 설정
        x_start, y_start = x, y
        x_end, y_end = x, y
        cropping = True

    elif event == cv2.EVENT_MOUSEMOVE:
        # 마우스 이동 시 현재 끝점 업데이트 (드래그 시각화용)
        if cropping:
            x_end, y_end = x, y

    elif event == cv2.EVENT_LBUTTONUP:
        # 마우스 버튼을 떼면 최종 ROI 확정
        x_end, y_end = x, y
        cropping = False
        
        # 좌상단(p1)과 우하단(p2) 좌표 계산 (음수 드래그 대응)
        p1 = (min(x_start, x_end), min(y_start, y_end))
        p2 = (max(x_start, x_end), max(y_start, y_end))
        
        crop_roi = (p1[0], p1[1], p2[0], p2[1])
        
        print("\n" + "="*50)
        print(f"🎯 ROI 설정 완료!")
        print(f"시작 좌표 (x_min, y_min): {p1}")
        print(f"끝   좌표 (x_max, y_max): {p2}")
        print(f"Width: {p2[0]-p1[0]}, Height: {p2[1]-p1[1]}")
        print("="*50)

def main():
    global cropping, x_start, y_start, x_end, y_end, crop_roi

    # 카메라 연결
    cap = cv2.VideoCapture(CAMERA_INDEX)

    if not cap.isOpened():
        print(f"오류: {CAMERA_INDEX}번 카메라를 열 수 없습니다.")
        return

    cv2.namedWindow(WINDOW_NAME)
    cv2.setMouseCallback(WINDOW_NAME, mouse_callback)

    print("--- 제어 방법 ---")
    print("[R] : ROI 초기화")
    print("[Q] : 프로그램 종료")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("프레임을 가져올 수 없습니다.")
            break

        display_frame = frame.copy()

        # 1. 드래그 중인 영역 표시 (녹색 실선)
        if cropping:
            cv2.rectangle(display_frame, (x_start, y_start), (x_end, y_end), (0, 255, 0), 2)
        
        # 2. 확정된 ROI 영역 표시 (빨간색 두꺼운 선)
        if crop_roi:
            x1, y1, x2, y2 = crop_roi
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
            info_text = f"ROI: ({x1}, {y1}) - ({x2}, {y2})"
            cv2.putText(display_frame, info_text, (10, frame.shape[0] - 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # 안내 문구
        cv2.putText(display_frame, "Drag to set ROI | R: Reset | Q: Quit", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow(WINDOW_NAME, display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            crop_roi = None
            x_start, y_start, x_end, y_end = 0, 0, 0, 0
            print("ROI 초기화됨.")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()