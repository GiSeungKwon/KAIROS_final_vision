import cv2
import numpy as np
import os
import sys

# ----------------------------------------------------
# 1. 설정 변수 (Configuration)
# ----------------------------------------------------
# 입력 이미지 경로
IMAGE_PATH = r"C:\Dev\KAIROS_Project\data\Anomaly_ESP32\WIN_20251211_19_51_13_Pro.jpg"

# ROI 설정 (학습 코드와 동일하게 유지)
ROI_START = (500, 300)  # (x_min, y_min)
ROI_END = (1800, 1000)  # (x_max, y_max)

# OpenCV 창 이름
WINDOW_NAME = "MyCobot HSV Masking Tool (Image)"
TRACKBAR_WINDOW_NAME = "HSV Controls"

# ----------------------------------------------------
# 2. 전역 변수 및 유틸리티 함수
# ----------------------------------------------------
def nothing(x):
    pass

def apply_roi_and_hsv_masking(image, hsv_low, hsv_high):
    x_min, y_min = ROI_START
    x_max, y_max = ROI_END

    if x_max <= x_min or y_max <= y_min:
        return np.zeros_like(image)

    # ROI 외부 제거
    masked_image_roi = image.copy()
    masked_image_roi[0:y_min, :] = 0
    masked_image_roi[y_max:, :] = 0
    masked_image_roi[:, 0:x_min] = 0
    masked_image_roi[:, x_max:] = 0

    # HSV 변환
    hsv = cv2.cvtColor(masked_image_roi, cv2.COLOR_BGR2HSV)

    # HSV 마스크
    hsv_mask = cv2.inRange(hsv, hsv_low, hsv_high)

    # 바이너리 결과 (3채널)
    final_binary_image = np.zeros_like(image)
    final_binary_image[hsv_mask > 0] = [255, 255, 255]

    return final_binary_image

# ----------------------------------------------------
# 3. 메인 실행
# ----------------------------------------------------
def main():
    # 1. 이미지 로드
    if not os.path.exists(IMAGE_PATH):
        print(f"❌ 이미지 파일을 찾을 수 없습니다:\n{IMAGE_PATH}")
        sys.exit(1)

    image = cv2.imread(IMAGE_PATH)
    if image is None:
        print("❌ 이미지 로드 실패")
        sys.exit(1)

    # 2. 윈도우 및 트랙바 생성
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 1200, 600)

    cv2.namedWindow(TRACKBAR_WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(TRACKBAR_WINDOW_NAME, 400, 300)

    cv2.createTrackbar('H_Low', TRACKBAR_WINDOW_NAME, 0, 179, nothing)
    cv2.createTrackbar('S_Low', TRACKBAR_WINDOW_NAME, 0, 255, nothing)
    cv2.createTrackbar('V_Low', TRACKBAR_WINDOW_NAME, 0, 255, nothing)
    cv2.createTrackbar('H_High', TRACKBAR_WINDOW_NAME, 179, 179, nothing)
    cv2.createTrackbar('S_High', TRACKBAR_WINDOW_NAME, 255, 255, nothing)
    cv2.createTrackbar('V_High', TRACKBAR_WINDOW_NAME, 255, 255, nothing)

    print("\n--- 🖼️ 이미지 HSV 마스킹 도구 ---")
    print(f"✅ 이미지 경로:\n{IMAGE_PATH}")
    print(f"✅ ROI: {ROI_START} ~ {ROI_END}")
    print("🖱️ 트랙바 조절 → 마스킹 결과 확인")
    print("   [q] 또는 [ESC] : 종료")
    print("----------------------------------")

    while True:
        # HSV 값 읽기
        h_low = cv2.getTrackbarPos('H_Low', TRACKBAR_WINDOW_NAME)
        s_low = cv2.getTrackbarPos('S_Low', TRACKBAR_WINDOW_NAME)
        v_low = cv2.getTrackbarPos('V_Low', TRACKBAR_WINDOW_NAME)
        h_high = cv2.getTrackbarPos('H_High', TRACKBAR_WINDOW_NAME)
        s_high = cv2.getTrackbarPos('S_High', TRACKBAR_WINDOW_NAME)
        v_high = cv2.getTrackbarPos('V_High', TRACKBAR_WINDOW_NAME)

        hsv_low = np.array([h_low, s_low, v_low])
        hsv_high = np.array([h_high, s_high, v_high])

        # 원본 + ROI 표시
        original_vis = image.copy()
        cv2.rectangle(original_vis, ROI_START, ROI_END, (0, 0, 255), 2)
        cv2.putText(
            original_vis,
            "Original Image (with ROI)",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )

        # 마스킹 결과
        processed_image = apply_roi_and_hsv_masking(image, hsv_low, hsv_high)
        cv2.putText(
            processed_image,
            "Processed Output (Mask)",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        # 나란히 표시
        combined = np.hstack([original_vis, processed_image])
        cv2.imshow(WINDOW_NAME, combined)

        # HSV 값 출력
        sys.stdout.write(
            f"\r🔍 HSV Range: [{h_low}, {s_low}, {v_low}] ~ [{h_high}, {s_high}, {v_high}] "
        )
        sys.stdout.flush()

        key = cv2.waitKey(30) & 0xFF
        if key == ord('q') or key == 27:
            print("\n👋 종료")
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
