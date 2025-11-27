import cv2
import numpy as np
import os
import glob

# --- 설정 변수 ---
# 이미지가 있는 폴더 경로
# ESP32 -> (260, 0) / (1325, 1080)
# IMAGE_DIR = r"C:\Dev\KAIROS_Project\Vision\0_data\ObjectClassification\ESP32"

# L298N -> (230, 28) / (1274, 1012)
# IMAGE_DIR = r"C:\Dev\KAIROS_Project\Vision\0_data\ObjectClassification\L298N"

# MB102 -> (194, 0) / (1240, 1080)
IMAGE_DIR = r"C:\Dev\KAIROS_Project\Vision\0_data\ObjectClassification\MB102"

# 이미지 파일 확장자 (jpg, png 등)
IMAGE_EXT = "jpg"
# 창의 크기 (원본 해상도 1920x1080에 맞춰 640x360으로 리사이즈하여 표시)
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 450
# ----------------

# Global 변수 설정
cropping = False
x_start, y_start, x_end, y_end = 0, 0, 0, 0
crop_roi = None # 최종 크롭 영역 (원본 이미지 기준)
current_img = None
display_img = None
file_list = []
file_index = 0

def mouse_callback(event, x, y, flags, param):
    """마우스 이벤트 콜백 함수: 크롭 영역 지정 및 드래그 처리"""
    global x_start, y_start, x_end, y_end, cropping, crop_roi, current_img, display_img

    # 현재 x, y 좌표는 표시용 이미지(Resize된 이미지) 기준입니다.
    
    if event == cv2.EVENT_LBUTTONDOWN:
        # 마우스 왼쪽 버튼을 누르면 크롭 시작
        x_start, y_start = x, y
        x_end, y_end = x, y
        cropping = True

    elif event == cv2.EVENT_MOUSEMOVE:
        # 마우스 이동 시 사각형 업데이트
        if cropping:
            x_end, y_end = x, y

    elif event == cv2.EVENT_LBUTTONUP:
        # 마우스 왼쪽 버튼을 떼면 크롭 종료
        x_end, y_end = x, y
        cropping = False

        # 표시용 이미지의 좌표를 원본 이미지 좌표로 변환
        h_orig, w_orig = current_img.shape[:2]
        h_disp, w_disp = display_img.shape[:2]
        
        # 스케일 팩터 계산
        scale_w = w_orig / w_disp
        scale_h = h_orig / h_disp
        
        # 원본 이미지 기준 좌표 계산 (좌상단: p1, 우하단: p2)
        p1 = (int(min(x_start, x_end) * scale_w), int(min(y_start, y_end) * scale_h))
        p2 = (int(max(x_start, x_end) * scale_w), int(max(y_start, y_end) * scale_h))
        
        # 크롭 ROI 설정 (원본 기준)
        crop_roi = (p1[0], p1[1], p2[0], p2[1]) # (x_min, y_min, x_max, y_max)
        print("\n--- 크롭 영역 설정 완료 (원본 이미지 기준) ---")
        print(f"좌상단 좌표 (x_min, y_min): {p1}")
        print(f"우하단 좌표 (x_max, y_max): {p2}")
        print("-" * 40)
        
        # ROI 설정 후 바로 이미지 업데이트
        update_image()

def update_image():
    """현재 인덱스의 이미지를 로드하고, ROI를 표시하여 화면에 업데이트"""
    global current_img, display_img, file_index, file_list, crop_roi
    
    if not file_list:
        return

    # 1. 이미지 로드
    file_path = file_list[file_index]
    current_img = cv2.imread(file_path)

    if current_img is None:
        print(f"오류: 이미지 파일을 로드할 수 없습니다: {file_path}")
        return

    # 2. 표시용 이미지(Resize) 생성
    h_orig, w_orig = current_img.shape[:2]
    display_img = cv2.resize(current_img.copy(), (WINDOW_WIDTH, WINDOW_HEIGHT))
    
    # 3. 크롭 ROI 표시 (crop_roi가 설정되어 있을 경우)
    if crop_roi:
        x_min_orig, y_min_orig, x_max_orig, y_max_orig = crop_roi
        
        # 원본 좌표를 표시용 이미지 좌표로 변환
        scale_w = WINDOW_WIDTH / w_orig
        scale_h = WINDOW_HEIGHT / h_orig
        
        x1_disp = int(x_min_orig * scale_w)
        y1_disp = int(y_min_orig * scale_h)
        x2_disp = int(x_max_orig * scale_w)
        y2_disp = int(y_max_orig * scale_h)
        
        # 사각형 그리기 (빨간색)
        cv2.rectangle(display_img, (x1_disp, y1_disp), (x2_disp, y2_disp), (0, 0, 255), 2)
        
        # 텍스트 정보 표시
        info_text = f"ROI: ({x_min_orig}, {y_min_orig}) - ({x_max_orig}, {y_max_orig})"
        cv2.putText(display_img, info_text, (10, WINDOW_HEIGHT - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # 4. 드래그 중인 사각형 표시 (cropping 중일 경우)
    elif cropping:
        cv2.rectangle(display_img, (x_start, y_start), (x_end, y_end), (0, 255, 0), 2)

    # 5. 현재 파일 정보 표시
    cv2.putText(display_img, f"File {file_index + 1}/{len(file_list)}: {os.path.basename(file_path)}", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    cv2.putText(display_img, "Controls: [A] Prev | [D] Next | [R] Reset ROI | [Q] Quit", 
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    cv2.imshow("Crop Area Checker", display_img)

def main():
    global file_list, file_index, crop_roi
    
    # 파일 목록 로드
    search_path = os.path.join(IMAGE_DIR, f"*.{IMAGE_EXT}")
    file_list = sorted(glob.glob(search_path))
    
    if not file_list:
        print(f"경고: {IMAGE_DIR} 경로에 .{IMAGE_EXT} 파일이 없습니다.")
        return

    # 윈도우 생성 및 콜백 연결
    cv2.namedWindow("Crop Area Checker")
    cv2.setMouseCallback("Crop Area Checker", mouse_callback)

    # 초기 이미지 로드 및 표시
    update_image()

    while True:
        key = cv2.waitKey(0) & 0xFF

        if key == ord('q'):
            # Q: 종료
            break
        elif key == ord('d'):
            # D: 다음 이미지
            if file_index < len(file_list) - 1:
                file_index += 1
                update_image()
            else:
                print("마지막 파일입니다.")
        elif key == ord('a'):
            # A: 이전 이미지
            if file_index > 0:
                file_index -= 1
                update_image()
            else:
                print("첫 번째 파일입니다.")
        elif key == ord('r'):
            # R: ROI 리셋
            crop_roi = None
            print("크롭 영역이 초기화되었습니다. 마우스로 다시 지정해주세요.")
            update_image()

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()