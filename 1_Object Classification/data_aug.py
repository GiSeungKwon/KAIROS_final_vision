import os
import cv2
import numpy as np

# 사용자님이 제시한 기본 경로
BASE_DIR = 'C:/Dev/KAIROS_Project' 
DATA_DIR = os.path.join(BASE_DIR, 'data')

# 입력 및 출력 폴더 설정
INPUT_FOLDERS = {
    'ESP32': 'Anomaly_ESP32',
    'MB102': 'Anomaly_MB102',
    'L298N': 'Anomaly_L298N'
}

OUTPUT_FOLDERS = {
    'ESP32': 'aug_Anomaly_ESP32',
    'MB102': 'aug_Anomaly_MB102',
    'L298N': 'aug_Anomaly_L298N'
}

# 증강 조건 설정
ROTATIONS = [0, 90, 180, 270] # 회전 각도 (도)
BRIGHTNESS_FACTORS = [0.8, 0.9, 1.0, 1.1, 1.2] # 밝기 변화 계수

# 최종 크롭 영역 (x_start, y_start, x_end, y_end)
# (420, 0) / (1600, 1060)
CROP_AREA = (420, 0, 1600, 1060) 

def apply_rotation(image, angle):
    """주어진 각도로 이미지를 회전합니다."""
    if angle == 0:
        return image
    
    # 이미지의 높이와 너비
    (h, w) = image.shape[:2]
    # 중심 좌표
    center = (w / 2, h / 2)
    
    # 회전 행렬 계산
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # 회전 적용
    rotated_image = cv2.warpAffine(image, M, (w, h))
    return rotated_image

def apply_brightness(image, factor):
    """밝기 변화 계수를 적용하여 이미지를 조정합니다."""
    # OpenCV는 0-255 범위의 BGR 이미지를 사용
    # G = B * factor (여기서 B는 밝기 factor)
    
    # 이미지의 모든 픽셀 값에 factor를 곱하고 클리핑 (0~255 범위 유지)
    augmented_image = np.clip(image * factor, 0, 255).astype(np.uint8)
    return augmented_image

def apply_crop(image, crop_area):
    """지정된 영역으로 이미지를 크롭합니다."""
    x_start, y_start, x_end, y_end = crop_area
    # 크롭 영역이 이미지 크기를 벗어나지 않도록 주의 (입력 이미지는 크롭 영역보다 크다고 가정)
    cropped_image = image[y_start:y_end, x_start:x_end]
    return cropped_image

def augment_data():
    """데이터 증강 프로세스를 실행합니다."""
    for key in INPUT_FOLDERS.keys():
        input_folder_name = INPUT_FOLDERS[key]
        output_folder_name = OUTPUT_FOLDERS[key]
        
        input_path = os.path.join(DATA_DIR, input_folder_name)
        output_path = os.path.join(DATA_DIR, output_folder_name)
        
        # 출력 폴더가 없으면 생성
        os.makedirs(output_path, exist_ok=True)
        print(f"--- {input_folder_name} -> {output_folder_name} 증강 시작 ---")
        
        # 입력 폴더의 파일 목록을 가져옵니다.
        # 이미지 파일(png, jpg, jpeg 등)만 처리하도록 확장자 확인 로직 추가 가능
        image_files = [f for f in os.listdir(input_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        total_count = 0
        
        for filename in image_files:
            original_filepath = os.path.join(input_path, filename)
            # 이미지 로드 (컬러로 읽기)
            # cv2.IMREAD_COLOR (1)을 사용하여 이미지 로드
            original_image = cv2.imread(original_filepath, cv2.IMREAD_COLOR)
            
            if original_image is None:
                print(f"경고: {original_filepath} 파일을 읽을 수 없습니다. 스킵합니다.")
                continue

            base_name, ext = os.path.splitext(filename)
            
            for rot_angle in ROTATIONS:
                # 1. 회전 적용
                rotated_img = apply_rotation(original_image, rot_angle)
                
                for factor in BRIGHTNESS_FACTORS:
                    # 2. 밝기 변화 적용
                    bright_img = apply_brightness(rotated_img, factor)
                    
                    # 3. 최종 크롭 적용
                    # 크롭은 모든 증강 후 마지막에 적용되어야 합니다.
                    final_img = apply_crop(bright_img, CROP_AREA)
                    
                    # 파일명 생성: 원본이름_R{각도}_B{계수}
                    # 예: image001_R0_B1.0.jpg
                    factor_str = str(factor).replace('.', '_') # 1.0 -> 1_0
                    new_filename = f"{base_name}_R{rot_angle}_B{factor_str}{ext}"
                    new_filepath = os.path.join(output_path, new_filename)
                    
                    # 이미지 저장
                    cv2.imwrite(new_filepath, final_img)
                    total_count += 1
        
        print(f"✅ {input_folder_name} 처리 완료. 총 {total_count}개의 증강 이미지 생성.")
        print(f"원본 200장 * 4(회전) * 5(밝기) = {200 * 4 * 5}장 예상")
        print("-" * 40)

if __name__ == "__main__":
    augment_data()