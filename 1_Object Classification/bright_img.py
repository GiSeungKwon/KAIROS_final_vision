import cv2
import os
import glob
import numpy as np
from typing import Dict, Tuple, List

# --- 1. 설정 변수 ---

# 각 모듈별 입력 폴더 및 출력 폴더 설정
# 입력 경로는 이전 단계의 Rotated 폴더입니다.
MODULE_PATHS: Dict[str, Tuple[str, str]] = {
    "ESP32": (
        r"C:\Dev\KAIROS_Project\Vision\0_data\ObjectClassification\Rotated_ESP32",
        r"C:\Dev\KAIROS_Project\Vision\0_data\ObjectClassification\Aug_ESP32"
    ),
    "L298N": (
        r"C:\Dev\KAIROS_Project\Vision\0_data\ObjectClassification\Rotated_L298N",
        r"C:\Dev\KAIROS_Project\Vision\0_data\ObjectClassification\Aug_L298N"
    ),
    "MB102": (
        r"C:\Dev\KAIROS_Project\Vision\0_data\ObjectClassification\Rotated_MB102",
        r"C:\Dev\KAIROS_Project\Vision\0_data\ObjectClassification\Aug_MB102"
    )
}

# 적용할 밝기 계수 리스트 (1.0은 원본, 0.9는 어둡게, 1.1은 밝게)
BRIGHTNESS_FACTORS: List[float] = [0.9, 1.1] 

IMAGE_EXT = "jpg" # 처리할 이미지 확장자
# --------------------

def adjust_brightness(image, factor: float):
    """
    이미지의 밝기를 지정된 계수(factor)만큼 조절합니다.
    """
    # NumPy 배열로 변환하여 픽셀 값에 계수를 곱합니다.
    # cv2.convertScaleAbs를 사용하여 자동으로 0~255 범위로 클리핑합니다.
    return cv2.convertScaleAbs(image, alpha=factor, beta=0)

def process_brightness_augmentation(module_name: str, input_dir: str, output_dir: str, factors: List[float]):
    """
    지정된 폴더의 이미지를 로드하여 증강(밝기 변화) 후 새 폴더에 저장하는 함수
    """
    
    print(f"\n--- {module_name} 모듈 밝기 증강 처리 시작 ---")
    print(f"원본 경로: {input_dir}")
    print(f"저장 경로: {output_dir}")

    # 1. 출력 폴더 생성 (없으면 자동 생성)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"-> 출력 폴더 생성 완료: {output_dir}")

    # 2. 이미지 파일 목록 가져오기
    search_path = os.path.join(input_dir, f"*.{IMAGE_EXT}")
    file_list = glob.glob(search_path)
    
    if not file_list:
        print(f"-> 경고: 입력 경로에 .{IMAGE_EXT} 파일이 없습니다. 처리를 건너뜁니다.")
        return

    processed_count = 0
    
    # 3. 파일 순회하며 증강 및 저장
    for file_path in file_list:
        try:
            img = cv2.imread(file_path)
            if img is None:
                continue

            file_name = os.path.basename(file_path)
            base_name, ext = os.path.splitext(file_name)

            # 3-1. 원본 이미지 (factor 1.0) 저장
            original_save_path = os.path.join(output_dir, file_name)
            # 원본 파일을 복사하여 Augmentation 폴더에 저장
            cv2.imwrite(original_save_path, img)
            processed_count += 1
            
            # 3-2. 밝기 변화 이미지 저장
            for factor in factors:
                augmented_img = adjust_brightness(img, factor)
                # 저장 파일명: '원본이름_bri계수.jpg' (예: _bri0.9, _bri1.1)
                factor_str = str(factor).replace('.', '') # 0.9 -> 09, 1.1 -> 11
                augmented_file_name = f"{base_name}_bri{factor_str}{ext}"
                augmented_save_path = os.path.join(output_dir, augmented_file_name)
                
                cv2.imwrite(augmented_save_path, augmented_img)
                processed_count += 1

        except Exception as e:
            print(f"  [오류 발생] 파일: {os.path.basename(file_path)}, 오류: {e}")
            
    # 최종 결과 보고: 원본 이미지 수 * (1(원본) + 2(밝기 계수))
    total_expected = len(file_list) * (1 + len(factors))
    print(f"--- {module_name} 처리 완료: 총 {processed_count}개 파일 저장 ---")
    if processed_count != total_expected:
        print(f"  [주의] 예상 저장 파일 수({total_expected})와 다릅니다.")


def main():
    """메인 함수: 설정된 모든 모듈에 대해 밝기 증강 일괄 처리 실행"""
    
    total_modules = len(MODULE_PATHS)
    print(f"*** 총 {total_modules}개 모듈의 밝기 증강 일괄 처리 시작 ***")
    
    for module_name, (input_dir, output_dir) in MODULE_PATHS.items():
        process_brightness_augmentation(module_name, input_dir, output_dir, BRIGHTNESS_FACTORS)
        
    print("\n*** 모든 데이터 증강 작업 완료 ***")


if __name__ == "__main__":
    main()