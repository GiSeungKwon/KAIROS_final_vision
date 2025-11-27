import cv2
import os
import glob
from typing import Dict, Tuple, List

# --- 1. 설정 변수 ---

# 각 모듈별 입력 폴더 및 출력 폴더 설정
# (입력 폴더 경로, 출력 폴더 경로)
MODULE_PATHS: Dict[str, Tuple[str, str]] = {
    "ESP32": (
        r"C:\Dev\KAIROS_Project\Vision\0_data\ObjectClassification\Croped_ESP32",
        r"C:\Dev\KAIROS_Project\Vision\0_data\ObjectClassification\Rotated_ESP32"
    ),
    "L298N": (
        r"C:\Dev\KAIROS_Project\Vision\0_data\ObjectClassification\Croped_L298N",
        r"C:\Dev\KAIROS_Project\Vision\0_data\ObjectClassification\Rotated_L298N"
    ),
    "MB102": (
        r"C:\Dev\KAIROS_Project\Vision\0_data\ObjectClassification\Croped_MB102",
        r"C:\Dev\KAIROS_Project\Vision\0_data\ObjectClassification\Rotated_MB102"
    )
}

# 적용할 회전 각도 리스트 (도 단위)
ROTATION_ANGLES: List[int] = [10, 20, 30] 

IMAGE_EXT = "jpg" # 처리할 이미지 확장자
# --------------------

def rotate_image(image, angle: int):
    """
    이미지를 지정된 각도만큼 회전시키고 원본 크기를 유지합니다.
    """
    (h, w) = image.shape[:2]
    # 이미지의 중심을 회전 기준으로 설정
    center = (w // 2, h // 2)

    # 회전 행렬 계산
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # 아핀 변환(Affine Transformation) 적용
    rotated = cv2.warpAffine(image, M, (w, h), borderValue=(0, 0, 0)) # 검은색(0,0,0)으로 비어있는 영역 채움
    return rotated

def process_rotation_augmentation(module_name: str, input_dir: str, output_dir: str, angles: List[int]):
    """
    지정된 폴더의 이미지를 로드하여 증강(회전) 후 새 폴더에 저장하는 함수
    """
    
    print(f"\n--- {module_name} 모듈 증강 처리 시작 ---")
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

            # 3-1. 원본 이미지 저장
            original_save_path = os.path.join(output_dir, file_name)
            cv2.imwrite(original_save_path, img)
            processed_count += 1
            
            # 3-2. 회전 증강 이미지 저장
            for angle in angles:
                rotated_img = rotate_image(img, angle)
                # 저장 파일명: '원본이름_rot각도.jpg'
                rotated_file_name = f"{base_name}_rot{angle}{ext}"
                rotated_save_path = os.path.join(output_dir, rotated_file_name)
                
                cv2.imwrite(rotated_save_path, rotated_img)
                processed_count += 1

        except Exception as e:
            print(f"  [오류 발생] 파일: {os.path.basename(file_path)}, 오류: {e}")
            
    # 최종 결과 보고: 원본 이미지 수 * (1(원본) + 3(회전 각도))
    total_expected = len(file_list) * (1 + len(angles))
    print(f"--- {module_name} 처리 완료: 총 {processed_count}개 파일 저장 ({len(file_list)}개 원본) ---")
    if processed_count != total_expected:
        print(f"  [주의] 예상 저장 파일 수({total_expected})와 다릅니다.")


def main():
    """메인 함수: 설정된 모든 모듈에 대해 회전 증강 일괄 처리 실행"""
    
    total_modules = len(MODULE_PATHS)
    print(f"*** 총 {total_modules}개 모듈의 회전 증강 일괄 처리 시작 ***")
    
    for module_name, (input_dir, output_dir) in MODULE_PATHS.items():
        process_rotation_augmentation(module_name, input_dir, output_dir, ROTATION_ANGLES)
        
    print("\n*** 모든 데이터 증강 작업 완료 ***")


if __name__ == "__main__":
    main()