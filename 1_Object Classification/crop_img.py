import cv2
import os
import glob
from typing import Dict, Tuple

# --- 1. 설정 변수 ---

# 각 모듈별 입력 폴더, 출력 폴더 및 크롭 좌표 (x1, y1, x2, y2)
# 좌표: (좌상단 x, 좌상단 y) / (우하단 x, 우하단 y)
CROP_CONFIG: Dict[str, Tuple[str, str, Tuple[int, int, int, int]]] = {
    "ESP32": (
        r"C:\Dev\KAIROS_Project\Vision\0_data\ObjectClassification\ESP32",
        r"C:\Dev\KAIROS_Project\Vision\0_data\ObjectClassification\Croped_ESP32",
        (260, 0, 1325, 1080) # (x1, y1, x2, y2)
    ),
    "L298N": (
        r"C:\Dev\KAIROS_Project\Vision\0_data\ObjectClassification\L298N",
        r"C:\Dev\KAIROS_Project\Vision\0_data\ObjectClassification\Croped_L298N",
        (230, 28, 1274, 1012)
    ),
    "MB102": (
        r"C:\Dev\KAIROS_Project\Vision\0_data\ObjectClassification\MB102",
        r"C:\Dev\KAIROS_Project\Vision\0_data\ObjectClassification\Croped_MB102",
        (194, 0, 1240, 1080)
    )
}

IMAGE_EXT = "jpg" # 처리할 이미지 확장자
# --------------------

def process_batch_crop(module_name: str, input_dir: str, output_dir: str, crop_coords: Tuple[int, int, int, int]):
    """
    지정된 폴더의 모든 이미지를 크롭 좌표에 따라 처리하고 새 폴더에 저장하는 함수
    """
    x1, y1, x2, y2 = crop_coords
    
    print(f"\n--- {module_name} 모듈 처리 시작 ---")
    print(f"입력 경로: {input_dir}")
    print(f"출력 경로: {output_dir}")
    print(f"크롭 좌표: ({x1}, {y1}) ~ ({x2}, {y2})")

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
    
    # 3. 파일 순회하며 크롭 및 저장
    for file_path in file_list:
        try:
            # 이미지 로드
            img = cv2.imread(file_path)
            if img is None:
                print(f"  [Skip] 이미지 로드 실패: {os.path.basename(file_path)}")
                continue

            # 크롭 수행: [y1:y2, x1:x2]
            cropped_img = img[y1:y2, x1:x2]

            # 파일 저장 경로 설정
            file_name = os.path.basename(file_path)
            save_path = os.path.join(output_dir, file_name)

            # 크롭된 이미지 저장
            cv2.imwrite(save_path, cropped_img)
            processed_count += 1

        except Exception as e:
            print(f"  [오류 발생] 파일: {os.path.basename(file_path)}, 오류: {e}")
            
    print(f"--- {module_name} 처리 완료: 총 {processed_count}개의 파일 저장 완료 ---")


def main():
    """메인 함수: 설정된 모든 모듈에 대해 일괄 처리 실행"""
    
    total_modules = len(CROP_CONFIG)
    print(f"*** 총 {total_modules}개 모듈의 크롭 일괄 처리 시작 ***")
    
    for module_name, (input_dir, output_dir, crop_coords) in CROP_CONFIG.items():
        process_batch_crop(module_name, input_dir, output_dir, crop_coords)
        
    print("\n*** 모든 크롭 일괄 처리 작업 완료 ***")


if __name__ == "__main__":
    main()