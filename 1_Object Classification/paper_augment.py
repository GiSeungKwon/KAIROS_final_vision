import cv2
import numpy as np
import random
import os
from pathlib import Path

# 설정 값
INPUT_DIR = r"C:\Dev\KAIROS_Project\data\Anomaly_L298N"
OUTPUT_DIR = r"C:\Dev\KAIROS_Project\data\Anomaly_augmented\Anomaly_img_L298N"
# CROP_AREA: (x1, y1, x2, y2)
CROP_AREA = (420, 0, 1600, 1060)
# 증강 적용 범위 (좌표는 원본 기준)
AUG_X1, AUG_Y1, AUG_X2, AUG_Y2 = 820, 374, 1152, 732

def crop_image(image):
    x1, y1, x2, y2 = CROP_AREA
    return image[y1:y2, x1:x2]

def get_random_pos_in_range():
    rx = random.randint(AUG_X1, AUG_X2)
    ry = random.randint(AUG_Y1, AUG_Y2)
    # Crop된 이미지 기준으로 좌표 변환
    return rx - CROP_AREA[0], ry - CROP_AREA[1]

def apply_color_mosaic(image, zeta=20):
    img_aug = image.copy()
    cx, cy = get_random_pos_in_range()
    r = 150  # 요청하신 대로 반지름을 150으로 설정 (지름 300)
    
    # 1. 원을 포함하는 사각형 영역(ROI) 추출
    y1, y2 = max(0, cy-r), min(image.shape[0], cy+r)
    x1, x2 = max(0, cx-r), min(image.shape[1], cx+r)
    
    roi = img_aug[y1:y2, x1:x2].copy()
    if roi.size == 0: return img_aug
    
    # 2. ROI 전체에 모자이크 적용
    small = cv2.resize(roi, (max(1, roi.shape[1]//zeta), max(1, roi.shape[0]//zeta)), interpolation=cv2.INTER_LINEAR)
    mosaic_roi = cv2.resize(small, (roi.shape[1], roi.shape[0]), interpolation=cv2.INTER_NEAREST)
    
    # 3. 무작위 색상 선택 (R, G, B 중 하나)
    # OpenCV는 BGR 순서이므로 (B, G, R)로 정의합니다.
    colors = [
        (0, 0, 255),   # Red
        (0, 255, 0),   # Green
        (255, 0, 0)    # Blue
    ]
    chosen_color = random.choice(colors)
    
    # 4. 모자이크 이미지에 색상 입히기 (색상 오버레이)
    # 모자이크된 결과물과 선택된 색상을 7:3 비율로 섞어 결함 느낌을 줍니다.
    colored_mosaic = cv2.addWeighted(mosaic_roi, 0.7, np.full(roi.shape, chosen_color, dtype=np.uint8), 0.3, 0)
    
    # 5. 원형 마스크 생성
    mask_cx, mask_cy = cx - x1, cy - y1
    mask = np.zeros((roi.shape[0], roi.shape[1]), dtype=np.uint8)
    cv2.circle(mask, (mask_cx, mask_cy), r, 255, -1)
    
    # 6. 마스크를 이용하여 원형 부분만 합성
    img_aug[y1:y2, x1:x2] = np.where(mask[:, :, None] == 255, colored_mosaic, roi)
    
    return img_aug

def apply_mosaic(image, zeta=20):
    img_aug = image.copy()
    cx, cy = get_random_pos_in_range()
    r = 100
    
    # 1. 원을 포함하는 사각형 영역(ROI) 추출
    y1, y2 = max(0, cy-r), min(image.shape[0], cy+r)
    x1, x2 = max(0, cx-r), min(image.shape[1], cx+r)
    
    roi = img_aug[y1:y2, x1:x2].copy()
    if roi.size == 0: return img_aug
    
    # 2. ROI 전체에 모자이크 적용
    small = cv2.resize(roi, (max(1, roi.shape[1]//zeta), max(1, roi.shape[0]//zeta)), interpolation=cv2.INTER_LINEAR)
    mosaic_roi = cv2.resize(small, (roi.shape[1], roi.shape[0]), interpolation=cv2.INTER_NEAREST)
    
    # 3. 원형 마스크 생성 (ROI와 같은 크기)
    # ROI 내에서의 중심점 계산
    mask_cx, mask_cy = cx - x1, cy - y1
    mask = np.zeros((roi.shape[0], roi.shape[1]), dtype=np.uint8)
    cv2.circle(mask, (mask_cx, mask_cy), r, 255, -1)
    
    # 4. 마스크를 이용하여 원형 부분만 합성
    # mask가 255인 곳은 mosaic_roi를, 0인 곳은 원본 roi를 유지
    img_aug[y1:y2, x1:x2] = np.where(mask[:, :, None] == 255, mosaic_roi, roi)
    
    return img_aug

def apply_liquify(image, eta=0.05):
    img_aug = image.copy()
    h, w = img_aug.shape[:2]
    
    # 50x50 영역의 중심점을 시작점과 끝점으로 무작위 선택
    src_x, src_y = get_random_pos_in_range()
    dst_x, dst_y = get_random_pos_in_range()
    
    rows, cols = img_aug.shape[:2]
    map_x, map_y = np.meshgrid(np.arange(cols), np.arange(rows))
    map_x = map_x.astype('float32')
    map_y = map_y.astype('float32')
    
    # 50x50 크기의 영향 범위를 주기 위한 가우시안 마스크 (시각적 반경 약 25)
    dist = np.sqrt((map_x - src_x)**2 + (map_y - src_y)**2)
    mask = np.exp(-dist**2 / (2 * (25)**2))
    
    map_x += (dst_x - src_x) * mask
    map_y += (dst_y - src_y) * mask
    
    return cv2.remap(img_aug, map_x, map_y, cv2.INTER_LINEAR)

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    image_files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(image_extensions)]
    
    print(f"총 {len(image_files)}개의 이미지를 처리합니다...")

    for filename in image_files:
        img_path = os.path.join(INPUT_DIR, filename)
        img = cv2.imread(img_path)
        
        if img is None: continue
        
        # 1. 먼저 Crop 적용
        img_cropped = crop_image(img)
        
        # 2. 증강 적용 (Mosaic, Liquify, Mosiquify)
        res_mosaic = apply_mosaic(img_cropped)
        res_liquify = apply_liquify(img_cropped)
        res_color_mosaic = apply_color_mosaic(img_cropped)

        # 3. 저장
        base_name = Path(filename).stem
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"{base_name}_mosaic.jpg"), res_mosaic)
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"{base_name}_liquify.jpg"), res_liquify)
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"{base_name}_color_mosaic.jpg"), res_color_mosaic)

    print("모든 증강 이미지가 성공적으로 저장되었습니다.")

if __name__ == "__main__":
    main()