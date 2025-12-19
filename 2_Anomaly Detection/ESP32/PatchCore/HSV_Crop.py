import cv2
import numpy as np
from pathlib import Path

class CropInspector:
    def __init__(self, folder_path):
        self.folder_path = Path(folder_path)
        # 이미지 파일 리스트 확보
        self.img_files = sorted(list(self.folder_path.glob("*.jpg")) + list(self.folder_path.glob("*.png")))
        self.current_idx = 0
        
        # HSV 설정값
        self.lower = np.array([0, 0, 70])
        self.upper = np.array([179, 255, 255])
        self.kernel = np.ones((5, 5), np.uint8)

    def process_image(self, img_path):
        img = cv2.imread(str(img_path))
        if img is None: return None
        
        display_img = img.copy()
        
        # 1. 전처리 및 컨투어 추출
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower, self.upper)
        mask_cleaned = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel, iterations=2)
        contours, _ = cv2.findContours(mask_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            largest_cnt = max(contours, key=cv2.contourArea)
            rect = cv2.minAreaRect(largest_cnt)
            box = cv2.boxPoints(rect)
            box = np.int64(box)

            # 2. Warping (수평 정렬 및 Crop)
            width = int(rect[1][0])
            height = int(rect[1][1])
            src_pts = box.astype("float32")
            dst_pts = np.array([[0, height-1], [0, 0], [width-1, 0], [width-1, height-1]], dtype="float32")
            
            M = cv2.getPerspectiveTransform(src_pts, dst_pts)
            warped = cv2.warpPerspective(img, M, (width, height))

            # 가로/세로 정규화 (항상 눕힌 모양으로)
            # if width < height:
            #     warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)
            
            final_crop = cv2.resize(warped, (224, 224))
            
            # 원본에 BBox 표시
            cv2.drawContours(display_img, [box], 0, (0, 255, 0), 3)
            return display_img, final_crop
        
        return display_img, np.zeros((224, 224, 3), dtype=np.uint8)

    def run(self):
        cv2.namedWindow("Crop Inspector", cv2.WINDOW_NORMAL)
        print(f"--- 검수 시작: {len(self.img_files)}개 이미지 ---")
        print("D: 다음 이미지 | A: 이전 이미지 | Q: 종료")

        while True:
            img_path = self.img_files[self.current_idx]
            original, cropped = self.process_image(img_path)

            # 화면 구성을 위해 두 이미지를 합치기 (원본 리사이즈)
            h, w = original.shape[:2]
            scale = 600 / h
            orig_res = cv2.resize(original, (int(w*scale), 600))
            
            # Crop 이미지를 원본 높이에 맞춰 표시하기 위해 패딩 추가
            crop_res = cv2.resize(cropped, (400, 400))
            canvas = np.zeros((600, 400, 3), dtype=np.uint8)
            canvas[100:500, :] = crop_res
            
            # 최종 결과 결합 (가로로 붙이기)
            result_view = np.hstack([orig_res, canvas])
            
            # 텍스트 정보 추가
            info_text = f"[{self.current_idx + 1}/{len(self.img_files)}] {img_path.name}"
            cv2.putText(result_view, info_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

            cv2.imshow("Crop Inspector", result_view)
            
            key = cv2.waitKey(0) & 0xFF
            if key == ord('d'): # 다음
                self.current_idx = (self.current_idx + 1) % len(self.img_files)
            elif key == ord('a'): # 이전
                self.current_idx = (self.current_idx - 1) % len(self.img_files)
            elif key == ord('q') or key == 27: # 종료
                break

        cv2.destroyAllWindows()

# --- 실행 ---
DATA_PATH = r"C:\Dev\KAIROS_Project\data\Anomaly_augmented\aug_Anomaly_ESP32"
inspector = CropInspector(DATA_PATH)
inspector.run()