import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
import cv2
import numpy as np
from PIL import Image

# --- 설정 변수 ---
MODULE = "ESP32"
DAY = "12191140"
MODEL_PATH = f"../../../../models/memoryBank_{MODULE}/{MODULE}_memory_bank_{DAY}.pt" 

class RealTimePatchCore:
    def __init__(self, memory_bank_path, device="cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.model.to(self.device).eval()
        
        self.features = []
        def hook(module, input, output): 
            self.features.append(output)
            
        self.model.layer1[-1].register_forward_hook(hook)
        self.model.layer2[-1].register_forward_hook(hook)

        print(f"Loading Memory Bank from {memory_bank_path}...")
        self.memory_bank = torch.load(memory_bank_path, map_location=self.device)
        
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # [변경] 카메라 전체 화면에서의 ROI (x, y, w, h)
        # 3번 카메라의 원본 해상도에 맞춰 이 좌표를 수정하세요.
        self.roi = (0, 0, 640, 480) 

    def embed(self, x):
        self.features = []
        with torch.no_grad(): _ = self.model(x)
        f1, f2 = self.features
        target_size = (f1.shape[2], f1.shape[3]) # (56, 56)
        
        f1 = F.avg_pool2d(f1, 3, 1, 1)
        f2 = F.interpolate(f2, size=target_size, mode="bilinear", align_corners=False)
        f2 = F.avg_pool2d(f2, 3, 1, 1)
        
        combined = torch.cat([f1, f2], dim=1)
        return combined.permute(0, 2, 3, 1).reshape(-1, combined.shape[1])
    
    def predict(self, frame):
        # 1. 원본 이미지 복사 (시각화용)
        overlay = frame.copy()
        h_orig, w_orig = frame.shape[:2]

        # 2. ROI 영역 추출 및 전처리
        rx, ry, rw, rh = self.roi
        # ROI가 이미지 범위를 벗어나지 않도록 방어 코드
        rx, ry = max(0, rx), max(0, ry)
        rw = min(rw, w_orig - rx)
        rh = min(rh, h_orig - ry)
        
        roi_bgr = frame[ry:ry+rh, rx:rx+rw]
        
        if roi_bgr.size == 0:
            return frame, 0.0

        # 모델 입력 규격(224x224)으로 변환
        roi_input = cv2.resize(roi_bgr, (224, 224))
        roi_rgb = cv2.cvtColor(roi_input, cv2.COLOR_BGR2RGB)
        input_tensor = self.transform(Image.fromarray(roi_rgb)).unsqueeze(0).to(self.device)
        
        # 3. 모델 추론
        with torch.no_grad():
            embedding = self.embed(input_tensor) 

        # NN 거리 계산
        distances = torch.cdist(embedding, self.memory_bank, p=2)
        min_distances, _ = torch.min(distances, dim=1)
        
        # 히트맵 생성 (PatchCore 기본 56x56)
        side_len = 56 
        anomaly_map = min_distances.reshape(side_len, side_len).cpu().numpy()
        
        # 4. 시각화: 히트맵을 원본 ROI 크기로 확대
        anomaly_map_resized = cv2.resize(anomaly_map, (rw, rh))
        anomaly_score = anomaly_map_resized.max()
        
        # 정규화 및 컬러맵 적용
        norm_map = cv2.normalize(anomaly_map_resized, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        heatmap = cv2.applyColorMap(norm_map, cv2.COLORMAP_JET)
        
        # 5. ROI 영역에 히트맵 덮어쓰기 (가중치 0.5)
        roi_overlay = cv2.addWeighted(roi_bgr, 0.5, heatmap, 0.5, 0)
        overlay[ry:ry+rh, rx:rx+rw] = roi_overlay
        
        # ROI 테두리 표시
        cv2.rectangle(overlay, (rx, ry), (rx+rw, ry+rh), (0, 255, 255), 2)
        
        return overlay, anomaly_score

def main():
    detector = RealTimePatchCore(MODEL_PATH)
    cap = cv2.VideoCapture(3)

    # 카메라 해상도 설정 (필요시)
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"--- Camera Info ---")
    print(f"Resolution: {int(width)} x {int(height)}")
    print(f"FPS: {fps}")
    print(f"ROI Setting: {detector.roi}")
    print(f"-------------------")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        display_img, score = detector.predict(frame)
        
        # 결과 텍스트 표시
        status = "ANOMALY" if score > 5.0 else "NORMAL"
        color = (0, 0, 255) if status == "ANOMALY" else (0, 255, 0)
        
        cv2.putText(display_img, f"Score: {score:.2f} ({status})", (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
        
        # 화면이 너무 크면 리사이즈해서 보기 (선택 사항)
        show_img = cv2.resize(display_img, (1280, 720))
        cv2.imshow("Full Screen Inspection (ROI only)", show_img)
        
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()