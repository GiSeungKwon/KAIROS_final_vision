import os
import sys
import numpy as np
import cv2
from PIL import Image
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms

# =================================================================
# 1. 설정 및 경로 정의
# =================================================================
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
ROI = (200, 50, 550, 400) 
CLASS_NAMES = ['ESP32', 'L298N', 'MB102']
NUM_CLASSES = len(CLASS_NAMES)

# [수정] 클래스별 개별 임계값 설정
# 실험 결과에 따라 이 수치들을 미세 조정하시면 됩니다.
AD_THRESHOLDS = {
    "ESP32": 4.5,
    "L298N": 4.5,
    "MB102": 4.5
}

CLASSIFIER_PATH = r"C:\Dev\KAIROS_Project\Vision\ano_classification.pth"
AD_MODEL_PATHS = {
    "ESP32": r"C:\Dev\KAIROS_Project\Vision\ESP32_memory_bank.pt",
    "L298N": r"C:\Dev\KAIROS_Project\Vision\L298N_memory_bank.pt",
    "MB102": r"C:\Dev\KAIROS_Project\Vision\MB102_memory_bank.pt"
}

# =================================================================
# 2. 통합 검사 클래스
# =================================================================
class IntegratedInspector:
    def __init__(self):
        # --- 2.1 Classification 모델 로드 ---
        print(f"Loading Classification Model: {CLASSIFIER_PATH}")
        self.classifier = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        num_ftrs = self.classifier.fc.in_features
        self.classifier.fc = nn.Linear(num_ftrs, NUM_CLASSES)
        self.classifier.load_state_dict(torch.load(CLASSIFIER_PATH, map_location=DEVICE))
        self.classifier.to(DEVICE).eval()

        # --- 2.2 PatchCore 백본 및 Hook 설정 ---
        self.ad_backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1).to(DEVICE)
        self.ad_backbone.eval()
        self.features = []

        def hook(module, input, output):
            self.features.append(output)

        # 1792차원 유지를 위한 Layer 1, 2, 3 Hook
        self.ad_backbone.layer1[-1].register_forward_hook(hook)
        self.ad_backbone.layer2[-1].register_forward_hook(hook)
        self.ad_backbone.layer3[-1].register_forward_hook(hook)

        # --- 2.3 Memory Banks 로드 ---
        self.memory_banks = {}
        for name, path in AD_MODEL_PATHS.items():
            if os.path.exists(path):
                print(f"Loading {name} Memory Bank...")
                self.memory_banks[name] = torch.load(path, map_location=DEVICE)
        
        # --- 2.4 전처리 설정 ---
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def embed(self, x):
        self.features = []
        with torch.no_grad():
            _ = self.ad_backbone(x)
        
        f1, f2, f3 = self.features
        target_size = (f1.shape[2], f1.shape[3]) 
        
        f1 = F.avg_pool2d(f1, 3, 1, 1)
        f2 = F.interpolate(f2, size=target_size, mode="bilinear", align_corners=False)
        f2 = F.avg_pool2d(f2, 3, 1, 1)
        f3 = F.interpolate(f3, size=target_size, mode="bilinear", align_corners=False)
        f3 = F.avg_pool2d(f3, 3, 1, 1)
        
        combined = torch.cat([f1, f2, f3], dim=1)
        return combined.permute(0, 2, 3, 1).reshape(-1, combined.shape[1])

    def inspect(self, frame):
        overlay = frame.copy()
        h_orig, w_orig = frame.shape[:2]

        rx, ry, rw, rh = ROI
        rx, ry = max(0, rx), max(0, ry)
        rw, rh = min(rw, w_orig - rx), min(rh, h_orig - ry)
        roi_bgr = frame[ry:ry+rh, rx:rx+rw]

        if roi_bgr.size == 0:
            return frame, "None", 0.0, 0.0

        roi_input = cv2.resize(roi_bgr, (224, 224))
        roi_rgb = cv2.cvtColor(roi_input, cv2.COLOR_BGR2RGB)
        input_tensor = self.transform(Image.fromarray(roi_rgb)).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            # 1. Classification
            cls_out = self.classifier(input_tensor)
            probs = F.softmax(cls_out, dim=1).squeeze()
            pred_idx = torch.argmax(probs).item()
            pred_class = CLASS_NAMES[pred_idx]
            confidence = probs[pred_idx].item()

            # 2. Anomaly Detection
            anomaly_score = 0.0
            if pred_class in self.memory_banks:
                embedding = self.embed(input_tensor)
                distances = torch.cdist(embedding, self.memory_banks[pred_class], p=2)
                min_distances, _ = torch.min(distances, dim=1)
                
                side_len = 56 
                anomaly_map = min_distances.reshape(side_len, side_len).cpu().numpy()
                anomaly_map_resized = cv2.resize(anomaly_map, (rw, rh))
                anomaly_score = anomaly_map_resized.max()
                
                norm_map = cv2.normalize(anomaly_map_resized, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                heatmap = cv2.applyColorMap(norm_map, cv2.COLORMAP_JET)
                roi_overlay = cv2.addWeighted(roi_bgr, 0.5, heatmap, 0.5, 0)
                overlay[ry:ry+rh, rx:rx+rw] = roi_overlay

        cv2.rectangle(overlay, (rx, ry), (rx+rw, ry+rh), (0, 255, 255), 2)
        return overlay, pred_class, confidence, anomaly_score

# =================================================================
# 3. 메인 루프
# =================================================================
def main():
    inspector = IntegratedInspector()
    cap = cv2.VideoCapture(3) 

    if not cap.isOpened():
        print("Error: Camera not found.")
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        cv2.imshow("Original Image", frame)

        start_time = time.time()
        display_img, cls_name, conf, ad_score = inspector.inspect(frame)
        fps = 1 / (time.time() - start_time)

        # [수정] 클래스별 개별 임계값을 가져와서 판정 (기본값 5.0)
        current_threshold = AD_THRESHOLDS.get(cls_name, 5.0)
        status = "ANOMALY" if ad_score > current_threshold else "NORMAL"
        color = (0, 0, 255) if status == "ANOMALY" else (0, 255, 0)

        # UI 텍스트 출력
        cv2.putText(display_img, f"Class: {cls_name} ({conf:.2f})", (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        # 현재 적용된 임계값(TH)도 같이 표시되도록 추가했습니다.
        cv2.putText(display_img, f"Score: {ad_score:.2f} (TH: {current_threshold})", (20, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
        cv2.putText(display_img, f"Status: {status}", (20, 130), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)
        cv2.putText(display_img, f"FPS: {fps:.1f}", (20, frame.shape[0]-20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

        cv2.imshow("KAIROS Integrated Inspection", display_img)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()