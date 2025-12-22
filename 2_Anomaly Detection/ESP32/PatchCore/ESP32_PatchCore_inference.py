import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
import cv2
import numpy as np
from pathlib import Path

MODULE = "ESP32"
DAY = "12191055"

class RealTimePatchCore:
    def __init__(self, memory_bank_path, device="cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        # 1. 모델 설정 (Train과 동일한 구조)
        self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.model.to(self.device)
        self.model.eval()
        
        self.features = []
        def hook(module, input, output):
            self.features.append(output)
        
        self.model.layer1[-1].register_forward_hook(hook)
        self.model.layer2[-1].register_forward_hook(hook)
        self.model.layer3[-1].register_forward_hook(hook)

        # 2. 메모리 뱅크 로드
        print(f"Loading Memory Bank from {memory_bank_path}...")
        self.memory_bank = torch.load(memory_bank_path).to(self.device) # (N, C)
        
        # 3. 전처리 설정
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def embed(self, x):
        self.features = []
        with torch.no_grad():
            _ = self.model(x)
        
        f1, f2, f3 = self.features
        target_size = (f1.shape[2], f1.shape[3])
        
        f1 = F.avg_pool2d(f1, 3, 1, 1)
        f2 = F.interpolate(f2, size=target_size, mode="bilinear", align_corners=False)
        f2 = F.avg_pool2d(f2, 3, 1, 1)
        f3 = F.interpolate(f3, size=target_size, mode="bilinear", align_corners=False)
        f3 = F.avg_pool2d(f3, 3, 1, 1)
        
        combined = torch.cat([f1, f2, f3], dim=1)
        return combined # (1, C, H, W)

    def predict(self, frame):
        # 전처리
        input_tensor = self.transform(frame).unsqueeze(0).to(self.device)
        
        # 특징 추출
        with torch.no_grad():
            feature_map = self.embed(input_tensor) # (1, C, H_f, W_f)
            
        B, C, H_f, W_f = feature_map.shape
        embedding = feature_map.permute(0, 2, 3, 1).reshape(-1, C) # (H_f*W_f, C)

        # Nearest Neighbor 거리 계산 (L2 Distance)
        # (H_f*W_f, 1, C) - (1, N_bank, C) -> 거리 계산
        # 메모리 효율을 위해 torch.cdist 사용
        distances = torch.cdist(embedding, self.memory_bank, p=2) # (H_f*W_f, N_bank)
        min_distances, _ = torch.min(distances, dim=1) # 각 패치별 최소 거리
        
        # 히트맵 리사이즈 (원래 이미지 크기로)
        anomaly_map = min_distances.reshape(H_f, W_f)
        anomaly_map_resized = F.interpolate(anomaly_map.unsqueeze(0).unsqueeze(0), 
                                            size=(frame.shape[0], frame.shape[1]), 
                                            mode="bilinear", align_corners=False).squeeze().cpu().numpy()
        
        # 이미지 전체의 이상 점수 (최대 거리값 활용)
        anomaly_score = anomaly_map_resized.max()
        
        return anomaly_map_resized, anomaly_score

def main():
    # 경로 설정
    MODEL_PATH = f"../../../../models/memoryBank_{MODULE}/{MODULE}_memory_bank_{DAY}.pt"
    detector = RealTimePatchCore(MODEL_PATH)

    # 카메라 스트리밍 시작
    cap = cv2.VideoCapture(3) # 3번 카메라 (웹캠)

    print("--- 실시간 탐지 시작 (종료하려면 'q'를 누르세요) ---")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        # 이상 탐지 수행
        anomaly_map, score = detector.predict(frame)

        # 히트맵 시각화 처리
        # 스코어 정규화 (0~255) - 임계값은 실험적으로 조정 필요 (예: 5.0 이상이면 이상)
        norm_anomaly_map = cv2.normalize(anomaly_map, None, 0, 255, cv2.NORM_MINMAX)
        norm_anomaly_map = np.uint8(norm_anomaly_map)
        heatmap = cv2.applyColorMap(norm_anomaly_map, cv2.COLORMAP_JET)

        # 원본과 히트맵 합성
        overlay = cv2.addWeighted(frame, 0.6, heatmap, 0.4, 0)

        # 텍스트 정보 표시
        status = "ANOMALY" if score > 4.2 else "NORMAL" # Threshold 5.0은 예시입니다.
        color = (0, 0, 255) if status == "ANOMALY" else (0, 255, 0)
        cv2.putText(overlay, f"Score: {score:.2f} ({status})", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        # 결과 창 출력
        cv2.imshow("Original", frame)
        cv2.imshow("Anomaly Heatmap", overlay)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()