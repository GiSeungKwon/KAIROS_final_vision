import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2

def get_optimized_mask(h, w, device, sigma=0.2, center=(0.2, -0.2)):
    # center=(y_shift, x_shift) : +는 아래/오른쪽, -는 위/왼쪽
    y_range = torch.linspace(-1, 1, h)
    x_range = torch.linspace(-1, 1, w)
    y_grid, x_grid = torch.meshgrid(y_range, x_range, indexing='ij')
    
    # 중심 이동 및 시그마 적용
    # 시그마 값이 작을수록 더 넓은 범위를 커버하게 수정 (수식 변경)
    mask = torch.exp(-((x_grid - center[1])**2 + (y_grid - center[0])**2) / (2 * sigma**2))
    return mask.to(device)

# --- 시각화 확인용 ---
def visualize_tuning(image_path):
    img = Image.open(image_path).convert("RGB").resize((224, 224))
    img_np = np.array(img)
    
    # 설정값 (이 값을 조절해보세요!)
    # 현재 이미지 기준으로 약간 왼쪽 아래로 이동: center=(0.1, -0.1)
    # 범위를 넓게: sigma=0.5
    opt_mask = get_optimized_mask(224, 224, "cpu", sigma=0.5, center=(0.1, -0.1))
    
    mask_np = opt_mask.numpy()
    heatmap = cv2.applyColorMap(np.uint8(255 * mask_np), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img_np, 0.6, heatmap, 0.4, 0)
    
    plt.imshow(overlay)
    plt.title(f"Optimized Focus (Sigma: 0.5, Center: 0.1, -0.1)")
    plt.show()

# 실제 파일로 테스트
IMG_PATH = r"C:\Dev\KAIROS_Project\data\Anomaly_ESP32\WIN_20251211_19_51_13_Pro.jpg"
visualize_tuning(IMG_PATH)