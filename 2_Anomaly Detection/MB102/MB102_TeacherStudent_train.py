import os
import yaml
from glob import glob
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from torch.utils.data import random_split

# --------------------------------------------------
# Config
# --------------------------------------------------
CONFIG_YAML_PATH = "MB102_config.yaml"


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# --------------------------------------------------
# Dataset
# --------------------------------------------------
class MB102Dataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_paths = sorted(glob(os.path.join(image_dir, "*.*")))
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img


# --------------------------------------------------
# Teacher (ResNet18, layer3)
# --------------------------------------------------
class TeacherNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.resnet18(pretrained=True)
        self.features = {}

        def hook_fn(name):
            def fn(_, __, output):
                self.features[name] = output
            return fn

        # 🔥 옵션 A: layer3만 사용
        self.model.layer3.register_forward_hook(hook_fn("layer3"))

    def forward(self, x):
        _ = self.model(x)
        return self.features["layer3"]  # [B, 256, 16, 16]

class StudentNet(nn.Module):
    def __init__(self, width_mult=0.5):
        super().__init__()
        backbone = models.mobilenet_v2(
            pretrained=False,
            width_mult=width_mult
        )

        self.features = backbone.features[:14]

        # 🔥 dummy forward로 채널 수 자동 계산
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 256, 256)
            c = self.features(dummy).shape[1]

        self.proj = nn.Conv2d(c, 256, kernel_size=1)

    def forward(self, x):
        feat = self.features(x)
        feat = self.proj(feat)
        return feat

def train_kd_anomaly_model():
    config = load_config(CONFIG_YAML_PATH)

    device = torch.device(
        "cuda" if torch.cuda.is_available() and config["device"] == "cuda" else "cpu"
    )

    print("\n--- MB102 모델 준비 중 ---")

    transform = transforms.Compose([
        transforms.Resize((config["image_size"], config["image_size"])),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    full_dataset = MB102Dataset(config["data_root"], transform)

    val_size = int(config["val_split"] * len(full_dataset))
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    dataloader = DataLoader(
        train_dataset, # 훈련 데이터셋 사용
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=0
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False, # 검증 시에는 shuffle 불필요
        num_workers=0
    )

    print(f"훈련 데이터셋 크기: {len(train_dataset)}개")
    print(f"검증 데이터셋 크기: {len(val_dataset)}개") # 추가

    teacher = TeacherNet().to(device)
    student = StudentNet(config["student_width_mult"]).to(device)

    teacher.eval()  # Teacher 고정
    student.train()

    optimizer = torch.optim.Adam(
        student.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"]
    )

    mse = nn.MSELoss()

    os.makedirs(config["output_dir"], exist_ok=True)
    best_loss = float('inf') # 가장 낮은 검증 손실을 추적

    print(f"\n--- MB102 KD 학습 시작 ({config['epochs']} Epoch) ---")

    for epoch in range(config["epochs"]):
        student.train()
        epoch_loss = 0.0

        for images in tqdm(dataloader, desc=f"Epoch {epoch+1}/{config['epochs']}"):
            images = images.to(device)

            with torch.no_grad():
                teacher_feat = teacher(images)   # [B, 256, 16, 16]

            student_feat = student(images)       # [B, 256, 16, 16]
            
            student_feat = F.interpolate(
                student_feat,
                size=teacher_feat.shape[2:],
                mode="bilinear",
                align_corners=False
            )
            
            kd_loss = mse(student_feat, teacher_feat)
            total_loss = kd_loss * config["kd_loss_weight"]

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            epoch_loss += total_loss.item()

        avg_train_loss = epoch_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{config['epochs']}] - KD Loss: {avg_train_loss:.6f}")

        student.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images in tqdm(val_dataloader, desc=f"Epoch {epoch+1}/{config['epochs']} [VAL  ]"):
                images = images.to(device)
                
                teacher_feat = teacher(images)
                student_feat = student(images)
                
                # Interpolation (기존과 동일)
                student_feat = F.interpolate(
                    student_feat,
                    size=teacher_feat.shape[2:],
                    mode="bilinear",
                    align_corners=False
                )
                
                kd_loss = mse(student_feat, teacher_feat)
                total_loss = kd_loss * config["kd_loss_weight"]
                val_loss += total_loss.item()
        
        avg_val_loss = val_loss / len(val_dataloader)

        print(f"Epoch [{epoch+1}/{config['epochs']}] - Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")
        
        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(config["output_dir"], f"student_kd_epoch_{epoch+1}.pth")
            torch.save(student.state_dict(), checkpoint_path)
            print(f"  [Checkpoint] {epoch+1} Epoch 모델 저장됨 → {checkpoint_path}")

        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            best_model_path = os.path.join(config["output_dir"], "student_kd_best.pth")
            torch.save(student.state_dict(), best_model_path)
            print(f"  [BEST] Validation Loss 경신 ({best_loss:.6f})! 모델 저장됨 → {best_model_path}")
            
    final_save_path = os.path.join(config["output_dir"], "student_kd_final.pth")
    torch.save(student.state_dict(), final_save_path)

    print(f"\n--- 학습 완료: 최종 Student 모델 저장됨 → {final_save_path} ---")


if __name__ == "__main__":
    train_kd_anomaly_model()