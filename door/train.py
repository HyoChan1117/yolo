"""MobileNetV3-small fine-tuning for door open / closed classification.

preprocess.py 실행 후 사용하세요.
학습 결과: door/models/best.pth

실행: python door/train.py
"""

from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

DATA_DIR = Path("dataset")
SAVE_DIR = Path("door/models")
EPOCHS = 20
BATCH = 16
LR = 1e-3
IMG_SIZE = 224
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SAVE_DIR.mkdir(parents=True, exist_ok=True)

tfm = {
    "train": transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.3, contrast=0.3),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    ),
    "val": transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    ),
}

for split in ("train", "val"):
    p = DATA_DIR / split
    if not p.exists():
        print(f"{p} 없음. door/preprocess.py 를 먼저 실행하세요.")
        exit()

ds = {s: datasets.ImageFolder(str(DATA_DIR / s), transform=tfm[s]) for s in ("train", "val")}
loaders = {
    s: DataLoader(ds[s], batch_size=BATCH, shuffle=(s == "train"), num_workers=0)
    for s in ("train", "val")
}

print(f"클래스: {ds['train'].classes}")
print(f"학습: {len(ds['train'])}장  검증: {len(ds['val'])}장")
print(f"디바이스: {DEVICE}\n")

model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, len(ds["train"].classes))
model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

best_acc = 0.0

for epoch in range(1, EPOCHS + 1):
    for phase in ("train", "val"):
        model.train() if phase == "train" else model.eval()
        total_loss, correct = 0.0, 0

        with torch.set_grad_enabled(phase == "train"):
            for images, labels in loaders[phase]:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, labels)
                if phase == "train":
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                total_loss += loss.item() * len(images)
                correct += (outputs.argmax(1) == labels).sum().item()

        n = len(ds[phase])
        acc = correct / n
        print(f"  epoch {epoch:3d} [{phase:5s}]  loss {total_loss / n:.4f}  acc {acc:.3f}")

        if phase == "val" and acc > best_acc:
            best_acc = acc
            torch.save(
                {"state_dict": model.state_dict(), "classes": ds["train"].classes},
                SAVE_DIR / "best.pth",
            )
            print(f"    -> best 저장 (acc={best_acc:.3f})")

    scheduler.step()

print(f"\n학습 완료! 최고 검증 정확도: {best_acc:.3f}")
print(f"모델: {SAVE_DIR / 'best.pth'}")
