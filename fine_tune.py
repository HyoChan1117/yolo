"""
YOLOv8 분류 모델 Fine-tuning
──────────────────────────────
preprocess_dataset.py 실행 후 사용하세요.

학습 결과: runs/classify/train/weights/best.pt
"""

from pathlib import Path
from ultralytics import YOLO

BASE_MODEL = "yolov8n-cls.pt"
DATA_DIR   = "dataset"
EPOCHS     = 50
IMG_SIZE   = 224
BATCH      = 16
DEVICE     = 0          # GPU 없으면 "cpu"

if __name__ == "__main__":
    for split in ["train", "val"]:
        path = Path(DATA_DIR) / split
        if not path.exists():
            print(f"  {path} 없음. preprocess_dataset.py 를 먼저 실행하세요.")
            exit()
        counts = {c.name: len(list(c.glob("*"))) for c in path.iterdir() if c.is_dir()}
        print(f"  [{split}] " + " / ".join(f"{k}: {v}장" for k, v in counts.items()))

    print(f"\nFine-tuning 시작 (epochs={EPOCHS}, imgsz={IMG_SIZE}, batch={BATCH})\n")

    model   = YOLO(BASE_MODEL)
    results = model.train(
        data    = DATA_DIR,
        epochs  = EPOCHS,
        imgsz   = IMG_SIZE,
        batch   = BATCH,
        device  = DEVICE,
        patience= 10,
        plots   = True,
    )

    best = Path(results.save_dir) / "weights" / "best.pt"
    print(f"\n학습 완료!")
    print(f"  최종 모델: {best}")
    print(f"\nyolo_webcam.py 의 MODEL_PATH 를 아래 경로로 변경하세요:")
    print(f'  MODEL_PATH = r"{best}"')
