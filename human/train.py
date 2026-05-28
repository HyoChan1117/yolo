import shutil
from pathlib import Path
from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("yolo11x.pt")

    results = model.train(
        data="C:/.code/yolo/yolo_dataset/data.yaml",
        epochs=100,
        imgsz=640,
        batch=4,
        project="runs/train",
        name="classroom-person",
        exist_ok=True,
    )

    best_pt = Path(results.save_dir) / "weights" / "best.pt"
    dest_pt = Path("human/models/yolo11x_fine.pt")
    shutil.copy2(best_pt, dest_pt)
    print(f"모델 저장 완료: {dest_pt}")
