"""
YOLOv8 Fine-tuning 스크립트
────────────────────────────
preprocess_dataset.py 실행 후 사용하세요.

학습 결과는 runs/classify/train/ 에 저장됩니다.
최종 모델: runs/classify/train/weights/best.pt
"""

from pathlib import Path
from ultralytics import YOLO

# ── 설정 ────────────────────────────────────────
BASE_MODEL = "yolov8n-cls.pt"   # 베이스 모델 (자동 다운로드)
DATA_DIR   = "dataset"           # preprocess_dataset.py 결과 폴더
EPOCHS     = 50
IMG_SIZE   = 640
BATCH      = 16
DEVICE     = 0                   # GPU 없으면 "cpu" 로 변경

if __name__ == "__main__":
    # ── 데이터 확인 ──────────────────────────────────
    for split in ["train", "val"]:
        path = Path(DATA_DIR) / split
        if not path.exists():
            print(f"❌ {path} 가 없어요. preprocess_dataset.py 를 먼저 실행하세요.")
            exit()

        counts = {cls.name: len(list(cls.glob("*"))) for cls in path.iterdir() if cls.is_dir()}
        print(f"  [{split}] " + " / ".join(f"{k}: {v}장" for k, v in counts.items()))

    # ── 학습 ─────────────────────────────────────────
    print(f"\n🚀 Fine-tuning 시작 (epochs={EPOCHS}, imgsz={IMG_SIZE}, batch={BATCH})\n")

    model = YOLO(BASE_MODEL)
    results = model.train(
        data    = DATA_DIR,
        epochs  = EPOCHS,
        imgsz   = IMG_SIZE,
        batch   = BATCH,
        device  = DEVICE,
        patience= 10,       # 10 epoch 개선 없으면 조기 종료
        plots   = True,     # 학습 곡선 저장
    )

    # ── 결과 요약 ────────────────────────────────────
    best = Path(results.save_dir) / "weights" / "best.pt"
    print(f"\n✅ 학습 완료!")
    print(f"   최종 모델: {best}")
    print(f"\n모델을 yolo_webcam.py 에 적용하려면:")
    print(f'   YOLO("{best}")')
