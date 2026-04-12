"""
데이터셋 전처리 스크립트
──────────────────────────
dataset/raw/{door_open, door_closed, invalid}/
  -> dataset/train / dataset/val  (8:2 분할)

fine_tune.py 실행 전에 반드시 먼저 실행하세요.
"""

import shutil
import random
from pathlib import Path

RAW_DIR   = Path("dataset/raw")
OUT_DIR   = Path("dataset")
VAL_RATIO = 0.2
SEED      = 42

random.seed(SEED)

for cls_dir in sorted(RAW_DIR.iterdir()):
    if not cls_dir.is_dir():
        continue
    cls = cls_dir.name
    images = list(cls_dir.glob("*.jpg"))
    if not images:
        print(f"  {cls}: 이미지 없음, 건너뜀")
        continue

    random.shuffle(images)
    n_val   = max(1, int(len(images) * VAL_RATIO))
    splits  = {"val": images[:n_val], "train": images[n_val:]}

    for split, files in splits.items():
        dest = OUT_DIR / split / cls
        dest.mkdir(parents=True, exist_ok=True)
        for f in files:
            shutil.copy(f, dest / f.name)
        print(f"  [{split}] {cls}: {len(files)}장")

print("\n전처리 완료 -> fine_tune.py 를 실행하세요.")
