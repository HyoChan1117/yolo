"""데이터셋 전처리: raw 이미지를 train / val 으로 분리합니다.

dataset/raw/{class}/ → dataset/train/{class}/ + dataset/val/{class}/

invalid 클래스는 학습에서 제외합니다.

실행: python door/preprocess.py
"""

import random
import shutil
from pathlib import Path

RAW_DIR = Path("dataset/raw")
SPLIT_DIR = Path("dataset")
VAL_RATIO = 0.2
SEED = 42
CLASSES = ["door_open", "door_closed"]  # invalid 제외

random.seed(SEED)

for cls in CLASSES:
    src = RAW_DIR / cls
    files = sorted(src.glob("*.jpg"))
    if not files:
        print(f"  {cls}: 이미지 없음, 건너뜀")
        continue

    random.shuffle(files)
    n_val = max(1, int(len(files) * VAL_RATIO))
    splits = {"val": files[:n_val], "train": files[n_val:]}

    for split, paths in splits.items():
        dst = SPLIT_DIR / split / cls
        dst.mkdir(parents=True, exist_ok=True)
        for p in paths:
            shutil.copy(p, dst / p.name)
        print(f"  [{split}/{cls}] {len(paths)}장")

print("\n분리 완료.")
