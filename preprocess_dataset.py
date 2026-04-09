"""
데이터셋 전처리 스크립트
────────────────────────
dataset/raw/ 의 이미지를 train/val 로 분리하고
YOLOv8 classify 학습 구조로 정리합니다.

실행 전 확인:
  dataset/raw/door_open/    ← 수집한 문 열림 이미지
  dataset/raw/door_closed/  ← 수집한 문 닫힘 이미지

실행 후 결과:
  dataset/train/door_open/
  dataset/train/door_closed/
  dataset/val/door_open/
  dataset/val/door_closed/
"""

import shutil
import random
import cv2
import numpy as np
from pathlib import Path

# ── 설정 ────────────────────────────────────────
RAW_DIR   = Path("dataset/raw")
OUT_DIR   = Path("dataset")
VAL_RATIO = 0.2    # 검증셋 비율
IMG_SIZE  = 640    # 리사이즈 크기
AUGMENT   = True   # 증강 여부
SEED      = 42

CLASSES = ["door_open", "door_closed"]
random.seed(SEED)

# ── 증강 함수 ────────────────────────────────────
def augment(img: np.ndarray) -> list[np.ndarray]:
    results = []

    # 좌우 반전
    results.append(cv2.flip(img, 1))

    # 밝기 조절
    for alpha in [0.7, 1.3]:
        results.append(np.clip(img * alpha, 0, 255).astype(np.uint8))

    # 가우시안 블러 (흔들림 시뮬레이션)
    results.append(cv2.GaussianBlur(img, (5, 5), 0))

    # 회전 (-10 ~ +10도)
    h, w = img.shape[:2]
    for angle in [-10, 10]:
        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        results.append(cv2.warpAffine(img, M, (w, h)))

    return results

# ── 메인 ─────────────────────────────────────────
total_saved = {"train": 0, "val": 0}

for cls in CLASSES:
    src_dir = RAW_DIR / cls
    images  = list(src_dir.glob("*.jpg")) + list(src_dir.glob("*.png"))

    if not images:
        print(f"⚠️  {cls}: 이미지 없음, 건너뜀")
        continue

    random.shuffle(images)
    n_val   = max(1, int(len(images) * VAL_RATIO))
    splits  = {"val": images[:n_val], "train": images[n_val:]}

    for split, files in splits.items():
        dst = OUT_DIR / split / cls
        dst.mkdir(parents=True, exist_ok=True)

        for src in files:
            img = cv2.imread(str(src))
            if img is None:
                continue
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

            # 원본 저장
            out_path = dst / src.name
            cv2.imwrite(str(out_path), img)
            total_saved[split] += 1

            # 증강은 train 에만 적용
            if AUGMENT and split == "train":
                for i, aug_img in enumerate(augment(img)):
                    aug_path = dst / f"{src.stem}_aug{i}{src.suffix}"
                    cv2.imwrite(str(aug_path), aug_img)
                    total_saved[split] += 1

    print(f"  {cls}: train {len(splits['train'])}장 → 증강 포함 {len(splits['train']) * (1 + (6 if AUGMENT else 0))}장 | val {len(splits['val'])}장")

print(f"\n✅ 전처리 완료 — train: {total_saved['train']}장 / val: {total_saved['val']}장")
print(f"   저장 위치: {OUT_DIR.resolve()}")
