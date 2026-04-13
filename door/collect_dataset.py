"""ROI 기반 문 상태 데이터셋 수집 스크립트
──────────────────────────────────────────
1. 첫 프레임에서 마우스로 ROI(세콤 센서 영역)를 드래그 선택
2. ROI 좌표를 roi.json 에 저장
3. 이후 루프에서 ROI 크롭 이미지를 라벨별로 저장

키보드:
  O -> door_open   (문 열림)
  C -> door_closed (문 닫힘)
  I -> invalid     (사람이 가리는 등 판단 불가)
  Q -> 종료

실행: python door/collect_dataset.py
"""

import json
import os
import time
from pathlib import Path

import cv2
from dotenv import load_dotenv

load_dotenv()

SAVE_DIR = Path("dataset/raw")
ROI_FILE = Path("roi.json")
CLASSES = ["door_open", "door_closed", "invalid"]
CAMERA_INDEX = int(os.getenv("CAMERA_INDEX", "0"))

for cls in CLASSES:
    (SAVE_DIR / cls).mkdir(parents=True, exist_ok=True)

cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    print("웹캠을 열 수 없어요.")
    exit()

# ── ROI 선택 ──────────────────────────────────────────────────
print("세콤 센서 위치를 드래그로 선택하세요. Enter/Space 로 확정, C 로 취소.")
ret, frame = cap.read()
roi = cv2.selectROI("ROI 선택 (드래그 후 Enter)", frame, showCrosshair=True, fromCenter=False)
cv2.destroyWindow("ROI 선택 (드래그 후 Enter)")

x, y, w, h = roi
if w == 0 or h == 0:
    print("ROI가 선택되지 않았어요.")
    cap.release()
    exit()

ROI_FILE.write_text(json.dumps({"x": x, "y": y, "w": w, "h": h}, indent=2), encoding="utf-8")
print(f"ROI 저장: x={x} y={y} w={w} h={h}")

# ── 수집 루프 ─────────────────────────────────────────────────
counts = {cls: len(list((SAVE_DIR / cls).glob("*.jpg"))) for cls in CLASSES}
print("\nO: 문 열림  C: 문 닫힘  I: 무효  Q: 종료\n")


def save(cls_name: str, crop) -> None:
    fn = SAVE_DIR / cls_name / f"{cls_name}_{int(time.time() * 1000)}.jpg"
    cv2.imwrite(str(fn), crop)
    counts[cls_name] += 1
    print(f"  저장: {fn.name}  ({cls_name}: {counts[cls_name]}장)")


while True:
    ret, frame = cap.read()
    if not ret:
        break

    crop = frame[y : y + h, x : x + w]
    display = frame.copy()

    cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 255), 2)
    cv2.putText(
        display,
        f"open:{counts['door_open']}  closed:{counts['door_closed']}  invalid:{counts['invalid']}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )
    cv2.putText(
        display,
        "O: open  C: closed  I: invalid  Q: quit",
        (10, display.shape[0] - 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (200, 200, 200),
        1,
    )

    cv2.imshow("데이터 수집", display)
    cv2.imshow("ROI", crop)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("o"):
        save("door_open", crop)
    elif key == ord("c"):
        save("door_closed", crop)
    elif key == ord("i"):
        save("invalid", crop)
    elif key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
print(
    f"\n수집 완료 — open:{counts['door_open']}장 / closed:{counts['door_closed']}장 / invalid:{counts['invalid']}장"
)
