"""
데이터셋 수집 스크립트
──────────────────────
1. 첫 프레임에서 마우스로 세콤 센서 ROI를 드래그 선택
2. ROI 좌표를 roi.json에 저장
3. 이후 루프에서 ROI 크롭 이미지를 라벨별로 저장

키보드:
  O -> 현재 ROI를 "door_open"   으로 저장
  C -> 현재 ROI를 "door_closed"  로 저장
  I -> 현재 ROI를 "invalid"      로 저장 (사람이 가리는 등 판단 불가 상황)
  Q -> 종료
"""

import cv2
import json
import time
from pathlib import Path

SAVE_DIR = Path("dataset/raw")
ROI_FILE = Path("roi.json")
CLASSES  = ["door_open", "door_closed", "invalid"]

for cls in CLASSES:
    (SAVE_DIR / cls).mkdir(parents=True, exist_ok=True)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("웹캠을 열 수 없어요.")
    exit()

# -- ROI 선택 ────────────────────────────────────────────────
print("세콤 센서 위치를 드래그로 선택하세요. Enter/Space 로 확정, C 로 취소.")
ret, frame = cap.read()
roi = cv2.selectROI("ROI 선택 (드래그 후 Enter)", frame, showCrosshair=True, fromCenter=False)
cv2.destroyWindow("ROI 선택 (드래그 후 Enter)")

x, y, w, h = roi
if w == 0 or h == 0:
    print("ROI가 선택되지 않았어요.")
    cap.release()
    exit()

roi_data = {"x": x, "y": y, "w": w, "h": h}
ROI_FILE.write_text(json.dumps(roi_data, indent=2))
print(f"ROI 저장 완료: {roi_data}")

# -- 수집 루프 ───────────────────────────────────────────────
counts = {cls: len(list((SAVE_DIR / cls).glob("*.jpg"))) for cls in CLASSES}
print("\n데이터 수집 시작!")
print("  O: 문 열림  |  C: 문 닫힘  |  I: 무효  |  Q: 종료\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    crop = frame[y:y+h, x:x+w]

    display = frame.copy()
    cv2.rectangle(display, (x, y), (x+w, y+h), (0, 255, 255), 2)
    cv2.putText(display, "ROI", (x, y - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.putText(display, f"door_open:   {counts['door_open']}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 255, 100), 2)
    cv2.putText(display, f"door_closed: {counts['door_closed']}",
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 255), 2)
    cv2.putText(display, f"invalid:     {counts['invalid']}",
                (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 2)
    cv2.putText(display, "O: open  C: closed  I: invalid  Q: quit",
                (10, display.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    cv2.imshow("데이터 수집 (O/C/Q)", display)
    cv2.imshow("ROI 미리보기", crop)
    key = cv2.waitKey(1) & 0xFF

    def save(cls):
        filename = SAVE_DIR / cls / f"{cls}_{int(time.time()*1000)}.jpg"
        cv2.imwrite(str(filename), crop)
        counts[cls] += 1
        print(f"  저장: {filename.name}  ({cls}: {counts[cls]}장)")

    if key == ord('o'):
        save("door_open")
    elif key == ord('c'):
        save("door_closed")
    elif key == ord('i'):
        save("invalid")
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print(f"\n수집 완료 -- open: {counts['door_open']}장 / closed: {counts['door_closed']}장 / invalid: {counts['invalid']}장")
