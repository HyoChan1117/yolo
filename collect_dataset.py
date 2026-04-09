"""
데이터셋 수집 스크립트
────────────────────
키보드 단축키:
  O 키 → 현재 프레임을 "door_open" 으로 저장
  C 키 → 현재 프레임을 "door_closed" 으로 저장
  Q 키 → 종료
"""

import cv2
import time
from pathlib import Path

# ── 저장 경로 설정 ──────────────────────────────
SAVE_DIR = Path("dataset/raw")
CLASSES  = ["door_open", "door_closed"]

for cls in CLASSES:
    (SAVE_DIR / cls).mkdir(parents=True, exist_ok=True)

# ── 웹캠 초기화 ─────────────────────────────────
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ 웹캠을 열 수 없어요.")
    exit()

counts = {cls: len(list((SAVE_DIR / cls).glob("*.jpg"))) for cls in CLASSES}
print("📸 데이터 수집 시작!")
print("  O: 문 열림  |  C: 문 닫힘  |  Q: 종료\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # HUD 표시
    display = frame.copy()
    cv2.putText(display, f"door_open:   {counts['door_open']}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 255, 100), 2)
    cv2.putText(display, f"door_closed: {counts['door_closed']}",
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 255), 2)
    cv2.putText(display, "O: open  C: closed  Q: quit",
                (10, display.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    cv2.imshow("데이터 수집 (O/C/Q)", display)
    key = cv2.waitKey(1) & 0xFF

    def save(cls):
        filename = SAVE_DIR / cls / f"{cls}_{int(time.time()*1000)}.jpg"
        cv2.imwrite(str(filename), frame)
        counts[cls] += 1
        print(f"  저장: {filename.name}  ({cls}: {counts[cls]}장)")

    if key == ord('o'):
        save("door_open")
    elif key == ord('c'):
        save("door_closed")
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print(f"\n✅ 수집 완료 — open: {counts['door_open']}장 / closed: {counts['door_closed']}장")
