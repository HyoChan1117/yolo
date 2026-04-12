"""
세콤 센서 ROI 기반 문 상태 감지 + Slack 알림
──────────────────────────────────────────────
- roi.json 에 저장된 ROI를 크롭 -> 분류 모델 추론
- 5초간 같은 상태가 유지되면 상태 확정 후 Slack 전송
  · 문 열림  -> "문 열었습니다"
  · 문 닫힘  -> "문 닫았습니다"

종료: 'q' 키
"""

import cv2
import json
import os
import time
import requests
from pathlib import Path
from dotenv import load_dotenv
from ultralytics import YOLO

load_dotenv()

# -- 설정 ────────────────────────────────────────────────────
MODEL_PATH    = r"runs/classify/train/weights/best.pt"
ROI_FILE      = Path("roi.json")
HOLD_DURATION = 5      # 상태 유지 시간 (초)
CONF_THRESH   = 0.7    # 신뢰도 임계값
SLACK_URL     = os.getenv("SLACK_WEBHOOK_URL", "")

# -- ROI 로드 ─────────────────────────────────────────────────
if not ROI_FILE.exists():
    print("roi.json 없음. collect_dataset.py 를 먼저 실행하세요.")
    exit()

roi = json.loads(ROI_FILE.read_text())
rx, ry, rw, rh = roi["x"], roi["y"], roi["w"], roi["h"]
print(f"ROI 로드: x={rx} y={ry} w={rw} h={rh}")

# -- 모델 로드 ────────────────────────────────────────────────
model = YOLO(MODEL_PATH)

# -- Slack 전송 ───────────────────────────────────────────────
def send_slack(message: str):
    if not SLACK_URL:
        return
    try:
        requests.post(SLACK_URL, json={"text": message}, timeout=5)
    except Exception:
        pass

# -- 웹캠 초기화 ──────────────────────────────────────────────
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("웹캠을 열 수 없어요.")
    exit()

# -- 상태 변수 ────────────────────────────────────────────────
door_state    = None   # 확정 상태: "door_open" | "door_closed" | None
pending_state = None   # 대기 중인 후보 상태
pending_start = None   # 후보 상태 시작 시각

# -- 메인 루프 ────────────────────────────────────────────────
while True:
    ret, frame = cap.read()
    if not ret:
        break

    now  = time.time()
    crop = frame[ry:ry+rh, rx:rx+rw]

    # 추론
    results    = model(crop, verbose=False)
    probs      = results[0].probs
    top_class  = results[0].names[probs.top1]
    confidence = probs.top1conf.item()

    # invalid 또는 신뢰도 미달이면 타이머·후보 상태를 그대로 동결
    is_invalid = (top_class == "invalid") or (confidence < CONF_THRESH)

    if not is_invalid:
        # 후보 변경 시 타이머 리셋
        if top_class != pending_state:
            pending_state = top_class
            pending_start = now

    elapsed = now - (pending_start or now)

    # 상태 확정 (invalid·신뢰도 미달 프레임은 건너뜀)
    if not is_invalid and pending_state and elapsed >= HOLD_DURATION and pending_state != door_state:
        door_state = pending_state
        ts = time.strftime("%H:%M:%S")
        if door_state == "door_open":
            send_slack(f"[{ts}] 문 열었습니다.")
            print(f"[{ts}] 문 열었습니다.")
        elif door_state == "door_closed":
            send_slack(f"[{ts}] 문 닫았습니다.")
            print(f"[{ts}] 문 닫았습니다.")

    # -- 화면 표시 ────────────────────────────────────────────
    display = frame.copy()

    # ROI 박스
    cv2.rectangle(display, (rx, ry), (rx+rw, ry+rh), (0, 255, 255), 2)
    cv2.putText(display, f"{top_class} {confidence:.0%}",
                (rx, ry - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

    # 상태 바
    if door_state == "door_open":
        bar_color, state_text = (30, 180, 30),  "OPEN   (문 열림)"
    elif door_state == "door_closed":
        bar_color, state_text = (30, 30, 180),  "CLOSED (문 닫힘)"
    else:
        bar_color, state_text = (80, 80, 80),   "waiting..."

    # invalid 중이면 상단 바 우측에 표시
    if is_invalid:
        state_text += "  |  INVALID"

    h_frame, w_frame = display.shape[:2]
    cv2.rectangle(display, (0, 0), (w_frame, 44), bar_color, -1)
    cv2.putText(display, state_text,
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    # 대기 진행 바
    bar_w = int(w_frame * min(elapsed / HOLD_DURATION, 1.0))
    cv2.rectangle(display, (0, h_frame - 6), (bar_w, h_frame), (0, 220, 255), -1)

    # ROI 크롭 우측 상단에 작게 표시
    thumb = cv2.resize(crop, (rw * 2, rh * 2)) if rw < 100 else crop.copy()
    th, tw = thumb.shape[:2]
    if tw < w_frame and th < h_frame - 44:
        display[50:50+th, w_frame-tw-10:w_frame-10] = thumb

    cv2.imshow("세콤 문 감지 (q: 종료)", display)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
