"""
🏫 강의실 문 상태 판단 시스템
----------------------------------------------
Fine-tuned YOLOv8 분류 모델로 문 상태를 판단:
  - door_closed 가 5초 유지 (이전에 open 확인된 경우) → "문 닫았습니다"
  - door_open   가 5초 유지 → "문 열었습니다"

종료: 'q' 키
"""

import cv2
import time
import os
import requests
from dotenv import load_dotenv
from ultralytics import YOLO

load_dotenv()

# ─────────────────────────────────────────────
# 설정값
# ─────────────────────────────────────────────
MODEL_PATH        = r"C:/.code/yolo/runs/classify/train3/weights/best.pt"
HOLD_DURATION     = 5     # 상태 유지 시간 (초)
CONF_THRESH       = 0.7   # 분류 신뢰도 임계값 (미달 시 판단 보류)
BRIGHTNESS_THRESH = 120    # 밝음/어두움 경계값 (0~255)
PERSON_CONF       = 0.5   # 사람 탐지 신뢰도
PERSON_MODEL_PATH = "yolov8n.pt"

SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL", "")

def send_slack(message: str):
    try:
        requests.post(SLACK_WEBHOOK_URL, json={"text": message}, timeout=5)
    except Exception:
        pass

# ─────────────────────────────────────────────
# 초기화
# ─────────────────────────────────────────────
model        = YOLO(MODEL_PATH)
person_model = YOLO(PERSON_MODEL_PATH)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    exit()

# 상태 추적 변수
door_state    = None   # 확정된 문 상태: "door_closed" | "door_open" | None
pending_state = None   # 유지 중인 후보 상태
pending_start = None   # 후보 상태 시작 시각

# ─────────────────────────────────────────────
# 메인 루프
# ─────────────────────────────────────────────
while True:
    ret, frame = cap.read()
    if not ret:
        break

    now = time.time()

    # ── 분류 모델 추론 ────────────────────────
    results    = model(frame, verbose=False)
    probs      = results[0].probs
    top_class  = results[0].names[probs.top1]   # "door_open" or "door_closed"
    confidence = probs.top1conf.item()

    # ── 밝기 & 사람 탐지 ──────────────────────
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    is_dark = gray.mean() < BRIGHTNESS_THRESH

    person_results = person_model(frame, conf=PERSON_CONF, verbose=False)
    person_count   = sum(1 for box in person_results[0].boxes if int(box.cls[0]) == 0)

    # ── 최종 후보 결정 ────────────────────────
    # 신뢰도 미달 → 보류
    # 모델+밝기+사람 세 조건 모두 충족해야 "closed"
    # 사람이 있으면 항상 "open"
    if confidence < CONF_THRESH:
        current_candidate = pending_state
    elif top_class == "door_closed" and is_dark and person_count == 0:
        current_candidate = "door_closed"
    else:
        current_candidate = "door_open"

    # ── 타이머 관리 ───────────────────────────
    if current_candidate != pending_state:
        pending_state = current_candidate
        pending_start = now

    elapsed = now - (pending_start or now)

    # ── 문 상태 확정 ──────────────────────────
    if elapsed >= HOLD_DURATION and current_candidate != door_state:
        prev_state = door_state
        door_state = current_candidate
        if door_state == "door_closed" and prev_state == "door_open":
            send_slack(f"🔒 [{time.strftime('%H:%M:%S')}] 문 닫았습니다.")
        elif door_state == "door_open":
            send_slack(f"🔓 [{time.strftime('%H:%M:%S')}] 문 열었습니다.")

    # ── 화면 시각화 ───────────────────────────
    annotated = results[0].plot().copy()
    h, w = annotated.shape[:2]

    # 상단 상태 바
    if door_state == "door_closed":
        bar_color, state_text = (30, 30, 180), "CLOSED"
    elif door_state == "door_open":
        bar_color, state_text = (30, 180, 30), "OPEN"
    else:
        bar_color, state_text = (80, 80, 80), "waiting..."

    cv2.rectangle(annotated, (0, 0), (w, 44), bar_color, -1)
    cv2.putText(annotated, state_text,
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    # 우측 상단: 신뢰도 / 밝기 / 사람
    cv2.putText(annotated, f"conf:  {confidence:.0%}",
                (w - 160, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
    cv2.putText(annotated, f"bright: {'dark' if is_dark else 'light'}",
                (w - 160, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
    person_color = (100, 255, 100) if person_count == 0 else (100, 100, 255)
    cv2.putText(annotated, f"person: {person_count}",
                (w - 160, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, person_color, 2)

    # 하단 진행 바
    bar_w = int(w * min(elapsed / HOLD_DURATION, 1.0))
    cv2.rectangle(annotated, (0, h - 6), (bar_w, h), (0, 220, 255), -1)

    cv2.imshow("강의실 모니터 (q: 종료)", annotated)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
