"""ROI + AI (MobileNetV3) 기반 문 상태 감지 + Slack 알림
──────────────────────────────────────────────────────────
흐름:
  1. roi.json 에서 ROI 좌표 로드
  2. 웹캠 프레임에서 ROI 크롭
  3. MobileNetV3 추론 → door_open / door_closed
  4. HOLD_DURATION 초 유지되면 상태 확정 후 Slack 전송

종료: q

실행: python door/webcam.py
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import cv2
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
from dotenv import load_dotenv
from torchvision import models, transforms

load_dotenv()

MODEL_PATH = Path("door/models/best.pth")
ROI_FILE = Path("roi.json")
HOLD_DURATION = float(os.getenv("HOLD_DURATION", "5.0"))
CONF_THRESH = float(os.getenv("DOOR_CONF_THRESH", "0.70"))
SLACK_URL = os.getenv("SLACK_WEBHOOK_URL", "")
CAMERA_INDEX = int(os.getenv("CAMERA_INDEX", "1"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── ROI 로드 ──────────────────────────────────────────────────
if not ROI_FILE.exists():
    print("roi.json 없음. door/collect_dataset.py 를 먼저 실행하세요.")
    exit()

roi = json.loads(ROI_FILE.read_text(encoding="utf-8"))
rx, ry, rw, rh = roi["x"], roi["y"], roi["w"], roi["h"]
print(f"ROI: x={rx} y={ry} w={rw} h={rh}")

# ── 모델 로드 ─────────────────────────────────────────────────
if not MODEL_PATH.exists():
    print(f"모델 없음: {MODEL_PATH}  door/train.py 를 먼저 실행하세요.")
    exit()

checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
CLASSES = checkpoint["classes"]

model = models.mobilenet_v3_small()
model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, len(CLASSES))
model.load_state_dict(checkpoint["state_dict"])
model.eval().to(DEVICE)
print(f"모델 로드 완료. 클래스: {CLASSES}")

transform = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


def predict(crop) -> tuple[str, float]:
    x = transform(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        probs = F.softmax(model(x), dim=1)[0]
    idx = int(probs.argmax())
    return CLASSES[idx], float(probs[idx])


def send_slack(message: str) -> None:
    if not SLACK_URL:
        return
    try:
        requests.post(SLACK_URL, json={"text": message}, timeout=5)
    except Exception:
        pass


# ── 웹캠 루프 ─────────────────────────────────────────────────
cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    print("웹캠을 열 수 없어요.")
    exit()

door_state: str | None = None
pending_state: str | None = None
pending_start: float | None = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    now = time.time()
    crop = frame[ry : ry + rh, rx : rx + rw]

    top_class, confidence = predict(crop)
    is_valid = confidence >= CONF_THRESH

    if is_valid and top_class != "invalid" and top_class != pending_state:
        pending_state = top_class
        pending_start = now

    elapsed = now - (pending_start or now)

    if is_valid and top_class == "invalid":
        elapsed = 0

    if is_valid and pending_state and elapsed >= HOLD_DURATION and pending_state != door_state:
        door_state = pending_state
        ts = time.strftime("%H:%M:%S")
        if door_state == "door_open":
            msg = f"[{ts}] 문 열렸습니다."
        elif door_state == "door_closed":
            msg = f"[{ts}] 문 닫혔습니다."
        else:
            continue
        print(msg)
        send_slack(msg)

    # ── 화면 표시 ─────────────────────────────────────────────
    display = frame.copy()
    cv2.rectangle(display, (rx, ry), (rx + rw, ry + rh), (0, 255, 255), 2)

    label_text = (
        f"{top_class} {confidence:.0%}" if is_valid else f"LOW CONF {confidence:.0%}"
    )
    cv2.putText(
        display,
        label_text,
        (rx, max(18, ry - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 255),
        2,
    )

    if door_state == "door_open":
        bar_color, state_text = (30, 180, 30), "OPEN"
    elif door_state == "door_closed":
        bar_color, state_text = (30, 30, 180), "CLOSED"
    else:
        bar_color, state_text = (80, 80, 80), "WAITING"

    if is_valid and pending_state and pending_state != door_state:
        state_text += f"  ({elapsed:.1f}s / {HOLD_DURATION}s)"

    h_frame, w_frame = display.shape[:2]
    cv2.rectangle(display, (0, 0), (w_frame, 44), bar_color, -1)
    cv2.putText(
        display, f"Door: {state_text}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2
    )

    bar_w = int(w_frame * min(elapsed / HOLD_DURATION, 1.0)) if pending_state else 0
    cv2.rectangle(display, (0, h_frame - 6), (bar_w, h_frame), (0, 220, 255), -1)

    thumb = cv2.resize(crop, (rw * 2, rh * 2)) if rw < 100 else crop.copy()
    th, tw = thumb.shape[:2]
    if tw < w_frame and th < h_frame - 50:
        display[50 : 50 + th, w_frame - tw - 10 : w_frame - 10] = thumb

    cv2.imshow("문 감지 (q: 종료)", display)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
