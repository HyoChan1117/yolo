"""YOLO ONNX 기반 사람 수 카운팅 웹캠 + Slack 슬래시 커맨드 (/인원)

실행: python human/webcam.py

Slack 슬래시 커맨드 설정:
  1. https://api.slack.com/apps 에서 앱 생성
  2. Slash Commands → /인원 추가
  3. Request URL: http://<ngrok-url>/slack/people-count
  4. ngrok 실행: ngrok http 5100
"""

from __future__ import annotations

import os
import sys
import threading
from pathlib import Path

import json

import cv2
from dotenv import load_dotenv
from flask import Flask, request, jsonify

sys.path.insert(0, str(Path(__file__).parent.parent))
from human.person_counter import PersonCounter

load_dotenv()

CAMERA_INDEX  = int(os.getenv("CAMERA_INDEX", "0"))
SLACK_TOKEN   = os.getenv("SLACK_VERIFICATION_TOKEN", "")  # 선택: 요청 검증용
SERVER_PORT   = int(os.getenv("PERSON_SERVER_PORT", "5100"))

# ── 공유 상태 ─────────────────────────────────────────────────
_count_lock   = threading.Lock()
_current_count: int = 0


def get_count() -> int:
    with _count_lock:
        return _current_count


def set_count(n: int) -> None:
    global _current_count
    with _count_lock:
        _current_count = n


# ── Flask 슬래시 커맨드 서버 ──────────────────────────────────
app = Flask(__name__)


DOOR_STATE_FILE = Path(__file__).parent.parent / "door_state.json"


@app.route("/slack/door-status", methods=["POST"])
def slack_door_status():
    if SLACK_TOKEN and request.form.get("token") != SLACK_TOKEN:
        return jsonify({"error": "unauthorized"}), 403

    state = None
    if DOOR_STATE_FILE.exists():
        try:
            state = json.loads(DOOR_STATE_FILE.read_text(encoding="utf-8")).get("state")
        except Exception:
            pass

    if state == "door_open":
        text = "문 상태: *열려 있음* :door:"
    elif state == "door_closed":
        text = "문 상태: *닫혀 있음* :lock:"
    else:
        text = "문 상태: *확인 중* (아직 판정 전)"

    return jsonify({"response_type": "in_channel", "text": text})


@app.route("/slack/people-count", methods=["POST"])
def slack_person_count():
    # Verification Token 검증 (Slack 앱에서 설정한 경우)
    if SLACK_TOKEN and request.form.get("token") != SLACK_TOKEN:
        return jsonify({"error": "unauthorized"}), 403

    count = get_count()
    return jsonify({
        "response_type": "in_channel",
        "text": f"현재 감지된 인원: *{count}명*",
    })


def _run_server() -> None:
    import logging
    log = logging.getLogger("werkzeug")
    log.setLevel(logging.ERROR)  # Flask 요청 로그 억제
    app.run(host="0.0.0.0", port=SERVER_PORT)


# ── 모델 로드 ─────────────────────────────────────────────────
person_counter = PersonCounter()

# Flask 서버를 데몬 스레드로 시작
threading.Thread(target=_run_server, daemon=True).start()
print(f"Slack 서버 시작: http://0.0.0.0:{SERVER_PORT}/slack/인원")
print("ngrok 사용: ngrok http", SERVER_PORT)

# ── 웹캠 루프 ─────────────────────────────────────────────────
cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    print("웹캠을 열 수 없어요.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    count, boxes, scores = person_counter.count(frame)
    set_count(count)

    display = frame.copy()

    for (x1, y1, x2, y2), score in zip(boxes, scores):
        cv2.rectangle(display, (x1, y1), (x2, y2), (0, 220, 120), 2)
        cv2.putText(
            display,
            f"person {score:.0%}",
            (x1, max(18, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 220, 120),
            2,
        )

    h_frame, w_frame = display.shape[:2]
    cv2.rectangle(display, (0, 0), (w_frame, 44), (40, 40, 40), -1)
    cv2.putText(
        display,
        f"People: {count}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2,
    )

    cv2.imshow("사람 카운팅 (q: 종료)", display)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
