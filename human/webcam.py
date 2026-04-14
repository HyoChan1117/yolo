"""YOLO 기반 사람 수 카운팅 웹캠

실행: python human/webcam.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import cv2
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent))
from human.person_counter import PersonCounter

load_dotenv()

CAMERA_INDEX = int(os.getenv("CAMERA_INDEX", "1"))

person_counter = PersonCounter()

cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    print("웹캠을 열 수 없어요.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    count, boxes, scores = person_counter.count(frame)
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
