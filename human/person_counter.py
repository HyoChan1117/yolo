"""YOLO-based person counting module."""

from __future__ import annotations

import os

from dotenv import load_dotenv
from ultralytics import YOLO

load_dotenv()

PERSON_MODEL_PATH = os.getenv("PERSON_MODEL_PATH", "yolov8n.pt")
PERSON_CONF_THRESH = float(os.getenv("PERSON_CONF_THRESH", "0.35"))


class PersonCounter:
    def __init__(
        self,
        model_path: str = PERSON_MODEL_PATH,
        conf_thresh: float = PERSON_CONF_THRESH,
    ):
        self.model = YOLO(model_path)
        self.conf_thresh = conf_thresh

    def count(
        self, frame
    ) -> tuple[int, list[tuple[int, int, int, int]], list[float]]:
        """Returns (count, boxes, confidences) for all detected persons."""
        result = self.model(frame, verbose=False)[0]
        boxes: list[tuple[int, int, int, int]] = []
        confidences: list[float] = []

        if result.boxes is None:
            return 0, boxes, confidences

        for box in result.boxes:
            cls_id = int(box.cls.item())
            conf = float(box.conf.item())
            if cls_id != 0 or conf < self.conf_thresh:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            boxes.append((x1, y1, x2, y2))
            confidences.append(conf)

        return len(boxes), boxes, confidences
