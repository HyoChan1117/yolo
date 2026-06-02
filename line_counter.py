"""IP 카메라 1대 YOLO 추론 (세로 화면) + 선 통과 카운팅
  선을 -1→+1 방향으로 건너면 IN +1
  선을 +1→-1 방향으로 건너면 OUT +1
실행: python line_counter.py
종료: q
"""

import json
import logging
import threading
import time
import cv2
import numpy as np
from ultralytics import YOLO
from dotenv import load_dotenv
import os

load_dotenv()

os.makedirs("logs", exist_ok=True)
_log_path = os.path.join("logs", time.strftime("line_counter_%Y%m%d.log"))
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(_log_path, encoding="utf-8", mode="a"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

MODEL_PATH = "human/models/medium.engine"
LINE_FILE = "line_counter_line.json"

USER = os.getenv("VIGI_USERNAME", "admin")
PASS = os.getenv("VIGI_PASSWORD", "")
STREAM = os.getenv("VIGI_STREAM", "stream1")

CAM1_IP = "192.168.50.111"
RTSP_URL = f"rtsp://{USER}:{PASS}@{CAM1_IP}:554/{STREAM}"

CONF = 0.3
IMGSZ = 640
PERSON_CLASS = 0  # COCO class 0 = person


def load_polyline(path: str) -> list[tuple[int, int]]:
    data = json.loads(open(path, encoding="utf-8").read())
    pts = data["lines"][0]
    return [(int(p[0]), int(p[1])) for p in pts]


def save_polyline(path: str, points: list[tuple[int, int]]) -> None:
    data = {"lines": [[[p[0], p[1]] for p in points]]}
    open(path, "w", encoding="utf-8").write(json.dumps(data))


def draw_line_interactive(frame) -> list[tuple[int, int]] | None:
    """첫 프레임에서 마우스로 폴리라인을 그리고 점 목록을 반환.
    좌클릭: 점 추가 / 우클릭: 마지막 점 취소 / Enter: 확정 / Esc: 취소
    """
    WIN = "Draw Line"
    points: list[tuple[int, int]] = []
    cursor: list[int] = [0, 0]

    def on_mouse(event, x, y, *_):
        cursor[0], cursor[1] = x, y
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
        elif event == cv2.EVENT_RBUTTONDOWN and points:
            points.pop()

    # 창을 먼저 띄운 뒤 마우스 콜백 등록 (Windows 이벤트 큐 안정화)
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.imshow(WIN, frame)
    cv2.waitKey(1)
    cv2.setMouseCallback(WIN, on_mouse)

    while True:
        canvas = frame.copy()
        for i in range(len(points) - 1):
            cv2.line(canvas, points[i], points[i + 1], (0, 200, 255), 2)
        if points:
            cv2.line(canvas, points[-1], (cursor[0], cursor[1]), (0, 200, 255), 1)
        for p in points:
            cv2.circle(canvas, p, 6, (0, 80, 255), -1)

        guide = f"pts:{len(points)}  LClick:add  RClick:undo  Enter:confirm  Esc:cancel"
        cv2.putText(canvas, guide, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.imshow(WIN, canvas)

        key = cv2.waitKey(16) & 0xFF
        if key == 13 and len(points) >= 2:  # Enter
            cv2.destroyWindow(WIN)
            return points
        elif key == 27:  # Esc
            cv2.destroyWindow(WIN)
            return None


def point_side(px: float, py: float, polyline: list[tuple[int, int]]) -> int:
    """가장 가까운 선분의 외적으로 오른쪽 +1, 왼쪽 -1 반환."""
    min_dist = float("inf")
    side = 1
    for i in range(len(polyline) - 1):
        ax, ay = polyline[i]
        bx, by = polyline[i + 1]
        abx, aby = bx - ax, by - ay
        t = max(0.0, min(1.0, ((px - ax) * abx + (py - ay) * aby) / (abx ** 2 + aby ** 2 + 1e-9)))
        cx, cy = ax + t * abx, ay + t * aby
        dist = (px - cx) ** 2 + (py - cy) ** 2
        if dist < min_dist:
            min_dist = dist
            cross = abx * (py - ay) - aby * (px - ax)
            side = 1 if cross > 0 else -1
    return side


def draw_polyline(frame, polyline):
    pts = np.array(polyline, dtype=np.int32)
    cv2.polylines(frame, [pts], isClosed=False, color=(0, 200, 255), thickness=3)


class CameraStream:
    def __init__(self, url: str, name: str):
        self.name = name
        self.cap = cv2.VideoCapture(url)
        self.frame = None
        self.running = False
        self._lock = threading.Lock()

    def start(self):
        if not self.cap.isOpened():
            print(f"[{self.name}] 카메라 연결 실패")
            return False
        self.running = True
        threading.Thread(target=self._read_loop, daemon=True).start()
        return True

    def _read_loop(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                with self._lock:
                    self.frame = frame
            else:
                self.running = False

    def get_frame(self):
        with self._lock:
            return self.frame.copy() if self.frame is not None else None

    def stop(self):
        self.running = False
        self.cap.release()


def main():
    model = YOLO(MODEL_PATH)
    print(f"모델 로드 완료: {MODEL_PATH}")

    cam = CameraStream(RTSP_URL, "CAM1")
    if not cam.start():
        print("카메라 연결 실패. IP 주소와 자격증명을 확인하세요.")
        return

    if os.path.exists(LINE_FILE):
        polyline = load_polyline(LINE_FILE)
        print(f"선 로드 완료: {polyline}")
    else:
        print(f"{LINE_FILE} 없음 → 첫 프레임에서 선을 직접 그려주세요.")
        # 첫 프레임이 들어올 때까지 대기
        first_frame = None
        while first_frame is None:
            first_frame = cam.get_frame()
            time.sleep(0.03)

        h, w = first_frame.shape[:2]
        if w > h:
            first_frame = cv2.rotate(first_frame, cv2.ROTATE_90_CLOCKWISE)

        polyline = draw_line_interactive(first_frame)
        if not polyline:
            print("선 그리기 취소. 종료합니다.")
            cam.stop()
            return
        save_polyline(LINE_FILE, polyline)
        print(f"선 저장 완료: {LINE_FILE}")

    print("스트림 시작 (q: 종료)")

    prev_side: dict[int, int] = {}
    count = 0  # IN: +1, OUT: -1, 0 미만 불가

    while True:
        frame = cam.get_frame()

        if frame is not None:
            h, w = frame.shape[:2]
            if w > h:
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

            results = model.track(frame, conf=CONF, imgsz=IMGSZ,
                                  classes=[PERSON_CLASS], persist=True, verbose=False)
            frame = results[0].plot(labels=False)

            boxes = results[0].boxes
            if boxes.id is not None:
                for box, track_id in zip(boxes, boxes.id.int().tolist()):
                    x1, _, x2, y2 = box.xyxy[0].tolist()
                    foot_x = (x1 + x2) / 2
                    foot_y = y2
                    side = point_side(foot_x, foot_y, polyline)

                    if track_id in prev_side and prev_side[track_id] != side:
                        if side == 1:
                            count += 1
                            logger.info("IN   count=%d", count)
                        else:
                            count = max(0, count - 1)
                            logger.info("OUT  count=%d", count)

                    prev_side[track_id] = side

            draw_polyline(frame, polyline)

            cv2.putText(frame, f"COUNT: {count}  (+i / -u)",
                        (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)

            cv2.imshow("Line Counter (q: 종료)", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("i"):
            count += 1
            logger.info("IN   manual       count=%d", count)
        elif key == ord("u"):
            count = max(0, count - 1)
            logger.info("OUT  manual       count=%d", count)

    cam.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
