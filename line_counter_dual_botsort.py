"""IP 카메라 2대 동시 YOLO 추론 + 선 1개 통과 카운팅 (통합 카운트) — BoT-SORT 버전
  CAM1: 192.168.50.111
  CAM2: 192.168.50.113
  선 파일: line_counter_line_dual.json
  선을 +1→-1 방향으로 건너면 IN +1, -1→+1 방향으로 건너면 OUT -1
실행: python line_counter_dual_botsort.py
종료: q  |  수동: i(+1)  u(-1)
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
_log_path = os.path.join("logs", time.strftime("line_counter_dual_botsort_%Y%m%d.log"))
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

MODEL_PATH = "human/models/yolo11m.pt"
LINE_FILE  = "line_counter_line_dual.json"

USER   = os.getenv("VIGI_USERNAME", "admin")
PASS   = os.getenv("VIGI_PASSWORD", "")
STREAM = os.getenv("VIGI_STREAM", "stream1")

CAMS_CFG = [
    {"ip": "192.168.50.111", "name": "CAM1"},
    {"ip": "192.168.50.113", "name": "CAM2"},
]

CONF         = 0.3
IMGSZ        = 640
PERSON_CLASS = 0

COLOR_LINE   = (0, 200, 255)
COLOR_NORMAL = (0, 200, 255)
COLOR_IN     = (0, 255,   0)
COLOR_OUT    = (0,  80, 255)


# ── 선 파일 ───────────────────────────────────────────────
# 형식: {"CAM1": [[x,y],...], "CAM2": [[x,y],...]}

def _parse_pts(pts) -> list[tuple[int, int]]:
    return [(int(p[0]), int(p[1])) for p in pts]

def load_lines(path: str) -> dict[str, list]:
    data = json.loads(open(path, encoding="utf-8").read())
    result = {}
    for name, pts in data.items():
        if isinstance(pts, list) and len(pts) >= 2 and isinstance(pts[0], list):
            result[name] = _parse_pts(pts)
    return result

def save_lines(path: str, lines: dict[str, list]) -> None:
    data = {name: [[p[0], p[1]] for p in pts] for name, pts in lines.items()}
    open(path, "w", encoding="utf-8").write(json.dumps(data))


# ── 공통 유틸 ─────────────────────────────────────────────

def draw_line_interactive(
    frame, cam_name: str
) -> list[tuple[int, int]] | None:
    """마우스로 폴리라인을 그리고 점 목록을 반환.
    좌클릭: 점 추가 / 우클릭: 마지막 점 취소 / Enter: 확정(2점 이상) / Esc: 취소
    """
    WIN = f"Draw [{cam_name}]"
    points, cursor = [], [0, 0]

    def on_mouse(event, x, y, *_):
        cursor[0], cursor[1] = x, y
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
        elif event == cv2.EVENT_RBUTTONDOWN and points:
            points.pop()

    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.imshow(WIN, frame)
    cv2.waitKey(1)
    cv2.setMouseCallback(WIN, on_mouse)

    while True:
        canvas = frame.copy()
        for i in range(len(points) - 1):
            cv2.line(canvas, points[i], points[i+1], COLOR_LINE, 2)
        if points:
            cv2.line(canvas, points[-1], tuple(cursor), COLOR_LINE, 1)
        for p in points:
            cv2.circle(canvas, p, 6, COLOR_LINE, -1)

        ok_hint = "Enter:ok" if len(points) >= 2 else f"need {2 - len(points)} more"
        cv2.putText(canvas,
                    f"[{cam_name}]  LClick:add  RClick:undo  {ok_hint}  Esc:cancel",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.imshow(WIN, canvas)

        key = cv2.waitKey(16) & 0xFF
        if key == 13 and len(points) >= 2:
            cv2.destroyWindow(WIN)
            return points
        elif key == 27:
            cv2.destroyWindow(WIN)
            return None


def point_side(px: float, py: float, polyline: list[tuple[int, int]]) -> int:
    """가장 가까운 선분의 외적으로 오른쪽 +1, 왼쪽 -1 반환."""
    min_dist, side = float("inf"), 1
    for i in range(len(polyline) - 1):
        ax, ay = polyline[i]; bx, by = polyline[i+1]
        abx, aby = bx - ax, by - ay
        t = max(0.0, min(1.0, ((px-ax)*abx + (py-ay)*aby) / (abx**2 + aby**2 + 1e-9)))
        dist = (px - ax - t*abx)**2 + (py - ay - t*aby)**2
        if dist < min_dist:
            min_dist = dist
            cross = abx*(py-ay) - aby*(px-ax)
            side = 1 if cross > 0 else -1
    return side


def draw_polyline(frame, polyline) -> None:
    cv2.polylines(frame, [np.array(polyline, dtype=np.int32)],
                  isClosed=False, color=COLOR_LINE, thickness=3)


# ── 공유 카운터 ───────────────────────────────────────────

class SharedCounter:
    def __init__(self):
        self.value = 0
        self._lock = threading.Lock()

    def increment(self) -> int:
        with self._lock:
            self.value += 1
            return self.value

    def decrement(self) -> int:
        with self._lock:
            self.value = max(0, self.value - 1)
            return self.value

    def adjust(self, delta: int) -> int:
        with self._lock:
            self.value = max(0, self.value + delta)
            return self.value


# ── 카메라 스트림 ────────────────────────────────────────

class CameraStream:
    def __init__(self, url, name):
        self.name = name
        self.cap  = cv2.VideoCapture(url)
        self.frame   = None
        self.running = False
        self._lock   = threading.Lock()

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


# ── 카운터 워커 (카메라 1대 담당) ───────────────────────

class CamCounter:
    def __init__(self, stream: CameraStream, model: YOLO,
                 polyline: list, name: str, counter: SharedCounter):
        self.stream   = stream
        self.model    = model
        self.polyline = polyline
        self.name     = name
        self.counter  = counter

        self._prev_side:   dict[int, int]   = {}
        self._flash_ids:   dict[int, float] = {}
        self._flash_col:   dict[int, tuple] = {}
        self._last_pos:    dict[int, tuple] = {}   # tid -> (x, y)
        self._lost_tracks: dict[int, tuple] = {}   # tid -> (side, x, y, lost_time)

        self._annotations: dict[int, tuple] = {}
        self._ann_lock   = threading.Lock()
        self._frame_out  = None
        self._frame_lock = threading.Lock()
        self.running = False

    def start(self):
        self.running = True
        threading.Thread(target=self._display_loop, daemon=True).start()
        threading.Thread(target=self._inference_loop, daemon=True).start()

    def stop(self):
        self.running = False

    def get_frame(self):
        with self._frame_lock:
            return self._frame_out.copy() if self._frame_out is not None else None

    def _display_loop(self):
        while self.running:
            frame = self.stream.get_frame()
            if frame is None:
                time.sleep(0.005)
                continue

            h, w = frame.shape[:2]
            if w > h:
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

            now = time.time()
            with self._ann_lock:
                self._annotations = {tid: v for tid, v in self._annotations.items() if v[5] > now}
                active = dict(self._annotations)

            draw_polyline(frame, self.polyline)

            for tid, (x1, y1, x2, y2, color, _) in active.items():
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                cv2.circle(frame, (x1, (y1+y2)//2), 5, color, -1)
                cv2.putText(frame, str(tid), (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            with self._frame_lock:
                self._frame_out = frame

    def _inference_loop(self):
        _REMATCH_DIST = 150   # 픽셀: lost track 재매칭 허용 거리
        _REMATCH_TIME = 2.0   # 초: lost track 유지 시간

        while self.running:
            frame = self.stream.get_frame()
            if frame is None:
                time.sleep(0.01)
                continue

            h, w = frame.shape[:2]
            if w > h:
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

            results = self.model.track(frame, conf=CONF, imgsz=IMGSZ,
                                       classes=[PERSON_CLASS], persist=True, verbose=False,
                                       tracker="botsort_custom.yaml")
            boxes = results[0].boxes
            now   = time.time()

            if boxes.id is not None:
                active_tids = set(boxes.id.int().tolist())

                # 이번 프레임에 없는 track → lost_tracks로 이동
                for tid in list(self._prev_side.keys()):
                    if tid not in active_tids:
                        lx, ly = self._last_pos.get(tid, (0, 0))
                        self._lost_tracks[tid] = (self._prev_side.pop(tid), lx, ly, now)

                # 오래된 lost track 정리
                self._lost_tracks = {t: v for t, v in self._lost_tracks.items()
                                     if now - v[3] < _REMATCH_TIME}

                new_anns: dict[int, tuple] = {}
                for box, tid in zip(boxes, boxes.id.int().tolist()):
                    x1, y1, _, y2 = box.xyxy[0].tolist()
                    left_x = x1
                    left_y = (y1 + y2) / 2
                    side = point_side(left_x, left_y, self.polyline)

                    self._last_pos[tid] = (left_x, left_y)

                    # 새 ID면 lost track과 위치 매칭해서 이전 측면 상속
                    if tid not in self._prev_side and self._lost_tracks:
                        best_tid = min(
                            self._lost_tracks,
                            key=lambda t: (left_x - self._lost_tracks[t][1])**2
                                        + (left_y - self._lost_tracks[t][2])**2
                        )
                        dist = ((left_x - self._lost_tracks[best_tid][1])**2
                              + (left_y - self._lost_tracks[best_tid][2])**2) ** 0.5
                        if dist < _REMATCH_DIST:
                            self._prev_side[tid] = self._lost_tracks.pop(best_tid)[0]
                            logger.info("[%s] track rematch: %d→%d (dist=%.1f)",
                                        self.name, best_tid, tid, dist)

                    if tid in self._prev_side and self._prev_side[tid] != side:
                        if side == 1:
                            cnt = self.counter.increment()
                            logger.info("[%s] IN   count=%d", self.name, cnt)
                            self._flash_ids[tid] = now + 1.5
                            self._flash_col[tid] = COLOR_IN
                        else:
                            cnt = self.counter.decrement()
                            logger.info("[%s] OUT  count=%d", self.name, cnt)
                            self._flash_ids[tid] = now + 1.5
                            self._flash_col[tid] = COLOR_OUT

                    self._prev_side[tid] = side

                    bx1, by1, bx2, by2 = map(int, box.xyxy[0].tolist())
                    if self._flash_ids.get(tid, 0) > now:
                        color  = self._flash_col.get(tid, COLOR_NORMAL)
                        expiry = self._flash_ids[tid]
                    else:
                        color  = COLOR_NORMAL
                        expiry = now + 1.0
                    new_anns[tid] = (bx1, by1, bx2, by2, color, expiry)

                self._flash_ids = {tid: t for tid, t in self._flash_ids.items() if t > now}
                with self._ann_lock:
                    self._annotations.update(new_anns)


# ── main ────────────────────────────────────────────────

def main():
    lines: dict[str, list] = {}
    if os.path.exists(LINE_FILE):
        lines = load_lines(LINE_FILE)
        if lines:
            print(f"선 파일 로드 완료: {LINE_FILE}")
        else:
            print(f"[경고] {LINE_FILE} 가 구형식이거나 비어있습니다. 선을 다시 그려주세요.")

    streams = []
    for cfg in CAMS_CFG:
        url = f"rtsp://{USER}:{PASS}@{cfg['ip']}:554/{STREAM}"
        s = CameraStream(url, cfg["name"])
        if not s.start():
            for st in streams:
                st.stop()
            return
        streams.append(s)

    for s, cfg in zip(streams, CAMS_CFG):
        if cfg["name"] not in lines:
            print(f"[{cfg['name']}] 선 없음 → 선을 그려주세요.")
            first_frame = None
            while first_frame is None:
                first_frame = s.get_frame()
                time.sleep(0.03)
            h, w = first_frame.shape[:2]
            if w > h:
                first_frame = cv2.rotate(first_frame, cv2.ROTATE_90_CLOCKWISE)
            pts = draw_line_interactive(first_frame, cfg["name"])
            if not pts:
                print("선 그리기 취소. 종료합니다.")
                for st in streams:
                    st.stop()
                return
            lines[cfg["name"]] = pts
            save_lines(LINE_FILE, lines)
            print(f"[{cfg['name']}] 선 저장 완료: {LINE_FILE}")

    counter = SharedCounter()
    workers = []
    for s, cfg in zip(streams, CAMS_CFG):
        print(f"[{cfg['name']}] 모델 로드 중...")
        model = YOLO(MODEL_PATH)
        worker = CamCounter(s, model, lines[cfg["name"]], cfg["name"], counter)
        worker.start()
        workers.append(worker)

    print("스트림 시작 (q: 종료 | i: +1  u: -1)")

    cached = [None, None]

    while True:
        for i, worker in enumerate(workers):
            f = worker.get_frame()
            if f is not None:
                cached[i] = f

        if cached[0] is not None and cached[1] is not None:
            f0, f1 = cached[0], cached[1]
            h = max(f0.shape[0], f1.shape[0])
            if f0.shape[0] != h:
                f0 = cv2.resize(f0, (int(f0.shape[1] * h / f0.shape[0]), h))
            if f1.shape[0] != h:
                f1 = cv2.resize(f1, (int(f1.shape[1] * h / f1.shape[0]), h))
            combined = np.hstack([f0, f1])

            cnt = counter.value
            label = f"COUNT: {cnt}"
            font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2
            (tw, th), bl = cv2.getTextSize(label, font, scale, thick)
            pad = 6
            cv2.rectangle(combined, (8, 8), (8 + tw + pad*2, 8 + th + bl + pad*2), (0, 0, 0), -1)
            cv2.putText(combined, label, (8 + pad, 8 + th + pad), font, scale, (0, 255, 0), thick)

            cv2.imshow("Line Counter Dual BoT-SORT (q: 종료)", combined)

        key = cv2.waitKey(1) & 0xFF
        if   key == ord("q"): break
        elif key == ord("i"):
            cnt = counter.adjust(+1)
            logger.info("IN   manual  count=%d", cnt)
        elif key == ord("u"):
            cnt = counter.adjust(-1)
            logger.info("OUT  manual  count=%d", cnt)

    for s in streams:
        s.stop()
    for w in workers:
        w.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
