import cv2
import numpy as np
from ultralytics import YOLO
from dotenv import load_dotenv
import threading
import os
import json

load_dotenv()

CAMERA_IPS = ["192.168.50.111", "192.168.50.112", "192.168.50.113", "192.168.50.114"]
USER     = os.getenv("VIGI_USERNAME", "admin")
PW       = os.getenv("VIGI_PASSWORD", "gsctest01A!")
ROI_FILE = "rois.json"

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"


def capture_sync(ip, barrier, results):
    url = f"rtsp://{USER}:{PW}@{ip}:554/stream1"
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        print(f"[{ip}] connection failed")
        barrier.wait()
        results[ip] = None
        return
    barrier.wait()  # 모든 카메라 연결 완료 후 동시에 버퍼 비우기
    for _ in range(5):  # 버퍼에 쌓인 오래된 프레임 드레인 (디코딩 없이 빠름)
        cap.grab()
    ret, frame = cap.retrieve()
    cap.release()
    results[ip] = frame if ret else None
    if not ret:
        print(f"[{ip}] capture failed")


def draw_roi(ip, frame, cam_index, total):
    """마우스 폴리곤 ROI 설정.
    Left click: add point | Right click: remove last | Enter/Space: confirm | ESC: full frame
    """
    pts = []
    win = f"ROI Setup [{cam_index}/{total}] {ip}"

    def mouse_cb(event, x, y, *_):
        if event == cv2.EVENT_LBUTTONDOWN:
            pts.append((x, y))
        elif event == cv2.EVENT_RBUTTONDOWN and pts:
            pts.pop()

    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 540, 960)
    cv2.imshow(win, frame)
    cv2.waitKey(1)                      # 창이 완전히 생성된 뒤에 콜백 등록
    cv2.setMouseCallback(win, mouse_cb)

    while True:
        display = frame.copy()

        for p in pts:
            cv2.circle(display, p, 6, (0, 255, 0), -1)
        if len(pts) >= 2:
            cv2.polylines(display, [np.array(pts)], False, (0, 255, 0), 2)
        if len(pts) >= 3:
            ov = display.copy()
            cv2.fillPoly(ov, [np.array(pts)], (0, 255, 0))
            cv2.addWeighted(ov, 0.2, display, 0.8, 0, display)
            cv2.polylines(display, [np.array(pts)], True, (0, 255, 0), 2)

        instructions = [
            f"Camera {cam_index}/{total}  |  {ip}",
            "Left click: add pt  Right click: remove pt",
            f"Points: {len(pts)}   Enter/Space(3+pts): confirm   ESC: use full frame",
        ]
        for i, line in enumerate(instructions):
            cv2.putText(display, line, (10, 30 + i * 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 200, 255), 2)

        cv2.imshow(win, display)
        key = cv2.waitKey(20) & 0xFF

        if key in (13, 32) and len(pts) >= 3:   # Enter or Space
            break
        elif key == 27:                          # ESC → full frame
            pts.clear()
            break

    cv2.destroyWindow(win)
    cv2.waitKey(1)
    return pts if len(pts) >= 3 else None


def load_rois():
    if not os.path.exists(ROI_FILE):
        return None
    with open(ROI_FILE) as f:
        data = json.load(f)
    return {ip: ([tuple(p) for p in pts] if pts else None) for ip, pts in data.items()}


def save_rois(rois):
    with open(ROI_FILE, "w") as f:
        json.dump(rois, f, indent=2)
    print(f"ROI saved: {ROI_FILE}  (delete this file to reset ROI)")


def in_roi(xyxy, roi_pts):
    """바운딩 박스 하단 중심점이 ROI 폴리곤 안에 있는지 확인."""
    if roi_pts is None:
        return True
    x1, _, x2, y2 = xyxy
    cx = int((x1 + x2) / 2)
    poly = np.array(roi_pts, dtype=np.int32)
    return cv2.pointPolygonTest(poly, (float(cx), float(y2)), False) >= 0


# ── 1. 캡처 ───────────────────────────────────────────────────────
print("Capturing from 4 cameras...")
frames = {}
barrier = threading.Barrier(len(CAMERA_IPS))
threads = [threading.Thread(target=capture_sync, args=(ip, barrier, frames)) for ip in CAMERA_IPS]
for t in threads:
    t.start()
for t in threads:
    t.join()

# ── 2. ROI 설정 (최초 1회) ────────────────────────────────────────
rois = load_rois()
if rois is None:
    print("\n[First run] Set ROI for each camera.")
    rois = {}
    for idx, ip in enumerate(CAMERA_IPS, 1):
        frame = frames.get(ip)
        if frame is None:
            rois[ip] = None
            continue
        pts = draw_roi(ip, frame, idx, len(CAMERA_IPS))
        rois[ip] = pts
        label = "full frame" if pts is None else f"polygon {len(pts)} pts"
        print(f"  [{ip}] {label}")
    save_rois(rois)
    print()

# ── 3. 탐지 ───────────────────────────────────────────────────────
model = YOLO("human/models/yolo11x_fine.pt", task="detect")

valid_ips    = [ip for ip in CAMERA_IPS if frames.get(ip) is not None]
valid_frames = [frames[ip] for ip in valid_ips]

batch_results = model(valid_frames, verbose=False, classes=[0], conf=0.20) if valid_frames else []
results_map   = dict(zip(valid_ips, batch_results))

annotated = {}
for ip in CAMERA_IPS:
    if frames.get(ip) is None:
        annotated[ip] = np.zeros((480, 640, 3), dtype=np.uint8)
        continue

    roi_pts = rois.get(ip)
    result  = results_map[ip]
    boxes   = result.boxes

    count = 0
    if boxes is not None:
        for i in range(len(boxes)):
            if in_roi(boxes.xyxy[i].tolist(), roi_pts):
                count += 1

    print(f"[{ip}] people in ROI: {count}")

    img = result.plot()

    if roi_pts:
        poly = np.array(roi_pts, dtype=np.int32)
        ov = img.copy()
        cv2.fillPoly(ov, [poly], (0, 255, 0))
        cv2.addWeighted(ov, 0.15, img, 0.85, 0, img)
        cv2.polylines(img, [poly], True, (0, 255, 0), 2)

    cv2.putText(img, f"{ip}  People(ROI): {count}", (10, 36),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    annotated[ip] = img

# ── 4. 1x4 가로 나열 표시 ─────────────────────────────────────────
W, H = 540, 960
tiles = [cv2.resize(annotated[ip], (W, H)) for ip in CAMERA_IPS]
grid  = np.hstack(tiles)

cv2.imwrite("result_grid.jpg", grid)
print("Saved: result_grid.jpg")

cv2.imshow("4-Camera Detection  (any key: exit)", grid)
cv2.waitKey(0)
cv2.destroyAllWindows()
