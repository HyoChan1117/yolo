"""YOLO-based person counting module.

백엔드 우선순위 (자동 선택):
  TensorRT (.engine) → ONNX Runtime (.onnx) → PyTorch (.pt)
  환경변수 BACKEND=trt|onnx|pt 로 강제 지정 가능
"""

from __future__ import annotations

import os
from pathlib import Path

import torch
from dotenv import load_dotenv
from ultralytics import YOLO

load_dotenv()

MODEL_DIR          = Path("human/models")
TRT_PATH           = MODEL_DIR / "yolov8n.engine"
ONNX_PATH          = MODEL_DIR / "yolov8n.onnx"
PT_PATH            = MODEL_DIR / "yolov8n.pt"
PERSON_CONF_THRESH = float(os.getenv("PERSON_CONF_THRESH", "0.35"))
BACKEND            = os.getenv("BACKEND", "auto").lower()  # auto | trt | onnx | pt

# ── 모델 로드 (TRT → ONNX → PT 자동 선택) ─────────────────────
_model: YOLO | None = None
_device: str = "cpu"


def _try_trt() -> bool:
    if not TRT_PATH.exists():
        return False
    try:
        global _model, _device
        _model = YOLO(str(TRT_PATH))
        _device = "0"
        print(f"[백엔드] TensorRT  ({TRT_PATH})")
        return True
    except Exception as e:
        print(f"[백엔드] TRT 로드 실패 ({e}) → ONNX 시도")
        return False


def _onnx_cuda_available() -> bool:
    """ONNX Runtime CUDA에 필요한 DLL이 실제로 로드 가능한지 확인."""
    import ctypes
    for dll in ("cublasLt64_12.dll", "cublas64_12.dll", "cudart64_12.dll"):
        try:
            ctypes.CDLL(dll)
            return True
        except OSError:
            continue
    return False


def _try_onnx() -> bool:
    if not ONNX_PATH.exists():
        return False
    try:
        global _model, _device
        _model = YOLO(str(ONNX_PATH))
        _device = "0" if _onnx_cuda_available() else "cpu"
        print(f"[백엔드] ONNX Runtime  ({ONNX_PATH})  device={_device}")
        return True
    except Exception as e:
        print(f"[백엔드] ONNX 로드 실패 ({e}) → PT 시도")
        return False


def _load_pt() -> None:
    global _model, _device
    if not PT_PATH.exists():
        print(f"[모델] {PT_PATH} 없음 → 자동 다운로드 중...")
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        _model = YOLO("yolov8n.pt")  # ultralytics 캐시에서 다운로드
        _model.save(str(PT_PATH))
        print(f"[모델] 저장 완료: {PT_PATH}")
    else:
        _model = YOLO(str(PT_PATH))
    _device = "0" if torch.cuda.is_available() else "cpu"
    print(f"[백엔드] PyTorch  ({PT_PATH})  device={_device}")


if BACKEND == "trt":
    if not _try_trt():
        print("[백엔드] TRT 강제 지정했지만 로드 실패.")
        exit()
elif BACKEND == "onnx":
    if not _try_onnx():
        print("[백엔드] ONNX 강제 지정했지만 로드 실패.")
        exit()
elif BACKEND == "pt":
    _load_pt()
else:  # auto
    if not _try_trt():
        if not _try_onnx():
            _load_pt()


class PersonCounter:
    def __init__(self, conf_thresh: float = PERSON_CONF_THRESH):
        self.conf_thresh = conf_thresh

    def count(
        self, frame
    ) -> tuple[int, list[tuple[int, int, int, int]], list[float]]:
        """Returns (count, boxes, confidences) for all detected persons."""
        result = _model(frame, verbose=False, device=_device)[0]
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
