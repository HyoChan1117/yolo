"""YOLO 사람 감지 모델 변환: .pt → .onnx

실행 순서:
  1. 모델 학습 또는 human/models/yolov8n.pt 준비
  2. python human/export_yolo_onnx.py   # .onnx 생성

요구사항:
  - ultralytics
  - onnx
  - onnxruntime-gpu (GPU 추론) 또는 onnxruntime (CPU 추론)

실행: python human/export_yolo_onnx.py [--model human/models/yolov8n.pt] [--imgsz 640] [--fp16] [--skip-verify]
"""

import argparse
from pathlib import Path


MODEL_DIR    = Path("human/models")
DEFAULT_PT   = MODEL_DIR / "yolov8n.pt"
DEFAULT_ONNX = MODEL_DIR / "yolov8n.onnx"
IMG_SIZE     = 640


# ──────────────────────────────────────────────
# 1. pt → onnx
# ──────────────────────────────────────────────
def export_onnx(pt_path: Path, onnx_path: Path, imgsz: int, fp16: bool) -> Path:
    """YOLOv8 .pt 모델을 ONNX 로 변환하고 저장 경로를 반환합니다."""
    from ultralytics import YOLO

    model = YOLO(str(pt_path))
    print(f"[ONNX] 변환 시작: {pt_path}  (imgsz={imgsz}, fp16={fp16})")

    out = model.export(
        format="onnx",
        imgsz=imgsz,
        half=fp16,
        batch=1,
        dynamic=False,
        opset=17,
    )

    generated = Path(out)
    if generated.resolve() != onnx_path.resolve():
        generated.rename(onnx_path)
        generated = onnx_path

    print(f"[ONNX] 저장 완료: {generated}")
    return generated


# ──────────────────────────────────────────────
# 2. onnxruntime 으로 ONNX 검증
# ──────────────────────────────────────────────
def verify_onnx(onnx_path: Path, imgsz: int) -> None:
    try:
        import numpy as np
        import onnxruntime as ort

        sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        inp_name = sess.get_inputs()[0].name
        dummy = np.zeros((1, 3, imgsz, imgsz), dtype=np.float32)
        out = sess.run(None, {inp_name: dummy})
        print(f"[ONNX 검증] 입력: {inp_name} {dummy.shape}  출력 shape: {out[0].shape}  ✓")
    except ImportError:
        print("[ONNX 검증] onnxruntime 미설치 — 건너뜁니다.")


# ──────────────────────────────────────────────
# 3. 추론 헬퍼 (ultralytics 래퍼)
# ──────────────────────────────────────────────
class ONNXPersonDetector:
    """yolov8n.onnx 로 사람을 감지하는 래퍼.

    PersonCounter 와 동일한 인터페이스를 제공하므로 교체해서 사용할 수 있습니다.
    ultralytics 가 전처리·후처리(NMS 포함)를 모두 처리합니다.
    """

    def __init__(self, onnx_path: Path, conf_thresh: float = 0.35):
        from ultralytics import YOLO
        self.model       = YOLO(str(onnx_path))
        self.conf_thresh = conf_thresh
        print(f"[ONNX] 모델 로드 완료: {onnx_path}")

    def count(
        self, frame
    ) -> tuple[int, list[tuple[int, int, int, int]], list[float]]:
        """Returns (count, boxes, confidences) for all detected persons."""
        result = self.model(frame, verbose=False)[0]
        boxes: list[tuple[int, int, int, int]] = []
        confs: list[float] = []

        if result.boxes is None:
            return 0, boxes, confs

        for box in result.boxes:
            if int(box.cls.item()) != 0:
                continue
            conf = float(box.conf.item())
            if conf < self.conf_thresh:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            boxes.append((x1, y1, x2, y2))
            confs.append(conf)

        return len(boxes), boxes, confs


# ──────────────────────────────────────────────
# main
# ──────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="YOLO 사람 감지 모델 ONNX 변환")
    parser.add_argument("--model",     default=str(DEFAULT_PT),
                        help=f"변환할 .pt 파일 경로 (기본: {DEFAULT_PT})")
    parser.add_argument("--output",    default=str(DEFAULT_ONNX),
                        help=f"출력 .onnx 경로 (기본: {DEFAULT_ONNX})")
    parser.add_argument("--imgsz",     type=int, default=IMG_SIZE,
                        help=f"추론 이미지 크기 (기본: {IMG_SIZE})")
    parser.add_argument("--fp16",      action="store_true",
                        help="FP16 으로 변환 (onnxruntime-gpu 필요)")
    parser.add_argument("--skip-onnx", action="store_true",
                        help="이미 .onnx 가 있으면 변환 단계 건너뜀")
    parser.add_argument("--skip-verify", action="store_true",
                        help="변환 후 검증 건너뜀")
    args = parser.parse_args()

    pt_path   = Path(args.model)
    onnx_path = Path(args.output)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: pt → onnx
    if args.skip_onnx and onnx_path.exists():
        print(f"[ONNX] 기존 파일 사용: {onnx_path}")
    else:
        if not pt_path.exists():
            raise FileNotFoundError(
                f"{pt_path} 가 없습니다. human/models/yolov8n.pt 를 준비하거나 --model 옵션을 확인하세요."
            )
        onnx_path = export_onnx(pt_path, onnx_path, args.imgsz, args.fp16)

    # Step 2: 검증
    if not args.skip_verify:
        verify_onnx(onnx_path, args.imgsz)

    print("\n변환 완료!")
    print(f"  ONNX   : {onnx_path}")
    print(f"  imgsz  : {args.imgsz}")
    print(f"  FP16   : {args.fp16}")
    print(f"\n.env 에서 PERSON_MODEL_PATH 를 아래로 변경하면 ONNX 모델을 사용합니다:")
    print(f"  PERSON_MODEL_PATH={onnx_path}")
    print("\nONNXPersonDetector 사용 예시:")
    print("  from human.export_yolo_onnx import ONNXPersonDetector")
    print(f"  detector = ONNXPersonDetector('{onnx_path}')")
    print("  count, boxes, confs = detector.count(frame)")


if __name__ == "__main__":
    main()
