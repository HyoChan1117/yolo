"""YOLO 사람 감지 모델 변환: .pt → .engine (TensorRT)

실행 순서:
  1. 모델 학습 또는 human/models/yolov8n.pt 준비
  2. python human/export_yolo_onnx.py   # .onnx 생성 (선택)
  3. python human/export_yolo_trt.py    # .engine 생성

요구사항:
  - TensorRT 8.x 이상 (NVIDIA GPU 필수)
  - ultralytics (tensorrt 백엔드 포함)

실행: python human/export_yolo_trt.py [--model human/models/yolov8n.pt] [--fp16] [--imgsz 640] [--workspace 4096]
"""

import argparse
from pathlib import Path


MODEL_DIR      = Path("human/models")
DEFAULT_PT     = MODEL_DIR / "yolov8n.pt"
DEFAULT_ENGINE = MODEL_DIR / "yolov8n.engine"
IMG_SIZE       = 640


# ──────────────────────────────────────────────
# 1. pt → TensorRT 엔진 빌드
# ──────────────────────────────────────────────
def build_engine(
    pt_path: Path,
    engine_path: Path,
    imgsz: int,
    fp16: bool,
    workspace_mb: int,
) -> Path:
    """YOLOv8 .pt 모델을 TensorRT 엔진으로 변환하고 저장 경로를 반환합니다."""
    from ultralytics import YOLO

    model = YOLO(str(pt_path))
    print(f"[TRT] 엔진 빌드 시작: {pt_path}")
    print(f"  imgsz={imgsz}, fp16={fp16}, workspace={workspace_mb}MB")
    print("[TRT] 빌드 중... (수 분 소요될 수 있습니다)")

    out = model.export(
        format="engine",
        imgsz=imgsz,
        half=fp16,
        batch=1,
        dynamic=False,
        workspace=workspace_mb,
    )

    generated = Path(out)
    if generated.resolve() != engine_path.resolve():
        generated.rename(engine_path)
        generated = engine_path

    print(f"[TRT] 엔진 저장 완료: {generated}")
    return generated


# ──────────────────────────────────────────────
# 2. TensorRT 엔진 검증
# ──────────────────────────────────────────────
def verify_engine(engine_path: Path, imgsz: int) -> None:
    """변환된 .engine 파일로 추론 테스트합니다."""
    import numpy as np
    from ultralytics import YOLO

    print(f"\n[TRT 검증] {engine_path} 로드 중...")
    model = YOLO(str(engine_path))

    dummy = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
    result = model(dummy, verbose=False)[0]
    n = len(result.boxes) if result.boxes is not None else 0
    print(f"[TRT 검증] 추론 성공 ✓  (감지 수: {n})")


# ──────────────────────────────────────────────
# 3. 추론 헬퍼 (ultralytics TRT 래퍼)
# ──────────────────────────────────────────────
class TRTPersonDetector:
    """yolov8n.engine 으로 사람을 감지하는 래퍼.

    PersonCounter 와 동일한 인터페이스를 제공하므로 교체해서 사용할 수 있습니다.
    ultralytics 가 전처리·후처리(NMS 포함)를 모두 처리합니다.
    """

    def __init__(self, engine_path: Path, conf_thresh: float = 0.35):
        from ultralytics import YOLO
        self.model       = YOLO(str(engine_path))
        self.conf_thresh = conf_thresh
        print(f"[TRT] 엔진 로드 완료: {engine_path}")

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
    parser = argparse.ArgumentParser(description="YOLO 사람 감지 모델 TensorRT 변환")
    parser.add_argument("--model",       default=str(DEFAULT_PT),
                        help=f"변환할 .pt 파일 경로 (기본: {DEFAULT_PT})")
    parser.add_argument("--output",      default=str(DEFAULT_ENGINE),
                        help=f"출력 .engine 경로 (기본: {DEFAULT_ENGINE})")
    parser.add_argument("--imgsz",       type=int, default=IMG_SIZE,
                        help=f"추론 이미지 크기 (기본: {IMG_SIZE})")
    parser.add_argument("--fp16",        action="store_true",
                        help="FP16 정밀도로 엔진 빌드 (속도 향상, 약간의 정확도 감소)")
    parser.add_argument("--workspace",   type=int, default=4096,
                        help="빌더 workspace 크기 (MB, 기본값: 4096)")
    parser.add_argument("--skip-verify", action="store_true",
                        help="변환 후 검증 건너뜀")
    args = parser.parse_args()

    pt_path     = Path(args.model)
    engine_path = Path(args.output)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    if not pt_path.exists():
        raise FileNotFoundError(
            f"{pt_path} 가 없습니다. human/models/yolov8n.pt 를 준비하거나 --model 옵션을 확인하세요."
        )

    engine_path = build_engine(
        pt_path, engine_path,
        imgsz=args.imgsz,
        fp16=args.fp16,
        workspace_mb=args.workspace,
    )

    if not args.skip_verify:
        verify_engine(engine_path, args.imgsz)

    print("\n변환 완료!")
    print(f"  Engine : {engine_path}")
    print(f"  imgsz  : {args.imgsz}")
    print(f"  FP16   : {args.fp16}")
    print(f"\n.env 에서 PERSON_MODEL_PATH 를 아래로 변경하면 TRT 엔진을 사용합니다:")
    print(f"  PERSON_MODEL_PATH={engine_path}")
    print("\nTRTPersonDetector 사용 예시:")
    print("  from human.export_yolo_trt import TRTPersonDetector")
    print(f"  detector = TRTPersonDetector('{engine_path}')")
    print("  count, boxes, confs = detector.count(frame)")


if __name__ == "__main__":
    main()
