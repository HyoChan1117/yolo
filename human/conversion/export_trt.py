"""YOLO 사람 감지 모델 변환: .onnx → .engine (TensorRT)

실행 순서:
  1. python human/conversion/export_onnx.py   # yolo11n.onnx 생성
  2. python human/conversion/export_trt.py    # yolo11n.engine 생성

요구사항:
  - TensorRT 8.x 이상 (NVIDIA GPU 필수)
  - tensorrt Python 패키지

실행: python human/conversion/export_trt.py [--model human/models/yolo11n.onnx] [--fp16] [--workspace 4096]
"""

import argparse
from pathlib import Path


MODEL_DIR      = Path("human/models")
DEFAULT_ONNX   = MODEL_DIR / "yolo11x.onnx"
DEFAULT_ENGINE = MODEL_DIR / "yolo11x.engine"
IMGSZ = 640

# ──────────────────────────────────────────────
# 1. onnx → TensorRT 엔진 빌드
# ──────────────────────────────────────────────
def build_engine(
    onnx_path: Path,
    engine_path: Path,
    fp16: bool,
    workspace_mb: int,
    imgsz: int = IMGSZ,
) -> Path:
    """YOLO .onnx 모델을 TensorRT 엔진으로 변환하고 저장 경로를 반환합니다."""
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)

    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(f"[TRT] ONNX 파싱 오류: {parser.get_error(i)}")
            raise RuntimeError("ONNX 파싱 실패")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_mb * 1024 * 1024)

    if fp16:
        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            print("[TRT] FP16 활성화")
        else:
            print("[TRT] 경고: 이 GPU는 FP16을 지원하지 않습니다. FP32로 빌드합니다.")

    # ONNX가 dynamic=True로 내보내진 경우 최적화 프로파일 필요
    inp = network.get_input(0)
    if -1 in inp.shape:
        profile = builder.create_optimization_profile()
        profile.set_shape(inp.name,
                          (1, 3, imgsz, imgsz),
                          (1, 3, imgsz, imgsz),
                          (1, 3, imgsz, imgsz))
        config.add_optimization_profile(profile)
        print(f"[TRT] 동적 입력 감지 — 최적화 프로파일 설정: imgsz={imgsz}")

    print(f"[TRT] 엔진 빌드 시작: {onnx_path}")
    print(f"  fp16={fp16}, workspace={workspace_mb}MB, imgsz={imgsz}")
    print("[TRT] 빌드 중... (수 분 소요될 수 있습니다)")

    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TensorRT 엔진 빌드 실패")

    with open(engine_path, "wb") as f:
        f.write(serialized)

    print(f"[TRT] 엔진 저장 완료: {engine_path}")
    return engine_path


# ──────────────────────────────────────────────
# 2. TensorRT 엔진 검증
# ──────────────────────────────────────────────
def verify_engine(engine_path: Path, imgsz: int) -> None:
    """변환된 .engine 파일로 추론 테스트합니다."""
    import numpy as np
    from ultralytics import YOLO

    print(f"\n[TRT 검증] {engine_path} 로드 중...")
    model = YOLO(str(engine_path), task="detect")
    model.overrides["imgsz"] = imgsz  # 엔진 빌드 크기와 일치시킴

    dummy = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
    result = model(dummy, imgsz=imgsz, verbose=False)[0]
    n = len(result.boxes) if result.boxes is not None else 0
    print(f"[TRT 검증] 추론 성공 ✓  (감지 수: {n})")


# ──────────────────────────────────────────────
# 3. 추론 헬퍼 (ultralytics TRT 래퍼)
# ──────────────────────────────────────────────
class TRTPersonDetector:
    """yolo11n.engine 으로 사람을 감지하는 래퍼."""

    def __init__(self, engine_path: Path, conf_thresh: float = 0.35):
        from ultralytics import YOLO
        self.model       = YOLO(str(engine_path), task="detect")
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
    parser = argparse.ArgumentParser(description="YOLO ONNX → TensorRT 엔진 변환")
    parser.add_argument("--model",       default=str(DEFAULT_ONNX),
                        help=f"변환할 .onnx 파일 경로 (기본: {DEFAULT_ONNX})")
    parser.add_argument("--output",      default=str(DEFAULT_ENGINE),
                        help=f"출력 .engine 경로 (기본: {DEFAULT_ENGINE})")
    parser.add_argument("--fp16",        action="store_true",
                        help="FP16 정밀도로 엔진 빌드 (속도 향상, 약간의 정확도 감소)")
    parser.add_argument("--workspace",   type=int, default=4096,
                        help="빌더 workspace 크기 (MB, 기본값: 4096)")
    parser.add_argument("--imgsz",       type=int, default=640,
                        help="검증용 이미지 크기 (기본: 640)")
    parser.add_argument("--skip-verify", action="store_true",
                        help="변환 후 검증 건너뜀")
    args = parser.parse_args()

    onnx_path   = Path(args.model)
    engine_path = Path(args.output)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    if not onnx_path.exists():
        raise FileNotFoundError(
            f"{onnx_path} 가 없습니다. 먼저 export_onnx.py 를 실행해 yolo11n.onnx 를 생성하세요."
        )

    build_engine(onnx_path, engine_path, fp16=args.fp16, workspace_mb=args.workspace, imgsz=args.imgsz)

    if not args.skip_verify:
        verify_engine(engine_path, args.imgsz)

    print("\n변환 완료!")
    print(f"  Engine : {engine_path}")
    print(f"  FP16   : {args.fp16}")


if __name__ == "__main__":
    main()
