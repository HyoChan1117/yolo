"""Door 분류 모델 변환: best.onnx → best.trt (TensorRT 엔진)

실행 순서:
  1. python door/train.py          # best.pth 생성
  2. python door/export_onnx.py    # best.onnx + best_classes.json 생성
  3. python door/export_trt.py     # best.trt 생성

요구사항:
  - TensorRT 8.x 이상 (NVIDIA GPU 필수)
  - tensorrt Python 패키지
  - pycuda

실행: python door/export_trt.py [--fp16] [--workspace 1024]
"""

import argparse
import json
from pathlib import Path

MODEL_DIR    = Path("door/models")
ONNX_PATH    = MODEL_DIR / "best.onnx"
TRT_PATH     = MODEL_DIR / "best.trt"
CLASSES_PATH = MODEL_DIR / "best_classes.json"
IMG_SIZE     = 224


# ──────────────────────────────────────────────
# 1. ONNX → TensorRT 엔진 빌드
# ──────────────────────────────────────────────
def build_engine(onnx_path: Path, trt_path: Path, fp16: bool, workspace_mb: int) -> None:
    """ONNX 모델을 TensorRT 엔진으로 빌드하여 저장합니다."""
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)

    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(f"[TRT] 파싱 오류: {parser.get_error(i)}")
            raise RuntimeError("ONNX 파싱 실패")
    print(f"[TRT] ONNX 파싱 완료: {onnx_path}")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_mb * 1024 * 1024)

    if fp16:
        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            print("[TRT] FP16 모드 활성화")
        else:
            print("[TRT] 경고: 이 GPU는 FP16 가속을 지원하지 않습니다. FP32로 빌드합니다.")

    # dynamic shape 프로파일 (배치 1 고정)
    profile = builder.create_optimization_profile()
    profile.set_shape("input",
                      min=(1, 3, IMG_SIZE, IMG_SIZE),
                      opt=(1, 3, IMG_SIZE, IMG_SIZE),
                      max=(1, 3, IMG_SIZE, IMG_SIZE))
    config.add_optimization_profile(profile)

    print("[TRT] 엔진 빌드 중... (수 분 소요될 수 있습니다)")
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TensorRT 엔진 빌드 실패")

    trt_path.write_bytes(bytes(serialized))
    print(f"[TRT] 엔진 저장 완료: {trt_path}")


# ──────────────────────────────────────────────
# 2. TensorRT 엔진 추론 헬퍼
# ──────────────────────────────────────────────
class TRTClassifier:
    """best.trt + best_classes.json 으로 추론하는 래퍼."""

    def __init__(self, trt_path: Path, classes_path: Path):
        import numpy as np
        import pycuda.autoinit  # noqa: F401  CUDA 컨텍스트 자동 초기화
        import tensorrt as trt

        self.np = np
        self.classes: list[str] = json.loads(classes_path.read_text(encoding="utf-8"))

        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        self.engine = runtime.deserialize_cuda_engine(trt_path.read_bytes())
        self.context = self.engine.create_execution_context()

        # 입출력 버퍼 할당
        self._allocate_buffers()
        print(f"[TRT] 엔진 로드 완료: {trt_path}")

    def _allocate_buffers(self):
        import pycuda.driver as cuda
        import tensorrt as trt

        self._host_inputs = []
        self._host_outputs = []
        self._cuda_inputs = []
        self._cuda_outputs = []
        self._bindings = []

        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = self.engine.get_tensor_shape(name)
            dtype = self.engine.get_tensor_dtype(name)
            np_dtype = self._trt_dtype_to_np(dtype)

            host_buf = cuda.pagelocked_empty(self._volume(shape), dtype=np_dtype)
            cuda_buf = cuda.mem_alloc(host_buf.nbytes)

            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self._host_inputs.append(host_buf)
                self._cuda_inputs.append(cuda_buf)
            else:
                self._host_outputs.append(host_buf)
                self._cuda_outputs.append(cuda_buf)
            self._bindings.append(int(cuda_buf))

    @staticmethod
    def _volume(shape) -> int:
        result = 1
        for s in shape:
            result *= s
        return result

    @staticmethod
    def _trt_dtype_to_np(dtype):
        import numpy as np
        import tensorrt as trt
        mapping = {
            trt.DataType.FLOAT: np.float32,
            trt.DataType.HALF:  np.float16,
            trt.DataType.INT32: np.int32,
            trt.DataType.INT8:  np.int8,
        }
        return mapping[dtype]

    def preprocess(self, bgr_crop):
        import cv2
        import numpy as np

        img = cv2.resize(cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2RGB), (IMG_SIZE, IMG_SIZE))
        img = img.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img = (img - mean) / std
        return img.transpose(2, 0, 1)[np.newaxis]  # (1, 3, H, W)

    def predict(self, bgr_crop) -> tuple[str, float]:
        """bgr_crop: OpenCV BGR 이미지 → (class_name, confidence)"""
        import numpy as np
        import pycuda.driver as cuda

        x = self.preprocess(bgr_crop).ravel()
        np.copyto(self._host_inputs[0], x)

        cuda.memcpy_htod(self._cuda_inputs[0], self._host_inputs[0])

        self.context.execute_v2(self._bindings)

        cuda.memcpy_dtoh(self._host_outputs[0], self._cuda_outputs[0])

        logits = self._host_outputs[0].astype(np.float32)
        exp = np.exp(logits - logits.max())
        probs = exp / exp.sum()
        idx = int(probs.argmax())
        return self.classes[idx], float(probs[idx])


# ──────────────────────────────────────────────
# main
# ──────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Door 모델 TensorRT 변환")
    parser.add_argument("--fp16", action="store_true",
                        help="FP16 정밀도로 엔진 빌드 (속도 향상, 약간의 정확도 감소)")
    parser.add_argument("--workspace", type=int, default=1024,
                        help="빌더 workspace 크기 (MB, 기본값: 1024)")
    args = parser.parse_args()

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    if not ONNX_PATH.exists():
        raise FileNotFoundError(
            f"{ONNX_PATH} 가 없습니다. door/export_onnx.py 를 먼저 실행하세요."
        )
    if not CLASSES_PATH.exists():
        raise FileNotFoundError(
            f"{CLASSES_PATH} 가 없습니다. door/export_onnx.py 를 먼저 실행하세요."
        )

    build_engine(ONNX_PATH, TRT_PATH, fp16=args.fp16, workspace_mb=args.workspace)

    classes: list[str] = json.loads(CLASSES_PATH.read_text(encoding="utf-8"))

    print("\n변환 완료!")
    print(f"  TRT    : {TRT_PATH}")
    print(f"  클래스 : {CLASSES_PATH}")
    print(f"  클래스 목록: {classes}")
    print(f"  FP16   : {args.fp16}")
    print("\nTRTClassifier 사용 예시:")
    print("  from door.export_trt import TRTClassifier")
    print(f"  clf = TRTClassifier('{TRT_PATH}', '{CLASSES_PATH}')")
    print("  label, conf = clf.predict(bgr_crop)")


if __name__ == "__main__":
    main()
