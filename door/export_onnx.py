"""Door 분류 모델 변환: best.pth → best.onnx

실행 순서:
  1. python door/train.py         # best.pth 생성
  2. python door/export_onnx.py   # best.onnx + best_classes.json 생성

요구사항:
  - onnx
  - onnxruntime-gpu (GPU 추론) 또는 onnxruntime (CPU 추론)

실행: python door/export_onnx.py [--skip-onnx]
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models

MODEL_DIR    = Path("door/models")
PTH_PATH     = MODEL_DIR / "best.pth"
ONNX_PATH    = MODEL_DIR / "best.onnx"
CLASSES_PATH = MODEL_DIR / "best_classes.json"
IMG_SIZE     = 224


# ──────────────────────────────────────────────
# 1. pth → onnx + classes.json
# ──────────────────────────────────────────────
def export_onnx(pth_path: Path, onnx_path: Path) -> list[str]:
    """학습된 .pth 를 ONNX 로 변환하고 클래스 목록을 반환합니다."""
    checkpoint = torch.load(pth_path, map_location="cpu")
    classes: list[str] = checkpoint["classes"]
    num_classes = len(classes)

    model = models.mobilenet_v3_small(weights=None)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    dummy = torch.zeros(1, 3, IMG_SIZE, IMG_SIZE)

    torch.onnx.export(
        model,
        dummy,
        str(onnx_path),
        opset_version=17,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        dynamo=False,
    )
    print(f"[ONNX] 저장 완료: {onnx_path}  (클래스: {classes})")

    CLASSES_PATH.write_text(json.dumps(classes, ensure_ascii=False), encoding="utf-8")
    print(f"[ONNX] 클래스 저장 완료: {CLASSES_PATH}")

    return classes


# ──────────────────────────────────────────────
# 2. onnxruntime으로 ONNX 검증
# ──────────────────────────────────────────────
def verify_onnx(onnx_path: Path) -> None:
    try:
        import numpy as np
        import onnxruntime as ort

        sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        dummy = np.random.randn(1, 3, IMG_SIZE, IMG_SIZE).astype(np.float32)
        out = sess.run(None, {"input": dummy})
        print(f"[ONNX 검증] 출력 shape: {out[0].shape}  ✓")
    except ImportError:
        print("[ONNX 검증] onnxruntime 미설치 — 건너뜁니다.")


# ──────────────────────────────────────────────
# 3. ONNX 추론 헬퍼
# ──────────────────────────────────────────────
class ONNXClassifier:
    """best.onnx + best_classes.json 으로 추론하는 래퍼."""

    def __init__(self, onnx_path: Path, classes_path: Path):
        import numpy as np
        import onnxruntime as ort

        self.np = np
        self.classes: list[str] = json.loads(classes_path.read_text(encoding="utf-8"))

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.sess = ort.InferenceSession(str(onnx_path), providers=providers)
        self.input_name = self.sess.get_inputs()[0].name

        active = self.sess.get_providers()[0]
        print(f"[ONNX] 추론 장치: {active}")

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
        np = self.np
        x = self.preprocess(bgr_crop)
        logits = self.sess.run(None, {self.input_name: x})[0][0]
        exp = np.exp(logits - logits.max())
        probs = exp / exp.sum()
        idx = int(probs.argmax())
        return self.classes[idx], float(probs[idx])


# ──────────────────────────────────────────────
# main
# ──────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Door 모델 ONNX 변환")
    parser.add_argument("--skip-onnx", action="store_true",
                        help=f"{ONNX_PATH} 가 이미 있으면 pth→onnx 단계를 건너뜁니다")
    args = parser.parse_args()

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: pth → onnx
    if args.skip_onnx and ONNX_PATH.exists():
        print(f"[ONNX] 기존 파일 사용: {ONNX_PATH}")
        classes: list[str] = json.loads(CLASSES_PATH.read_text(encoding="utf-8"))
    else:
        if not PTH_PATH.exists():
            raise FileNotFoundError(f"{PTH_PATH} 가 없습니다. door/train.py 를 먼저 실행하세요.")
        classes = export_onnx(PTH_PATH, ONNX_PATH)

    # Step 2: ONNX 검증
    verify_onnx(ONNX_PATH)

    print("\n변환 완료!")
    print(f"  ONNX   : {ONNX_PATH}")
    print(f"  클래스 : {CLASSES_PATH}")
    print(f"  클래스 목록: {classes}")
    print("\nONNXClassifier 사용 예시:")
    print("  from door.export_onnx import ONNXClassifier")
    print(f"  clf = ONNXClassifier('{ONNX_PATH}', '{CLASSES_PATH}')")
    print("  label, conf = clf.predict(bgr_crop)")


if __name__ == "__main__":
    main()
