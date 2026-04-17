"""
Door 분류 모델 변환: best.pth -> best.onnx

Jetson Nano (JetPack 4.6.1 / TensorRT 8.2.1) 호환성을 고려한 보수적 버전
- opset_version=12
- dynamic_axes 사용 안 함
- 입력 shape 고정: 1x3x224x224

실행:
  python3 door/export_onnx.py
  python3 door/export_onnx.py --skip-verify

필요:
  pip install onnx onnxruntime
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models

MODEL_DIR = Path("door/models")
PTH_PATH = MODEL_DIR / "best.pth"
ONNX_PATH = MODEL_DIR / "best.onnx"
CLASSES_PATH = MODEL_DIR / "best_classes.json"

IMG_SIZE = 224
BATCH_SIZE = 1
ONNX_OPSET = 12


def build_model(num_classes: int) -> nn.Module:
    """
    학습 시 사용한 구조와 동일하게 모델을 생성합니다.
    현재는 MobileNetV3 Small 분류기 기준입니다.
    """
    model = models.mobilenet_v3_small(weights=None)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    return model


def export_onnx(pth_path: Path, onnx_path: Path) -> list[str]:
    """
    .pth 체크포인트를 ONNX로 변환하고 클래스 목록을 저장합니다.
    """
    if not pth_path.exists():
        raise FileNotFoundError(f"{pth_path} 가 없습니다. 먼저 학습을 완료하세요.")

    checkpoint = torch.load(pth_path, map_location="cpu")

    if "classes" not in checkpoint:
        raise KeyError("checkpoint에 'classes' 키가 없습니다.")

    if "state_dict" not in checkpoint:
        raise KeyError("checkpoint에 'state_dict' 키가 없습니다.")

    classes: list[str] = checkpoint["classes"]
    num_classes = len(classes)

    model = build_model(num_classes)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    dummy = torch.randn(BATCH_SIZE, 3, IMG_SIZE, IMG_SIZE, dtype=torch.float32)

    onnx_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        dummy,
        str(onnx_path),
        export_params=True,
        opset_version=ONNX_OPSET,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamo=False,
    )

    CLASSES_PATH.write_text(
        json.dumps(classes, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"[ONNX] 저장 완료: {onnx_path}")
    print(f"[CLASS] 저장 완료: {CLASSES_PATH}")
    print(f"[CLASS] 목록: {classes}")
    print(f"[ONNX] opset: {ONNX_OPSET}")
    print(f"[ONNX] input shape: {BATCH_SIZE}x3x{IMG_SIZE}x{IMG_SIZE}")

    return classes


def verify_onnx(onnx_path: Path) -> None:
    """
    ONNX Runtime으로 최소 동작 검증을 수행합니다.
    주의: 이 검증은 TensorRT 호환성을 보장하지 않습니다.
    """
    try:
        import numpy as np
        import onnx
        import onnxruntime as ort
    except ImportError:
        print("[VERIFY] onnx / onnxruntime 미설치 - 검증 건너뜀")
        return

    if not onnx_path.exists():
        raise FileNotFoundError(f"{onnx_path} 가 없습니다.")

    print("[VERIFY] ONNX 구조 검사 시작")
    model = onnx.load(str(onnx_path))
    onnx.checker.check_model(model)
    print("[VERIFY] ONNX checker 통과")

    print("[VERIFY] ONNX Runtime 추론 검사 시작")
    sess = ort.InferenceSession(
        str(onnx_path),
        providers=["CPUExecutionProvider"],
    )

    dummy = np.random.randn(BATCH_SIZE, 3, IMG_SIZE, IMG_SIZE).astype("float32")
    outputs = sess.run(None, {"input": dummy})

    if not outputs:
        raise RuntimeError("ONNX Runtime 출력이 비어 있습니다.")

    print(f"[VERIFY] 출력 shape: {outputs[0].shape}")
    print("[VERIFY] ONNX Runtime 검사 통과")


def main() -> None:
    parser = argparse.ArgumentParser(description="Door 모델 ONNX 변환")
    parser.add_argument(
        "--skip-verify",
        action="store_true",
        help="ONNX Runtime 검증을 건너뜁니다.",
    )
    args = parser.parse_args()

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    export_onnx(PTH_PATH, ONNX_PATH)

    if not args.skip_verify:
        verify_onnx(ONNX_PATH)

    print("\n변환 완료")
    print(f"  PTH     : {PTH_PATH}")
    print(f"  ONNX    : {ONNX_PATH}")
    print(f"  CLASSES : {CLASSES_PATH}")


if __name__ == "__main__":
    main()