"""Phase 10 service fixtures: a synthetic, correctly hash-bound model bundle.

The ONNX model is tiny and deterministic: logit = (mean pixel - 128) / 8,
so bright media -> "likely manipulated", dark -> "likely real". The bundle
has the real artifact schemas, and every hash and binding is computed the way
the export/calibration/gate code does it, so verification code paths are
exercised for real. Faces come from a MockFaceDetector (no real faces are
needed for API tests)."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

from configuard.calibration.artifact import SCHEMA as CAL_SCHEMA
from configuard.config import ValidationLimits
from configuard.crops.store import canonical_json
from configuard.media.face_detector import MockFaceDetector, default_yunet_model_path
from configuard.media.types import BoundingBox, FaceLandmarks
from configuard.quality.gate import GateThresholds, save_thresholds
from configuard.service.artifacts import file_sha256, signals_version
from configuard.service.config import ServiceConfig, hash_api_key

CKPT = "c" * 64
MCFG = "d" * 64


def _onnx_model(path: Path) -> None:
    """Same head structure as the exported student (ReLU features -> GAP -> 1x1 conv -> ReLU -> Gemm),
    so the Grad-CAM explainer attaches to it. Two complementary ReLU channels give
    logit = (mean pixel - 128) / 8 exactly, on a 7x7 feature grid (32x32-pixel cells)."""
    x = helper.make_tensor_value_info("pixels", TensorProto.FLOAT, ["N", 3, 224, 224])
    y = helper.make_tensor_value_info("logit", TensorProto.FLOAT, ["N"])
    k = np.full((2, 3, 32, 32), 1.0 / (3 * 32 * 32), np.float32)
    k[1] *= -1
    inits = [numpy_helper.from_array(k, "k"), numpy_helper.from_array(np.array([-128.0, 128.0], np.float32), "kb"),
             numpy_helper.from_array(np.eye(2, dtype=np.float32).reshape(2, 2, 1, 1), "w1"),
             numpy_helper.from_array(np.zeros(2, np.float32), "b1"),
             numpy_helper.from_array(np.array([[0.125, -0.125]], np.float32), "w2"),
             numpy_helper.from_array(np.zeros(1, np.float32), "b2"),
             numpy_helper.from_array(np.array([1], np.int64), "sq")]
    nodes = [
        helper.make_node("Conv", ["pixels", "k", "kb"], ["c"], kernel_shape=[32, 32], strides=[32, 32]),
        helper.make_node("Relu", ["c"], ["feat"]),
        helper.make_node("GlobalAveragePool", ["feat"], ["gap"]),
        helper.make_node("Conv", ["gap", "w1", "b1"], ["h"], kernel_shape=[1, 1]),
        helper.make_node("Relu", ["h"], ["hr"]),
        helper.make_node("Flatten", ["hr"], ["f"], axis=1),
        helper.make_node("Gemm", ["f", "w2", "b2"], ["g"], transB=1),
        helper.make_node("Squeeze", ["g", "sq"], ["logit"]),
    ]
    g = helper.make_graph(nodes, "gap_head", [x], [y], inits)
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 18)])
    m.ir_version = 9
    onnx.save(m, str(path))


def _cal(levels: dict) -> dict:
    body = {"schema": CAL_SCHEMA, "checkpoint_sha256": CKPT, "model_config": {}, "model_config_sha256": MCFG,
            "model_provenance": {}, "calibration_provenance": {"synthetic": True}, "default_alpha": 0.05, "levels": levels}
    return body | {"content_sha256": hashlib.sha256(canonical_json(body)).hexdigest()}


def _conf(alphas, q=0.2):
    return [{"alpha": a, "mode": "mondrian", "q_real": q, "q_fake": q, "n_real": 71, "n_fake": 284} for a in alphas]


def build_bundle(root: Path, gate: GateThresholds | None = None) -> tuple[Path, Path]:
    pkg = root / "pkg"
    pkg.mkdir(parents=True, exist_ok=True)
    _onnx_model(pkg / "student_fp32.onnx")
    static = _cal({"frame": {"temperature": 1.0, "conformal": _conf([0.05])},
                   "video": {"temperature": 1.0, "conformal": _conf([0.05])}})
    adaptive = _cal({f"video_k{k}": {"temperature": 1.0, "conformal": _conf([0.015, 0.02, 0.05])} for k in (4, 8, 16)})
    for name, art in (("calibration.json", static), ("adaptive_calibration.json", adaptive)):
        (pkg / name).write_bytes(canonical_json(art))
    files = {n: {"bytes": (pkg / n).stat().st_size, "sha256": file_sha256(pkg / n)}
             for n in ("student_fp32.onnx", "calibration.json", "adaptive_calibration.json")}
    body = {"schema": "p8-export-1", "checkpoint_sha256": CKPT, "model_config_sha256": MCFG, "files": files,
            "variants": {"fp32": "student_fp32.onnx"},
            "calibration": {n: {"checkpoint_sha256": CKPT, "content_sha256": a["content_sha256"], "model_config_sha256": MCFG}
                            for n, a in (("calibration.json", static), ("adaptive_calibration.json", adaptive))}}
    manifest = body | {"content_sha256": hashlib.sha256(canonical_json(body)).hexdigest()}
    (pkg / "export_manifest.json").write_bytes(canonical_json(manifest))
    gate_path = root / "quality_gate.json"
    save_thresholds(gate_path, gate or LENIENT, {
        "export_manifest_sha256": manifest["content_sha256"], "onnx_fp32_sha256": files["student_fp32.onnx"]["sha256"],
        "adaptive_calibration_content_sha256": adaptive["content_sha256"], "signals_version": signals_version()})
    return pkg, gate_path


LENIENT = GateThresholds(sharpness_min=-99.0, hf_ratio_min=-99.0, blockiness_max=99.0, face_px_min=0.0)
STRICT = GateThresholds(sharpness_min=99.0, hf_ratio_min=99.0, blockiness_max=-99.0, face_px_min=0.0)


def face_detector(delay: float = 0.0, box=(40.0, 40.0, 120.0, 120.0)):
    b = BoundingBox(*box)
    lm = FaceLandmarks(right_eye=(b.x + 36, b.y + 45), left_eye=(b.x + 84, b.y + 45), nose_tip=(b.x + 60, b.y + 66),
                       right_mouth_corner=(b.x + 40, b.y + 90), left_mouth_corner=(b.x + 80, b.y + 90))

    class Slow(MockFaceDetector):
        def detect(self, image):
            if delay:
                time.sleep(delay)
            return super().detect(image)

    return lambda: Slow(fixed_detections=[(b, lm, 0.99)])


def make_config(root: Path, pkg: Path, gate: Path, **kw) -> ServiceConfig:
    limits = ValidationLimits(max_image_size_mb=1, max_video_size_mb=3, max_video_duration_seconds=5)
    base = dict(environment="testing", validation=limits, package_dir=pkg, gate_path=gate,
                yunet_path=default_yunet_model_path(), device="cpu", max_concurrent_inference=1, max_queue=1,
                request_timeout_s=20.0, upload_timeout_s=10.0, cpu_threads_per_session=1, ready_recheck_s=0.0,
                temp_dir=root / "tmp", log_level="INFO", allow_explanations=False, c2pa_enabled=False)
    base.update(kw)
    return ServiceConfig(**base)


def image_bytes(value: int, ext: str = ".png", size=(240, 240), noise: bool = True) -> bytes:
    rng = np.random.default_rng(value)
    img = np.full((*size, 3), value, np.uint8)
    if noise:
        img = np.clip(img.astype(int) + rng.integers(-3, 4, img.shape), 0, 255).astype(np.uint8)
    return cv2.imencode(ext, img)[1].tobytes()


def make_video(path: Path, color: str = "white", seconds: float = 2.0, size: str = "240x240", fps: int = 12) -> Path:
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", f"color=c={color}:s={size}:r={fps}:d={seconds}",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path)], check=True, timeout=60)
    return path


@pytest.fixture(scope="session")
def bundle_root(tmp_path_factory) -> Path:
    return tmp_path_factory.mktemp("bundle")


@pytest.fixture(scope="session")
def bundle(bundle_root) -> tuple[Path, Path]:
    return build_bundle(bundle_root)


@pytest.fixture(scope="session")
def videos(tmp_path_factory) -> dict[str, Path]:
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not available")
    d = tmp_path_factory.mktemp("videos")
    return {"bright": make_video(d / "bright.mp4", "white"), "dark": make_video(d / "dark.mp4", "0x101010"),
            "long": make_video(d / "long.mp4", "white", seconds=7.0), "short": make_video(d / "short.mp4", "white", 0.5)}


API_KEY = "test-key-0123456789"
API_KEY_SHA = hash_api_key(API_KEY)


def leftover(cfg: ServiceConfig) -> list[str]:
    root = Path(cfg.temp_dir)
    return [p.name for p in root.iterdir()] if root.exists() else []


def json_dumps(o) -> str:
    return json.dumps(o)
