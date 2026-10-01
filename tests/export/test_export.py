"""Phase 8: ONNX student graph contract, INT8 calibration guard, package hash checks."""

from __future__ import annotations

import hashlib
import json

import cv2
import numpy as np
import pytest
import torch

from configuard.export.onnx_student import CropReader, export, load_pixels, quantize_int8, run, session
from configuard.export.package import ExportMismatchError, build_package, load_package, variant_path
from configuard.models.registry import create_encoder
from configuard.teacher.cache import ProtectedSplitError

PRE = {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}


@pytest.fixture(scope="module")
def tiny_export(tmp_path_factory):
    torch.manual_seed(0)
    model = create_encoder("mobilenetv4_conv_small", pretrained=False).eval()
    d = tmp_path_factory.mktemp("exp")
    return model, export(model, PRE, d / "m32.onnx"), d


def _crops(d, n, split="train"):
    rows = []
    for i in range(n):
        img = np.random.default_rng(i).integers(0, 255, (224, 224, 3), dtype=np.uint8)
        (d / f"c{i}.png").write_bytes(cv2.imencode(".png", img)[1].tobytes())
        rows.append({"crop_path": f"c{i}.png", "metadata": {"split": split}})
    return rows


def test_fp32_graph_matches_pytorch_with_normalisation_inside(tiny_export, tmp_path):
    model, path, _ = tiny_export
    rows = _crops(tmp_path, 3)
    px = load_pixels([tmp_path / r["crop_path"] for r in rows])
    assert px.dtype == np.float32 and px.shape == (3, 3, 224, 224) and px.max() <= 255
    mean = torch.tensor(PRE["mean"]).view(1, 3, 1, 1) * 255
    std = torch.tensor(PRE["std"]).view(1, 3, 1, 1) * 255
    with torch.inference_mode():
        ref = model.forward_logits((torch.from_numpy(px) - mean) / std).numpy()
    out = run(session(path, "cpu"), px)
    assert out.shape == (3,) and np.abs(out - ref).max() < 1e-3
    assert run(session(path, "cpu"), px[:1]).shape == (1,)  # dynamic batch


def test_int8_calibration_refuses_non_train_rows_and_quantises(tiny_export, tmp_path):
    _, path, d = tiny_export
    with pytest.raises(ProtectedSplitError):
        CropReader(_crops(tmp_path, 2, split="val"), tmp_path)
    with pytest.raises(ProtectedSplitError):
        CropReader(_crops(tmp_path, 2, split="test"), tmp_path)
    reader = CropReader(_crops(tmp_path, 8), tmp_path, batch=4)  # 8 rows = 2 equal batches
    assert reader.get_next()["pixels"].shape == (4, 3, 224, 224)
    reader.rewind()
    q = quantize_int8(path, d / "m8.onnx", reader)
    out = run(session(q, "cpu"), load_pixels([tmp_path / "c0.png"]))
    assert np.isfinite(out).all() and q.stat().st_size < path.stat().st_size / 2


def _run_dir(tmp_path, ck_bytes=b"checkpoint"):
    run = tmp_path / "run"
    run.mkdir()
    (run / "best.pt").write_bytes(ck_bytes)
    sha = hashlib.sha256(ck_bytes).hexdigest()
    for name in ("calibration.json", "adaptive_calibration.json"):
        (run / name).write_text(json.dumps({"checkpoint_sha256": sha, "content_sha256": name, "model_config_sha256": "cfg"}))
    return run, sha


def test_package_roundtrip_and_mismatch_refusals(tmp_path):
    run, sha = _run_dir(tmp_path)
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "a.onnx").write_bytes(b"onnx-a")
    m = build_package(pkg, run, {"fp32": "a.onnx"}, {"selection": {"cpu_default": "fp32"}})
    assert load_package(pkg, run / "best.pt")["checkpoint_sha256"] == sha
    assert variant_path(pkg, m, "fp32").name == "a.onnx"
    with pytest.raises(KeyError):
        variant_path(pkg, m, "int8")
    other = tmp_path / "other.pt"
    other.write_bytes(b"different checkpoint")
    with pytest.raises(ExportMismatchError, match="reference checkpoint"):
        load_package(pkg, other)
    (pkg / "a.onnx").write_bytes(b"tampered")
    with pytest.raises(ExportMismatchError, match="a.onnx"):
        load_package(pkg)
    (pkg / "a.onnx").write_bytes(b"onnx-a")
    man = json.loads((pkg / "export_manifest.json").read_text())
    man["selection"]["cpu_default"] = "int8"
    (pkg / "export_manifest.json").write_text(json.dumps(man))
    with pytest.raises(ExportMismatchError, match="modified"):
        load_package(pkg)


def test_package_refuses_calibration_from_another_checkpoint(tmp_path):
    run, _ = _run_dir(tmp_path)
    (run / "calibration.json").write_text(json.dumps({"checkpoint_sha256": "f" * 64, "content_sha256": "x",
                                                      "model_config_sha256": "cfg"}))
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "a.onnx").write_bytes(b"onnx-a")
    with pytest.raises(ExportMismatchError, match="fitted for checkpoint"):
        build_package(pkg, run, {"fp32": "a.onnx"}, {})


def test_shape_pinned_runner_keeps_one_session_per_batch(tiny_export):
    from configuard.export.onnx_student import ShapePinnedRunner

    _, path, _ = tiny_export
    px = (np.random.default_rng(1).random((8, 3, 224, 224)) * 255).astype(np.float32)
    ref = run(session(path, "cpu"), px)
    pinned = ShapePinnedRunner(path, "cpu", pin=True)
    out = np.concatenate([pinned(px[:4]), pinned(px[4:])])
    assert pinned.sessions == 1 and np.abs(out - ref).max() < 1e-4  # same shape twice -> one session
    pinned(px)  # batch 8
    assert pinned.sessions == 2
    single = ShapePinnedRunner(path, "cpu")  # CPU default: one shared session
    single(px[:4]), single(px)
    assert single.sessions == 1
