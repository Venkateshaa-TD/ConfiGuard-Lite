"""Torch-free verification of the deployable model bundle.

Equivalent to configuard.export.package.load_package + calibration loading,
minus the reference PyTorch checkpoint (a CPU server ships without it):
- export_manifest.json content hash; every listed file's size + SHA-256;
- each calibration artifact's content hash, its checkpoint / model-config
  binding to the manifest;
- the Phase 9 v1 quality-gate artifact: content hash, schema (v1 only: the
  9b/9c variants were rejected) and its binding to this manifest, the ONNX
  FP32 file, the adaptive calibration and the installed signals code;
- the pinned YuNet face detector.
Errors carry short codes and package-relative file names, never full paths.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from configuard.calibration.artifact import SCHEMA as CALIBRATION_SCHEMA
from configuard.calibration.artifact import Calibrator
from configuard.crops.store import canonical_json
from configuard.quality.gate import SCHEMA as GATE_V1_SCHEMA
from configuard.quality.gate import GateThresholds, QualityGateMismatchError, load_thresholds

EXPORT_SCHEMA = "p8-export-1"
MODEL_NAME = "student_distilled_p80"


class ArtifactMismatchError(Exception):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def signals_version() -> str:
    """Same definition as scripts/quality_gate.py: first 16 hex of SHA-256(signals.py)."""
    import configuard.quality.signals as s

    return hashlib.sha256(Path(s.__file__).read_bytes()).hexdigest()[:16]


def _read_json(path: Path, code: str) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ArtifactMismatchError(code) from exc


@dataclass(frozen=True)
class ModelBundle:
    manifest: dict[str, Any]
    onnx_path: Path
    onnx_sha256: str
    adaptive: Calibrator
    static: Calibrator
    gate: GateThresholds
    gate_sha256: str
    yunet_sha256: str

    @property
    def version(self) -> str:
        return (f"{MODEL_NAME}+onnx-fp32:{self.onnx_sha256[:12]}"
                f"/cal:{self.adaptive.artifact['content_sha256'][:8]}"
                f"/gate-v1:{self.gate_sha256[:8]}")

    def describe(self) -> dict[str, Any]:
        return {"name": MODEL_NAME, "version": self.version, "runtime": "onnx-fp32",
                "onnx_sha256": self.onnx_sha256, "export_manifest_sha256": self.manifest["content_sha256"],
                "adaptive_calibration_sha256": self.adaptive.artifact["content_sha256"],
                "frame_calibration_sha256": self.static.artifact["content_sha256"],
                "quality_gate": "phase9-v1", "quality_gate_sha256": self.gate_sha256}


def verify_bundle(package_dir: Path, gate_path: Path, yunet_path: Path) -> ModelBundle:
    package_dir = Path(package_dir)
    m = _read_json(package_dir / "export_manifest.json", "manifest_unreadable")
    if m.get("schema") != EXPORT_SCHEMA:
        raise ArtifactMismatchError("manifest_schema")
    body = {k: v for k, v in m.items() if k != "content_sha256"}
    if hashlib.sha256(canonical_json(body)).hexdigest() != m.get("content_sha256"):
        raise ArtifactMismatchError("manifest_content_hash")
    for name, rec in m["files"].items():
        p = package_dir / name
        if not p.is_file() or p.stat().st_size != rec["bytes"] or file_sha256(p) != rec["sha256"]:
            raise ArtifactMismatchError(f"file_hash:{name}")
    cals: dict[str, Calibrator] = {}
    for name, rec in m["calibration"].items():
        art = _read_json(package_dir / name, f"calibration_unreadable:{name}")
        cbody = {k: v for k, v in art.items() if k != "content_sha256"}
        if art.get("schema") != CALIBRATION_SCHEMA or hashlib.sha256(canonical_json(cbody)).hexdigest() != art.get("content_sha256"):
            raise ArtifactMismatchError(f"calibration_content_hash:{name}")
        if (art["content_sha256"] != rec["content_sha256"] or art["checkpoint_sha256"] != m["checkpoint_sha256"]
                or art["model_config_sha256"] != m["model_config_sha256"]):
            raise ArtifactMismatchError(f"calibration_binding:{name}")
        cals[name] = Calibrator(art)
    if "adaptive_calibration.json" not in cals or "calibration.json" not in cals:
        raise ArtifactMismatchError("calibration_missing")
    fp32 = m["variants"].get("fp32")
    if fp32 is None or fp32 not in m["files"]:
        raise ArtifactMismatchError("onnx_fp32_missing")
    onnx_sha = m["files"][fp32]["sha256"]
    adaptive = cals["adaptive_calibration.json"]
    try:
        gate = load_thresholds(gate_path, {
            "export_manifest_sha256": m["content_sha256"], "onnx_fp32_sha256": onnx_sha,
            "adaptive_calibration_content_sha256": adaptive.artifact["content_sha256"],
            "signals_version": signals_version()})
    except (OSError, ValueError) as exc:
        raise ArtifactMismatchError("gate_unreadable") from exc
    except QualityGateMismatchError as exc:
        raise ArtifactMismatchError("gate_binding") from exc
    if not isinstance(gate, GateThresholds) or gate.schema != GATE_V1_SCHEMA:
        raise ArtifactMismatchError("gate_not_production_v1")
    gate_sha = _read_json(Path(gate_path), "gate_unreadable")["content_sha256"]
    from configuard.media.face_detector import FaceDetectorError, verify_yunet_model

    try:
        yunet_sha = verify_yunet_model(yunet_path)
    except FaceDetectorError as exc:
        raise ArtifactMismatchError("face_detector_hash") from exc
    return ModelBundle(m, package_dir / fp32, onnx_sha, adaptive, cals["calibration.json"], gate, gate_sha, yunet_sha)
