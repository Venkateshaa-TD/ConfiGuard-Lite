"""Calibration artifact: per-level temperature + conformal thresholds, bound to
one exact student checkpoint.

The JSON records the checkpoint's SHA-256, the training config (and its
SHA-256), the crop/teacher/partition provenance, and a SHA-256 over its
own content. `load_calibration` refuses (CalibrationMismatchError) if:
- the content hash does not match (edited file);
- the checkpoint file's SHA-256 differs from the recorded one;
- the checkpoint's embedded config/provenance differ from the recorded ones;
- the schema is unknown.
`Calibrator` then maps raw student logits to calibrated P(fake) and a
Verdict, with frame/image and video levels kept separate.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from configuard.calibration.core import prediction_sets, sigmoid, verdicts_from_sets
from configuard.crops.store import atomic_write_bytes, canonical_json
from configuard.io_types import Verdict

SCHEMA = "p6c-calibration-1"
LEVELS = ("frame", "video")


class CalibrationMismatchError(Exception):
    """Calibration artifact does not belong to this checkpoint/config, or was altered."""


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _checkpoint_meta(checkpoint: str | Path) -> dict[str, Any]:
    import torch

    ck = torch.load(checkpoint, map_location="cpu", weights_only=True)
    return {"config": ck["config"], "provenance": ck["provenance"]}


def build_artifact(checkpoint: str | Path, levels: dict[str, dict[str, Any]], default_alpha: float,
                   calibration_provenance: dict[str, Any], required_levels: tuple[str, ...] = LEVELS,
                   extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """`required_levels`: ("frame", "video") for Phase 6c; ("video_k4", "video_k8",
    "video_k16") for the Phase 6d adaptive stages. `extra` (e.g. the stopping
    policy) is covered by the content hash like everything else."""
    if set(levels) != set(required_levels):
        raise ValueError(f"levels must be exactly {required_levels}")
    meta = _checkpoint_meta(checkpoint)
    body = {
        "schema": SCHEMA, "checkpoint_sha256": file_sha256(checkpoint),
        "model_config": meta["config"], "model_config_sha256": hashlib.sha256(canonical_json(meta["config"])).hexdigest(),
        "model_provenance": meta["provenance"], "calibration_provenance": calibration_provenance,
        "default_alpha": default_alpha, "levels": levels,
    } | (extra or {})
    return body | {"content_sha256": hashlib.sha256(canonical_json(body)).hexdigest()}


def save_artifact(path: str | Path, artifact: dict[str, Any]) -> None:
    atomic_write_bytes(Path(path), canonical_json(artifact))


def load_calibration(path: str | Path, checkpoint: str | Path) -> "Calibrator":
    art = json.loads(Path(path).read_text(encoding="utf-8"))
    if art.get("schema") != SCHEMA:
        raise CalibrationMismatchError(f"{path}: unknown schema {art.get('schema')!r}")
    body = {k: v for k, v in art.items() if k != "content_sha256"}
    if hashlib.sha256(canonical_json(body)).hexdigest() != art.get("content_sha256"):
        raise CalibrationMismatchError(f"{path}: content hash mismatch (file was modified)")
    actual = file_sha256(checkpoint)
    if actual != art["checkpoint_sha256"]:
        raise CalibrationMismatchError(f"{checkpoint}: sha256 {actual[:12]} != calibrated {art['checkpoint_sha256'][:12]}")
    meta = _checkpoint_meta(checkpoint)
    if json.loads(canonical_json(meta["config"])) != art["model_config"] or \
            json.loads(canonical_json(meta["provenance"])) != art["model_provenance"]:
        raise CalibrationMismatchError(f"{checkpoint}: config/provenance differ from the calibration artifact")
    return Calibrator(art)


@dataclass(frozen=True)
class CalibratedOutput:
    p_fake: np.ndarray
    verdicts: list[Verdict]


class Calibrator:
    def __init__(self, artifact: dict[str, Any]) -> None:
        self.artifact = artifact

    def temperature(self, level: str) -> float:
        return float(self.artifact["levels"][level]["temperature"])

    def thresholds(self, level: str, alpha: float | None = None, mode: str = "mondrian") -> dict[str, Any]:
        alpha = self.artifact["default_alpha"] if alpha is None else alpha
        for t in self.artifact["levels"][level]["conformal"]:
            if t["mode"] == mode and abs(t["alpha"] - alpha) < 1e-12:
                return t
        raise KeyError(f"no {mode} conformal thresholds for level={level} alpha={alpha}")

    def calibrate(self, logits: np.ndarray, level: str) -> np.ndarray:
        if level not in self.artifact["levels"]:
            raise ValueError(f"level must be one of {sorted(self.artifact['levels'])}")
        return sigmoid(np.asarray(logits, np.float64) / self.temperature(level))

    def predict(self, logits: np.ndarray, level: str, alpha: float | None = None,
                mode: str = "mondrian") -> CalibratedOutput:
        """level='frame' for single images/frames; level='video' for the mean frame logit of a video."""
        p = self.calibrate(logits, level)
        has_real, has_fake = prediction_sets(p, self.thresholds(level, alpha, mode))
        return CalibratedOutput(p, verdicts_from_sets(has_real, has_fake))
