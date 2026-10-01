"""Media-quality safety gate over the adaptive 4/8/16 verdict.

Contract: the gated verdict is ALWAYS either the ungated verdict or
UNCERTAIN. The gate never turns anything into likely real / likely
manipulated, never changes which frames the analyzer scores, and never
re-scores a frame (quality comes from the crops the scorer already decoded).

A confident verdict becomes UNCERTAIN when, over the frames the analyzer
actually used:
1. MAJORITY: >= `majority` (default 0.5) of them fail the same check
   -> that reason code (LOW_SHARPNESS, LOW_RESOLUTION, HEAVY_COMPRESSION);
   SMALL_FACE fires from the detector's source face width for the video;
2. QUALITY_DEPENDENT_VERDICT: at least one used frame fails a check AND the
   verdict recomputed from the passing frames alone (same stage temperature
   and conformal thresholds) is not the same singleton. This closes the
   "one bad frame flips the decision" bypass.
An UNCERTAIN verdict stays UNCERTAIN (reason codes are still reported).

Thresholds are fitted on official TRAIN crops only (configuard.quality.fit)
and stored in a hash-bound artifact.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from configuard.adaptive.analyzer import AdaptiveResult
from configuard.adaptive.policy import StagePolicy, decide_stage, level_name
from configuard.calibration.artifact import Calibrator
from configuard.crops.store import atomic_write_bytes, canonical_json
from configuard.io_types import Verdict
from configuard.quality.signals import SIGNALS, crop_signals

LOW_SHARPNESS, LOW_RESOLUTION, HEAVY_COMPRESSION, SMALL_FACE = (
    "LOW_SHARPNESS", "LOW_RESOLUTION", "HEAVY_COMPRESSION", "SMALL_FACE")
QUALITY_DEPENDENT = "QUALITY_DEPENDENT_VERDICT"
FRAME_CODES = (LOW_SHARPNESS, LOW_RESOLUTION, HEAVY_COMPRESSION)
SCHEMA = "p9-quality-gate-1"


class QualityGateMismatchError(Exception):
    """Gate artifact edited or bound to other model/calibration files."""


@dataclass(frozen=True)
class GateThresholds:
    sharpness_min: float
    hf_ratio_min: float
    blockiness_max: float
    face_px_min: float
    majority: float = 0.5
    percentile: float = 0.0  # tail percentile the thresholds came from (provenance)

    def frame_flags(self, q: np.ndarray) -> np.ndarray:
        """q (n, 3) -> (n, 3) bool in FRAME_CODES order."""
        q = np.atleast_2d(q)
        return np.stack([q[:, 0] < self.sharpness_min, q[:, 1] < self.hf_ratio_min, q[:, 2] > self.blockiness_max], 1)


@dataclass(frozen=True)
class GatedResult:
    verdict: Verdict
    base_verdict: Verdict
    reasons: list[str]
    gated: bool
    failing_fraction: dict[str, float]
    frames_used: int
    passing_frames: int
    base: AdaptiveResult | None = field(default=None, repr=False)

    def to_dict(self) -> dict[str, Any]:
        return {"verdict": self.verdict.value, "base_verdict": self.base_verdict.value, "reasons": self.reasons,
                "gated": self.gated, "failing_fraction": self.failing_fraction, "frames_used": self.frames_used,
                "passing_frames": self.passing_frames}


def apply_gate(result: AdaptiveResult, quality_by_slot: dict[int, np.ndarray], face_px: float | None,
               thr: GateThresholds, calibrator: Calibrator, policy: StagePolicy) -> GatedResult:
    slots = [t["slot"] for t in result.timeline]
    logits = np.array([t["logit"] for t in result.timeline], np.float64)
    if not slots:
        return GatedResult(Verdict.UNCERTAIN, result.verdict, [], False, {}, 0, 0, result)
    q = np.stack([np.asarray(quality_by_slot[s], np.float32) for s in slots])
    flags = thr.frame_flags(q)
    frac = {c: float(flags[:, i].mean()) for i, c in enumerate(FRAME_CODES)}
    reasons = [c for c in FRAME_CODES if frac[c] >= thr.majority]
    if face_px is not None and face_px < thr.face_px_min:
        reasons.append(SMALL_FACE)
    bad = flags.any(axis=1)
    if result.verdict is not Verdict.UNCERTAIN and not reasons and bad.any():
        good = ~bad
        stage = result.final_stage
        level = level_name(stage)
        if good.any():
            d = decide_stage(float(logits[good].mean()), stage, calibrator.temperature(level),
                             calibrator.thresholds(level, policy.alpha_spending[stage], policy.mode))
            same = d.singleton and d.verdict is result.verdict
        else:
            same = False
        if not same:
            reasons = [QUALITY_DEPENDENT] + [c for c in FRAME_CODES if frac[c] > 0]
    gated = result.verdict is not Verdict.UNCERTAIN and bool(reasons)
    verdict = Verdict.UNCERTAIN if gated else result.verdict
    assert verdict in (result.verdict, Verdict.UNCERTAIN)  # the gate can only downgrade
    return GatedResult(verdict, result.verdict, reasons, gated, frac, len(slots), int((~bad).sum()), result)


class QualityAwareScorer:
    """Adaptive-analyzer scorer that decodes each requested crop ONCE, computes its
    quality signals from those same pixels, and runs the detector (e.g. an ONNX
    ShapePinnedRunner) on them. Quality never enters the detector's input."""

    def __init__(self, runner, crop_paths_by_slot: dict[int, str | Path]) -> None:
        self.runner, self.paths = runner, crop_paths_by_slot
        self.quality: dict[int, np.ndarray] = {}
        self.frames_scored = 0

    def __call__(self, slots: Sequence[int]) -> np.ndarray:
        import cv2

        imgs = [cv2.imdecode(np.fromfile(str(self.paths[s]), np.uint8), cv2.IMREAD_COLOR) for s in slots]
        for s, img in zip(slots, imgs):
            self.quality[s] = crop_signals(img)
        pixels = np.stack([img[:, :, ::-1].transpose(2, 0, 1) for img in imgs]).astype(np.float32)
        self.frames_scored += len(slots)
        return np.asarray(self.runner(pixels), np.float32)


def save_thresholds(path: str | Path, thr: GateThresholds, binding: dict[str, Any]) -> dict[str, Any]:
    body = {"schema": SCHEMA, "signals": list(SIGNALS), "thresholds": thr.__dict__, "binding": binding}
    art = body | {"content_sha256": hashlib.sha256(canonical_json(body)).hexdigest()}
    atomic_write_bytes(Path(path), canonical_json(art))
    return art


def load_thresholds(path: str | Path, expected_binding: dict[str, Any] | None = None) -> GateThresholds:
    art = json.loads(Path(path).read_text(encoding="utf-8"))
    body = {k: v for k, v in art.items() if k != "content_sha256"}
    if art.get("schema") != SCHEMA or hashlib.sha256(canonical_json(body)).hexdigest() != art.get("content_sha256"):
        raise QualityGateMismatchError(f"{path}: unknown schema or edited content")
    if expected_binding is not None:
        for k, v in expected_binding.items():
            if art["binding"].get(k) != v:
                raise QualityGateMismatchError(f"{path}: bound to {k}={art['binding'].get(k)!r}, expected {v!r}")
    return GateThresholds(**art["thresholds"])


class GatedVideoAnalyzer:
    """Adaptive 4/8/16 analysis + quality gate in one call (Phase 9 production path).

    `runner` scores pixel batches (e.g. configuard.export.onnx_student.ShapePinnedRunner
    over the package's default ONNX FP32 model). Frames are decoded once, scored
    once and their quality is computed from the same pixels; the gate can only
    downgrade. `enabled=False` returns the ungated verdict unchanged (reasons are
    still reported), which preserves the previous pipeline exactly."""

    def __init__(self, calibrator: Calibrator, policy: StagePolicy, thresholds: GateThresholds, enabled: bool = True) -> None:
        from configuard.adaptive.analyzer import AdaptiveVideoAnalyzer

        self.analyzer = AdaptiveVideoAnalyzer(calibrator, policy)
        self.calibrator, self.policy, self.thresholds, self.enabled = calibrator, policy, thresholds, enabled

    def analyze(self, runner, crop_paths_by_slot: dict[int, str | Path], face_px: float | None) -> GatedResult:
        scorer = QualityAwareScorer(runner, crop_paths_by_slot)
        result = self.analyzer.analyze(scorer, available=set(crop_paths_by_slot))
        assert scorer.frames_scored == result.frames_used  # no frame is ever scored twice
        g = apply_gate(result, scorer.quality, face_px, self.thresholds, self.calibrator, self.policy)
        if self.enabled:
            return g
        return GatedResult(result.verdict, result.verdict, g.reasons, False, g.failing_fraction, g.frames_used,
                           g.passing_frames, result)
