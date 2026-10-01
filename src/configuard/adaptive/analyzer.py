"""Adaptive video analyzer: 4 -> 8 -> 16 frames, never re-scoring a frame.

A `FrameScorer` maps a list of slot indices to per-frame logits (and,
optionally, embeddings). The analyzer keeps every scored frame in a
per-video cache, so escalating from 4 to 8 frames scores only the 4 new
odd-multiple-of-2 slots, and 8 -> 16 only the 8 odd slots. The same code
path runs over precomputed logits (`ArrayScorer`, for evaluation) and over
live crops with the student (`StudentCropScorer`).

Every result records the stopping reason, frames used, confidence, each
stage's decision, and an evidence timeline (one entry per scored frame in
temporal slot order: its logit, raw P(fake), and the stage that added it).
"""

from __future__ import annotations

import time
from collections.abc import Callable, Collection, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from configuard.adaptive.policy import FULL, StageDecision, StagePolicy, decide_stage, level_name, stage_slots
from configuard.calibration.artifact import Calibrator
from configuard.calibration.core import sigmoid
from configuard.io_types import Verdict


class FrameScorer(Protocol):
    def __call__(self, slots: Sequence[int]) -> np.ndarray: ...


@dataclass(frozen=True)
class AdaptiveResult:
    verdict: Verdict
    stopping_reason: str
    frames_used: int
    final_stage: int
    p_fake: float
    confidence: float
    stages: list[dict[str, Any]]
    timeline: list[dict[str, Any]]
    scorer_calls: int
    latency_ms: float | None = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["verdict"] = self.verdict.value
        return d


class AdaptiveVideoAnalyzer:
    def __init__(self, calibrator: Calibrator, policy: StagePolicy) -> None:
        self.calibrator, self.policy = calibrator, policy
        for k in policy.stages:  # fail fast if the artifact lacks a stage/alpha
            calibrator.thresholds(level_name(k), policy.alpha_spending[k], policy.mode)

    def _decide(self, logits: dict[int, float], k: int, alpha: float) -> StageDecision:
        score = float(np.mean([logits[s] for s in stage_slots(k)]))
        level = level_name(k)
        return decide_stage(score, k, self.calibrator.temperature(level),
                            self.calibrator.thresholds(level, alpha, self.policy.mode))

    def analyze(self, scorer: FrameScorer, available: Collection[int] | None = None) -> AdaptiveResult:
        """`available`: slot indices that have a usable crop (None = all 16). A stage
        whose nested set is incomplete ends the analysis as "uncertain"."""
        t0 = time.perf_counter()
        logits: dict[int, float] = {}
        added_at: dict[int, int] = {}
        decisions: list[StageDecision] = []
        calls = 0
        reason = ""
        for k in self.policy.stages:
            need = [s for s in stage_slots(k) if s not in logits]
            if available is not None and any(s not in available for s in need):
                reason = f"insufficient_frames_for_k{k}"
                break
            if need:
                new = np.asarray(scorer(need), np.float64)
                calls += 1
                if new.shape != (len(need),) or not np.isfinite(new).all():
                    raise ValueError(f"scorer returned {new.shape} for {len(need)} slots")
                for s, z in zip(need, new):
                    logits[s], added_at[s] = float(z), k
            d = self._decide(logits, k, self.policy.alpha_spending[k])
            decisions.append(d)
            if k < self.policy.stages[-1] and d.singleton:
                reason = f"confident_singleton_k{k}"
                break
            if k == self.policy.stages[-1]:
                reason = "final_k16_singleton" if d.singleton else (
                    "final_k16_uncertain_both" if d.has_real else "final_k16_uncertain_empty")
        if not decisions:
            return AdaptiveResult(Verdict.UNCERTAIN, reason, len(logits), 0, float("nan"), 0.0, [], [], calls)
        last = decisions[-1]
        verdict = last.verdict if not reason.startswith("insufficient") else Verdict.UNCERTAIN
        timeline = [{"slot": s, "position": s / FULL, "logit": logits[s], "p_fake_raw": float(sigmoid(logits[s])),
                     "added_at_stage": added_at[s]} for s in sorted(logits)]
        stages = [{"stage": d.stage, "score": d.score, "p_fake": d.p_fake, "alpha": d.alpha,
                   "set": [n for n, inc in (("real", d.has_real), ("fake", d.has_fake)) if inc],
                   "verdict": d.verdict.value} for d in decisions]
        return AdaptiveResult(verdict, reason, len(logits), last.stage, last.p_fake, last.confidence, stages,
                              timeline, calls, (time.perf_counter() - t0) * 1000)

    def fixed(self, scorer: FrameScorer, k: int, alpha: float) -> StageDecision:
        """Non-adaptive baseline: score exactly the k-set once and decide at level alpha."""
        slots = stage_slots(k)
        logits = dict(zip(slots, np.asarray(scorer(slots), np.float64).tolist()))
        return self._decide(logits, k, alpha)


class ArrayScorer:
    """Scorer over precomputed per-slot logits (evaluation). Counts scored frames."""

    def __init__(self, logits_by_slot: Sequence[float]) -> None:
        self.logits = np.asarray(logits_by_slot, np.float64)
        self.frames_scored = 0

    def __call__(self, slots: Sequence[int]) -> np.ndarray:
        self.frames_scored += len(slots)
        return self.logits[list(slots)]


class StudentCropScorer:
    """Live scorer: reads + decodes the crops for the requested slots and runs the
    student on them as one batch (no augmentation)."""

    def __init__(self, model: Any, norm: Any, crop_paths_by_slot: dict[int, str | Path], device: str = "cuda",
                 amp: bool = True, decode: Callable[[str | Path], np.ndarray] | None = None) -> None:
        self.model, self.norm, self.paths, self.device = model, norm, crop_paths_by_slot, device
        self.amp = amp and device == "cuda"
        self.decode = decode or _decode_rgb_chw
        self.frames_scored = 0

    def __call__(self, slots: Sequence[int]) -> np.ndarray:
        import torch

        x = torch.from_numpy(np.stack([self.decode(self.paths[s]) for s in slots])).to(self.device)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16, enabled=self.amp):
            z = self.model.forward_logits(self.norm(x))
        if self.device == "cuda":
            torch.cuda.synchronize()
        self.frames_scored += len(slots)
        return z.float().cpu().numpy()


def _decode_rgb_chw(path: str | Path) -> np.ndarray:
    import cv2

    img = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise OSError(f"could not decode {path}")
    return np.ascontiguousarray(img[:, :, ::-1].transpose(2, 0, 1))
