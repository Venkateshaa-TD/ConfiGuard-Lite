"""Stage-wise stopping policy over the nested 4/8/16 frame sets.

Stage k uses only the frames in the k-set (slot % (16 // k) == 0, the
Phase 5d nested contract), and its video score is the mean frame logit
over those frames. Each stage has its own temperature T_k and mondrian
conformal thresholds at its own alpha_k (alpha spending):

- stages 4 and 8: stop ONLY on a singleton prediction set (a confident
  "likely real" or "likely manipulated"); otherwise escalate;
- stage 16: return the singleton verdict, or "uncertain" for an empty or
  two-label set.

Union bound: if every stage's set misses the true label with probability
<= alpha_k, the label the procedure finally commits to is wrong with
probability <= sum(alpha_k) (= 0.05 by default). This holds only for data
exchangeable with the calibration partition. Under domain shift (already
measured inside FF++ in Phase 6c) the coverage is EMPIRICAL, NOT GUARANTEED.

Feasibility: a mondrian threshold at level alpha needs n_class >= 1/alpha - 1
calibration units per class. With 71 real conformal-calibration videos,
alpha_k >= 1/72 ~ 0.0139, which is why the default spend is 0.015/0.015/0.02.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from configuard.calibration.core import prediction_sets, sigmoid
from configuard.io_types import Verdict

STAGES = (4, 8, 16)
FULL = 16


@dataclass(frozen=True)
class StagePolicy:
    stages: tuple[int, ...] = STAGES
    alpha_spending: dict[int, float] = field(default_factory=lambda: {4: 0.015, 8: 0.015, 16: 0.02})
    mode: str = "mondrian"

    def __post_init__(self) -> None:
        if tuple(sorted(self.stages)) != self.stages or self.stages[-1] != FULL:
            raise ValueError("stages must be increasing and end at 16")
        if set(self.alpha_spending) != set(self.stages):
            raise ValueError("alpha_spending needs one alpha per stage")

    @property
    def total_alpha(self) -> float:
        return float(sum(self.alpha_spending.values()))

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["alpha_spending"] = {str(k): v for k, v in self.alpha_spending.items()}
        d["stages"] = list(self.stages)
        return d


def min_supported_alpha(n_per_class: int) -> float:
    return 1.0 / (n_per_class + 1)


def stage_slots(k: int, full: int = FULL) -> list[int]:
    """Nested contract: 4-set = slots % 4 == 0, 8-set = even slots, 16 = all."""
    if full % k:
        raise ValueError(f"stage {k} does not divide {full}")
    return list(range(0, full, full // k))


def level_name(k: int) -> str:
    return f"video_k{k}"


@dataclass(frozen=True)
class StageDecision:
    stage: int
    score: float  # mean frame logit over the stage's frames
    p_fake: float  # temperature-scaled
    has_real: bool
    has_fake: bool
    alpha: float

    @property
    def singleton(self) -> bool:
        return self.has_real != self.has_fake

    @property
    def verdict(self) -> Verdict:
        if not self.singleton:
            return Verdict.UNCERTAIN
        return Verdict.LIKELY_MANIPULATED if self.has_fake else Verdict.LIKELY_REAL

    @property
    def confidence(self) -> float:
        return max(self.p_fake, 1.0 - self.p_fake)


def decide_stage(score: float, k: int, temperature: float, thresholds: dict[str, Any]) -> StageDecision:
    p = float(sigmoid(score / temperature))
    has_real, has_fake = prediction_sets([p], thresholds)
    return StageDecision(k, float(score), p, bool(has_real[0]), bool(has_fake[0]), float(thresholds["alpha"]))
