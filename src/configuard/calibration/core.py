"""Temperature scaling, split-conformal prediction sets, and selective metrics.

Temperature scaling: p = sigmoid(z / T), T > 0 fitted by minimising NLL
(log-T grid + golden-section refinement; NLL is unimodal in log T). It
never changes the ranking or the sign of z, so 0.5-threshold decisions,
AUROC and risk-coverage ordering are identical to raw; only the
probabilities move.

Split conformal (binary, nonconformity s = 1 - p(true class)):
- "mondrian" (label-conditional, default): one threshold per class from
  that class's calibration samples, so coverage >= 1 - alpha holds for
  reals and for fakes separately (for exchangeable units).
- "marginal": one threshold from all calibration samples.
q = the ceil((n + 1)(1 - alpha))-th smallest score (1.0 if that exceeds n).
Class y is in the set iff 1 - p(y) <= q_y. {fake} -> likely manipulated,
{real} -> likely real, {real, fake} or {} -> uncertain.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from configuard.io_types import Verdict

LOG_T_RANGE = (-4.0, 4.0)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(np.asarray(x, np.float64), -60, 60)))


def nll(y: np.ndarray, p: np.ndarray) -> float:
    eps = 1e-12
    return float(-np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps)))


def fit_temperature(logits: np.ndarray, y: np.ndarray) -> float:
    z, y = np.asarray(logits, np.float64), np.asarray(y, np.float64)
    if len(z) == 0 or len(np.unique(y)) < 2:
        raise ValueError("temperature fitting needs both classes")
    f = lambda lt: nll(y, sigmoid(z / math.exp(lt)))  # noqa: E731
    grid = np.linspace(*LOG_T_RANGE, 161)
    i = int(np.argmin([f(g) for g in grid]))
    a, b = grid[max(0, i - 1)], grid[min(len(grid) - 1, i + 1)]
    g = (math.sqrt(5) - 1) / 2
    c, d = b - g * (b - a), a + g * (b - a)
    for _ in range(80):
        if f(c) < f(d):
            b = d
        else:
            a = c
        c, d = b - g * (b - a), a + g * (b - a)
    return float(math.exp((a + b) / 2))


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    s = np.sort(np.asarray(scores, np.float64))
    n = len(s)
    if n == 0:
        raise ValueError("no calibration scores")
    k = math.ceil((n + 1) * (1 - alpha))
    return 1.0 if k > n else float(s[k - 1])


def fit_conformal(p_fake: np.ndarray, y: np.ndarray, alpha: float, mode: str = "mondrian") -> dict[str, Any]:
    p, y = np.asarray(p_fake, np.float64), np.asarray(y).astype(int)
    s_true = np.where(y == 1, 1 - p, p)  # 1 - p(true class)
    if mode == "mondrian":
        q_real, q_fake = conformal_quantile(s_true[y == 0], alpha), conformal_quantile(s_true[y == 1], alpha)
    elif mode == "marginal":
        q_real = q_fake = conformal_quantile(s_true, alpha)
    else:
        raise ValueError(f"unknown conformal mode {mode!r}")
    return {"alpha": alpha, "mode": mode, "q_real": q_real, "q_fake": q_fake,
            "n_real": int((y == 0).sum()), "n_fake": int((y == 1).sum())}


def prediction_sets(p_fake: np.ndarray, thresholds: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """(has_real, has_fake) boolean arrays."""
    p = np.asarray(p_fake, np.float64)
    return p <= thresholds["q_real"], (1 - p) <= thresholds["q_fake"]


def verdicts_from_sets(has_real: np.ndarray, has_fake: np.ndarray) -> list[Verdict]:
    return [Verdict.LIKELY_MANIPULATED if f and not r else Verdict.LIKELY_REAL if r and not f else Verdict.UNCERTAIN
            for r, f in zip(has_real, has_fake)]


def ece(y: np.ndarray, p: np.ndarray, bins: int = 15) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1], right=True), 0, bins - 1)
    return float(sum((idx == b).mean() * abs(p[idx == b].mean() - y[idx == b].mean())
                     for b in range(bins) if (idx == b).any()))


def probability_metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    y, p = np.asarray(y, np.float64), np.asarray(p, np.float64)
    return {"ece": ece(y, p), "nll": nll(y, p), "brier": float(np.mean((p - y) ** 2)),
            "accuracy@0.5": float(((p >= 0.5) == (y == 1)).mean())}


def risk_coverage(y: np.ndarray, p: np.ndarray, points: int = 21) -> dict[str, Any]:
    """Abstain on the least confident first (confidence = max(p, 1 - p))."""
    y, p = np.asarray(y).astype(int), np.asarray(p, np.float64)
    order = np.argsort(-np.maximum(p, 1 - p), kind="mergesort")
    err = ((p >= 0.5).astype(int) != y)[order]
    n = len(y)
    risks = np.cumsum(err) / np.arange(1, n + 1)
    curve = []
    for c in np.linspace(1.0 / points, 1.0, points):
        k = max(1, int(round(c * n)))
        curve.append({"coverage": round(k / n, 4), "selective_risk": float(risks[k - 1])})
    return {"aurc": float(risks.mean()), "curve": curve}


def conformal_metrics(y: np.ndarray, p: np.ndarray, thresholds: dict[str, Any]) -> dict[str, Any]:
    y = np.asarray(y).astype(int)
    has_real, has_fake = prediction_sets(p, thresholds)
    covered = np.where(y == 1, has_fake, has_real)
    single = has_real ^ has_fake
    pred_fake = has_fake & ~has_real
    correct = single & (pred_fake == (y == 1))
    return {
        "alpha": thresholds["alpha"], "mode": thresholds["mode"],
        "coverage": float(covered.mean()),
        "coverage_real": float(covered[y == 0].mean()) if (y == 0).any() else None,
        "coverage_fake": float(covered[y == 1].mean()) if (y == 1).any() else None,
        "abstention_rate": float((~single).mean()),
        "abstention_real": float((~single)[y == 0].mean()) if (y == 0).any() else None,
        "abstention_fake": float((~single)[y == 1].mean()) if (y == 1).any() else None,
        "empty_set_rate": float((~has_real & ~has_fake).mean()),
        "selective_accuracy": float(correct.sum() / single.sum()) if single.any() else None,
        "decided": int(single.sum()), "n": int(len(y)),
    }


def confidence_abstention_at(y: np.ndarray, p: np.ndarray, abstention_rate: float) -> dict[str, float]:
    """Baseline selective classifier: abstain on the `abstention_rate` least confident
    (max(p, 1 - p)) items, predict at 0.5 otherwise. Used to compare conformal
    decisions against plain confidence ranking at the same abstention rate."""
    y, p = np.asarray(y).astype(int), np.asarray(p, np.float64)
    n = len(y)
    keep = max(1, int(round((1.0 - abstention_rate) * n)))
    order = np.argsort(-np.maximum(p, 1 - p), kind="mergesort")[:keep]
    acc = float(((p[order] >= 0.5).astype(int) == y[order]).mean())
    return {"abstention_rate": 1.0 - keep / n, "selective_accuracy": acc}
