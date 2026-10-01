"""Small, dependency-free statistics for the Phase 5d shortcut audit:
rank AUC, Spearman correlation, mean with a normal-approximation 95% CI,
and a deterministic multinomial logistic regression (numpy, full-batch
gradient descent, fixed iterations, zero init) for "is this factor
predictive?" probes."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np


def _ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    sorted_vals = values[order]
    i = 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and sorted_vals[j + 1] == sorted_vals[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def auc(positive: Sequence[float], negative: Sequence[float]) -> float | None:
    """P(score_pos > score_neg) + 0.5 P(tie) (Mann-Whitney). 0.5 = no signal."""
    pos, neg = np.asarray(positive, float), np.asarray(negative, float)
    if len(pos) == 0 or len(neg) == 0:
        return None
    ranks = _ranks(np.concatenate([pos, neg]))
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def spearman(x: Sequence[float], y: Sequence[float]) -> float | None:
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3:
        return None
    rx, ry = _ranks(x), _ranks(y)
    if rx.std() == 0 or ry.std() == 0:
        return 0.0
    return float(np.corrcoef(rx, ry)[0, 1])


def mean_ci(values: Sequence[float]) -> dict[str, float | int | None]:
    v = np.asarray(values, float)
    if len(v) == 0:
        return {"n": 0, "mean": None, "ci95": None, "sd": None}
    sd = float(v.std(ddof=1)) if len(v) > 1 else 0.0
    half = 1.96 * sd / math.sqrt(len(v)) if len(v) > 1 else 0.0
    return {"n": int(len(v)), "mean": float(v.mean()), "ci95": [float(v.mean() - half), float(v.mean() + half)], "sd": sd}


def fit_softmax(x: np.ndarray, y: np.ndarray, classes: int, l2: float = 1e-3, steps: int = 3000, lr: float = 0.5):
    """Returns (weights, bias, mean, std). Features are standardised with
    TRAIN statistics only."""
    mu, sd = x.mean(axis=0), x.std(axis=0) + 1e-9
    z = (x - mu) / sd
    w = np.zeros((z.shape[1], classes))
    b = np.zeros(classes)
    onehot = np.eye(classes)[y]
    for _ in range(steps):
        logits = z @ w + b
        logits -= logits.max(axis=1, keepdims=True)
        p = np.exp(logits)
        p /= p.sum(axis=1, keepdims=True)
        grad = p - onehot
        w -= lr * (z.T @ grad / len(z) + l2 * w)
        b -= lr * grad.mean(axis=0)
    return w, b, mu, sd


def predict_proba(model, x: np.ndarray) -> np.ndarray:
    w, b, mu, sd = model
    logits = ((x - mu) / sd) @ w + b
    logits -= logits.max(axis=1, keepdims=True)
    p = np.exp(logits)
    return p / p.sum(axis=1, keepdims=True)
