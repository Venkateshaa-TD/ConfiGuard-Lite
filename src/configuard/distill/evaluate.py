"""Frame- and video-level validation metrics for a binary real/fake scorer.

Threshold-free: AUROC, AUPRC (average precision). Calibration: ECE
(15 equal-width bins on P(fake)), Brier score, NLL. Threshold 0.5: accuracy,
balanced accuracy. Video scores are the mean frame logit per sample_id
(the Phase 4 mean-aggregation contract). Per-manipulation results compare
one method's fakes against all originals of the same split.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from typing import Any

import numpy as np

from configuard.distill.data import REAL, method_of
from configuard.training.metrics import compute_auroc, compute_average_precision

METHODS = ("Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures")


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60, 60)))


def expected_calibration_error(y: np.ndarray, p: np.ndarray, bins: int = 15) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1], right=True), 0, bins - 1)
    ece = 0.0
    for b in range(bins):
        mask = idx == b
        if mask.any():
            ece += mask.mean() * abs(p[mask].mean() - y[mask].mean())
    return float(ece)


def binary_metrics(y: np.ndarray, logits: np.ndarray) -> dict[str, Any]:
    y = np.asarray(y, np.float64)
    logits = np.asarray(logits, np.float64)
    p = sigmoid(logits)
    eps = 1e-7
    pred = p >= 0.5
    pos, neg = y == 1, y == 0
    tpr = float(pred[pos].mean()) if pos.any() else None
    tnr = float((~pred[neg]).mean()) if neg.any() else None
    return {
        "n": int(len(y)), "n_fake": int(pos.sum()), "n_real": int(neg.sum()),
        "auroc": compute_auroc(y.astype(int), logits), "auprc": compute_average_precision(y.astype(int), logits),
        "accuracy": float((pred == pos).mean()),
        "balanced_accuracy": None if tpr is None or tnr is None else (tpr + tnr) / 2,
        "tpr_fake": tpr, "tnr_real": tnr,
        "ece": expected_calibration_error(y, p), "brier": float(np.mean((p - y) ** 2)),
        "nll": float(-np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))),
        "mean_p_fake": {"real": float(p[neg].mean()) if neg.any() else None,
                        "fake": float(p[pos].mean()) if pos.any() else None},
    }


def video_scores(rows: Sequence[dict[str, Any]], logits: np.ndarray) -> tuple[list[str], np.ndarray, np.ndarray, list[str]]:
    groups: dict[str, list[float]] = defaultdict(list)
    meta: dict[str, tuple[float, str]] = {}
    for r, z in zip(rows, logits):
        groups[r["sample_id"]].append(float(z))
        meta[r["sample_id"]] = (1.0 if r["label"] == "fake" else 0.0, method_of(r))
    ids = sorted(groups)
    return (ids, np.array([meta[i][0] for i in ids]), np.array([np.mean(groups[i]) for i in ids]),
            [meta[i][1] for i in ids])


def _per_method(y: np.ndarray, z: np.ndarray, methods: Sequence[str]) -> dict[str, Any]:
    methods = np.asarray(methods)
    real = methods == REAL
    out = {}
    for m in METHODS:
        mask = real | (methods == m)
        if (methods == m).any():
            r = binary_metrics(y[mask], z[mask])
            out[m] = {k: r[k] for k in ("auroc", "auprc", "tpr_fake", "n_fake")}
    return out


def evaluate_logits(rows: Sequence[dict[str, Any]], logits: np.ndarray) -> dict[str, Any]:
    if len(rows) != len(logits):
        raise ValueError("rows and logits must align")
    logits = np.asarray(logits, np.float64)
    y = np.array([1.0 if r["label"] == "fake" else 0.0 for r in rows])
    methods = [method_of(r) for r in rows]
    _, vy, vz, vm = video_scores(rows, logits)
    frame, video = binary_metrics(y, logits), binary_metrics(vy, vz)
    frame["per_method"], video["per_method"] = _per_method(y, logits, methods), _per_method(vy, vz, vm)
    return {"frame": frame, "video": video}
