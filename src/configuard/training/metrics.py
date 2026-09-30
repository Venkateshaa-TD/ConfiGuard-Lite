"""Validation metrics, numpy-only (no scikit-learn dependency).

Threshold-FREE metrics (AUROC, average precision) are kept structurally
separate from threshold-DEPENDENT ones (confusion matrix, sensitivity,
specificity, balanced accuracy, F1) - see ValidationMetrics.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _rankdata_average(values: np.ndarray) -> np.ndarray:
    """Assigns 1-indexed ranks, averaging ranks of tied values (matches
    scipy.stats.rankdata(method="average") without the scipy dependency)."""
    sorter = np.argsort(values, kind="mergesort")
    sorted_vals = values[sorter]
    n = len(values)
    rank_in_sorted = np.empty(n, dtype=np.float64)

    pos = 0
    while pos < n:
        end = pos
        while end + 1 < n and sorted_vals[end + 1] == sorted_vals[pos]:
            end += 1
        average_rank = (pos + end) / 2.0 + 1.0
        rank_in_sorted[pos : end + 1] = average_rank
        pos = end + 1

    ranks = np.empty(n, dtype=np.float64)
    ranks[sorter] = rank_in_sorted
    return ranks


def compute_auroc(y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    """Mann-Whitney U formulation. Returns None (undefined) if either
    class is absent from y_true - callers must handle this explicitly
    (see docs/DECISIONS.md for the checkpoint-selection fallback)."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score, dtype=np.float64)
    n_pos = int((y_true == 1).sum())
    n_neg = int((y_true == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return None

    ranks = _rankdata_average(y_score)
    sum_ranks_pos = ranks[y_true == 1].sum()
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def compute_average_precision(y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    """Step-function AP = sum_n (R_n - R_{n-1}) * P_n, sweeping thresholds
    at every distinct score in descending order. Returns None if there
    are no positive examples (undefined)."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score, dtype=np.float64)
    n_pos = int((y_true == 1).sum())
    if n_pos == 0:
        return None

    order = np.argsort(-y_score, kind="mergesort")
    y_true_sorted = y_true[order]

    tp_cumsum = np.cumsum(y_true_sorted == 1)
    fp_cumsum = np.cumsum(y_true_sorted == 0)
    precision = tp_cumsum / (tp_cumsum + fp_cumsum)
    recall = tp_cumsum / n_pos

    recall_prev = np.concatenate(([0.0], recall[:-1]))
    delta_recall = recall - recall_prev
    return float(np.sum(delta_recall * precision))


@dataclass(frozen=True)
class ConfusionMatrix:
    true_positive: int
    false_positive: int
    true_negative: int
    false_negative: int

    def to_dict(self) -> dict[str, int]:
        return {
            "true_positive": self.true_positive, "false_positive": self.false_positive,
            "true_negative": self.true_negative, "false_negative": self.false_negative,
        }


def compute_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> ConfusionMatrix:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return ConfusionMatrix(
        true_positive=int(((y_pred == 1) & (y_true == 1)).sum()),
        false_positive=int(((y_pred == 1) & (y_true == 0)).sum()),
        true_negative=int(((y_pred == 0) & (y_true == 0)).sum()),
        false_negative=int(((y_pred == 0) & (y_true == 1)).sum()),
    )


@dataclass(frozen=True)
class ThresholdDependentMetrics:
    """All of these depend on the chosen decision threshold - kept
    separate from ThresholdFreeMetrics by design (task 15)."""

    threshold: float
    confusion_matrix: ConfusionMatrix
    sensitivity: float  # recall on the positive (fake) class: TP / (TP + FN)
    specificity: float  # TN / (TN + FP)
    balanced_accuracy: float  # (sensitivity + specificity) / 2
    precision: float
    f1: float


def compute_threshold_dependent_metrics(
    y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.5
) -> ThresholdDependentMetrics:
    y_pred = (np.asarray(y_score) >= threshold).astype(int)
    cm = compute_confusion_matrix(y_true, y_pred)

    sensitivity = cm.true_positive / (cm.true_positive + cm.false_negative) if (cm.true_positive + cm.false_negative) else 0.0
    specificity = cm.true_negative / (cm.true_negative + cm.false_positive) if (cm.true_negative + cm.false_positive) else 0.0
    precision = cm.true_positive / (cm.true_positive + cm.false_positive) if (cm.true_positive + cm.false_positive) else 0.0
    f1 = 2 * precision * sensitivity / (precision + sensitivity) if (precision + sensitivity) else 0.0
    balanced_accuracy = (sensitivity + specificity) / 2.0

    return ThresholdDependentMetrics(
        threshold=threshold, confusion_matrix=cm, sensitivity=sensitivity, specificity=specificity,
        balanced_accuracy=balanced_accuracy, precision=precision, f1=f1,
    )


@dataclass(frozen=True)
class ThresholdFreeMetrics:
    auroc: float | None
    average_precision: float | None


@dataclass(frozen=True)
class ValidationMetrics:
    threshold_free: ThresholdFreeMetrics
    threshold_dependent: ThresholdDependentMetrics
    num_samples: int

    def to_dict(self) -> dict:
        return {
            "num_samples": self.num_samples,
            "auroc": self.threshold_free.auroc,
            "average_precision": self.threshold_free.average_precision,
            "threshold": self.threshold_dependent.threshold,
            "sensitivity": self.threshold_dependent.sensitivity,
            "specificity": self.threshold_dependent.specificity,
            "balanced_accuracy": self.threshold_dependent.balanced_accuracy,
            "precision": self.threshold_dependent.precision,
            "f1": self.threshold_dependent.f1,
            "confusion_matrix": self.threshold_dependent.confusion_matrix.to_dict(),
        }


def compute_validation_metrics(
    y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.5
) -> ValidationMetrics:
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    return ValidationMetrics(
        threshold_free=ThresholdFreeMetrics(
            auroc=compute_auroc(y_true, y_score),
            average_precision=compute_average_precision(y_true, y_score),
        ),
        threshold_dependent=compute_threshold_dependent_metrics(y_true, y_score, threshold),
        num_samples=len(y_true),
    )
