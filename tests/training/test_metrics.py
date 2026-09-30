"""Phase 5: validation metrics."""

from __future__ import annotations

import numpy as np

from configuard.training.metrics import (
    compute_auroc,
    compute_average_precision,
    compute_confusion_matrix,
    compute_threshold_dependent_metrics,
    compute_validation_metrics,
)


def test_auroc_perfect_separation_is_one():
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_score = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    assert compute_auroc(y_true, y_score) == 1.0


def test_auroc_inverted_separation_is_zero():
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_score = np.array([0.9, 0.8, 0.7, 0.3, 0.2, 0.1])
    assert compute_auroc(y_true, y_score) == 0.0


def test_auroc_all_tied_scores_is_half():
    y_true = np.array([0, 1, 0, 1])
    y_score = np.array([0.5, 0.5, 0.5, 0.5])
    assert compute_auroc(y_true, y_score) == 0.5


def test_auroc_undefined_when_one_class_absent():
    y_true = np.array([1, 1, 1])
    y_score = np.array([0.1, 0.5, 0.9])
    assert compute_auroc(y_true, y_score) is None


def test_auroc_undefined_when_all_negative():
    y_true = np.array([0, 0, 0])
    y_score = np.array([0.1, 0.5, 0.9])
    assert compute_auroc(y_true, y_score) is None


def test_average_precision_perfect_is_one():
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_score = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    assert compute_average_precision(y_true, y_score) == 1.0


def test_average_precision_undefined_when_no_positives():
    y_true = np.array([0, 0, 0])
    y_score = np.array([0.1, 0.5, 0.9])
    assert compute_average_precision(y_true, y_score) is None


def test_confusion_matrix_counts():
    y_true = np.array([1, 1, 0, 0])
    y_pred = np.array([1, 0, 0, 1])
    cm = compute_confusion_matrix(y_true, y_pred)
    assert cm.true_positive == 1
    assert cm.false_negative == 1
    assert cm.true_negative == 1
    assert cm.false_positive == 1


def test_threshold_dependent_metrics_perfect_classifier():
    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([0.1, 0.2, 0.8, 0.9])
    metrics = compute_threshold_dependent_metrics(y_true, y_score, threshold=0.5)
    assert metrics.sensitivity == 1.0
    assert metrics.specificity == 1.0
    assert metrics.balanced_accuracy == 1.0
    assert metrics.f1 == 1.0


def test_threshold_dependent_metrics_all_wrong():
    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([0.9, 0.8, 0.2, 0.1])
    metrics = compute_threshold_dependent_metrics(y_true, y_score, threshold=0.5)
    assert metrics.sensitivity == 0.0
    assert metrics.specificity == 0.0
    assert metrics.f1 == 0.0


def test_threshold_dependent_metrics_handles_empty_class_without_division_error():
    y_true = np.array([0, 0, 0])
    y_score = np.array([0.1, 0.6, 0.9])
    metrics = compute_threshold_dependent_metrics(y_true, y_score, threshold=0.5)
    assert metrics.sensitivity == 0.0  # no positives at all - defined as 0, not NaN/crash


def test_validation_metrics_separates_threshold_free_and_dependent():
    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([0.1, 0.2, 0.8, 0.9])
    result = compute_validation_metrics(y_true, y_score)
    assert result.threshold_free.auroc == 1.0
    assert result.threshold_dependent.threshold == 0.5
    assert result.num_samples == 4

    flat = result.to_dict()
    assert "auroc" in flat and "average_precision" in flat  # threshold-free
    assert "sensitivity" in flat and "specificity" in flat and "f1" in flat  # threshold-dependent
    assert "confusion_matrix" in flat


def test_different_thresholds_change_only_dependent_metrics():
    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([0.3, 0.4, 0.6, 0.7])
    low = compute_validation_metrics(y_true, y_score, threshold=0.1)
    high = compute_validation_metrics(y_true, y_score, threshold=0.9)
    assert low.threshold_free.auroc == high.threshold_free.auroc  # unaffected by threshold
    assert low.threshold_dependent.sensitivity != high.threshold_dependent.sensitivity
