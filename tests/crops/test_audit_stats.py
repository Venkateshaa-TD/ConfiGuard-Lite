"""Phase 5d: audit statistics helpers."""

from __future__ import annotations

import numpy as np
import pytest

from configuard.crops.audit_stats import auc, fit_softmax, mean_ci, predict_proba, spearman


def test_auc_basic_cases():
    assert auc([3, 4, 5], [0, 1, 2]) == 1.0
    assert auc([0, 1, 2], [3, 4, 5]) == 0.0
    assert auc([1, 1], [1, 1]) == 0.5  # ties count half
    assert auc([], [1]) is None


def test_spearman_and_mean_ci():
    assert spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    ci = mean_ci([1.0, 2.0, 3.0])
    assert ci["mean"] == 2.0 and ci["ci95"][0] < 2.0 < ci["ci95"][1]


def test_softmax_probe_is_deterministic_and_finds_signal():
    rng = np.random.default_rng(0)
    y = np.repeat(np.arange(3), 50)
    x = np.c_[y + rng.normal(0, 0.3, len(y)), rng.normal(0, 1, len(y))]
    m1, m2 = fit_softmax(x, y, 3, steps=500), fit_softmax(x, y, 3, steps=500)
    assert np.array_equal(m1[0], m2[0])
    assert (predict_proba(m1, x).argmax(axis=1) == y).mean() > 0.9
    noise = rng.normal(0, 1, (150, 2))
    acc = (predict_proba(fit_softmax(noise[:75], y[::2], 3, steps=500), noise[75:]).argmax(axis=1) == y[1::2]).mean()
    assert acc < 0.6  # no signal -> near chance
