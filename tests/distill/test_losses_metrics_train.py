"""Phase 6b: distillation loss, evaluation metrics, and a tiny CPU training run."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from configuard.distill.evaluate import binary_metrics, evaluate_logits, expected_calibration_error
from configuard.distill.losses import distillation_loss
from configuard.distill.train import DistillConfig, StaleRunError, StudentTrainer

from .conftest import CROP_TAG, TEACHER_TAG


def test_alpha_zero_is_plain_bce_and_ignores_teacher():
    z, y = torch.tensor([0.3, -1.2, 2.0]), torch.tensor([1.0, 0.0, 1.0])
    parts = distillation_loss(z, y, torch.tensor([9.0, 9.0, 9.0]), alpha=0.0, temperature=4.0)
    assert parts.total.item() == pytest.approx(F.binary_cross_entropy_with_logits(z, y).item())


def test_soft_term_is_minimised_at_the_teacher_logit():
    m = torch.tensor([1.5, -0.7])
    z = m.clone().requires_grad_(True)
    distillation_loss(z, torch.tensor([1.0, 0.0]), m, alpha=1.0, temperature=2.0).total.backward()
    assert torch.allclose(z.grad, torch.zeros(2), atol=1e-6)
    with pytest.raises(ValueError):
        distillation_loss(z, torch.tensor([1.0, 0.0]), m, alpha=1.5, temperature=2.0)


def test_metrics_known_values():
    y = np.array([0, 0, 1, 1])
    m = binary_metrics(y, np.array([-2.0, -1.0, 1.0, 2.0]))
    assert m["auroc"] == 1.0 and m["auprc"] == 1.0 and m["accuracy"] == 1.0
    assert expected_calibration_error(np.array([0.0, 1.0]), np.array([0.5, 0.5])) == pytest.approx(0.0)
    assert expected_calibration_error(np.array([0.0, 0.0, 1.0]), np.array([0.9, 0.9, 0.1])) == pytest.approx(0.9)
    assert binary_metrics(y, np.zeros(4))["brier"] == pytest.approx(0.25)


def test_video_and_per_method_aggregation():
    rows = [{"label": "real", "sample_id": "r", "metadata": {"method": None}}] * 2 + [
        {"label": "fake", "sample_id": "d", "metadata": {"method": "Deepfakes"}}] * 2 + [
        {"label": "fake", "sample_id": "n", "metadata": {"method": "NeuralTextures"}}] * 2
    out = evaluate_logits(rows, np.array([-1.0, -3.0, 2.0, 0.0, -4.0, 0.0]))
    assert out["video"]["n"] == 3
    assert out["video"]["per_method"]["Deepfakes"]["auroc"] == 1.0  # mean 1.0 > -2.0
    assert out["video"]["per_method"]["NeuralTextures"]["auroc"] == 0.5  # mean -2.0 ties the real video
    assert set(out["frame"]["per_method"]) == {"Deepfakes", "NeuralTextures"}


def _cfg(**kw):
    base = dict(run_name="tiny", crop_tag=CROP_TAG, teacher_tag=TEACHER_TAG, pretrained=False, batch_size=8,
                num_workers=0, samples_per_epoch=16, max_epochs=2, warmup_steps=1, amp=False,
                early_stopping_patience=5)
    return DistillConfig.from_dict(base | kw)


def _trainer(store, tmp_path, cfg):
    return StudentTrainer(cfg, store, store.parent / "teacher", tmp_path / "runs", device="cpu")


def test_unknown_config_keys_are_rejected():
    with pytest.raises(ValueError, match="Unknown"):
        DistillConfig.from_dict({"run_name": "x", "learning_rate": 1})


def test_tiny_training_run_resume_and_stale_refusal(synthetic_store, tmp_path):
    t = _trainer(synthetic_store, tmp_path, _cfg(alpha=0.5, temperature=2.0))
    s1 = t.fit(max_epochs=1)
    assert s1["epochs_run"] == 1 and s1["val_best"]["frame"]["n"] == 5 * 2 * 2
    ck = torch.load(t.run_dir / "best.pt", map_location="cpu", weights_only=True)  # no pickle needed
    assert ck["provenance"]["teacher_tag"] == TEACHER_TAG and ck["config"]["alpha"] == 0.5
    lines = (t.run_dir / "epochs.jsonl").read_text().splitlines()
    assert len(lines) == 1 and json.loads(lines[0])["train"]["soft"] > 0

    resumed = _trainer(synthetic_store, tmp_path, _cfg(alpha=0.5, temperature=2.0)).fit()
    assert resumed["resumed"] and resumed["epochs_run"] == 2
    assert len((t.run_dir / "epochs.jsonl").read_text().splitlines()) == 2

    with pytest.raises(StaleRunError):
        _trainer(synthetic_store, tmp_path, _cfg(alpha=0.9, temperature=2.0))


def test_baseline_and_distilled_share_init_and_batches(synthetic_store, tmp_path):
    a = _trainer(synthetic_store, tmp_path, _cfg(run_name="a", alpha=0.0))
    b = _trainer(synthetic_store, tmp_path, _cfg(run_name="b", alpha=0.9, temperature=4.0))
    sa, sb = a.model.state_dict(), b.model.state_dict()
    assert all(torch.equal(sa[k], sb[k]) for k in sa)
    a.train_sampler.set_epoch(1)
    b.train_sampler.set_epoch(1)
    assert a.train_sampler.indices() == b.train_sampler.indices()
    xa = next(iter(a.train_loader))[0]
    xb = next(iter(b.train_loader))[0]
    assert torch.equal(xa, xb)
