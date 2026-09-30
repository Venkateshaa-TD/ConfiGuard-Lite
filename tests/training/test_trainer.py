"""Phase 5: the training loop end-to-end on synthetic data.

ENGINEERING TESTS ONLY: every model here is trained on synthetic tinted
checkerboards (configuard.training.synthetic) - passing says the pipeline
mechanics work, nothing about deepfake detection.

Uses pretrained=False (no dependency on the D-drive weight cache) and
tmp_path for checkpoints/logs. The real D-drive runs are the smoke CLI
runs recorded in docs/EXPERIMENT_LOG.md.
"""

from __future__ import annotations

import shutil
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from configuard.datasets.manifest import write_manifest
from configuard.training import synthetic
from configuard.training.checkpoint import CheckpointMismatchError, load_checkpoint
from configuard.training.config import TrainingConfig
from configuard.training.logging_utils import read_jsonl
from configuard.training.runner import SMOKE_FACE_CONFIG, build_trainer, run_smoke
from configuard.training.splits import SplitLeakageError
from configuard.training.trainer import (
    NonFiniteLossError,
    Trainer,
    TrainingOutOfMemoryError,
    evaluate_checkpoint,
)

CUDA = torch.cuda.is_available()
SIX_GB_MB = 6 * 1024


@pytest.fixture
def data_root(tmp_path: Path) -> Path:
    root = tmp_path / "data"
    synthetic.write_synthetic_split(root, "train", 4, 4)
    synthetic.write_synthetic_split(root, "val", 2, 2)
    return root


def _config(data_root: Path, **overrides) -> TrainingConfig:
    base = TrainingConfig(
        encoder_name="mobilenetv4_conv_small", pretrained=False, seed=7, run_name="t",
        train_manifest=str(data_root / "train_manifest.jsonl"), val_manifest=str(data_root / "val_manifest.jsonl"),
        media_root=str(data_root / "media"), batch_size=4, epochs=2, lr=1e-3, warmup_steps=1,
        grad_accum_steps=1, amp=True, early_stopping_patience=None, log_every_n_steps=1, device="cpu",
    )
    return replace(base, **overrides)


def _trainer(config: TrainingConfig, tmp_path: Path) -> Trainer:
    return build_trainer(
        config, detector=synthetic.full_frame_detector(), face_config=SMOKE_FACE_CONFIG,
        checkpoint_dir=tmp_path / "ckpt", output_dir=tmp_path / "out", face_cache_dir=tmp_path / "crops",
    )


def _state_equal(a: torch.nn.Module, b: torch.nn.Module) -> bool:
    sa, sb = a.state_dict(), b.state_dict()
    return sa.keys() == sb.keys() and all(torch.equal(sa[k], sb[k]) for k in sa)


# ---------------------------------------------------------------- overfitting
def test_tiny_synthetic_set_can_be_overfitted(tmp_path, data_root):
    """Task 19/20: the obvious synthetic signal must be learnable to
    (near) zero loss - otherwise the loop, loss, or labels are broken.

    Full batch without replacement, so every batch holds both classes:
    balanced sampling draws WITH replacement, and on an 8-sample set that
    sometimes yields a single-class batch, where train-mode BatchNorm
    normalizes the class cue away and the loss spikes (measured, see
    docs/KNOWN_ISSUES.md). Eval-mode AUROC is checked (ranking), not the
    0.5-threshold metrics: after 12 from-scratch steps BN running stats
    haven't converged, which shifts the logit offset but not the order."""
    config = _config(
        data_root, epochs=12, val_manifest="", lr=1e-3, warmup_steps=0, batch_size=8,
        balance_source=False, balance_class=False,
    )
    trainer = _trainer(config, tmp_path)
    results = trainer.fit()

    assert results[-1].train_loss < 0.01 < results[0].train_loss
    assert all(b.train_loss <= a.train_loss * 1.5 for a, b in zip(results, results[1:]))
    metrics = evaluate_checkpoint(
        trainer.checkpoint_path("latest"), trainer.encoder, trainer.train_loader, torch.device("cpu")
    )
    assert metrics.threshold_free.auroc == 1.0


# ---------------------------------------------------------------- resume
def test_interrupted_training_resumes_bit_exactly_on_cpu(tmp_path, data_root):
    config = _config(data_root, epochs=3, grad_accum_steps=2)
    uninterrupted = _trainer(config, tmp_path / "a")
    full_results = uninterrupted.fit()

    interrupted = _trainer(config, tmp_path / "b")
    interrupted.fit(stop_after_epochs=2)
    assert interrupted.state.epoch == 2

    resumed = _trainer(config, tmp_path / "b")
    resumed_results = resumed.fit(resume_from=interrupted.checkpoint_path("latest"))

    assert [r.epoch for r in resumed_results] == [2]
    assert resumed.state.global_step == uninterrupted.state.global_step
    assert resumed_results[0].train_loss == full_results[2].train_loss
    assert _state_equal(resumed.encoder, uninterrupted.encoder)
    assert resumed.optimizer.state_dict()["state"].keys() == uninterrupted.optimizer.state_dict()["state"].keys()
    assert resumed.scheduler.get_last_lr() == uninterrupted.scheduler.get_last_lr()


def test_resume_restores_early_stopping_and_best_state(tmp_path, data_root):
    config = _config(data_root, epochs=3)
    first = _trainer(config, tmp_path)
    first.fit(stop_after_epochs=2)
    resumed = _trainer(config, tmp_path)
    resumed.resume(first.checkpoint_path("latest"))
    assert resumed.state == first.state


def test_resume_of_finished_run_runs_no_more_epochs(tmp_path, data_root):
    config = _config(data_root, epochs=1)
    first = _trainer(config, tmp_path)
    first.fit()
    again = _trainer(config, tmp_path)
    assert again.fit(resume_from=first.checkpoint_path("latest")) == []


@pytest.mark.parametrize(
    ("override", "expected"),
    [
        ({"lr": 5e-4}, "config.lr"),
        ({"batch_size": 2}, "config.batch_size"),
        ({"seed": 8}, "config.seed"),
        ({"epochs": 4}, "config.epochs"),
    ],
)
def test_resume_refuses_mismatched_config(tmp_path, data_root, override, expected):
    trainer = _trainer(_config(data_root, epochs=1), tmp_path)
    trainer.fit()
    other = _trainer(_config(data_root, **{"epochs": 1, **override}), tmp_path)
    with pytest.raises(CheckpointMismatchError, match=expected):
        other.resume(trainer.checkpoint_path("latest"))


def test_resume_refuses_changed_training_data(tmp_path, data_root):
    trainer = _trainer(_config(data_root, epochs=1), tmp_path)
    trainer.fit()

    manifest = data_root / "train_manifest.jsonl"
    samples = synthetic.build_synthetic_image_samples(data_root / "media", 5, 4, split="train")
    write_manifest(samples, manifest)  # one extra real sample -> different checksum
    other = _trainer(_config(data_root, epochs=1), tmp_path)
    with pytest.raises(CheckpointMismatchError, match="manifest_checksum"):
        other.resume(trainer.checkpoint_path("latest"))


def test_resume_refuses_changed_validation_data(tmp_path, data_root):
    trainer = _trainer(_config(data_root, epochs=1), tmp_path)
    trainer.fit()
    synthetic.write_synthetic_split(data_root, "val", 3, 2)
    other = _trainer(_config(data_root, epochs=1), tmp_path)
    with pytest.raises(CheckpointMismatchError, match="val_manifest_checksum"):
        other.resume(trainer.checkpoint_path("latest"))


def test_resume_refuses_different_preprocessing(tmp_path, data_root):
    trainer = _trainer(_config(data_root, epochs=1), tmp_path)
    trainer.fit()
    other = build_trainer(
        _config(data_root, epochs=1), detector=synthetic.full_frame_detector(),
        face_config=replace(SMOKE_FACE_CONFIG, margin_ratio=0.2),
        checkpoint_dir=tmp_path / "ckpt", output_dir=tmp_path / "out", face_cache_dir=tmp_path / "crops",
    )
    with pytest.raises(CheckpointMismatchError, match="preprocessing_version"):
        other.resume(trainer.checkpoint_path("latest"))


def test_resume_refuses_different_encoder(tmp_path, data_root):
    trainer = _trainer(_config(data_root, epochs=1), tmp_path)
    trainer.fit()
    other = _trainer(_config(data_root, epochs=1, encoder_name="efficientnet_b0"), tmp_path)
    with pytest.raises(CheckpointMismatchError, match="encoder_name"):
        other.resume(trainer.checkpoint_path("latest"))


# ---------------------------------------------------------------- checkpoints
def test_checkpoint_contains_every_required_field(tmp_path, data_root):
    trainer = _trainer(_config(data_root, epochs=1), tmp_path)
    trainer.fit()
    ckpt = load_checkpoint(trainer.checkpoint_path("latest"))

    assert ckpt.model_state and ckpt.optimizer_state["state"]
    assert ckpt.scheduler_state is not None
    assert ckpt.epoch == 1 and ckpt.global_step == 2
    assert {"python", "numpy", "torch_cpu"} <= set(ckpt.rng_state)
    p = ckpt.provenance
    assert p.config == trainer.config.to_dict()
    assert p.model_id == "timm/mobilenetv4_conv_small.e1200_r224_in1k"
    assert p.model_revision == "c9f31ac64483d7f0590db9edccb4418392a96eea"
    assert p.preprocessing_version == SMOKE_FACE_CONFIG.version_tag
    assert len(p.manifest_checksum) == 64 and p.val_manifest_checksum and len(p.val_manifest_checksum) == 64
    assert p.git_commit  # this test runs inside the repo's git checkout
    assert "val" in ckpt.metrics and "train_loss" in ckpt.metrics
    assert ckpt.train_state["epoch"] == 1


def test_best_and_latest_checkpoints_written_atomically(tmp_path, data_root):
    trainer = _trainer(_config(data_root, epochs=2), tmp_path)
    trainer.fit()
    assert trainer.checkpoint_path("latest").exists()
    assert trainer.checkpoint_path("best").exists()
    assert list((tmp_path / "ckpt").glob(".tmp_*")) == []


def test_amp_scaler_state_is_checkpointed_on_cuda_only(tmp_path, data_root):
    trainer = _trainer(_config(data_root, epochs=1), tmp_path)
    trainer.fit()
    assert trainer.use_amp is False  # AMP only engages on CUDA
    assert load_checkpoint(trainer.checkpoint_path("latest")).scaler_state is None


# ---------------------------------------------------------------- selection
def test_best_checkpoint_selected_by_val_auroc(tmp_path, data_root):
    trainer = _trainer(_config(data_root, epochs=2), tmp_path)
    results = trainer.fit()
    assert all(r.selection_metric_name == "val_auroc" for r in results)
    top = max(r.selection_value for r in results)
    tied = [r for r in results if r.selection_value == top]
    # exact AUROC ties are broken by lower validation loss
    assert trainer.state.best_epoch == min(tied, key=lambda r: (r.val_loss, r.epoch)).epoch
    assert trainer.state.best_metric == top
    assert load_checkpoint(trainer.checkpoint_path("best")).train_state["best_epoch"] == trainer.state.best_epoch


def test_selection_falls_back_to_val_loss_when_auroc_undefined(tmp_path, data_root):
    only_real = [s for s in synthetic.build_synthetic_image_samples(data_root / "media", 2, 0, split="val")]
    write_manifest(only_real, data_root / "val_manifest.jsonl")
    trainer = _trainer(_config(data_root, epochs=1), tmp_path)
    [result] = trainer.fit()
    assert result.val_metrics.threshold_free.auroc is None
    assert result.selection_metric_name == "neg_val_loss"
    assert result.selection_value == -result.val_loss


def test_selection_falls_back_to_train_loss_without_validation_set(tmp_path, data_root):
    trainer = _trainer(_config(data_root, epochs=1, val_manifest=""), tmp_path)
    [result] = trainer.fit()
    assert result.selection_metric_name == "neg_train_loss"
    assert result.val_metrics is None


def test_early_stopping_stops_after_patience(tmp_path, data_root):
    trainer = _trainer(_config(data_root, epochs=6, early_stopping_patience=2), tmp_path)
    trainer._selection_value = lambda *args: (0.5, "val_auroc")  # never improves after epoch 0
    results = trainer.fit()
    assert [r.epoch for r in results] == [0, 1, 2]
    assert trainer.state.epochs_without_improvement == 2


# ---------------------------------------------------------------- optimization details
def test_gradient_accumulation_counts_optimizer_steps(tmp_path, data_root):
    # 8 samples / batch 2 = 4 micro-batches; accum 3 -> 1 full step + 1 flushed partial step
    trainer = _trainer(_config(data_root, epochs=1, batch_size=2, grad_accum_steps=3), tmp_path)
    trainer.fit()
    assert trainer.state.global_step == 2


def test_freeze_backbone_trains_only_the_head(tmp_path, data_root):
    trainer = _trainer(_config(data_root, epochs=1, freeze_backbone=True), tmp_path)
    backbone_before = [p.detach().clone() for p in trainer.encoder.backbone.parameters()]
    head_before = trainer.encoder.head.weight.detach().clone()
    trainer.fit()
    assert all(torch.equal(a, b) for a, b in zip(backbone_before, trainer.encoder.backbone.parameters()))
    assert not torch.equal(head_before, trainer.encoder.head.weight)
    assert trainer.encoder.trainable_parameter_count() == trainer.encoder.head.weight.numel() + 1


def test_non_finite_loss_fails_safely_before_any_checkpoint(tmp_path, data_root, monkeypatch):
    def nan_logits(encoder, pixel_values):
        return encoder.forward_logits(pixel_values) * float("nan")

    monkeypatch.setattr("configuard.training.trainer.compute_logits_for_batch", nan_logits)
    trainer = _trainer(_config(data_root, epochs=1), tmp_path)
    with pytest.raises(NonFiniteLossError, match="Non-finite"):
        trainer.fit()
    assert not trainer.checkpoint_path("latest").exists()
    assert trainer.state.global_step == 0


def test_cuda_oom_gives_clear_diagnostic_and_never_changes_batch_size(tmp_path, data_root, monkeypatch):
    def oom(encoder, pixel_values):
        raise torch.cuda.OutOfMemoryError("CUDA out of memory. Tried to allocate 2.00 GiB")

    monkeypatch.setattr("configuard.training.trainer.compute_logits_for_batch", oom)
    trainer = _trainer(_config(data_root, epochs=1, batch_size=4), tmp_path)
    with pytest.raises(TrainingOutOfMemoryError, match="NOT changed automatically") as info:
        trainer.fit()
    assert "batch_size=4" in str(info.value)
    assert trainer.config.batch_size == 4 and trainer.train_loader.batch_size == 4
    assert isinstance(info.value.__cause__, torch.cuda.OutOfMemoryError)


# ---------------------------------------------------------------- logging / metrics
def test_structured_jsonl_and_csv_logs_record_required_fields(tmp_path, data_root):
    trainer = _trainer(_config(data_root, epochs=2), tmp_path)
    trainer.fit()
    records = read_jsonl(trainer.logger.jsonl_path)
    epoch_records = [r for r in records if r["split"] == "epoch"]
    train_records = [r for r in records if r["split"] == "train"]
    assert len(epoch_records) == 2 and train_records

    rec = epoch_records[-1]
    assert rec["loss"] is not None and rec["lr"] is not None and rec["epoch_time_seconds"] > 0
    assert "gpu_memory_mb" in rec  # None on CPU by design
    m = rec["metrics"]
    assert m["val_time_seconds"] >= 0 and m["mean_step_time_seconds"] > 0
    for key in ("auroc", "average_precision", "balanced_accuracy", "f1", "sensitivity", "specificity",
                "confusion_matrix"):
        assert key in m["val"]
    assert trainer.logger.csv_path("epoch").exists() and trainer.logger.csv_path("train").exists()
    header = trainer.logger.csv_path("epoch").read_text(encoding="utf-8").splitlines()[0]
    assert "metrics.val.auroc" in header and "metrics.val.confusion_matrix.true_positive" in header


def test_evaluation_only_path_matches_trainer_validation(tmp_path, data_root):
    trainer = _trainer(_config(data_root, epochs=1), tmp_path)
    [result] = trainer.fit()
    fresh = _trainer(_config(data_root, epochs=1), tmp_path / "fresh").encoder
    metrics = evaluate_checkpoint(trainer.checkpoint_path("latest"), fresh, trainer.val_loader, torch.device("cpu"))
    assert metrics.num_samples == 4
    assert metrics.threshold_free.auroc == pytest.approx(result.val_metrics.threshold_free.auroc)


def test_evaluation_refuses_encoder_mismatch(tmp_path, data_root):
    trainer = _trainer(_config(data_root, epochs=1), tmp_path)
    trainer.fit()
    other = _trainer(_config(data_root, epochs=1, encoder_name="efficientnet_b0"), tmp_path / "o").encoder
    with pytest.raises(ValueError, match="encoder"):
        evaluate_checkpoint(trainer.checkpoint_path("latest"), other, trainer.val_loader, torch.device("cpu"))


# ---------------------------------------------------------------- split hygiene
def test_build_trainer_refuses_source_leaking_between_train_and_val(tmp_path, data_root):
    train = synthetic.build_synthetic_image_samples(data_root / "media", 4, 4, split="train")
    write_manifest(train[:2], data_root / "val_manifest.jsonl")  # val reuses train sources
    with pytest.raises(SplitLeakageError, match="spans multiple splits|appears in both"):
        _trainer(_config(data_root), tmp_path)


# ---------------------------------------------------------------- both backbones
def test_efficientnet_b0_forward_backward_training_smoke_cpu(tmp_path, data_root):
    trainer = _trainer(_config(data_root, encoder_name="efficientnet_b0", epochs=1), tmp_path)
    [result] = trainer.fit()
    assert np.isfinite(result.train_loss)
    assert trainer.state.global_step == 2


def test_run_smoke_cpu_end_to_end(tmp_path):
    summary = run_smoke(
        device="cpu", encoder_name="mobilenetv4_conv_small", checkpoint_dir=tmp_path / "ckpt",
        output_dir=tmp_path / "out", epochs=2, batch_size=4, num_train_per_class=4, num_val_per_class=2,
        pretrained=False,
    )
    assert summary["disclaimer"].startswith("ENGINEERING_TEST_ONLY")
    assert summary["resume_consistency"]["max_abs_param_diff"] == 0.0
    assert summary["checkpoints"]["latest_bytes"] > 0
    assert Path(summary["summary_path"]).exists()
    for path in (summary["checkpoints"]["latest"], summary["logs"]["jsonl"]):
        assert Path(path).resolve().is_relative_to(tmp_path.resolve())


# ---------------------------------------------------------------- RTX 4050 (CUDA) smoke
@pytest.mark.skipif(not CUDA, reason="CUDA not available")
@pytest.mark.parametrize("encoder_name", ["mobilenetv4_conv_small", "efficientnet_b0"])
def test_cuda_mixed_precision_training_smoke(tmp_path, data_root, encoder_name):
    trainer = _trainer(_config(data_root, encoder_name=encoder_name, epochs=2, device="cuda"), tmp_path)
    results = trainer.fit()
    assert trainer.use_amp is True
    assert all(np.isfinite(r.train_loss) for r in results)
    peak = max(r.peak_gpu_reserved_mb for r in results)
    assert 0 < peak < 0.8 * SIX_GB_MB
    assert load_checkpoint(trainer.checkpoint_path("latest")).scaler_state is not None


@pytest.mark.skipif(not CUDA, reason="CUDA not available")
def test_cuda_resume_matches_uninterrupted_run(tmp_path, data_root):
    config = _config(data_root, epochs=2, device="cuda")
    full = _trainer(config, tmp_path / "a")
    full.fit()
    part = _trainer(config, tmp_path / "b")
    part.fit(stop_after_epochs=1)
    resumed = _trainer(config, tmp_path / "b")
    resumed.fit(resume_from=part.checkpoint_path("latest"))
    assert resumed.state.global_step == full.state.global_step
    diffs = [
        (a.float() - b.float()).abs().max().item()
        for a, b in zip(full.encoder.state_dict().values(), resumed.encoder.state_dict().values())
        if a.is_floating_point()
    ]
    # cuDNN deterministic mode is set, but exact GPU equality isn't guaranteed by CUDA; require near-equality.
    assert max(diffs) < 1e-4



@pytest.mark.skipif(not CUDA, reason="CUDA not available")
def test_amp_skipped_optimizer_steps_are_counted_and_most_steps_apply(tmp_path, data_root):
    """Regression: with GradScaler's default init scale every step of a
    6-step smoke run was skipped (fp16 overflow) and nothing reported it."""
    trainer = _trainer(_config(data_root, epochs=2, device="cuda"), tmp_path)
    trainer.fit()
    state = trainer.state
    assert 0 <= state.skipped_optimizer_steps < state.global_step
    assert trainer.scheduler.last_epoch == state.global_step - state.skipped_optimizer_steps
    assert trainer.optimizer.state_dict()["state"]  # parameters were really updated
    record = read_jsonl(trainer.logger.jsonl_path)[-1]
    assert record["metrics"]["optimizer_steps_skipped_amp"] == state.skipped_optimizer_steps


def test_cpu_training_never_skips_steps(tmp_path, data_root):
    trainer = _trainer(_config(data_root, epochs=1), tmp_path)
    trainer.fit()
    assert trainer.state.skipped_optimizer_steps == 0
    assert trainer.scheduler.last_epoch == trainer.state.global_step


def test_saturated_auroc_tie_is_broken_by_lower_val_loss(tmp_path, data_root):
    trainer = _trainer(_config(data_root, epochs=3), tmp_path)
    trainer._selection_value = lambda metrics, val_loss, train_loss: (1.0, "val_auroc")  # AUROC saturated
    results = trainer.fit()
    assert trainer.state.best_epoch == min(results, key=lambda r: r.val_loss).epoch

# ---------------------------------------------------------------- video + workers
def _write_video_split(root: Path, split: str) -> None:
    import subprocess

    from configuard.datasets.schema import Sample, SampleLabel, SampleMediaType

    media = root / "media"
    media.mkdir(parents=True, exist_ok=True)
    samples = []
    for label, pattern in ((SampleLabel.REAL, "testsrc"), (SampleLabel.FAKE, "testsrc2")):
        name = f"{split}_{label.value}.mp4"
        subprocess.run(
            ["ffmpeg", "-y", "-f", "lavfi", "-i", f"{pattern}=duration=2:size=64x64:rate=10",
             "-pix_fmt", "yuv420p", str(media / name)],
            capture_output=True, check=True, timeout=60,
        )
        samples.append(Sample(
            sample_id=f"synthetic:{name}", dataset_name="synthetic", dataset_version="v1",
            media_type=SampleMediaType.VIDEO, media_path=name, label=label,
            source_id=f"{split}_{name}", identity_id=f"{split}_{name}",
        ))
    write_manifest(samples, root / f"{split}_manifest.jsonl")


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not available")
def test_video_frame_training_end_to_end_cpu(tmp_path):
    root = tmp_path / "video"
    _write_video_split(root, "train")
    _write_video_split(root, "val")
    config = _config(root, epochs=1, batch_size=2, frames_per_video=4)
    trainer = _trainer(config, tmp_path)
    pixel_values, _ = next(iter(trainer.train_loader))
    assert pixel_values.shape == (2, 4, 3, 224, 224)  # (B, frames, C, H, W)
    [result] = trainer.fit()
    assert np.isfinite(result.train_loss) and result.val_metrics.num_samples == 2


def test_multiprocess_dataloader_workers_are_deterministic(tmp_path, data_root):
    """num_workers > 0 uses spawned worker processes on Windows: the
    dataset, detector, cache and worker_init_fn must all pickle, and the
    run must match a single-process run exactly."""
    single = _trainer(_config(data_root, epochs=1, num_workers=0), tmp_path / "w0")
    multi = _trainer(_config(data_root, epochs=1, num_workers=2), tmp_path / "w2")
    [a] = single.fit()
    [b] = multi.fit()
    assert a.train_loss == b.train_loss
    assert _state_equal(single.encoder, multi.encoder)
