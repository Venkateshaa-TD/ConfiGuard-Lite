"""Phase 5: split-leakage guard, structured logger, runner dataset guards."""

from __future__ import annotations

import csv
from dataclasses import replace
from pathlib import Path

import pytest

from configuard.datasets.schema import SampleMediaType
from configuard.models.encoder import DeepfakeVisualEncoder
from configuard.models.registry import ENCODER_SPECS
from configuard.training import synthetic
from configuard.training.dataloader import build_dataloader
from configuard.training.datasets import ManifestImageDataset
from configuard.training.logging_utils import ExperimentLogger, LogRecord, read_jsonl
from configuard.training.paths import resolve_cache_dir
from configuard.training.runner import build_manifest_dataset
from configuard.training.splits import SplitLeakageError, assert_no_cross_split_leakage, find_cross_split_leakage


def test_clean_splits_have_no_leakage(tmp_path: Path):
    train = synthetic.build_synthetic_image_samples(tmp_path, 3, 3, split="train")
    val = synthetic.build_synthetic_image_samples(tmp_path, 2, 2, split="val")
    calibration = synthetic.build_synthetic_image_samples(tmp_path, 1, 1, split="calib")
    assert find_cross_split_leakage({"train": train, "val": val, "calibration": calibration}) == []


def test_frames_sharing_a_source_across_splits_are_rejected(tmp_path: Path):
    """Two frame-level samples of the same source video in train and test."""
    train = synthetic.build_synthetic_image_samples(tmp_path, 1, 0, split="train")
    frame_in_test = replace(train[0], sample_id="synthetic:frame_7_of_same_video")
    with pytest.raises(SplitLeakageError, match="source_id"):
        assert_no_cross_split_leakage({"train": train, "test": [frame_in_test]})


def test_same_sample_in_two_splits_is_rejected(tmp_path: Path):
    train = synthetic.build_synthetic_image_samples(tmp_path, 1, 1, split="train")
    problems = find_cross_split_leakage({"train": train, "val": train[:1]})
    assert any("appears in both" in p for p in problems)


def test_identity_shared_across_splits_is_rejected(tmp_path: Path):
    train = synthetic.build_synthetic_image_samples(tmp_path, 1, 0, split="train")
    val = synthetic.build_synthetic_image_samples(tmp_path, 1, 0, split="val")
    val = [replace(val[0], identity_id=train[0].identity_id)]
    assert any("identity" in p for p in find_cross_split_leakage({"train": train, "val": val}))


def test_logger_writes_jsonl_and_per_split_csv_and_appends_on_resume(tmp_path: Path):
    logger = ExperimentLogger(tmp_path, "run")
    logger.log(LogRecord(epoch=0, step=1, split="train", loss=0.5, lr=1e-3))
    logger.log(LogRecord(epoch=0, step=1, split="epoch", loss=0.5, metrics={"val": {"auroc": 0.9, "cm": {"tp": 1}}}))

    resumed = ExperimentLogger(tmp_path, "run")  # new process after an interruption
    resumed.log(LogRecord(epoch=1, step=2, split="epoch", loss=0.4, metrics={"val": {"auroc": 0.95, "cm": {"tp": 2}}}))

    assert [r["split"] for r in read_jsonl(logger.jsonl_path)] == ["train", "epoch", "epoch"]
    with logger.csv_path("epoch").open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert [r["metrics.val.auroc"] for r in rows] == ["0.9", "0.95"]
    assert rows[1]["metrics.val.cm.tp"] == "2"
    assert "metrics.val.auroc" not in logger.csv_path("train").read_text(encoding="utf-8")


def test_mixed_image_and_video_manifest_is_refused(tmp_path: Path):
    images = synthetic.build_synthetic_image_samples(tmp_path, 1, 1)
    mixed = images + [replace(images[0], sample_id="v", media_type=SampleMediaType.VIDEO, media_path="v.mp4")]
    with pytest.raises(ValueError, match="mixes IMAGE and VIDEO"):
        build_manifest_dataset(mixed, tmp_path, 8, None, None, None, None)  # type: ignore[arg-type]


def test_dataloader_refuses_samples_not_matching_dataset(tmp_path: Path, full_frame_detector, face_cache, face_config):
    samples = synthetic.build_synthetic_image_samples(tmp_path, 2, 2)
    preprocess = DeepfakeVisualEncoder(ENCODER_SPECS["mobilenetv4_conv_small"], pretrained=False).resolve_preprocess_config()
    dataset = ManifestImageDataset(samples, tmp_path, full_frame_detector, face_cache, face_config, preprocess)
    with pytest.raises(ValueError, match="one-to-one"):
        build_dataloader(dataset, samples[:3], batch_size=2, seed=0)


def test_resolve_cache_dir_uses_env_var(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("CONFIGUARD_CACHE_DIR", str(tmp_path / "cache"))
    assert resolve_cache_dir() == tmp_path / "cache"
