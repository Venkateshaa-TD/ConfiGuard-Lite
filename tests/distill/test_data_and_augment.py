"""Phase 6b: crop rows, teacher alignment, balanced sampler, augmentation."""

from __future__ import annotations

import json
from collections import Counter

import numpy as np
import pytest

from configuard.distill.augment import AugmentConfig, augment, augment_rng
from configuard.distill.data import (
    CropDataset,
    EpochSampler,
    TeacherLogitMismatchError,
    balanced_weights,
    load_crop_rows,
    load_teacher_margins,
    method_of,
)
from configuard.teacher.cache import ProtectedSplitError

from .conftest import CROP_TAG, TEACHER_TAG


def test_rows_and_teacher_margins_align(synthetic_store):
    rows, sha = load_crop_rows(synthetic_store, CROP_TAG, "train")
    m = load_teacher_margins(synthetic_store.parent / "teacher", TEACHER_TAG, "train", rows, sha)
    assert len(rows) == len(m) == 5 * 3 * 2
    assert all((mm > 0) == (r["label"] == "fake") for r, mm in zip(rows, m))


def test_test_split_is_refused_before_reading(synthetic_store):
    with pytest.raises(ProtectedSplitError):
        load_crop_rows(synthetic_store, CROP_TAG, "test")
    with pytest.raises(ProtectedSplitError):
        load_teacher_margins(synthetic_store.parent / "teacher", TEACHER_TAG, "test", [], "x")


def test_mismatched_teacher_cache_is_rejected(synthetic_store):
    rows, sha = load_crop_rows(synthetic_store, CROP_TAG, "train")
    teacher = synthetic_store.parent / "teacher"
    with pytest.raises(TeacherLogitMismatchError):
        load_teacher_margins(teacher, TEACHER_TAG, "train", rows, "0" * 64)  # other crop manifest
    with pytest.raises(TeacherLogitMismatchError):
        load_teacher_margins(teacher, TEACHER_TAG, "train", rows[::-1], sha)  # misaligned rows
    path = teacher / TEACHER_TAG / "train" / "teacher_logits.jsonl"
    lines = path.read_text().splitlines()
    bad = json.loads(lines[0]) | {"teacher_tag": "t6a-other"}
    path.write_text("\n".join([json.dumps(bad)] + lines[1:]) + "\n")
    with pytest.raises(TeacherLogitMismatchError):
        load_teacher_margins(teacher, TEACHER_TAG, "train", rows, sha)


def test_tampered_crop_manifest_is_rejected(synthetic_store):
    p = synthetic_store / "manifests" / CROP_TAG / "full" / "crops_val.jsonl"
    p.write_text(p.read_text() + "\n")
    with pytest.raises(TeacherLogitMismatchError, match="summary"):
        load_crop_rows(synthetic_store, CROP_TAG, "val")


def test_balanced_weights_give_half_real_and_equal_methods():
    rows = [{"label": "real", "metadata": {"method": None}}] * 10 + [
        {"label": "fake", "metadata": {"method": m}} for m, n in
        (("Deepfakes", 40), ("Face2Face", 10), ("FaceSwap", 20), ("NeuralTextures", 5)) for _ in range(n)]
    w = balanced_weights(rows)
    mass = Counter()
    for r, wi in zip(rows, w):
        mass[method_of(r)] += wi
    assert mass["original"] == pytest.approx(0.5)
    assert all(mass[m] == pytest.approx(0.125) for m in ("Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"))


def test_sampler_is_seeded_per_epoch():
    w = np.full(100, 0.01)
    a, b = EpochSampler(100, 50, 7, w), EpochSampler(100, 50, 7, w)
    assert a.indices() == b.indices()
    first = a.indices()
    a.set_epoch(1)
    assert a.indices() != first
    assert list(EpochSampler(5, 5, 7, None)) == [(i, 0) for i in range(5)]


def test_augment_is_deterministic_mild_and_label_free():
    img = np.random.default_rng(0).integers(0, 255, (224, 224, 3), dtype=np.uint8)
    cfg = AugmentConfig(blur_prob=1.0, hjitter_prob=1.0)
    out1, out2 = augment(img, cfg, augment_rng(1, 2, 3)), augment(img, cfg, augment_rng(1, 2, 3))
    assert out1.shape == img.shape and out1.dtype == np.uint8
    assert np.array_equal(out1, out2)
    assert not np.array_equal(out1, augment(img, cfg, augment_rng(1, 2, 4)))
    # vertical geometry untouched: jitter only maps x, so a horizontally constant image is unchanged
    stripes = np.repeat(np.arange(224, dtype=np.uint8)[:, None, None], 224, axis=1).repeat(3, axis=2)
    jit = augment(stripes, AugmentConfig(blur_prob=0.0, hjitter_prob=1.0), augment_rng(0, 0, 0))
    assert np.array_equal(jit, stripes)
    noop = augment(img, AugmentConfig(blur_prob=0.0, hjitter_prob=0.0), augment_rng(0, 0, 0))
    assert np.array_equal(noop, img)


def test_dataset_val_is_unaugmented_and_train_is_reproducible(synthetic_store):
    rows, sha = load_crop_rows(synthetic_store, CROP_TAG, "train")
    m = load_teacher_margins(synthetic_store.parent / "teacher", TEACHER_TAG, "train", rows, sha)
    plain = CropDataset(rows, synthetic_store, m, None, seed=1)
    aug = CropDataset(rows, synthetic_store, m, AugmentConfig(blur_prob=1.0, hjitter_prob=1.0), seed=1)
    x0, y0, t0, i0 = plain[(3, 0)]
    assert x0.shape == (3, 224, 224) and str(x0.dtype) == "torch.uint8" and i0 == 3
    assert np.array_equal(plain[(3, 0)][0].numpy(), plain[(3, 5)][0].numpy())  # no epoch dependence without augment
    assert np.array_equal(aug[(3, 2)][0].numpy(), aug[(3, 2)][0].numpy())
    assert not np.array_equal(aug[(3, 2)][0].numpy(), aug[(3, 3)][0].numpy())
    assert y0 == (1.0 if rows[3]["label"] == "fake" else 0.0) and t0 == pytest.approx(m[3])
