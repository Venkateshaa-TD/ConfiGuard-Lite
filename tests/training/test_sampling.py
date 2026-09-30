"""Phase 5: source-balanced / class-balanced sampling, worker seeding."""

from __future__ import annotations

import random

import numpy as np
import torch

from configuard.datasets.schema import SampleLabel
from configuard.training.sampling import (
    build_weighted_sampler,
    compute_balanced_weights,
    make_worker_init_fn,
)
from tests.training.conftest import build_synthetic_image_manifest


def test_balanced_weights_favor_minority_class(tmp_path):
    media_root = tmp_path / "media"
    media_root.mkdir()
    samples = build_synthetic_image_manifest(media_root, num_real=8, num_fake=2)
    weights = compute_balanced_weights(samples, balance_source=False, balance_class=True)

    real_weights = [w for s, w in zip(samples, weights) if s.label is SampleLabel.REAL]
    fake_weights = [w for s, w in zip(samples, weights) if s.label is SampleLabel.FAKE]
    assert all(fw > rw for fw in fake_weights for rw in real_weights)


def test_balanced_weights_favor_minority_source(tmp_path):
    media_root = tmp_path / "media"
    media_root.mkdir()
    # 4 real samples from ONE shared source + 4 fake samples each from a unique source
    samples = build_synthetic_image_manifest(media_root, num_real=4, num_fake=4)
    samples_shared_source = [
        s if s.label is SampleLabel.FAKE else __import__("dataclasses").replace(s, source_id="shared")
        for s in samples
    ]
    weights = compute_balanced_weights(samples_shared_source, balance_source=True, balance_class=False)
    shared_source_weights = [w for s, w in zip(samples_shared_source, weights) if s.source_id == "shared"]
    unique_source_weights = [w for s, w in zip(samples_shared_source, weights) if s.source_id != "shared"]
    assert all(uw > sw for uw in unique_source_weights for sw in shared_source_weights)


def test_weighted_sampler_is_deterministic_given_a_seed(tmp_path):
    media_root = tmp_path / "media"
    media_root.mkdir()
    samples = build_synthetic_image_manifest(media_root, num_real=4, num_fake=4)

    sampler1 = build_weighted_sampler(samples, seed=42)
    sampler2 = build_weighted_sampler(samples, seed=42)
    assert list(sampler1) == list(sampler2)


def test_weighted_sampler_different_seed_usually_differs(tmp_path):
    media_root = tmp_path / "media"
    media_root.mkdir()
    samples = build_synthetic_image_manifest(media_root, num_real=10, num_fake=10)

    sampler1 = build_weighted_sampler(samples, seed=1)
    sampler2 = build_weighted_sampler(samples, seed=2)
    assert list(sampler1) != list(sampler2)


def test_worker_init_fn_is_deterministic_per_worker():
    init_fn = make_worker_init_fn(base_seed=100)

    init_fn(0)
    state_a = (random.random(), np.random.rand(), torch.rand(1).item())

    init_fn(0)
    state_b = (random.random(), np.random.rand(), torch.rand(1).item())

    assert state_a == state_b  # same worker_id + base_seed -> identical RNG stream


def test_worker_init_fn_differs_per_worker_id():
    init_fn = make_worker_init_fn(base_seed=100)

    init_fn(0)
    value_worker_0 = random.random()

    init_fn(1)
    value_worker_1 = random.random()

    assert value_worker_0 != value_worker_1
