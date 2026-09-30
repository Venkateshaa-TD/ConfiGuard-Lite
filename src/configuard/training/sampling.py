"""Deterministic sampling: fixed-seed source-balanced and class-balanced
weighted sampling, plus deterministic DataLoader worker seeding.
"""

from __future__ import annotations

import random
from collections import Counter
from functools import partial
from typing import Callable

import numpy as np
import torch
from torch.utils.data import WeightedRandomSampler

from configuard.datasets.schema import Sample


def compute_balanced_weights(
    samples: list[Sample], balance_source: bool = True, balance_class: bool = True
) -> list[float]:
    """Per-sample weight = product of inverse group frequencies for each
    enabled balancing axis, so WeightedRandomSampler draws sources/classes
    roughly uniformly regardless of how imbalanced the manifest is."""
    source_counts = Counter(s.source_id for s in samples)
    class_counts = Counter(s.label for s in samples)

    weights = []
    for sample in samples:
        weight = 1.0
        if balance_source:
            weight /= source_counts[sample.source_id]
        if balance_class:
            weight /= class_counts[sample.label]
        weights.append(weight)
    return weights


def build_weighted_sampler(
    samples: list[Sample], seed: int, balance_source: bool = True, balance_class: bool = True
) -> WeightedRandomSampler:
    weights = compute_balanced_weights(samples, balance_source, balance_class)
    generator = torch.Generator().manual_seed(seed)
    return WeightedRandomSampler(weights, num_samples=len(samples), replacement=True, generator=generator)


def _worker_init_fn(worker_id: int, base_seed: int) -> None:
    seed = (base_seed + worker_id) % (2**32)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_worker_init_fn(base_seed: int) -> Callable[[int], None]:
    """Deterministic per-worker seeding: worker N gets seed (base_seed + N),
    so re-running with the same seed reproduces the same per-worker RNG
    streams regardless of num_workers."""
    return partial(_worker_init_fn, base_seed=base_seed)
