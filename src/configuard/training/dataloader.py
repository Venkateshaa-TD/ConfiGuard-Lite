"""Builds deterministic DataLoaders: fixed seeds, per-worker seeding,
optional source/class-balanced sampling, configurable batch size/workers.
"""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, Dataset

from configuard.datasets.schema import Sample
from configuard.training.sampling import build_weighted_sampler, make_worker_init_fn


def build_dataloader(
    dataset: Dataset,
    samples: list[Sample],
    *,
    batch_size: int,
    seed: int,
    num_workers: int = 0,
    balance_source: bool = True,
    balance_class: bool = True,
    shuffle_if_unbalanced: bool = True,
) -> DataLoader:
    """`samples` must be exactly the dataset's items, in order (e.g.
    ManifestImageDataset.samples) - balancing weights are per dataset
    index. For evaluation pass balance_*=False, shuffle_if_unbalanced=False."""
    if len(samples) != len(dataset):  # type: ignore[arg-type]
        raise ValueError(
            f"samples ({len(samples)}) must match the dataset's items ({len(dataset)}) one-to-one - "  # type: ignore[arg-type]
            "pass dataset.samples, not the unfiltered manifest."
        )
    generator = torch.Generator().manual_seed(seed)

    use_sampler = balance_source or balance_class
    sampler = (
        build_weighted_sampler(samples, seed, balance_source=balance_source, balance_class=balance_class)
        if use_sampler
        else None
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=(sampler is None and shuffle_if_unbalanced),
        num_workers=num_workers,
        worker_init_fn=make_worker_init_fn(seed) if num_workers > 0 else None,
        generator=generator,
        drop_last=False,
        persistent_workers=False,
    )
