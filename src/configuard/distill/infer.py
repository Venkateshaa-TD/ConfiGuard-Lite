"""Batch inference of a saved student checkpoint over crop rows (no augmentation)."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from configuard.distill.data import CropDataset, EpochSampler, single_thread_workers, worker_init
from configuard.distill.train import Normalizer
from configuard.models.registry import create_encoder


def load_student(checkpoint: str | Path, device: str = "cpu") -> tuple[torch.nn.Module, Normalizer, dict[str, Any]]:
    ck = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = create_encoder(ck["config"]["encoder_name"], pretrained=False)
    model.load_state_dict(ck["model"])
    model.eval().to(device)
    norm = Normalizer(tuple(ck["preprocess"]["mean"]), tuple(ck["preprocess"]["std"])).to(device)
    return model, norm, ck


@torch.inference_mode()
def predict_rows(model: torch.nn.Module, norm: Normalizer, rows: Sequence[dict[str, Any]], crop_root: str | Path,
                 device: str = "cuda", batch_size: int = 128, num_workers: int = 6, amp: bool = True) -> np.ndarray:
    """(N,) float32 student logits, aligned with `rows`."""
    ds = CropDataset(rows, crop_root, np.zeros(len(rows), np.float32), None, seed=0)
    single_thread_workers()
    loader = DataLoader(ds, batch_size=batch_size, sampler=EpochSampler(len(rows), len(rows), 0, None),
                        num_workers=num_workers, pin_memory=device == "cuda",
                        worker_init_fn=worker_init if num_workers else None)
    out = np.empty(len(rows), np.float32)
    use_amp = amp and device == "cuda"
    for x, _, _, idx in loader:
        with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
            z = model.forward_logits(norm(x.to(device, non_blocking=True)))
        out[idx.numpy()] = z.float().cpu().numpy()
    return out
