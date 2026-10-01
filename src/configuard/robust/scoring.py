"""Memory-safe, resumable scoring of stress-suite conditions.

Conditions are scored strictly one after another through a single
sequential DataLoader (few workers), and each condition's logits are
written atomically to `<out_dir>/<condition>.npy` the moment its last crop
is scored. A rerun loads the finished files and scores only what is left.
A `RamGuard` is checked every few batches; if available RAM drops below
its floor the loader's workers are shut down, completed conditions stay on
disk, and LowMemoryError propagates to the caller.
"""

from __future__ import annotations

import io
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from configuard.crops.store import atomic_write_bytes
from configuard.distill.data import CropDataset, EpochSampler, single_thread_workers, worker_init
from configuard.memory_guard import RamGuard


def _save(path: Path, arr: np.ndarray) -> None:
    buf = io.BytesIO()
    np.save(buf, arr)
    atomic_write_bytes(path, buf.getvalue())


def load_done(out_dir: Path, names: Sequence[str], sizes: dict[str, int]) -> dict[str, np.ndarray]:
    done = {}
    for n in names:
        f = out_dir / f"{n}.npy"
        if f.exists():
            arr = np.load(f)
            if arr.shape == (sizes[n],) and np.isfinite(arr).all():
                done[n] = arr
    return done


@torch.inference_mode()
def score_conditions(model: torch.nn.Module, norm: Callable[[torch.Tensor], torch.Tensor],
                     rows_by_condition: dict[str, list[dict[str, Any]]], out_dir: str | Path, *,
                     device: str = "cuda", num_workers: int = 4, batch_size: int = 128,
                     guard: RamGuard | None = None, check_every: int = 10,
                     log: Callable[[str], None] = print) -> dict[str, np.ndarray]:
    """Rows must carry absolute crop paths. Returns {condition: (N,) logits} for all conditions."""
    out_dir = Path(out_dir)
    names = list(rows_by_condition)
    sizes = {n: len(r) for n, r in rows_by_condition.items()}
    results = load_done(out_dir, names, sizes)
    todo = [n for n in names if n not in results]
    if todo:
        log(f"resuming: {len(results)} done, {len(todo)} to score ({', '.join(todo)})")
    if not todo:
        return results
    if guard is not None:
        guard.check()
    flat = [r for n in todo for r in rows_by_condition[n]]
    bounds, start = [], 0
    for n in todo:
        bounds.append((n, start, start + sizes[n]))
        start += sizes[n]
    single_thread_workers()
    loader = DataLoader(CropDataset(flat, "", np.zeros(len(flat), np.float32), None, seed=0),
                        batch_size=batch_size, sampler=EpochSampler(len(flat), len(flat), 0, None),
                        num_workers=num_workers, pin_memory=device == "cuda",
                        worker_init_fn=worker_init if num_workers else None)
    buf = np.empty(len(flat), np.float32)
    filled = 0
    current = 0  # index into bounds
    use_amp = device == "cuda"
    it = iter(loader)
    try:
        for b, (x, _, _, idx) in enumerate(it):
            with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                z = model.forward_logits(norm(x.to(device, non_blocking=True)))
            buf[idx.numpy()] = z.float().cpu().numpy()
            filled = int(idx.max()) + 1  # sequential sampler + in-order loader
            while current < len(bounds) and filled >= bounds[current][2]:
                n, lo, hi = bounds[current]
                results[n] = buf[lo:hi].copy()
                _save(out_dir / f"{n}.npy", results[n])
                log(f"  {n}: saved ({hi - lo} crops)")
                current += 1
            if guard is not None and b % check_every == 0:
                guard.check()
    finally:
        shutdown = getattr(it, "_shutdown_workers", None)
        if shutdown is not None:
            shutdown()
    return results
