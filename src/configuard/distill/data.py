"""Crop rows, cached teacher logits, the crop Dataset and the balanced sampler.

- Rows come from the Phase 5d model-facing crop manifests, whose SHA-256
  must match the Phase 5d extraction summary. Only `train` and `val` are
  accepted; `test` raises ProtectedSplitError before anything is read.
- Teacher logits come from the Phase 6a consolidated cache. They are
  aligned row by row and must agree on crop SHA-256, sample id and slot,
  with the cache's `meta.json` naming the same crop-manifest SHA-256 and
  teacher tag. GenD itself is never loaded.
- The sampler yields (index, epoch) pairs so that augmentation can be
  seeded per (seed, epoch, index) while DataLoader workers stay persistent.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from configuard.distill.augment import AugmentConfig, augment, augment_rng
from configuard.teacher.cache import ProtectedSplitError, require_allowed_split

REAL = "original"


class TeacherLogitMismatchError(Exception):
    """Cached teacher logits do not belong to these crop rows / this teacher."""


def method_of(row: dict[str, Any]) -> str:
    return row["metadata"]["method"] or REAL


def load_crop_rows(store: str | Path, crop_tag: str, split: str) -> tuple[list[dict[str, Any]], str]:
    """Rows of crops_<split>.jsonl plus its SHA-256 (checked against the Phase 5d summary)."""
    require_allowed_split(split)
    manifests = Path(store) / "manifests" / crop_tag / "full"
    name = f"crops_{split}.jsonl"
    data = (manifests / name).read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    summary = json.loads((manifests / "extraction_summary.json").read_text(encoding="utf-8"))
    if digest != summary["manifest_sha256"][name]:
        raise TeacherLogitMismatchError(f"{name} sha256 {digest} != Phase 5d summary")
    rows = [json.loads(line) for line in data.decode("utf-8").splitlines()]
    bad = [r["sample_id"] for r in rows if r["metadata"]["split"] != split]
    if bad:
        raise ProtectedSplitError(f"{len(bad)} rows of {name} are not in split {split!r}")
    return rows, digest


def load_teacher_margins(cache_dir: str | Path, teacher_tag: str, split: str,
                         rows: Sequence[dict[str, Any]], manifest_sha256: str) -> np.ndarray:
    """(N,) float32 teacher margins logit[fake] - logit[real], aligned to `rows`.
    P_teacher(fake) = sigmoid(margin), identical to GenD's 2-way softmax."""
    require_allowed_split(split)
    d = Path(cache_dir) / teacher_tag / split
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    if meta["teacher_tag"] != teacher_tag or meta["crop_manifest_sha256"] != manifest_sha256:
        raise TeacherLogitMismatchError(
            f"{d}: cache is for tag {meta['teacher_tag']} / manifest {meta['crop_manifest_sha256']}, "
            f"expected {teacher_tag} / {manifest_sha256}")
    lines = (d / "teacher_logits.jsonl").read_text(encoding="utf-8").splitlines()
    if len(lines) != len(rows):
        raise TeacherLogitMismatchError(f"{d}: {len(lines)} teacher rows for {len(rows)} crops")
    out = np.empty(len(rows), np.float32)
    for i, (line, row) in enumerate(zip(lines, rows)):
        t = json.loads(line)
        if (t["crop_sha256"], t["sample_id"], t["slot"], t["teacher_tag"]) != (
                row["crop_sha256"], row["sample_id"], row["slot"], teacher_tag):
            raise TeacherLogitMismatchError(f"{d}: row {i} does not match crop {row['crop_path']}")
        out[i] = t["logits"][1] - t["logits"][0]
    if not np.isfinite(out).all():
        raise TeacherLogitMismatchError(f"{d}: non-finite teacher logits")
    return out


def balanced_weights(rows: Sequence[dict[str, Any]], balance_class: bool = True,
                     balance_source: bool = True) -> np.ndarray:
    """weight = 1/n(label) * 1/n(method | label): with both on, real draws 1/2 and
    each manipulation 1/8 of the mass. The source axis is the manipulation method."""
    labels = [r["label"] for r in rows]
    methods = [(r["label"], method_of(r)) for r in rows]
    n_label, n_method = Counter(labels), Counter(methods)
    per_label_methods = Counter(lab for lab, _ in n_method)
    w = np.ones(len(rows), np.float64)
    for i, (lab, m) in enumerate(zip(labels, methods)):
        if balance_class and balance_source:
            w[i] = 1.0 / (len(n_label) * per_label_methods[lab] * n_method[m])
        elif balance_class:
            w[i] = 1.0 / n_label[lab]
        elif balance_source:
            w[i] = 1.0 / n_method[m]
    return w / w.sum()


class EpochSampler(Sampler):
    """Weighted with-replacement draws (training) or sequential order (eval).
    Each epoch's draw depends only on (seed, epoch), so it is reproducible and
    identical across runs that share the seed."""

    def __init__(self, num_rows: int, num_samples: int, seed: int, weights: np.ndarray | None = None) -> None:
        self.num_rows, self.num_samples, self.seed = num_rows, num_samples, seed
        self.weights = None if weights is None else torch.as_tensor(weights, dtype=torch.float64)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def indices(self) -> list[int]:
        if self.weights is None:
            return list(range(self.num_rows))
        g = torch.Generator().manual_seed(self.seed * 100_003 + self.epoch)
        return torch.multinomial(self.weights, self.num_samples, replacement=True, generator=g).tolist()

    def __iter__(self) -> Iterator[tuple[int, int]]:
        return iter([(i, self.epoch) for i in self.indices()])

    def __len__(self) -> int:
        return self.num_rows if self.weights is None else self.num_samples


class CropDataset(Dataset):
    """Item (index, epoch) -> (uint8 RGB CHW crop, label 0/1, teacher margin, index).
    Normalisation happens on the device (configuard.distill.train.Normalizer)."""

    def __init__(self, rows: Sequence[dict[str, Any]], crop_root: str | Path, teacher_margins: np.ndarray,
                 augment_cfg: AugmentConfig | None, seed: int) -> None:
        if len(teacher_margins) != len(rows):
            raise TeacherLogitMismatchError("teacher margins must align with rows")
        self.paths = [str(Path(crop_root) / r["crop_path"]) for r in rows]
        self.labels = np.array([r["label"] == "fake" for r in rows], np.float32)
        self.margins = np.asarray(teacher_margins, np.float32)
        self.augment_cfg, self.seed = augment_cfg, seed

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, item: tuple[int, int] | int) -> tuple[torch.Tensor, float, float, int]:
        index, epoch = item if isinstance(item, tuple) else (item, -1)
        img = cv2.imdecode(np.fromfile(self.paths[index], np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise OSError(f"could not decode {self.paths[index]}")
        if self.augment_cfg is not None:
            img = augment(img, self.augment_cfg, augment_rng(self.seed, epoch, index))
        rgb = np.ascontiguousarray(img[:, :, ::-1].transpose(2, 0, 1))
        return torch.from_numpy(rgb), float(self.labels[index]), float(self.margins[index]), index
