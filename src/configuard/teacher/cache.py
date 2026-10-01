"""Resumable, stale-rejecting cache of frozen-teacher logits for crop manifests.

Layout (one root per teacher tag + crop-manifest identity, on D:):
    <root>/<teacher_tag>/<split>/meta.json            what this cache is for
    <root>/<teacher_tag>/<split>/shard_<i:05d>.npz    rows [i*S, (i+1)*S) of the manifest
    <root>/<teacher_tag>/<split>/teacher_logits.jsonl consolidated, manifest order

Shard i always covers the same manifest rows, so a resumed run rebuilds
exactly the shards that are missing. Each shard stores the crop SHA-256s
it covers, and it is REJECTED (StaleTeacherCacheError) if:
- `meta.json` names another teacher tag, crop manifest SHA-256 or shard size;
- a shard's tag differs;
- a shard's crop SHA-256 list differs from the manifest rows it should cover;
- the logits are non-finite or misshapen.
Crop bytes are re-hashed as they are read for inference, and a mismatch
with the manifest's `crop_sha256` stops the run.

The official TEST split is refused unconditionally (ProtectedSplitError):
this module never opens it.
"""

from __future__ import annotations

import hashlib
import io
import json
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from configuard.crops.store import atomic_write_bytes, canonical_json

ALLOWED_SPLITS = ("train", "val")
PROTECTED_SPLITS = ("test",)


class StaleTeacherCacheError(Exception):
    """Existing cache entries do not match the current teacher/config/manifest."""


class ProtectedSplitError(Exception):
    """An attempt was made to touch a protected split (test)."""


class CropIntegrityError(Exception):
    """A crop's bytes do not match its manifest SHA-256."""


@dataclass(frozen=True)
class TeacherCacheConfig:
    teacher_repo: str
    teacher_revision: str
    teacher_weights_sha256: str
    head: str
    preprocessing: str
    attn_implementation: str
    autocast_dtype: str  # "float16" or "none"
    batch_size: int
    transformers_version: str
    torch_version: str
    shard_size: int = 1024
    schema: str = "p6a-1"

    @property
    def tag(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True).encode("utf-8")
        return "t6a-" + hashlib.sha256(payload).hexdigest()[:16]


def require_allowed_split(split: str) -> None:
    if split in PROTECTED_SPLITS:
        raise ProtectedSplitError(f"split {split!r} is protected and must stay untouched")
    if split not in ALLOWED_SPLITS:
        raise ValueError(f"unknown split {split!r}; allowed: {ALLOWED_SPLITS}")


def _npz_bytes(**arrays: np.ndarray) -> bytes:
    buf = io.BytesIO()
    np.savez(buf, **arrays)
    return buf.getvalue()


LogitFn = Callable[[list[bytes]], np.ndarray]  # PNG bytes -> (n, 2) float32 logits


class TeacherLogitCache:
    def __init__(self, root: str | Path, config: TeacherCacheConfig, split: str, rows: Sequence[dict[str, Any]],
                 manifest_sha256: str, crop_root: str | Path) -> None:
        require_allowed_split(split)
        bad = [r["sample_id"] for r in rows if r["metadata"]["split"] != split]
        if bad:
            raise ProtectedSplitError(f"{len(bad)} rows are not in split {split!r} (e.g. {bad[:2]})")
        self.config, self.split, self.rows = config, split, list(rows)
        self.crop_root = Path(crop_root)
        self.dir = Path(root) / config.tag / split
        self.meta = {
            "teacher_tag": config.tag, "teacher_config": asdict(config), "split": split,
            "crop_manifest_sha256": manifest_sha256, "rows": len(self.rows), "shard_size": config.shard_size,
        }
        meta_path = self.dir / "meta.json"
        if meta_path.exists():
            existing = json.loads(meta_path.read_text(encoding="utf-8"))
            if existing != self.meta:
                raise StaleTeacherCacheError(
                    f"{self.dir}: cache was built for tag {existing.get('teacher_tag')} / manifest "
                    f"{existing.get('crop_manifest_sha256')}; refusing to mix with {config.tag} / {manifest_sha256}")
        else:
            atomic_write_bytes(meta_path, canonical_json(self.meta))

    @property
    def num_shards(self) -> int:
        return (len(self.rows) + self.config.shard_size - 1) // self.config.shard_size

    def shard_rows(self, i: int) -> list[dict[str, Any]]:
        s = self.config.shard_size
        return self.rows[i * s:(i + 1) * s]

    def shard_path(self, i: int) -> Path:
        return self.dir / f"shard_{i:05d}.npz"

    def load_shard(self, i: int) -> np.ndarray | None:
        """Validated logits for shard i, or None if not cached yet."""
        path = self.shard_path(i)
        if not path.exists():
            return None
        with np.load(path, allow_pickle=False) as z:
            tag, shas, logits = str(z["teacher_tag"]), [str(s) for s in z["crop_sha256"]], z["logits"]
        expected = [r["crop_sha256"] for r in self.shard_rows(i)]
        if tag != self.config.tag:
            raise StaleTeacherCacheError(f"{path}: teacher tag {tag} != {self.config.tag}")
        if shas != expected:
            raise StaleTeacherCacheError(f"{path}: crop SHA-256 list does not match manifest rows")
        if logits.shape != (len(expected), 2) or logits.dtype != np.float32 or not np.isfinite(logits).all():
            raise StaleTeacherCacheError(f"{path}: malformed logits {logits.shape} {logits.dtype}")
        return logits

    def read_crop(self, row: dict[str, Any]) -> bytes:
        data = (self.crop_root / row["crop_path"]).read_bytes()
        if hashlib.sha256(data).hexdigest() != row["crop_sha256"]:
            raise CropIntegrityError(f"{row['crop_path']}: bytes do not match manifest crop_sha256")
        return data

    def fill(self, logit_fn: LogitFn, progress: Callable[[int, int], None] | None = None) -> dict[str, int]:
        """Compute missing shards (atomic writes). Returns counts."""
        done = computed = 0
        for i in range(self.num_shards):
            if self.load_shard(i) is not None:
                done += 1
                continue
            rows = self.shard_rows(i)
            logits = np.asarray(logit_fn([self.read_crop(r) for r in rows]), dtype=np.float32)
            if logits.shape != (len(rows), 2) or not np.isfinite(logits).all():
                raise RuntimeError(f"teacher produced malformed logits for shard {i}: {logits.shape}")
            atomic_write_bytes(self.shard_path(i), _npz_bytes(
                teacher_tag=np.array(self.config.tag), crop_sha256=np.array([r["crop_sha256"] for r in rows]),
                logits=logits))
            computed += 1
            if progress:
                progress(done + computed, self.num_shards)
        return {"shards_total": self.num_shards, "shards_resumed": done, "shards_computed": computed}

    def consolidate(self) -> Path:
        """Write teacher_logits.jsonl in manifest order (refuses if incomplete)."""
        lines = []
        for i in range(self.num_shards):
            logits = self.load_shard(i)
            if logits is None:
                raise StaleTeacherCacheError(f"shard {i} missing; run fill() first")
            for row, (l0, l1) in zip(self.shard_rows(i), logits.tolist()):
                lines.append(json.dumps({
                    "crop_sha256": row["crop_sha256"], "crop_path": row["crop_path"], "sample_id": row["sample_id"],
                    "slot": row["slot"], "teacher_tag": self.config.tag, "logits": [l0, l1],
                }, sort_keys=True) + "\n")
        out = self.dir / "teacher_logits.jsonl"
        atomic_write_bytes(out, "".join(lines).encode("utf-8"))
        return out

    def all_logits(self) -> np.ndarray:
        return np.concatenate([self.load_shard(i) for i in range(self.num_shards)])
