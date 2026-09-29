"""Versioned face-crop cache, keyed by (input SHA-256, frame index, track
ID, preprocessing config version). Writes are atomic (temp file + os.replace
in the same directory, so it's a same-filesystem rename on both POSIX and
Windows) and the cache supports a configurable byte-size limit with
oldest-first eviction.
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


class CacheError(Exception):
    """Raised only for genuine I/O/encoding failures, never for a cache miss."""


@dataclass(frozen=True)
class CacheKey:
    input_sha256: str
    frame_index: int
    track_id: int
    config_version: str

    def relative_path(self) -> Path:
        # Two-level fan-out on the hash prefix keeps any one directory from
        # accumulating too many entries.
        return (
            Path(self.config_version)
            / self.input_sha256[:2]
            / self.input_sha256
            / f"track{self.track_id}_frame{self.frame_index:06d}.png"
        )


class FaceCropCache:
    """A directory-backed cache of aligned face crops (PNG-encoded)."""

    def __init__(self, root: str | Path, max_bytes: int | None = None):
        self.root = Path(root)
        self.max_bytes = max_bytes
        self.root.mkdir(parents=True, exist_ok=True)

    def path_for(self, key: CacheKey) -> Path:
        return self.root / key.relative_path()

    def get(self, key: CacheKey) -> np.ndarray | None:
        path = self.path_for(key)
        if not path.exists():
            return None
        try:
            buffer = np.fromfile(str(path), dtype=np.uint8)
        except OSError:
            return None
        if buffer.size == 0:
            return None
        return cv2.imdecode(buffer, cv2.IMREAD_COLOR)

    def put(self, key: CacheKey, image: np.ndarray) -> Path:
        path = self.path_for(key)
        path.parent.mkdir(parents=True, exist_ok=True)

        ok, encoded = cv2.imencode(".png", image)
        if not ok:
            raise CacheError(f"Failed to PNG-encode crop for cache key {key!r}")

        fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=".tmp_", suffix=".png")
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(encoded.tobytes())
            os.replace(tmp_name, path)  # atomic on the same volume (POSIX + Windows)
        except OSError:
            if os.path.exists(tmp_name):
                os.remove(tmp_name)
            raise

        self._enforce_limit()
        return path

    def contains(self, key: CacheKey) -> bool:
        return self.path_for(key).exists()

    def _enforce_limit(self) -> None:
        if self.max_bytes is None:
            return
        entries = [(p, p.stat().st_size, p.stat().st_mtime) for p in self.root.rglob("*.png")]
        total = sum(size for _, size, _ in entries)
        if total <= self.max_bytes:
            return
        # Oldest-first (mtime ascending) eviction until back under the limit.
        for path, size, _ in sorted(entries, key=lambda e: e[2]):
            if total <= self.max_bytes:
                break
            try:
                path.unlink()
                total -= size
            except OSError:
                continue
