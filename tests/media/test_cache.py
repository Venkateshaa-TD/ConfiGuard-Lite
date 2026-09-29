"""Phase 2: versioned, atomic face-crop cache."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from configuard.media.cache import CacheKey, FaceCropCache


def _key(**overrides) -> CacheKey:
    defaults = dict(input_sha256="a" * 64, frame_index=0, track_id=0, config_version="v1-abc123")
    defaults.update(overrides)
    return CacheKey(**defaults)


def _image(value: int = 100) -> np.ndarray:
    return np.full((32, 32, 3), value, dtype=np.uint8)


def test_cache_miss_returns_none(tmp_path: Path):
    cache = FaceCropCache(tmp_path)
    assert cache.get(_key()) is None
    assert cache.contains(_key()) is False


def test_cache_put_then_get_round_trips(tmp_path: Path):
    cache = FaceCropCache(tmp_path)
    key = _key()
    cache.put(key, _image(200))

    assert cache.contains(key) is True
    retrieved = cache.get(key)
    assert retrieved is not None
    assert retrieved.shape == (32, 32, 3)
    assert abs(int(retrieved.mean()) - 200) <= 2  # PNG is lossless; allow for encode rounding


def test_cache_key_path_includes_all_key_fields(tmp_path: Path):
    cache = FaceCropCache(tmp_path)
    key = CacheKey(input_sha256="b" * 64, frame_index=7, track_id=2, config_version="v1-xyz")
    path = cache.path_for(key)

    assert "v1-xyz" in path.parts
    assert "b" * 64 in path.parts
    assert "track2_frame000007.png" in path.name


def test_different_config_version_is_a_different_cache_entry(tmp_path: Path):
    cache = FaceCropCache(tmp_path)
    key_v1 = _key(config_version="v1-aaa")
    key_v2 = _key(config_version="v1-bbb")
    cache.put(key_v1, _image(50))

    assert cache.get(key_v1) is not None
    assert cache.get(key_v2) is None  # different version = cache miss, not stale hit


def test_different_frame_index_or_track_id_is_a_different_entry(tmp_path: Path):
    cache = FaceCropCache(tmp_path)
    cache.put(_key(frame_index=0), _image(10))
    cache.put(_key(frame_index=1), _image(20))
    cache.put(_key(track_id=1), _image(30))

    assert cache.get(_key(frame_index=0)) is not None
    assert cache.get(_key(frame_index=1)) is not None
    assert cache.get(_key(track_id=1)) is not None
    assert cache.get(_key(frame_index=2)) is None


def test_write_is_atomic_no_leftover_temp_files(tmp_path: Path):
    cache = FaceCropCache(tmp_path)
    cache.put(_key(), _image())
    leftover_tmp = list(tmp_path.rglob(".tmp_*"))
    assert leftover_tmp == []


def test_cache_reuse_avoids_recompute(tmp_path: Path):
    """Simulates the pattern preprocess.py uses: check cache.get() before
    doing expensive work, only cache.put() on a miss."""
    cache = FaceCropCache(tmp_path)
    key = _key()
    compute_calls = 0

    def get_or_compute():
        nonlocal compute_calls
        cached = cache.get(key)
        if cached is not None:
            return cached
        compute_calls += 1
        result = _image(77)
        cache.put(key, result)
        return result

    first = get_or_compute()
    second = get_or_compute()
    assert compute_calls == 1
    assert first.shape == second.shape


def test_eviction_enforces_byte_limit(tmp_path: Path):
    # Each 32x32 PNG is a few hundred bytes; force a tiny limit so eviction
    # is guaranteed to trigger deterministically.
    cache = FaceCropCache(tmp_path, max_bytes=1)
    for i in range(5):
        cache.put(_key(frame_index=i), _image(i * 10))

    total_bytes = sum(p.stat().st_size for p in tmp_path.rglob("*.png"))
    assert total_bytes <= 1 or len(list(tmp_path.rglob("*.png"))) <= 1


def test_no_limit_keeps_everything(tmp_path: Path):
    cache = FaceCropCache(tmp_path, max_bytes=None)
    for i in range(5):
        cache.put(_key(frame_index=i), _image(i * 10))
    assert len(list(tmp_path.rglob("*.png"))) == 5
