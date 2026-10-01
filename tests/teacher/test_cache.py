"""Phase 6a: teacher-logit cache - resume, stale rejection, crop integrity,
protected test split."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import pytest

from configuard.teacher.cache import (
    CropIntegrityError,
    ProtectedSplitError,
    StaleTeacherCacheError,
    TeacherCacheConfig,
    TeacherLogitCache,
    require_allowed_split,
)

CFG = TeacherCacheConfig(
    teacher_repo="x/y", teacher_revision="r", teacher_weights_sha256="0" * 64, head="LinearNorm",
    preprocessing="bgr->rgb/255; clip mean/std", attn_implementation="sdpa", autocast_dtype="float16",
    batch_size=4, transformers_version="t", torch_version="p", shard_size=3,
)


def make_rows(tmp: Path, n: int, split: str = "train") -> list[dict]:
    rows = []
    for i in range(n):
        img = np.full((8, 8, 3), i * 10 % 255, np.uint8)
        data = cv2.imencode(".png", img)[1].tobytes()
        rel = f"crops/c{i:03d}.png"
        (tmp / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp / rel).write_bytes(data)
        rows.append({"crop_path": rel, "crop_sha256": hashlib.sha256(data).hexdigest(), "sample_id": f"s{i // 2}",
                     "slot": i % 2, "label": "fake" if i % 3 else "real", "metadata": {"split": split}})
    return rows


class CountingTeacher:
    def __init__(self):
        self.calls = 0

    def __call__(self, crops: list[bytes]) -> np.ndarray:
        self.calls += 1
        return np.array([[len(c) % 7, -float(len(c) % 5)] for c in crops], np.float32)


def test_fill_resume_and_consolidate_are_deterministic(tmp_path):
    rows = make_rows(tmp_path, 8)
    cache = TeacherLogitCache(tmp_path / "cache", CFG, "train", rows, "m" * 64, tmp_path)
    teacher = CountingTeacher()
    assert cache.fill(teacher) == {"shards_total": 3, "shards_resumed": 0, "shards_computed": 3}
    first = cache.consolidate().read_bytes()
    assert len(first.splitlines()) == 8
    assert [json.loads(l)["crop_sha256"] for l in first.splitlines()] == [r["crop_sha256"] for r in rows]

    cache.shard_path(1).unlink()  # simulate an interrupted run
    again = TeacherLogitCache(tmp_path / "cache", CFG, "train", rows, "m" * 64, tmp_path)
    assert again.fill(teacher) == {"shards_total": 3, "shards_resumed": 2, "shards_computed": 1}
    assert teacher.calls == 4
    assert again.consolidate().read_bytes() == first


def test_changed_teacher_or_manifest_is_refused(tmp_path):
    rows = make_rows(tmp_path, 5)
    TeacherLogitCache(tmp_path / "cache", CFG, "train", rows, "m" * 64, tmp_path).fill(CountingTeacher())
    with pytest.raises(StaleTeacherCacheError, match="refusing"):
        TeacherLogitCache(tmp_path / "cache", CFG, "train", rows, "n" * 64, tmp_path)
    # A different teacher config is a different tag directory, never mixed in.
    other = replace(CFG, autocast_dtype="none")
    assert other.tag != CFG.tag
    assert TeacherLogitCache(tmp_path / "cache", other, "train", rows, "m" * 64, tmp_path).load_shard(0) is None


def test_tampered_or_misplaced_shard_is_rejected(tmp_path):
    rows = make_rows(tmp_path, 6)
    cache = TeacherLogitCache(tmp_path / "cache", CFG, "train", rows, "m" * 64, tmp_path)
    cache.fill(CountingTeacher())
    cache.shard_path(0).write_bytes(cache.shard_path(1).read_bytes())  # rows 3-5 logits filed as rows 0-2
    with pytest.raises(StaleTeacherCacheError, match="does not match manifest rows"):
        cache.load_shard(0)
    with np.load(cache.shard_path(1)) as z:
        bad = {k: z[k] for k in z.files}
    bad["teacher_tag"] = np.array("t6a-other")
    np.savez(cache.shard_path(1), **bad)
    with pytest.raises(StaleTeacherCacheError, match="teacher tag"):
        cache.load_shard(1)


def test_crop_bytes_must_match_manifest(tmp_path):
    rows = make_rows(tmp_path, 3)
    (tmp_path / rows[1]["crop_path"]).write_bytes(b"changed")
    cache = TeacherLogitCache(tmp_path / "cache", CFG, "train", rows, "m" * 64, tmp_path)
    with pytest.raises(CropIntegrityError):
        cache.fill(CountingTeacher())
    assert not cache.shard_path(0).exists()


def test_test_split_is_protected(tmp_path):
    with pytest.raises(ProtectedSplitError):
        require_allowed_split("test")
    with pytest.raises(ProtectedSplitError):
        TeacherLogitCache(tmp_path / "c", CFG, "test", make_rows(tmp_path, 2, "test"), "m" * 64, tmp_path)
    with pytest.raises(ProtectedSplitError, match="not in split"):
        TeacherLogitCache(tmp_path / "c", CFG, "val", make_rows(tmp_path, 2, "test"), "m" * 64, tmp_path)
    assert not (tmp_path / "c").exists() or not any((tmp_path / "c").rglob("*.npz"))
