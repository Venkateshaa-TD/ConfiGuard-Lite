"""Synthetic Phase 5d-style crop store + Phase 6a-style teacher cache.
ENGINEERING DATA ONLY: tinted noise crops, never a detection claim."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

CROP_TAG, TEACHER_TAG = "p5d-test", "t6a-test"
METHODS = ("Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures")


def _write(path: Path, data: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def build_store(root: Path, videos_per_class: dict[str, int], frames: int = 2, size: int = 224) -> Path:
    store = root / "store"
    rng = np.random.default_rng(0)
    manifests = store / "manifests" / CROP_TAG / "full"
    shas = {}
    for split, n in videos_per_class.items():
        rows, tlines = [], []
        for method in (None, *METHODS):
            for v in range(n):
                sid = f"{split}:{method or 'original'}:{v}"
                for slot in range(frames):
                    img = rng.integers(0, 255, (size, size, 3), dtype=np.uint8)
                    img[..., 2] = 200 if method else 40  # obvious learnable tint (engineering only)
                    rel = f"crops/{split}/{method or 'orig'}/{v}_{slot}.png"
                    sha = _write(store / rel, cv2.imencode(".png", img)[1].tobytes())
                    rows.append({"crop_path": rel, "crop_sha256": sha, "label": "fake" if method else "real",
                                 "metadata": {"method": method, "split": split, "source_id": str(v)},
                                 "sample_id": sid, "slot": slot})
                    margin = 3.0 if method else -3.0
                    tlines.append({"crop_path": rel, "crop_sha256": sha, "sample_id": sid, "slot": slot,
                                   "teacher_tag": TEACHER_TAG, "logits": [-margin / 2, margin / 2]})
        text = "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows)
        shas[f"crops_{split}.jsonl"] = _write(manifests / f"crops_{split}.jsonl", text.encode())
        tdir = root / "teacher" / TEACHER_TAG / split
        _write(tdir / "teacher_logits.jsonl", "".join(json.dumps(t) + "\n" for t in tlines).encode())
        _write(tdir / "meta.json", json.dumps({"teacher_tag": TEACHER_TAG,
                                               "crop_manifest_sha256": shas[f"crops_{split}.jsonl"]}).encode())
    _write(manifests / "extraction_summary.json", json.dumps({"manifest_sha256": shas}).encode())
    return store


@pytest.fixture
def synthetic_store(tmp_path: Path) -> Path:
    return build_store(tmp_path, {"train": 3, "val": 2})
