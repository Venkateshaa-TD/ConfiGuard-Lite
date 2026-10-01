"""Phase 7: embedding cache, video grouping, residual GRU head."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from configuard.adaptive.policy import STAGES, stage_slots
from configuard.memory_guard import process_rss_mb
from configuard.teacher.cache import ProtectedSplitError
from configuard.temporal.embeddings import EmbeddingCache, StaleEmbeddingCacheError, extract, to_videos
from configuard.temporal.gru import GRUTrainConfig, ResidualGRUHead, predict, train_gru


def rows_for(n_videos=6, d_split="val"):
    rows = []
    for v in range(n_videos):
        for s in reversed(range(16)):  # deliberately out of slot order
            rows.append({"crop_path": f"{v}_{s}.png", "crop_sha256": f"{v:03d}{s:03d}", "sample_id": f"vid{v}", "slot": s,
                         "label": "fake" if v % 2 else "real",
                         "nested_levels": [k for k in STAGES if s in stage_slots(k)],
                         "metadata": {"split": d_split, "method": "NeuralTextures" if v % 2 else None}})
    return rows


def test_to_videos_orders_slots_and_checks_nested_levels():
    rows = rows_for(3)
    feats = np.array([[r["slot"], int(r["sample_id"][-1])] for r in rows], np.float16)
    logits = np.array([r["slot"] / 10 for r in rows], np.float32)
    vs = to_videos(rows, feats, logits)
    assert vs.features.shape == (3, 16, 2) and list(vs.features[1, :, 0]) == list(range(16))
    emb4, fl4 = vs.stage(4)
    assert list(emb4[0, :, 0]) == [0, 4, 8, 12] and fl4.shape == (3, 4)
    assert vs.mean_logit(16)[0] == pytest.approx(0.75) and vs.methods[1] == "NeuralTextures"
    bad = rows_for(1)
    next(r for r in bad if r["slot"] == 0)["nested_levels"] = [16]  # slot 0 belongs to 4/8/16
    with pytest.raises(ValueError):
        to_videos(bad, feats[:16], logits[:16])


def test_embedding_cache_roundtrip_and_stale_rejection(tmp_path):
    rows = rows_for(2)
    cache = EmbeddingCache(tmp_path, "a" * 64)
    assert cache.load("val", rows) is None
    f, z = np.ones((32, 4), np.float16), np.zeros(32, np.float32)
    cache.save("val", rows, f, z)
    f2, z2 = cache.load("val", rows)
    assert np.array_equal(f2, f) and np.array_equal(z2, z)
    with pytest.raises(StaleEmbeddingCacheError):
        cache.load("val", rows[::-1])  # different row order
    other = EmbeddingCache(tmp_path, "b" * 64)
    assert other.load("val", rows) is None  # another checkpoint never sees this cache


def test_extract_refuses_test_rows():
    with pytest.raises(ProtectedSplitError):
        extract(None, None, rows_for(1, d_split="test"), ".", device="cpu", num_workers=0)


def test_untrained_head_equals_mean_frame_logit_and_handles_4_8_16():
    head = ResidualGRUHead(in_dim=8).eval()
    assert head.parameter_count() < 200_000
    for k in STAGES:
        emb, fl = torch.randn(5, k, 8), torch.randn(5, k)
        assert torch.allclose(head(emb, fl), fl.mean(dim=1))
    with pytest.raises(ValueError):
        ResidualGRUHead(in_dim=8, hidden=256)


def test_gru_learns_a_temporal_cue_the_mean_cannot_see():
    rng = np.random.default_rng(0)

    def make(n):
        rows = []
        for v in range(n):
            for s in range(16):
                rows.append({"crop_sha256": f"{v}-{s}", "sample_id": f"v{v}", "slot": s, "label": "fake" if v % 2 else "real",
                             "nested_levels": [k for k in STAGES if s in stage_slots(k)],
                             "metadata": {"split": "val", "method": "Deepfakes" if v % 2 else None}})
        y = np.array([v % 2 for v in range(n) for _ in range(16)])
        sign = np.where(y == 1, 1.0, -1.0)
        slot = np.tile(np.arange(16), n)
        feats = rng.normal(0, 1, (n * 16, 4)).astype(np.float16)
        feats[:, 0] = (sign * (slot / 15.0 - 0.5)).astype(np.float16)  # fakes trend up over time, reals down
        return to_videos(rows, feats, np.zeros(n * 16, np.float32))  # mean frame logit carries no signal

    train, val = make(200), make(80)
    head, hist = train_gru(train, val, GRUTrainConfig(max_epochs=12, patience=12, batch_size=32), device="cpu",
                           log=lambda m: None)
    assert hist[0]["epoch"] == 0 and max(h["val_video_auroc"][16] for h in hist) > 0.9
    assert predict(head, val, 4).shape == (80,)
    assert process_rss_mb() > 0
