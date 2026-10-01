"""Phase 6e: degradations, curriculum, config wiring, and the stress suite."""

from __future__ import annotations

import hashlib
import inspect
import json
import shutil

import cv2
import numpy as np
import pytest

from configuard.distill.data import CropDataset
from configuard.distill.train import DistillConfig
from configuard.robust.degrade import (
    RobustAugmentConfig,
    gamma,
    h264_style,
    jpeg,
    resize_down_up,
    robust_degrade,
)
from configuard.teacher.cache import ProtectedSplitError

FFMPEG = shutil.which("ffmpeg") is not None


def face_like(seed=0, size=224):
    rng = np.random.default_rng(seed)
    img = cv2.GaussianBlur(rng.integers(0, 255, (size, size, 3), dtype=np.uint8), (0, 0), 3)
    cv2.circle(img, (size // 2, size // 2), size // 3, (180, 150, 120), -1)
    return img


def psnr(a, b):
    return cv2.PSNR(a, b)


def test_primitives_keep_shape_and_get_worse_with_severity():
    img = face_like()
    for out in (jpeg(img, 50), h264_style(img, 30), resize_down_up(img, 0.5), gamma(img, 1.2)):
        assert out.shape == img.shape and out.dtype == np.uint8
    assert psnr(img, h264_style(img, 22)) > psnr(img, h264_style(img, 30)) > psnr(img, h264_style(img, 38))
    assert psnr(img, jpeg(img, 90)) > psnr(img, jpeg(img, 40))
    assert psnr(img, resize_down_up(img, 0.9)) > psnr(img, resize_down_up(img, 0.45))
    assert np.array_equal(gamma(img, 1.0), img)


def test_robust_degrade_is_label_free_deterministic_and_bounded():
    assert "label" not in inspect.signature(robust_degrade).parameters
    cfg = RobustAugmentConfig()
    img = face_like(1)
    a, pa = robust_degrade(img, cfg, np.random.default_rng([1, 2, 3]), 2)
    b, pb = robust_degrade(img, cfg, np.random.default_rng([1, 2, 3]), 2)
    assert np.array_equal(a, b) and pa == pb
    counts = {k: 0 for k in ("gamma", "resize", "blur", "noise", "jpeg", "h264_qp")}
    for i in range(400):
        _, p = robust_degrade(img, cfg, np.random.default_rng([0, 9, i]), 9)
        for k in p:
            counts[k] += 1
        assert not ("jpeg" in p and "h264_qp" in p)  # one compression at most
        if "jpeg" in p:
            assert 40 <= p["jpeg"] <= 90
        if "h264_qp" in p:
            assert 22 <= p["h264_qp"] <= 36
        if "resize" in p:
            assert 0.45 <= p["resize"] <= 0.9
        if "gamma" in p:
            assert np.exp(-0.22) - 1e-9 <= p["gamma"] <= np.exp(0.22) + 1e-9
    assert abs(counts["jpeg"] / 400 - cfg.p_jpeg) < 0.08 and abs(counts["h264_qp"] / 400 - cfg.p_h264) < 0.08


def test_curriculum_caps_severity_mild_first():
    cfg = RobustAugmentConfig()
    assert cfg.cap(0) == 0.5 and cfg.cap(2) == 0.75 and cfg.cap(4) == 1.0 and cfg.cap(10) == 1.0
    worst_q = min(robust_degrade(face_like(), cfg, np.random.default_rng([0, 0, i]), 0)[1].get("jpeg", 99)
                  for i in range(300))
    assert worst_q >= 65  # epoch 0: at most half-way to the moderate end (90 -> 65)
    with pytest.raises(ValueError):
        RobustAugmentConfig(p_jpeg=0.7, p_h264=0.5)


def test_config_wiring_keeps_old_configs_and_roundtrips():
    base = {"run_name": "x"}
    assert "robust_augment" not in DistillConfig.from_dict(base).to_dict()
    cfg = DistillConfig.from_dict(base | {"robust_augment": RobustAugmentConfig().to_dict()})
    assert cfg.robust_config() == RobustAugmentConfig()
    d = cfg.to_dict()
    assert DistillConfig.from_dict(json.loads(json.dumps(d))).to_dict() == d
    with pytest.raises(TypeError):
        DistillConfig.from_dict(base | {"robust_augment": {"p_jpgg": 0.1}})


def test_dataset_degrades_training_views_only(tmp_path):
    img = face_like(2)
    (tmp_path / "c.png").write_bytes(cv2.imencode(".png", img)[1].tobytes())
    rows = [{"crop_path": "c.png", "label": "fake"}, {"crop_path": "c.png", "label": "real"}]
    always = RobustAugmentConfig(p_jpeg=1.0, p_h264=0.0, jpeg_quality=(40.0, 40.0))
    ds = CropDataset(rows, tmp_path, np.zeros(2, np.float32), None, seed=0, robust_cfg=always)
    clean = img[:, :, ::-1].transpose(2, 0, 1)
    assert np.array_equal(ds[(0, -1)][0].numpy(), clean)  # eval items (epoch -1) untouched
    assert not np.array_equal(ds[(0, 3)][0].numpy(), clean)
    # same index/epoch -> same pixels; the label never enters the degradation
    assert np.array_equal(ds[(0, 3)][0].numpy(), ds[(0, 3)][0].numpy())


def _val_rows(store, n_videos=2, frames=16, split="val"):
    rows = []
    for v in range(n_videos):
        for s in range(frames):
            img = face_like(100 * v + s)
            data = cv2.imencode(".png", img)[1].tobytes()
            rel = f"crops/{v}_{s}.png"
            (store / rel).parent.mkdir(parents=True, exist_ok=True)
            (store / rel).write_bytes(data)
            rows.append({"crop_path": rel, "crop_sha256": hashlib.sha256(data).hexdigest(), "label": "fake" if v else "real",
                         "sample_id": f"vid{v}", "slot": s, "nested_levels": [16],
                         "metadata": {"split": split, "method": "Deepfakes" if v else None}})
    return rows


@pytest.mark.skipif(not FFMPEG, reason="ffmpeg not installed")
def test_stress_suite_build_resume_determinism_and_refusals(tmp_path):
    from configuard.robust.stress import StaleStressSuiteError, StressSuite

    store = tmp_path / "store"
    rows = _val_rows(store)
    conds = ({"name": "x264_crf30", "ops": [["x264", 30]], "severity": "moderate", "in_train_range": True},
             {"name": "noise_s4", "ops": [["noise", 4.0]], "severity": "moderate", "in_train_range": True})
    s = StressSuite(tmp_path / "suite", store, rows, "v" * 64, conds)
    assert [s.build(c, workers=2)["status"] for c in conds] == ["built", "built"]
    assert [s.build(c, workers=2)["status"] for c in conds] == ["resumed", "resumed"]
    first = [r["crop_sha256"] for r in s.condition_rows("x264_crf30")]
    noise1 = [r["crop_sha256"] for r in s.condition_rows("noise_s4")]
    assert len(first) == 32 and s.condition_rows("clean")[0]["crop_path"].endswith("0_0.png")
    shutil.rmtree(s.root / "x264_crf30")
    shutil.rmtree(s.root / "noise_s4")
    s.build(conds[0], workers=2)
    s.build(conds[1], workers=2)
    assert [r["crop_sha256"] for r in s.condition_rows("x264_crf30")] == first  # deterministic rebuild
    assert [r["crop_sha256"] for r in s.condition_rows("noise_s4")] == noise1
    assert StressSuite(tmp_path / "suite", store, rows, "v" * 64, conds[:1]).root != s.root  # other conditions -> other tag
    (s.root / "suite_config.json").write_text("{}")  # same tag dir, different content -> refused
    with pytest.raises(StaleStressSuiteError):
        StressSuite(tmp_path / "suite", store, rows, "v" * 64, conds)
    with pytest.raises(ProtectedSplitError):
        StressSuite(tmp_path / "suite2", store, _val_rows(tmp_path / "t", 1, 2, split="test"), "v" * 64, conds)


class _MeanModel:
    def forward_logits(self, x):
        return x.float().mean(dim=(1, 2, 3)) / 255.0


def test_scoring_saves_each_condition_stops_safely_and_resumes(tmp_path):
    import torch

    from configuard.memory_guard import LowMemoryError, RamGuard, available_ram_gb
    from configuard.robust.scoring import score_conditions

    assert available_ram_gb() > 0
    rows = {}
    for c, val in (("clean", 10), ("jpeg", 120), ("noise", 240)):
        rows[c] = []
        for i in range(2):
            f = tmp_path / f"{c}{i}.png"
            f.write_bytes(cv2.imencode(".png", np.full((8, 8, 3), val, np.uint8))[1].tobytes())
            rows[c].append({"crop_path": str(f), "label": "real"})
    out, calls = tmp_path / "out", []
    class TripsSecondCheck(RamGuard):  # passes the start check, then "runs out" of RAM
        n = 0

        def check(self):
            self.n += 1
            if self.n >= 2:
                raise LowMemoryError("simulated")
            return 99.0

    with pytest.raises(LowMemoryError):
        score_conditions(_MeanModel(), lambda x: x, rows, out, device="cpu", num_workers=0, batch_size=2,
                         guard=TripsSecondCheck(), log=calls.append)
    with pytest.raises(LowMemoryError):  # a floor above any real RAM refuses before scoring anything new
        score_conditions(_MeanModel(), lambda x: x, rows, out, device="cpu", num_workers=0, batch_size=2,
                         guard=RamGuard(floor_gb=1e9), log=calls.append)
    assert sorted(p.name for p in out.glob("*.npy")) == ["clean.npy"]  # completed condition preserved
    res = score_conditions(_MeanModel(), lambda x: x, rows, out, device="cpu", num_workers=0, batch_size=2,
                           guard=RamGuard(floor_gb=0.0), log=calls.append)
    assert any("1 done, 2 to score" in m for m in calls)
    assert res["clean"] == pytest.approx([10 / 255] * 2, rel=1e-5) and res["noise"] == pytest.approx([240 / 255] * 2, rel=1e-5)
    again: list[str] = []
    score_conditions(_MeanModel(), lambda x: x, rows, out, device="cpu", num_workers=0, guard=None, log=again.append)
    assert again == []  # everything cached, nothing rescored
    del torch
