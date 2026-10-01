"""Deterministic clean/degraded development stress suite (official VAL crops only).

Each condition writes a degraded PNG for every val crop under
`<cache>/robust_stress/<suite_tag>/<condition>/` plus a per-condition
manifest. The suite tag hashes the condition list, the val crop manifest
SHA-256, the ffmpeg/x264 settings and the schema; a root holding another tag is
refused (StaleStressSuiteError). Conditions are resumable: a finished
condition (manifest present, every file present with the recorded size) is
skipped. Degradations are deterministic: H.264 conditions run REAL libx264
(single-threaded, fixed preset/CRF) over each video's 16 crops in slot order;
noise is seeded by (condition, crop SHA-256). Severity levels deliberately go
beyond the training augmentation range ("severe"). Degradations are applied to
aligned face crops, not to full frames before face detection, which is an
approximation of real-world re-encoding (docs/KNOWN_ISSUES.md).
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections import defaultdict
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from configuard.crops.store import atomic_write_bytes, canonical_json
from configuard.robust.degrade import add_noise, gamma, gaussian_blur, jpeg, resize_down_up
from configuard.teacher.cache import ProtectedSplitError

SCHEMA = "p6e-stress-1"
X264 = {"preset": "medium", "pix_fmt": "yuv420p", "threads": 1, "fps": 8}

# name, op chain [(op, param)], severity, inside the training range?
CONDITIONS: tuple[dict[str, Any], ...] = (
    {"name": "jpeg_q75", "ops": [["jpeg", 75]], "severity": "mild", "in_train_range": True},
    {"name": "jpeg_q50", "ops": [["jpeg", 50]], "severity": "moderate", "in_train_range": True},
    {"name": "jpeg_q30", "ops": [["jpeg", 30]], "severity": "severe", "in_train_range": False},
    {"name": "x264_crf23", "ops": [["x264", 23]], "severity": "mild", "in_train_range": True},
    {"name": "x264_crf30", "ops": [["x264", 30]], "severity": "moderate", "in_train_range": True},
    {"name": "x264_crf37", "ops": [["x264", 37]], "severity": "severe", "in_train_range": False},
    {"name": "resize_0.75", "ops": [["resize", 0.75]], "severity": "mild", "in_train_range": True},
    {"name": "resize_0.5", "ops": [["resize", 0.5]], "severity": "moderate", "in_train_range": True},
    {"name": "resize_0.33", "ops": [["resize", 0.33]], "severity": "severe", "in_train_range": False},
    {"name": "blur_s1.0", "ops": [["blur", 1.0]], "severity": "moderate", "in_train_range": True},
    {"name": "blur_s2.0", "ops": [["blur", 2.0]], "severity": "severe", "in_train_range": False},
    {"name": "noise_s4", "ops": [["noise", 4.0]], "severity": "moderate", "in_train_range": True},
    {"name": "noise_s10", "ops": [["noise", 10.0]], "severity": "severe", "in_train_range": False},
    {"name": "gamma_0.7", "ops": [["gamma", 0.7]], "severity": "severe", "in_train_range": False},
    {"name": "gamma_1.4", "ops": [["gamma", 1.4]], "severity": "severe", "in_train_range": False},
    {"name": "social_resize0.5_jpeg60", "ops": [["resize", 0.5], ["jpeg", 60]], "severity": "moderate",
     "in_train_range": True},
    {"name": "stream_resize0.5_x264crf30", "ops": [["resize", 0.5], ["x264", 30]], "severity": "moderate",
     "in_train_range": True},
)


class StaleStressSuiteError(Exception):
    """The suite root was built for another condition list / val manifest / settings."""


def ffmpeg_version() -> str:
    out = subprocess.run(["ffmpeg", "-hide_banner", "-version"], capture_output=True, text=True, check=True)
    return out.stdout.splitlines()[0]


def suite_tag(val_manifest_sha256: str, ffmpeg: str, conditions: Sequence[dict[str, Any]] = CONDITIONS) -> str:
    payload = {"schema": SCHEMA, "val": val_manifest_sha256, "ffmpeg": ffmpeg, "x264": X264,
               "conditions": list(conditions)}
    return "p6e-" + hashlib.sha256(canonical_json(payload)).hexdigest()[:16]


def x264_roundtrip(frames: Sequence[np.ndarray], crf: int) -> list[np.ndarray]:
    """Encode an ordered BGR uint8 sequence with libx264 and decode it back."""
    h, w = frames[0].shape[:2]
    enc = subprocess.run(
        ["ffmpeg", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", str(X264["fps"]),
         "-i", "-", "-c:v", "libx264", "-preset", X264["preset"], "-crf", str(crf), "-pix_fmt", X264["pix_fmt"],
         "-threads", str(X264["threads"]), "-f", "h264", "-"],
        input=b"".join(np.ascontiguousarray(f).tobytes() for f in frames), capture_output=True, check=True)
    dec = subprocess.run(["ffmpeg", "-v", "error", "-f", "h264", "-i", "-", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"],
                         input=enc.stdout, capture_output=True, check=True)
    out = np.frombuffer(dec.stdout, np.uint8)
    if out.size != len(frames) * h * w * 3:
        raise RuntimeError(f"x264 round trip returned {out.size} bytes for {len(frames)} frames")
    return list(out.reshape(len(frames), h, w, 3))


def apply_frame_op(img: np.ndarray, op: str, param: float, seed_key: str) -> np.ndarray:
    if op == "jpeg":
        return jpeg(img, int(param))
    if op == "resize":
        return resize_down_up(img, float(param))
    if op == "blur":
        return gaussian_blur(img, float(param))
    if op == "gamma":
        return gamma(img, float(param))
    if op == "noise":
        seed = int.from_bytes(hashlib.sha256(seed_key.encode()).digest()[:8], "little")
        return add_noise(img, float(param), np.random.default_rng(seed))
    raise ValueError(f"unknown op {op!r}")


def degrade_video(frames: Sequence[np.ndarray], ops: Sequence[Sequence[Any]], keys: Sequence[str]) -> list[np.ndarray]:
    """Apply an op chain to one video's ordered frames. x264 acts on the sequence."""
    out = list(frames)
    for op, param in ops:
        if op == "x264":
            out = x264_roundtrip(out, int(param))
        else:
            out = [apply_frame_op(f, op, param, f"{k}:{op}:{param}") for f, k in zip(out, keys)]
    return out


class StressSuite:
    def __init__(self, root: str | Path, crop_store: str | Path, val_rows: Sequence[dict[str, Any]],
                 val_manifest_sha256: str, conditions: Sequence[dict[str, Any]] = CONDITIONS) -> None:
        bad = {r["metadata"]["split"] for r in val_rows} - {"val"}
        if bad:  # development suite = official val only; test stays sealed
            raise ProtectedSplitError(f"stress suite accepts val rows only, got {sorted(bad)}")
        self.crop_store, self.rows, self.conditions = Path(crop_store), list(val_rows), list(conditions)
        self.tag = suite_tag(val_manifest_sha256, ffmpeg_version(), conditions)
        self.root = Path(root) / self.tag
        cfg = {"schema": SCHEMA, "tag": self.tag, "val_manifest_sha256": val_manifest_sha256,
               "conditions": self.conditions, "x264": X264, "rows": len(self.rows)}
        path = self.root / "suite_config.json"
        if path.exists():
            if json.loads(path.read_text(encoding="utf-8")) != json.loads(canonical_json(cfg)):
                raise StaleStressSuiteError(f"{self.root} holds a different suite")
        else:
            atomic_write_bytes(path, canonical_json(cfg))

    def manifest_path(self, name: str) -> Path:
        return self.root / name / "manifest.jsonl"

    def is_complete(self, name: str) -> bool:
        mp = self.manifest_path(name)
        if not mp.exists():
            return False
        rows = [json.loads(line) for line in mp.read_text(encoding="utf-8").splitlines()]
        return len(rows) == len(self.rows) and all(
            (self.root / r["crop_path"]).is_file() and (self.root / r["crop_path"]).stat().st_size == r["bytes"]
            for r in rows)

    def build(self, cond: dict[str, Any], workers: int = 12,
              guard: Callable[[], bool] | None = None) -> dict[str, Any]:
        name = cond["name"]
        if self.is_complete(name):
            return {"condition": name, "status": "resumed"}
        videos: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for r in self.rows:
            videos[r["sample_id"]].append(r)

        def one(sid: str) -> list[dict[str, Any]]:
            vr = sorted(videos[sid], key=lambda r: r["slot"])
            frames = [cv2.imdecode(np.fromfile(str(self.crop_store / r["crop_path"]), np.uint8), cv2.IMREAD_COLOR)
                      for r in vr]
            out = degrade_video(frames, cond["ops"], [r["crop_sha256"] for r in vr])
            recs = []
            for r, img in zip(vr, out):
                data = cv2.imencode(".png", img, [cv2.IMWRITE_PNG_COMPRESSION, 1])[1].tobytes()
                rel = f"{name}/{r['crop_sha256'][:2]}/{r['crop_sha256']}.png"
                atomic_write_bytes(self.root / rel, data)
                recs.append({"crop_path": rel, "crop_sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data),
                             "source_crop_sha256": r["crop_sha256"], "label": r["label"], "sample_id": r["sample_id"],
                             "slot": r["slot"], "nested_levels": r["nested_levels"], "metadata": r["metadata"]})
            return recs

        out: dict[str, list[dict[str, Any]]] = {}
        with ThreadPoolExecutor(workers) as pool:
            for i, (sid, recs) in enumerate(zip(videos, pool.map(one, list(videos)))):
                out[sid] = recs
                if guard is not None and i % 50 == 0 and not guard():
                    raise RuntimeError("free-space floor reached while building the stress suite; rerun to resume")
        by_src = {rec["source_crop_sha256"]: rec for recs in out.values() for rec in recs}
        lines = [json.dumps(by_src[r["crop_sha256"]], sort_keys=True) + "\n" for r in self.rows]  # val order
        atomic_write_bytes(self.manifest_path(name), "".join(lines).encode("utf-8"))
        return {"condition": name, "status": "built", "bytes": sum(rec["bytes"] for rec in by_src.values())}

    def condition_rows(self, name: str) -> list[dict[str, Any]]:
        """Rows (val order) with ABSOLUTE crop paths, ready for configuard.distill.infer.predict_rows(crop_root="")."""
        if name == "clean":
            return [r | {"crop_path": str(self.crop_store / r["crop_path"])} for r in self.rows]
        if not self.is_complete(name):
            raise StaleStressSuiteError(f"condition {name} is not built")
        rows = [json.loads(line) for line in self.manifest_path(name).read_text(encoding="utf-8").splitlines()]
        return [r | {"crop_path": str(self.root / r["crop_path"])} for r in rows]
