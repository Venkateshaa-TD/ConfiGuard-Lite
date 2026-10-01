"""Phase 6a: cache frozen GenD CLIP-L/14 teacher logits for the FF++ c23
Phase 5d crops - TRAIN and VAL only. The test split is never opened.

- Reuses the Phase 5d crop store and manifests as they are: no
  re-extraction, no video re-hashing. Only the small crop manifests are
  hashed (and checked against the Phase 5d summary), and each crop's
  bytes are checked against its manifest SHA-256 as it is read.
- The teacher is loaded from the pinned snapshot, frozen, and verified
  (assert_frozen) before any inference; inference runs under
  torch.inference_mode with fp16 autocast at the benchmarked batch size.
- Resumable (fixed-index shards) and stale-rejecting (teacher tag +
  crop-manifest SHA-256 + per-shard crop SHA-256 lists).

Usage:
    .venv/Scripts/python.exe scripts/cache_teacher_logits.py                 # train + val
    .venv/Scripts/python.exe scripts/cache_teacher_logits.py --limit-shards 2  # smoke trial
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from configuard.env_loader import load_dotenv  # noqa: E402

load_dotenv(REPO_ROOT / ".env")

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from configuard.crops.audit_stats import auc  # noqa: E402
from configuard.storage_guard import FreeSpaceGuard  # noqa: E402
from configuard.teacher.cache import ALLOWED_SPLITS, TeacherCacheConfig, TeacherLogitCache, require_allowed_split  # noqa: E402
from configuard.teacher.gend import (  # noqa: E402
    ATTN_IMPLEMENTATION,
    FAKE_INDEX,
    GEND_REPO_ID,
    GEND_REVISION,
    GEND_WEIGHTS_SHA256,
    assert_frozen,
    bgr_crops_to_tensor,
    load_gend_teacher,
)
from configuard.training.paths import resolve_cache_dir  # noqa: E402

METHODS = ("Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures")


def log(msg: str, logfile: Path) -> None:
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with logfile.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def fmt(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.4f}"


def mean_or_none(x: np.ndarray) -> float | None:
    return float(x.mean()) if x.size else None


def metrics(rows: list[dict], logits: np.ndarray) -> dict:
    """Frame- and video-level teacher quality (P(fake) = softmax[:, 1])."""
    p = torch.from_numpy(logits).softmax(dim=1)[:, FAKE_INDEX].numpy()
    fake = np.array([r["label"] == "fake" for r in rows])
    method = np.array([r["metadata"]["method"] or "original" for r in rows])
    out = {"frames": len(rows), "frame_auc": auc(p[fake], p[~fake]),
           "frame_acc_at_0.5": float(((p > 0.5) == fake).mean()),
           "mean_p_fake": {"real": mean_or_none(p[~fake]), "fake": mean_or_none(p[fake])},
           "frame_auc_vs_original": {m: auc(p[method == m], p[method == "original"]) for m in METHODS}}
    by_video: dict[str, list[float]] = {}
    lab: dict[str, bool] = {}
    for r, x in zip(rows, logits[:, FAKE_INDEX] - logits[:, 0]):
        by_video.setdefault(r["sample_id"], []).append(float(x))
        lab[r["sample_id"]] = r["label"] == "fake"
    v = {k: np.mean(x) for k, x in by_video.items()}
    out["videos"] = len(v)
    out["video_auc_mean_logit_margin"] = auc([v[k] for k in v if lab[k]], [v[k] for k in v if not lab[k]])
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--splits", default="train,val")
    parser.add_argument("--crop-store", type=Path, default=None)
    parser.add_argument("--cache-root", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=64, help="benchmarked safe size for the RTX 4050 (fp16)")
    parser.add_argument("--shard-size", type=int, default=1024)
    parser.add_argument("--limit-shards", type=int, default=0, help="smoke trial: only the first N shards")
    parser.add_argument("--floor-gb", type=float, default=40.0)
    args = parser.parse_args()

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    for s in splits:
        require_allowed_split(s)  # refuses "test" before anything is opened
    cache_dir = resolve_cache_dir()
    store = args.crop_store or cache_dir / "ffpp_face_crops" / "store"
    root = args.cache_root or cache_dir / "teacher_logits" / "gend_clip_l14"
    if args.limit_shards:
        root = root / "_trial"  # a partial cache must never be mistaken for (or block) the full one
    (root / "reports").mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    logfile = root / "reports" / f"cache_{'trial' if args.limit_shards else 'full'}_{stamp}.log"
    guard = FreeSpaceGuard(root.anchor, args.floor_gb, 1.0)
    if not guard.ok():
        log(f"REFUSED: free space {guard.free_gb():.1f} GB below floor", logfile)
        return 2

    crop_tag = json.loads((store / "store_config.json").read_text(encoding="utf-8"))["config_tag"]
    manifests = store / "manifests" / crop_tag / "full"
    summary = json.loads((manifests / "extraction_summary.json").read_text(encoding="utf-8"))
    if not torch.cuda.is_available():
        log("REFUSED: CUDA not available (teacher caching is GPU-only on this machine)", logfile)
        return 2

    import transformers

    config = TeacherCacheConfig(
        teacher_repo=GEND_REPO_ID, teacher_revision=GEND_REVISION, teacher_weights_sha256=GEND_WEIGHTS_SHA256,
        head="LinearNorm", preprocessing="png bgr->rgb, /255, CLIP mean/std, 224x224 (no resize)",
        attn_implementation=ATTN_IMPLEMENTATION, autocast_dtype="float16", batch_size=args.batch_size,
        transformers_version=transformers.__version__, torch_version=torch.__version__, shard_size=args.shard_size,
    )
    t0 = time.time()
    model = load_gend_teacher(device="cuda")  # verifies the pinned SHA-256
    assert_frozen(model)
    log(f"teacher {GEND_REPO_ID}@{GEND_REVISION[:12]} loaded+verified in {time.time() - t0:.1f}s; tag {config.tag}; "
        f"crop tag {crop_tag}; trainable params {sum(p.requires_grad for p in model.parameters())}", logfile)
    decode_pool = ThreadPoolExecutor(max_workers=6)

    def logit_fn(blobs: list[bytes]) -> np.ndarray:
        imgs = list(decode_pool.map(lambda b: cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR), blobs))
        out = []
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            for i in range(0, len(imgs), args.batch_size):
                x = bgr_crops_to_tensor(imgs[i:i + args.batch_size]).cuda(non_blocking=True)
                out.append(model(x).float().cpu().numpy())
        assert_frozen(model)
        return np.concatenate(out)

    report = {"generated_utc": datetime.now(timezone.utc).isoformat(), "teacher_tag": config.tag,
              "teacher_config": config.__dict__, "crop_tag": crop_tag, "splits": {}}
    for split in splits:
        name = f"crops_{split}.jsonl"
        path = manifests / name
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != summary["manifest_sha256"][name]:
            log(f"REFUSED: {name} sha256 {digest} != Phase 5d summary {summary['manifest_sha256'][name]}", logfile)
            return 2
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        if args.limit_shards:
            rows = rows[: args.limit_shards * args.shard_size]
        cache = TeacherLogitCache(root, config, split, rows, digest, store)
        started = time.time()

        def progress(done: int, total: int, split: str = split, started: float = started) -> None:
            rate = done / max(1e-9, time.time() - started)
            log(f"{split}: shard {done}/{total}, ETA {(total - done) / max(rate, 1e-9) / 60:.1f} min, "
                f"D: free {guard.free_gb():.1f} GB", logfile)
            if not guard.ok():
                raise SystemExit("free-space floor reached; rerun to resume")

        counts = cache.fill(logit_fn, progress)
        out = cache.consolidate()
        logits = cache.all_logits()
        m = metrics(rows, logits)
        elapsed = time.time() - started
        report["splits"][split] = {"crop_manifest": name, "crop_manifest_sha256": digest, **counts,
                                   "consolidated": str(out), "consolidated_sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
                                   "seconds": round(elapsed, 1), "metrics": m}
        log(f"{split}: {counts}; {len(rows)} frames in {elapsed:.0f}s; frame AUC {fmt(m['frame_auc'])}, "
            f"video AUC {fmt(m['video_auc_mean_logit_margin'])}, acc@0.5 {fmt(m['frame_acc_at_0.5'])}; "
            f"per-method AUC {json.dumps({k: fmt(v) for k, v in m['frame_auc_vs_original'].items()})}", logfile)
    report["test_split_touched"] = False
    assert set(report["splits"]) <= set(ALLOWED_SPLITS)
    out = root / "reports" / f"teacher_cache_{'trial' if args.limit_shards else 'full'}_{stamp}.json"
    out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    log(f"report {out}; D: free {guard.free_gb():.1f} GB", logfile)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
