"""Phase 6b: compare the baseline and distilled students on VALIDATION only.

Reads each run's summary.json, best.pt and val_best_logits.npy (no test
split, no GenD load: teacher val numbers come from the cached logits) and
reports:
- val frame/video metrics side by side, plus a paired video-level
  bootstrap of the AUROC difference;
- checkpoint size, parameter count, and peak training VRAM;
- inference latency (CPU bs1/bs32, GPU fp32 bs1 and fp16 bs64) and peak
  inference VRAM.

Usage:
    .venv/Scripts/python.exe scripts/compare_students.py --runs student_baseline,student_distilled
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from configuard.env_loader import load_dotenv  # noqa: E402

load_dotenv(REPO_ROOT / ".env")

import numpy as np  # noqa: E402
import torch  # noqa: E402

from configuard.distill.data import load_crop_rows  # noqa: E402
from configuard.distill.evaluate import video_scores  # noqa: E402
from configuard.models.registry import create_encoder  # noqa: E402
from configuard.training.metrics import compute_auroc  # noqa: E402
from configuard.training.paths import resolve_cache_dir, resolve_checkpoint_dir  # noqa: E402


def load_student(path: Path) -> torch.nn.Module:
    ck = torch.load(path, map_location="cpu", weights_only=True)
    model = create_encoder(ck["config"]["encoder_name"], pretrained=False)
    model.load_state_dict(ck["model"])
    return model.eval()


def time_it(fn, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(iters):
        t = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t)
    return statistics.median(times) * 1000


def latency(model: torch.nn.Module) -> dict:
    out = {"cpu_threads": torch.get_num_threads()}
    with torch.inference_mode():
        for bs in (1, 32):
            x = torch.rand(bs, 3, 224, 224)
            out[f"cpu_bs{bs}_ms"] = time_it(lambda: model(x), 5, 30 if bs == 1 else 10)
        g = model.cuda()
        x1 = torch.rand(1, 3, 224, 224, device="cuda")

        def gpu1():
            g(x1)
            torch.cuda.synchronize()

        out["gpu_fp32_bs1_ms"] = time_it(gpu1, 10, 100)
        x64 = torch.rand(64, 3, 224, 224, device="cuda")
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        def gpu64():
            with torch.autocast("cuda", dtype=torch.float16):
                g(x64)
            torch.cuda.synchronize()

        ms = time_it(gpu64, 5, 30)
        out["gpu_fp16_bs64_ms"] = ms
        out["gpu_fp16_bs64_img_per_s"] = 64 / (ms / 1000)
        out["gpu_fp16_bs64_peak_vram_mb"] = torch.cuda.max_memory_allocated() / 2**20
        model.cpu()
    return out


def paired_bootstrap(rows, za, zb, n: int = 2000, seed: int = 0) -> dict:
    ids, y, va, _ = video_scores(rows, za)
    _, _, vb, _ = video_scores(rows, zb)
    rng = np.random.default_rng(seed)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    deltas = []
    for _ in range(n):  # stratified resample of videos
        idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        deltas.append(compute_auroc(y[idx], vb[idx]) - compute_auroc(y[idx], va[idx]))
    d = np.array(deltas)
    return {"delta_video_auroc_b_minus_a": float(compute_auroc(y, vb) - compute_auroc(y, va)),
            "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
            "p_b_le_a": float((d <= 0).mean()), "resamples": n, "videos": len(ids)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs", default="student_baseline,student_distilled")
    args = ap.parse_args()
    root = resolve_checkpoint_dir() / "distill"
    names = args.runs.split(",")
    report: dict = {"generated": datetime.now().isoformat(timespec="seconds"), "split": "val", "runs": {}}
    logits = {}
    for name in names:
        d = root / name
        s = json.loads((d / "summary.json").read_text(encoding="utf-8"))
        epochs = [json.loads(line) for line in (d / "epochs.jsonl").read_text(encoding="utf-8").splitlines()]
        model = load_student(d / "best.pt")
        logits[name] = np.load(d / "val_best_logits.npy")
        report["runs"][name] = {
            "alpha": s["config"]["alpha"], "temperature": s["config"]["temperature"],
            "best_epoch": s["best_epoch"], "epochs_run": s["epochs_run"], "stopped_early": s["stopped_early"],
            "val": s["val_best"], "params": sum(p.numel() for p in model.parameters()),
            "best_pt_bytes": (d / "best.pt").stat().st_size,
            "train_peak_vram_mb": max(e["peak_vram_mb"] for e in epochs),
            "train_img_per_s_median": statistics.median(e["train_img_per_s"] for e in epochs),
            "latency": latency(model),
        }
        report["teacher_val"] = s["teacher_val"]
    crop_tag = json.loads((root / names[0] / "config.json").read_text(encoding="utf-8"))["crop_tag"]
    rows, _ = load_crop_rows(resolve_cache_dir() / "ffpp_face_crops" / "store", crop_tag, "val")
    if len(names) == 2:
        report["paired_bootstrap"] = paired_bootstrap(rows, logits[names[0]], logits[names[1]])
    out = root / "reports" / f"compare_{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1, default=float), encoding="utf-8")
    for name, r in report["runs"].items():
        f, v = r["val"]["frame"], r["val"]["video"]
        print(f"{name}: frame AUROC {f['auroc']:.4f} AUPRC {f['auprc']:.4f} ECE {f['ece']:.4f} | video AUROC "
              f"{v['auroc']:.4f} AUPRC {v['auprc']:.4f} ECE {v['ece']:.4f} bal-acc {v['balanced_accuracy']:.4f} | "
              f"{r['best_pt_bytes'] / 2**20:.1f} MiB, CPU bs1 {r['latency']['cpu_bs1_ms']:.1f} ms, "
              f"GPU fp16 {r['latency']['gpu_fp16_bs64_img_per_s']:.0f} img/s, train VRAM {r['train_peak_vram_mb']:.0f} MB")
    if "paired_bootstrap" in report:
        print("paired bootstrap:", json.dumps(report["paired_bootstrap"]))
    print("report", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
