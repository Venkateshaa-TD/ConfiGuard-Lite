"""Phase 6e: build the deterministic val stress suite and compare students on it.

    build     write every degraded condition of the official VAL crops to D:
              (resumable; refuses to cross the free-space floor)
    evaluate  score each run on clean + all conditions (logits cached per
              checkpoint SHA-256), report clean / worst-case / delta AUROC,
              FPR, per-method results, training cost, the robust-model decision
              rule, and that the 6c/6d calibration artifacts are INVALID for
              the robust checkpoint.

Never opens the test split. Never loads GenD (cached logits only in training).

Usage:
    .venv/Scripts/python.exe scripts/robust_eval.py build
    .venv/Scripts/python.exe scripts/robust_eval.py evaluate --runs student_distilled_p80,student_distilled_robust_p80
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

from configuard.calibration.artifact import CalibrationMismatchError, file_sha256, load_calibration  # noqa: E402
from configuard.distill.data import load_crop_rows  # noqa: E402
from configuard.distill.evaluate import evaluate_logits, video_scores  # noqa: E402
from configuard.distill.infer import load_student  # noqa: E402
from configuard.memory_guard import LowMemoryError, RamGuard, available_ram_gb  # noqa: E402
from configuard.robust.scoring import score_conditions  # noqa: E402
from configuard.robust.stress import CONDITIONS, StressSuite  # noqa: E402
from configuard.storage_guard import FreeSpaceGuard  # noqa: E402
from configuard.training.paths import resolve_cache_dir, resolve_checkpoint_dir  # noqa: E402

CROP_TAG = "p5d-b451b5ca770c8923"
FLOOR_GB, MARGIN_GB = 40.0, 2.0
MAX_CLEAN_VIDEO_AUROC_LOSS = 0.01
METHODS = ("Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures")


def suite() -> StressSuite:
    cache = resolve_cache_dir()
    store = cache / "ffpp_face_crops" / "store"
    rows, sha = load_crop_rows(store, CROP_TAG, "val")
    return StressSuite(cache / "robust_stress", store, rows, sha)


def cmd_build(args) -> int:
    s = suite()
    guard = FreeSpaceGuard(s.root.anchor, FLOOR_GB, MARGIN_GB)
    print(f"suite {s.tag} at {s.root}; D: free {guard.free_gb():.1f} GB", flush=True)
    for cond in CONDITIONS:
        if not guard.ok():
            print(f"REFUSED: free space {guard.free_gb():.1f} GB at floor {FLOOR_GB}+{MARGIN_GB}")
            return 2
        t = time.time()
        r = s.build(cond, workers=args.workers, guard=guard.ok)
        print(f"{cond['name']}: {r['status']} {r.get('bytes', 0) / 2**30:.2f} GiB in {time.time() - t:.0f}s; "
              f"D: free {guard.free_gb():.1f} GB", flush=True)
    return 0


def fpr_video(rows, logits) -> float:
    _, y, z, _ = video_scores(rows, logits)
    return float((z[y == 0] >= 0).mean())  # raw P(fake) >= 0.5 on real videos (models are uncalibrated here)


def score_run(run: str, s: StressSuite, workers: int, guard: RamGuard) -> dict:
    run_dir = resolve_checkpoint_dir() / "distill" / run
    ckpt = run_dir / "best.pt"
    sha = file_sha256(ckpt)
    out_dir = run_dir / "stress" / s.tag / sha[:12]  # per-condition logits, keyed by checkpoint hash
    names = ["clean"] + [c["name"] for c in CONDITIONS]
    rows = {n: s.condition_rows(n) for n in names}
    model, norm, _ = load_student(ckpt, "cuda")
    t = time.time()
    # Conditions are scored sequentially and saved one by one; 12 spawn workers had
    # run the 24 GB laptop out of RAM, so 4 workers and a RAM floor are used here.
    logits = score_conditions(model, norm, rows, out_dir, num_workers=workers, guard=guard,
                              log=lambda m: print(f"[{time.strftime('%H:%M:%S')}] {run} {m} "
                                                  f"(RAM avail {available_ram_gb():.1f} GB)", flush=True))
    print(f"{run}: all conditions ready in {time.time() - t:.0f}s; min RAM seen {guard.min_seen_gb:.1f} GB", flush=True)
    del model
    torch.cuda.empty_cache()
    out = {}
    for n in names:
        e = evaluate_logits(rows[n], logits[n])
        out[n] = {"frame_auroc": e["frame"]["auroc"], "video_auroc": e["video"]["auroc"],
                  "frame_auprc": e["frame"]["auprc"], "video_auprc": e["video"]["auprc"],
                  "video_fpr@0.5": fpr_video(rows[n], logits[n]),
                  "video_tpr@0.5": e["video"]["tpr_fake"], "video_balanced_acc@0.5": e["video"]["balanced_accuracy"],
                  "frame_nll": e["frame"]["nll"], "video_ece": e["video"]["ece"],
                  "per_method_video_auroc": {m: e["video"]["per_method"][m]["auroc"] for m in METHODS},
                  "per_method_frame_auroc": {m: e["frame"]["per_method"][m]["auroc"] for m in METHODS}}
    return {"checkpoint_sha256": sha, "conditions": out, "training": training_cost(run_dir)}


def training_cost(run_dir: Path) -> dict:
    s = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    ep = [json.loads(line) for line in (run_dir / "epochs.jsonl").read_text(encoding="utf-8").splitlines()]
    secs = sum(e["train_s"] + e["val_s"] for e in ep)
    return {"epochs_run": s["epochs_run"], "best_epoch": s["best_epoch"], "train_val_minutes": secs / 60,
            "median_train_img_per_s": statistics.median(e["train_img_per_s"] for e in ep),
            "peak_vram_mb": max(e["peak_vram_mb"] for e in ep)}


def summarize(res: dict) -> dict:
    c = res["conditions"]
    deg = {k: v for k, v in c.items() if k != "clean"}
    sev = {cd["name"]: cd["severity"] for cd in CONDITIONS}
    worst_v = min(deg, key=lambda k: deg[k]["video_auroc"])
    worst_f = min(deg, key=lambda k: deg[k]["frame_auroc"])
    by_sev = {}
    for level in ("mild", "moderate", "severe"):
        ks = [k for k in deg if sev[k] == level]
        by_sev[level] = {"mean_video_auroc": float(np.mean([deg[k]["video_auroc"] for k in ks])),
                         "mean_frame_auroc": float(np.mean([deg[k]["frame_auroc"] for k in ks]))}
    return {"clean_video_auroc": c["clean"]["video_auroc"], "clean_frame_auroc": c["clean"]["frame_auroc"],
            "worst_video_auroc": deg[worst_v]["video_auroc"], "worst_video_condition": worst_v,
            "worst_frame_auroc": deg[worst_f]["frame_auroc"], "worst_frame_condition": worst_f,
            "mean_degraded_video_auroc": float(np.mean([v["video_auroc"] for v in deg.values()])),
            "mean_degraded_frame_auroc": float(np.mean([v["frame_auroc"] for v in deg.values()])),
            "max_degraded_video_fpr": max(v["video_fpr@0.5"] for v in deg.values()),
            "mean_degraded_video_fpr": float(np.mean([v["video_fpr@0.5"] for v in deg.values()])),
            "by_severity": by_sev,
            "delta_video_auroc": {k: v["video_auroc"] - c["clean"]["video_auroc"] for k, v in deg.items()}}


def cmd_evaluate(args) -> int:
    if not torch.cuda.is_available():
        print("REFUSED: CUDA not available")
        return 2
    s = suite()
    for cond in CONDITIONS:
        if not s.is_complete(cond["name"]):
            print(f"REFUSED: condition {cond['name']} not built; run `build` first")
            return 2
    base, robust = args.runs.split(",")
    report = {"generated": datetime.now().isoformat(timespec="seconds"), "suite_tag": s.tag, "split": "val",
              "conditions": list(CONDITIONS), "runs": {}}
    guard = RamGuard(args.ram_floor_gb)
    print(f"available RAM {guard.check():.1f} GB (floor {args.ram_floor_gb} GB); workers {args.workers}", flush=True)
    for run in (base, robust):
        try:
            r = score_run(run, s, args.workers, guard)
        except LowMemoryError as e:
            print(f"STOPPED SAFELY: {e}. Completed conditions are saved; rerun the same command to resume.")
            return 3
        r["summary"] = summarize(r)
        report["runs"][run] = r
    b, rb = report["runs"][base]["summary"], report["runs"][robust]["summary"]
    clean_loss = b["clean_video_auroc"] - rb["clean_video_auroc"]
    improves = rb["worst_video_auroc"] > b["worst_video_auroc"] and rb["mean_degraded_video_auroc"] > b["mean_degraded_video_auroc"]
    report["decision"] = {
        "rule": f"prefer robust iff worst-case AND mean degraded video AUROC improve and clean video AUROC loss <= {MAX_CLEAN_VIDEO_AUROC_LOSS}",
        "clean_video_auroc_loss": clean_loss, "robustness_improves": improves,
        "prefer_robust": bool(improves and clean_loss <= MAX_CLEAN_VIDEO_AUROC_LOSS)}
    # Existing calibration artifacts are bound to the base checkpoint: they must be refused for the robust one.
    rdir, bdir = (resolve_checkpoint_dir() / "distill" / n for n in (robust, base))
    inval = {}
    for art in ("calibration.json", "adaptive_calibration.json"):
        if (bdir / art).exists():
            try:
                load_calibration(bdir / art, rdir / "best.pt")
                inval[art] = "ACCEPTED (unexpected)"
            except CalibrationMismatchError as e:
                inval[art] = f"refused: {type(e).__name__}"
    report["calibration_artifacts_vs_robust_checkpoint"] = inval
    out = resolve_checkpoint_dir() / "distill" / "reports" / f"robust_eval_{datetime.now():%Y%m%d-%H%M%S}.json"
    out.write_text(json.dumps(report, indent=1, default=float), encoding="utf-8")

    names = ["clean"] + [c["name"] for c in CONDITIONS]
    print(f"{'condition':28s} {'base vAUC':>9s} {'robust vAUC':>11s} {'base fAUC':>9s} {'robust fAUC':>11s} {'base FPR':>8s} {'rob FPR':>8s}")
    for n in names:
        x, y = report["runs"][base]["conditions"][n], report["runs"][robust]["conditions"][n]
        print(f"{n:28s} {x['video_auroc']:9.4f} {y['video_auroc']:11.4f} {x['frame_auroc']:9.4f} {y['frame_auroc']:11.4f} "
              f"{x['video_fpr@0.5']:8.4f} {y['video_fpr@0.5']:8.4f}")
    for run in (base, robust):
        print(run, json.dumps({k: v for k, v in report["runs"][run]["summary"].items() if k != "delta_video_auroc"}, default=float))
        print("  training", json.dumps(report["runs"][run]["training"], default=float))
    print("decision", json.dumps(report["decision"]))
    print("calibration artifacts vs robust checkpoint:", inval)
    print("report", out)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--workers", type=int, default=12)
    e = sub.add_parser("evaluate")
    e.add_argument("--runs", default="student_distilled_p80,student_distilled_robust_p80")
    e.add_argument("--workers", type=int, default=4, choices=range(0, 7), help="max 6 (RAM)")
    e.add_argument("--ram-floor-gb", type=float, default=4.0)
    args = ap.parse_args()
    return cmd_build(args) if args.cmd == "build" else cmd_evaluate(args)


if __name__ == "__main__":
    raise SystemExit(main())
