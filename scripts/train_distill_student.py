"""Phase 6b: train a MobileNetV4-Conv-Small student on the FF++ c23 crops
(train) with early stopping on val. Never opens the test split and never
loads GenD: distillation uses the Phase 6a cached logits only.

Modes:
    train  one run from the YAML (+ overrides); resumes from last.pt
    pilot  short-budget grid over (alpha, temperature) plus a baseline,
           ranked by val frame AUROC; writes pilot_report.json

Usage:
    .venv/Scripts/python.exe scripts/train_distill_student.py pilot --epochs 3 --samples-per-epoch 20000
    .venv/Scripts/python.exe scripts/train_distill_student.py train --run-name student_baseline --alpha 0
    .venv/Scripts/python.exe scripts/train_distill_student.py train --run-name student_distilled --alpha 0.5 --temperature 2
    # Phase 6c: same settings, trained on the 80% final_train partition only
    .venv/Scripts/python.exe scripts/train_distill_student.py train --run-name student_distilled_p80 --alpha 0.5 --temperature 2 --train-partition final_train
    # Phase 6e: same, plus class-independent compression/quality degradations (default RobustAugmentConfig)
    .venv/Scripts/python.exe scripts/train_distill_student.py train --run-name student_distilled_robust_p80 --alpha 0.5 --temperature 2 --train-partition final_train --robust
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from configuard.env_loader import load_dotenv  # noqa: E402

load_dotenv(REPO_ROOT / ".env")

import torch  # noqa: E402

from configuard.calibration.partitions import default_partitions_path  # noqa: E402
from configuard.distill.train import DistillConfig, StudentTrainer  # noqa: E402
from configuard.robust.degrade import RobustAugmentConfig  # noqa: E402
from configuard.storage_guard import FreeSpaceGuard  # noqa: E402
from configuard.training.paths import resolve_cache_dir, resolve_checkpoint_dir  # noqa: E402

DEFAULT_CONFIG = REPO_ROOT / "configs" / "distill" / "mobilenetv4_student.yaml"


def setup_logging(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("[%(asctime)s] %(message)s", "%H:%M:%S")
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    for h in (logging.StreamHandler(sys.stdout), logging.FileHandler(path, encoding="utf-8")):
        h.setFormatter(fmt)
        root.addHandler(h)


def run(cfg: DistillConfig, run_root: Path, max_epochs: int | None = None) -> dict:
    cache = resolve_cache_dir()
    parts = default_partitions_path(cache, cfg.crop_tag) if cfg.train_partition else None
    trainer = StudentTrainer(cfg, cache / "ffpp_face_crops" / "store", cache / "teacher_logits" / "gend_clip_l14",
                             run_root, device="cuda", partitions_path=parts)
    try:
        return trainer.fit(max_epochs)
    finally:
        trainer.close()


def brief(s: dict) -> dict:
    f, v = s["val_best"]["frame"], s["val_best"]["video"]
    return {"run": s["run_name"], "alpha": s["config"]["alpha"], "T": s["config"]["temperature"],
            "best_epoch": s["best_epoch"], "frame_auroc": f["auroc"], "frame_nll": f["nll"], "frame_ece": f["ece"],
            "video_auroc": v["auroc"], "per_method_frame_auroc": {m: r["auroc"] for m, r in f["per_method"].items()}}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("mode", choices=("train", "pilot"))
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--run-name")
    ap.add_argument("--alpha", type=float)
    ap.add_argument("--temperature", type=float)
    ap.add_argument("--max-epochs", type=int)
    ap.add_argument("--train-partition", help="Phase 6c: e.g. final_train (needs scripts/calibrate_student.py split)")
    ap.add_argument("--robust", action="store_true", help="Phase 6e: default RobustAugmentConfig degradations")
    ap.add_argument("--epochs", type=int, default=3, help="pilot: epochs per run")
    ap.add_argument("--samples-per-epoch", type=int, help="pilot default 20000; train default from config")
    ap.add_argument("--alphas", default="0.5,0.9")
    ap.add_argument("--temperatures", default="1,2,4")
    ap.add_argument("--floor-gb", type=float, default=40.0)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("REFUSED: CUDA not available")
        return 2
    root = resolve_checkpoint_dir() / "distill"
    guard = FreeSpaceGuard(root.anchor, args.floor_gb, 1.0)
    if not guard.ok():
        print(f"REFUSED: free space {guard.free_gb():.1f} GB below floor")
        return 2
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    setup_logging(root / "reports" / f"{args.mode}_{stamp}.log")
    log = logging.getLogger("configuard.distill")

    if args.mode == "train":
        cfg = DistillConfig.from_yaml(args.config, run_name=args.run_name, alpha=args.alpha,
                                      temperature=args.temperature, max_epochs=args.max_epochs,
                                      samples_per_epoch=args.samples_per_epoch, train_partition=args.train_partition,
                                      robust_augment=RobustAugmentConfig().to_dict() if args.robust else None)
        t0 = time.time()
        s = run(cfg, root)
        log.info("done in %.1f min: %s", (time.time() - t0) / 60, json.dumps(brief(s), default=float))
        return 0

    spe = args.samples_per_epoch or 20000
    grid = [(0.0, 1.0)] + [(float(a), float(t)) for a in args.alphas.split(",") for t in args.temperatures.split(",")]
    pilot_root = root / "pilot"
    results = []
    for alpha, temp in grid:
        name = "pilot_baseline" if alpha == 0 else f"pilot_a{alpha:g}_t{temp:g}"
        cfg = DistillConfig.from_yaml(args.config, run_name=name, alpha=alpha, temperature=temp,
                                      max_epochs=args.epochs, samples_per_epoch=spe, early_stopping_patience=args.epochs)
        t0 = time.time()
        results.append(brief(run(cfg, pilot_root)) | {"minutes": round((time.time() - t0) / 60, 1)})
        log.info("pilot %s: %s", name, json.dumps(results[-1], default=float))
    distilled = [r for r in results if r["alpha"] > 0]
    best = max(distilled, key=lambda r: (r["frame_auroc"], -r["frame_nll"]))
    report = {"generated": stamp, "budget": {"epochs": args.epochs, "samples_per_epoch": spe},
              "selection": "max val frame AUROC, tie -> lower val frame NLL", "results": results,
              "chosen": {"alpha": best["alpha"], "temperature": best["T"]}}
    out = pilot_root / f"pilot_report_{stamp}.json"
    out.write_text(json.dumps(report, indent=1, default=float), encoding="utf-8")
    log.info("pilot chosen alpha=%s T=%s -> %s", best["alpha"], best["T"], out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
