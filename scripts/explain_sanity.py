"""Phase 11: sanity checks for the Grad-CAM evidence hints on the real student.

On one crop per video (slot 0) from official TRAIN and VAL (never test):
  completeness       |sum(M) + c - logit| (exact decomposition)
  single-cell        occlude each of the 49 cells (blur sigma 6); Spearman between
  occlusion          the CAM cell values and the measured logit drop
  runtime check      pass rate of the per-explanation faithfulness rule
                     (top-10 cells vs 3 random 10-cell sets) and mean drops
  randomization      Spearman between the CAM and the CAM of a model whose
                     classifier weights are randomised (Adebayo et al. 2018); a
                     faithful map must change (low correlation)
Usage: .venv/Scripts/python.exe scripts/explain_sanity.py [--per-split 200]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from configuard.env_loader import load_dotenv  # noqa: E402

load_dotenv(REPO_ROOT / ".env")

import cv2  # noqa: E402
import numpy as np  # noqa: E402

CROP_TAG = "p5d-b451b5ca770c8923"


def spearman(a, b) -> float:
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--per-split", type=int, default=200)
    args = ap.parse_args()
    from configuard.distill.data import load_crop_rows
    from configuard.service.explain import BLUR_SIGMA, GRID, CamExplainer
    from configuard.training.paths import resolve_cache_dir, resolve_checkpoint_dir

    store = resolve_cache_dir() / "ffpp_face_crops" / "store"
    onnx_path = resolve_checkpoint_dir() / "export" / "student_p80" / "student_fp32.onnx"
    ex = CamExplainer(onnx_path, threads=0)
    rng = np.random.default_rng(0)
    w2_rand = rng.normal(0, ex.w2.std(), ex.w2.shape)
    report = {"generated": datetime.now().isoformat(timespec="seconds"), "per_split": args.per_split, "splits": {}}
    for split in ("train", "val"):
        rows, _ = load_crop_rows(store, CROP_TAG, split)
        first = sorted({r["sample_id"]: r for r in rows if r["slot"] == 0}.values(), key=lambda r: r["sample_id"])
        pick = [first[i] for i in np.random.default_rng(1).choice(len(first), args.per_split, replace=False)]
        crops = [cv2.imdecode(np.fromfile(str(store / r["crop_path"]), np.uint8), cv2.IMREAD_COLOR) for r in pick]
        t = time.perf_counter()
        comp, cell_rho, rand_rho, checks = [], [], [], []
        for b in range(0, len(crops), 16):
            chunk = crops[b:b + 16]
            px = np.stack([c[:, :, ::-1].transpose(2, 0, 1) for c in chunk]).astype(np.float32)
            cm = ex.cell_maps(px)
            comp += list(np.abs(cm.maps.sum(axis=(1, 2)) + cm.offsets - cm.logits))
            signs = np.where(cm.logits >= 0, 1.0, -1.0)
            checks += ex.faithfulness(chunk, cm.maps, cm.logits, signs)
            w2 = ex.w2
            ex.w2 = w2_rand
            rand_maps = ex.cell_maps(px).maps
            ex.w2 = w2
            for crop, m, z, rm in zip(chunk, cm.maps, cm.logits, rand_maps):
                blurred = cv2.GaussianBlur(crop, (0, 0), BLUR_SIGMA)
                occ = np.stack([ex._occlude(crop, blurred, np.array([c])) for c in range(GRID * GRID)])
                drops = z - ex.logits(occ[:, :, :, ::-1].transpose(0, 3, 1, 2).astype(np.float32))
                cell_rho.append(spearman(m.ravel(), drops))
                rand_rho.append(spearman(m.ravel(), rm.ravel()))
        passed = [c["passed"] for c in checks]
        report["splits"][split] = {
            "n": len(crops), "completeness_max_abs_error": float(np.max(comp)),
            "single_cell_occlusion_spearman_median": float(np.median(cell_rho)),
            "single_cell_occlusion_spearman_p10": float(np.percentile(cell_rho, 10)),
            "runtime_check_pass_rate": float(np.mean(passed)),
            "mean_drop_top_cells": float(np.mean([c["evidence_drop_top_cells"] for c in checks])),
            "mean_drop_random_max": float(np.mean([c["evidence_drop_random_max"] for c in checks])),
            "randomized_classifier_spearman_median": float(np.median(rand_rho)),
            "seconds": round(time.perf_counter() - t, 1)}
        print(split, json.dumps(report["splits"][split]), flush=True)
    out = resolve_checkpoint_dir() / "explain" / f"explain_sanity_{datetime.now():%Y%m%d-%H%M%S}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print("report", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
