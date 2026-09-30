"""Evaluation-only CLI: loads a checkpoint's model weights and reports
validation metrics on a manifest - no training, no optimizer/scheduler.

Usage:
    .venv/Scripts/python.exe scripts/evaluate.py --checkpoint D:/.../<run>_best.pt \
        --manifest D:/.../val.jsonl --media-root D:/.../media
    # synthetic smoke-run checkpoints were built with the full-frame mock detector:
    .venv/Scripts/python.exe scripts/evaluate.py --checkpoint ... --manifest ... --media-root ... --synthetic

The encoder is read from the checkpoint's provenance. Metrics JSON is
printed and written to CONFIGUARD_OUTPUT_DIR/eval/. Face crops are cached
under CONFIGUARD_CACHE_DIR. Never downloads anything (HF_HUB_OFFLINE).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from configuard.env_loader import load_dotenv  # noqa: E402

load_dotenv(REPO_ROOT / ".env")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from configuard.datasets.manifest import read_manifest  # noqa: E402
from configuard.dependency_safety import assert_dependency_safety  # noqa: E402
from configuard.media.cache import FaceCropCache  # noqa: E402
from configuard.media.face_detector import YuNetFaceDetector  # noqa: E402
from configuard.media.types import PreprocessingConfig as FacePreprocessingConfig  # noqa: E402
from configuard.models.device import resolve_device  # noqa: E402
from configuard.models.registry import create_encoder  # noqa: E402
from configuard.training import synthetic  # noqa: E402
from configuard.training.checkpoint import load_checkpoint  # noqa: E402
from configuard.training.dataloader import build_dataloader  # noqa: E402
from configuard.training.paths import resolve_cache_dir, resolve_output_dir  # noqa: E402
from configuard.training.runner import SMOKE_FACE_CONFIG, build_manifest_dataset  # noqa: E402
from configuard.training.trainer import evaluate_checkpoint  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="ConfiGuard-Lite evaluation-only CLI")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--media-root", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--threshold", type=float, default=0.5, help="Only affects threshold-dependent metrics")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--synthetic", action="store_true",
                        help="Use the full-frame mock detector the synthetic smoke runs used (engineering test only)")
    args = parser.parse_args()

    assert_dependency_safety(require_cuda=False)

    device = resolve_device(args.device)
    checkpoint = load_checkpoint(args.checkpoint, map_location="cpu")
    provenance = checkpoint.provenance
    saved_config = provenance.config or {}

    if args.synthetic:
        detector, face_config = synthetic.full_frame_detector(), SMOKE_FACE_CONFIG
    else:
        detector = YuNetFaceDetector()
        face_config = FacePreprocessingConfig(detector_name=detector.name, detector_version=detector.version)
    if face_config.version_tag != provenance.preprocessing_version:
        print(f"[WARNING] preprocessing version differs from the checkpoint's "
              f"({face_config.version_tag} vs {provenance.preprocessing_version}) - metrics may not be comparable.")

    # pretrained=False: every weight is overwritten by the checkpoint anyway.
    encoder = create_encoder(provenance.encoder_name, pretrained=False)
    model_preprocess = encoder.resolve_preprocess_config()
    samples = read_manifest(args.manifest)
    dataset = build_manifest_dataset(
        samples, args.media_root, int(saved_config.get("frames_per_video", 8)), detector,
        FaceCropCache(resolve_cache_dir() / "face_crops"), face_config, model_preprocess,
    )
    loader = build_dataloader(
        dataset, dataset.samples, batch_size=args.batch_size, seed=0,
        balance_source=False, balance_class=False, shuffle_if_unbalanced=False,
    )

    start = time.perf_counter()
    metrics = evaluate_checkpoint(args.checkpoint, encoder, loader, device, threshold=args.threshold)
    elapsed = time.perf_counter() - start

    result = {
        "checkpoint": str(Path(args.checkpoint)),
        "encoder_name": provenance.encoder_name,
        "checkpoint_epochs_completed": checkpoint.epoch,
        "manifest": str(Path(args.manifest)),
        "device": str(device),
        "eval_time_seconds": elapsed,
        "metrics": metrics.to_dict(),
    }
    if args.synthetic:
        result["disclaimer"] = synthetic.SYNTHETIC_RESULT_DISCLAIMER

    out_dir = resolve_output_dir() / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"eval_{Path(args.checkpoint).stem}_{time.strftime('%Y%m%d-%H%M%S')}.json"
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    result["written_to"] = str(out_path)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
