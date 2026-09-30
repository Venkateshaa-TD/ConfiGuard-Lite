"""Configuration-driven training CLI for MobileNetV4-Conv-Small (default)
or EfficientNet-B0.

Usage:
    # real training (manifests built with configuard.datasets over local data)
    .venv/Scripts/python.exe scripts/train.py --config configs/train/mobilenetv4_conv_small.yaml \
        --train-manifest D:/.../train.jsonl --val-manifest D:/.../val.jsonl --media-root D:/.../media
    # resume exactly from the last completed epoch
    .venv/Scripts/python.exe scripts/train.py --config ... --resume-from D:/.../<run>_latest.pt

    # synthetic smoke training (ENGINEERING TEST ONLY - not deepfake accuracy)
    .venv/Scripts/python.exe scripts/train.py --smoke cpu
    .venv/Scripts/python.exe scripts/train.py --smoke cuda                      # RTX 4050, AMP
    .venv/Scripts/python.exe scripts/train.py --smoke cuda --encoder efficientnet_b0

Checkpoints go to CONFIGUARD_CHECKPOINT_DIR, logs/outputs to
CONFIGUARD_OUTPUT_DIR, face-crop caches to CONFIGUARD_CACHE_DIR (all from
.env; refused if inside the repo). Never downloads a dataset or model:
HF_HUB_OFFLINE is forced so only the cached Phase 4 weights load.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import replace
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from configuard.env_loader import load_dotenv  # noqa: E402

load_dotenv(REPO_ROOT / ".env")  # MUST run before importing timm/torch/huggingface_hub
# Training never downloads a model: serve only the cached Phase 4 weights.
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import torch  # noqa: E402

from configuard.dependency_safety import assert_dependency_safety  # noqa: E402
from configuard.media.face_detector import YuNetFaceDetector  # noqa: E402
from configuard.media.types import PreprocessingConfig as FacePreprocessingConfig  # noqa: E402
from configuard.models.registry import list_encoder_names  # noqa: E402
from configuard.training.config import TrainingConfig  # noqa: E402
from configuard.training.paths import resolve_cache_dir, resolve_checkpoint_dir, resolve_output_dir  # noqa: E402
from configuard.training.runner import build_trainer, run_smoke  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="ConfiGuard-Lite training CLI")
    parser.add_argument("--config", help="Training config YAML (configs/train/*.yaml)")
    parser.add_argument("--train-manifest", help="Overrides config.train_manifest")
    parser.add_argument("--val-manifest", help="Overrides config.val_manifest")
    parser.add_argument("--media-root", help="Overrides config.media_root")
    parser.add_argument("--resume-from", default=None, help="Checkpoint to resume from (refused on any mismatch)")
    parser.add_argument("--smoke", choices=["cpu", "cuda"], help="Synthetic smoke training on this device")
    parser.add_argument("--encoder", choices=list_encoder_names(), default="mobilenetv4_conv_small",
                        help="Encoder for --smoke (config files set their own)")
    parser.add_argument("--epochs", type=int, default=5, help="Epochs for --smoke")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for --smoke")
    args = parser.parse_args()

    # Fails loudly if torch went CPU-only, torchvision broke, or CUDA vanished.
    safety = assert_dependency_safety(require_cuda=args.smoke == "cuda" or torch.cuda.is_available())
    print(f"dependency safety OK: torch={safety.torch_version} torchvision={safety.torchvision_version} "
          f"cuda_available={safety.cuda_available}")

    if args.smoke:
        summary = run_smoke(
            device=args.smoke, encoder_name=args.encoder,
            checkpoint_dir=resolve_checkpoint_dir(), output_dir=resolve_output_dir(),
            epochs=args.epochs, batch_size=args.batch_size,
        )
        print(json.dumps(summary, indent=2, default=str))
        return 0

    if not args.config:
        parser.error("--config is required unless --smoke is given")
    config = TrainingConfig.from_yaml(args.config)
    overrides = {
        k: v for k, v in {
            "train_manifest": args.train_manifest, "val_manifest": args.val_manifest, "media_root": args.media_root,
        }.items() if v
    }
    config = replace(config, **overrides)
    if not config.train_manifest or not config.media_root:
        parser.error("train_manifest and media_root must be set (config or --train-manifest/--media-root)")

    detector = YuNetFaceDetector()
    face_config = FacePreprocessingConfig(detector_name=detector.name, detector_version=detector.version)
    trainer = build_trainer(
        config, detector=detector, face_config=face_config,
        checkpoint_dir=resolve_checkpoint_dir(config.checkpoint_dir or None),
        output_dir=resolve_output_dir(config.output_dir or None),
        face_cache_dir=resolve_cache_dir() / "face_crops",
    )
    results = trainer.fit(resume_from=args.resume_from)

    print(f"Training complete: {len(results)} epoch(s) run this invocation "
          f"(best {trainer.state.best_metric_name}={trainer.state.best_metric} at epoch {trainer.state.best_epoch}).")
    for r in results:
        auroc = r.val_metrics.threshold_free.auroc if r.val_metrics else None
        print(f"  epoch={r.epoch} train_loss={r.train_loss:.4f} val_loss={r.val_loss} val_auroc={auroc} "
              f"step={r.mean_step_time_seconds * 1000:.1f}ms val_time={r.val_time_seconds:.1f}s "
              f"peak_gpu_mb={r.peak_gpu_memory_mb}")
    print(f"checkpoints: {trainer.checkpoint_path('latest')} / {trainer.checkpoint_path('best')}")
    print(f"logs: {trainer.logger.jsonl_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
