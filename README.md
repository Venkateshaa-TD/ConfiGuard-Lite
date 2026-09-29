# ConfiGuard-Lite

Efficient and uncertainty-aware deepfake image and video detection, designed
to train within a 6 GB VRAM laptop GPU budget and deploy on CPU-only
servers with optional GPU acceleration.

See `docs/PROJECT_PLAN.md` for the full requirements and phase roadmap,
`docs/ARCHITECTURE.md` for the system design, and `CLAUDE.md` for the
operating rules this project is developed under.

## Status

Phase 4 (pretrained baseline models and ONNX verification) — see
`docs/PHASE_STATUS.md` for current status. The end-to-end pipeline
(`configuard.pipeline`) runs but uses a **deterministic dummy predictor**,
not a trained model. The face preprocessing subsystem (`configuard.media`),
the dataset registry (`configuard.datasets`), and two pretrained visual
encoders (`configuard.models`) are all real and independently tested, but
none of them are wired into the main pipeline or a real training loop yet
— see `docs/ARCHITECTURE.md` for what's real vs. placeholder. **No
training dataset has been downloaded.** Two small ImageNet-pretrained
backbones (MobileNetV4-Conv-Small, EfficientNet-B0) *have* been
downloaded, per explicit authorization — see `docs/DATASETS.md` for full
provenance. **Every prediction `configuard.models` produces is untrained
and uncalibrated** — see the warning in `docs/MODEL_CARD.md`.

## Requirements

- Windows 10/11 (Linux/Docker compatibility is a design goal, not yet built)
- Python 3.11 (3.14 is not yet supported by the ML dependency stack — see
  `docs/DECISIONS.md`)
- Git
- NVIDIA GPU + recent driver, optional (CPU-only inference is supported)
- FFmpeg (install with `winget install --id Gyan.FFmpeg -e`, then open a new
  terminal so the updated `PATH` takes effect)
- The YuNet face detector ONNX model, for real face detection (optional -
  `configuard.media.face_detector.MockFaceDetector` works without it, e.g.
  for tests). See `docs/DATASETS.md` for the official download source,
  license, and SHA-256 to verify against; place it at
  `models/face_detection/face_detection_yunet_2026may.onnx` (gitignored)
  or point `CONFIGUARD_YUNET_MODEL_PATH` at it.
- The two pretrained backbone weights (MobileNetV4-Conv-Small,
  EfficientNet-B0), for real encoder construction (optional - tests use
  `pretrained=False`, no download needed). Set `HF_HOME`/`HF_HUB_CACHE`/
  `TORCH_HOME` in `.env` first (see `.env.example`), then run
  `scripts\download_baseline_models.py`. See `docs/DATASETS.md` for full
  provenance.

## Setup

```powershell
# 1. Create the virtual environment (Python 3.11)
py -3.11 -m venv .venv

# 2. Install PyTorch matching your hardware
#    GPU (NVIDIA, CUDA 12.1-compatible driver):
.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu121
#    CPU-only:
.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu

# 3. Install the remaining dependencies
.venv\Scripts\python.exe -m pip install -r requirements-dev.txt

# 4. Verify PyTorch still has the GPU build after step 3 (installing new
#    packages can silently re-resolve/downgrade torch - see docs/KNOWN_ISSUES.md)
.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

## Verify your environment

```powershell
.venv\Scripts\python.exe scripts\verify_environment.py
```

This reports Python version, Git/FFmpeg availability, and PyTorch/CUDA/GPU
detection without exposing any usernames, tokens, or secrets.

## Run tests

```powershell
.venv\Scripts\python.exe -m pytest -v
```

## Run the pipeline on a file (Phase 1: dummy prediction)

```python
import json
from configuard.config import load_config
from configuard.pipeline import run_pipeline
from configuard.io_types import result_to_dict

config = load_config("configs/development.yaml")
result = run_pipeline("path/to/image_or_video.mp4", config.validation)
print(json.dumps(result_to_dict(result), indent=2))
```

Invalid, corrupted, oversized, or wrong-type files raise
`configuard.pipeline.PipelineRejectedError` (carries the full
`ValidationResult` with human-readable `error_messages()`), rather than
producing a prediction.

## Run face preprocessing on a file (Phase 2)

```python
from configuard.media.cache import FaceCropCache
from configuard.media.face_detector import YuNetFaceDetector
from configuard.media.preprocess import preprocess_image, preprocess_video
from configuard.media.types import PreprocessingConfig

detector = YuNetFaceDetector()  # requires the model file - see Requirements above
cache = FaceCropCache("cache/face_crops", max_bytes=2_000_000_000)
config = PreprocessingConfig(detector_name=detector.name, detector_version=detector.version)

image_result = preprocess_image("photo.jpg", detector, cache, config)
video_result = preprocess_video("clip.mp4", requested_frame_count=8, detector=detector, cache=cache, config=config)
```

Benchmark each stage separately:

```powershell
.venv\Scripts\python.exe scripts\benchmark_preprocessing.py
```

## Build a dataset manifest (Phase 3)

```python
from configuard.datasets.registry import DEFAULT_REGISTRY
from configuard.datasets.manifest import write_manifest, validate_manifest_file
from configuard.datasets.splitting import SplitConfig, split_samples, write_split_audit_report
from configuard.datasets.duplicates import build_duplicate_report

adapter = DEFAULT_REGISTRY.get_adapter("faceforensics++")  # or celeb-df-v2 / dfdc / df40 / deeperforensics-1.0
samples = adapter.build_manifest("/path/to/your/local/FaceForensics++")  # never downloaded for you - see docs/DATASETS.md
write_manifest(samples, "manifest.jsonl")

report = validate_manifest_file("manifest.jsonl", media_root="/path/to/your/local/FaceForensics++")
split_report = split_samples(samples, SplitConfig(seed=42))
write_split_audit_report(split_report, "split_audit.json")
dup_report = build_duplicate_report(samples, media_root="/path/to/your/local/FaceForensics++")
```

A missing/incomplete local dataset root raises
`configuard.datasets.adapters.DatasetAccessError` with a clear message —
adapters never download anything.

Check your configured storage paths before pointing anything real at them:

```powershell
.venv\Scripts\python.exe scripts\check_storage.py
```

## Run a pretrained encoder (Phase 4 — UNTRAINED/uncalibrated output)

```python
import numpy as np
from configuard.models.registry import create_encoder
from configuard.models.inference import infer_image, infer_video_fixed_frames

encoder = create_encoder("mobilenetv4_conv_small", pretrained=True)  # or "efficientnet_b0"
config = encoder.resolve_preprocess_config()

image = np.zeros((224, 224, 3), dtype=np.uint8)  # a real aligned face crop, e.g. from configuard.media
result = infer_image(encoder, image, config)
print(result.probability, result.disclaimer)  # ALWAYS UNTRAINED_UNCALIBRATED right now

frames = [image] * 8  # ordered 4/8/16-frame sample, e.g. from configuard.media.sampling
video_result = infer_video_fixed_frames(encoder, frames, config)
```

Download the two authorized pretrained backbones (requires `HF_HOME`/
`HF_HUB_CACHE`/`TORCH_HOME` set in `.env` first):

```powershell
.venv\Scripts\python.exe scripts\download_baseline_models.py
.venv\Scripts\python.exe scripts\export_onnx_models.py     # FP32 ONNX + parity check
.venv\Scripts\python.exe scripts\benchmark_models.py         # latency/memory/FLOPs
```

## Project layout

See `docs/ARCHITECTURE.md` for the full repository layout and design.

## Configuration

Non-secret defaults live in `configs/*.yaml`. Machine-specific paths and
secrets go in a local `.env` file (copy `.env.example` to `.env` — `.env` is
never committed). Point `CONFIGUARD_DATA_DIR` / `CONFIGUARD_CACHE_DIR` /
`CONFIGUARD_CHECKPOINT_DIR` / `CONFIGUARD_OUTPUT_DIR` at a location
outside the repo (and outside any cloud-synced folder like OneDrive) with
enough free space — run `scripts\check_storage.py` to verify before
storing anything real. This machine's configured layout is recorded in
`docs/ARCHITECTURE.md`.
