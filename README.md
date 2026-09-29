# ConfiGuard-Lite

Efficient and uncertainty-aware deepfake image and video detection, designed
to train within a 6 GB VRAM laptop GPU budget and deploy on CPU-only
servers with optional GPU acceleration.

See `docs/PROJECT_PLAN.md` for the full requirements and phase roadmap,
`docs/ARCHITECTURE.md` for the system design, and `CLAUDE.md` for the
operating rules this project is developed under.

## Status

Phase 3 (dataset registry and leakage-safe data splits) — see
`docs/PHASE_STATUS.md` for current status. The end-to-end pipeline
(`configuard.pipeline`) runs but uses a **deterministic dummy predictor**,
not a trained model. The face preprocessing subsystem (`configuard.media`)
and the dataset registry (`configuard.datasets`) are both real and
independently tested, but neither is wired into the pipeline or into any
training loop yet — see `docs/ARCHITECTURE.md` for what's real vs.
placeholder. **No training dataset has been downloaded** - the registry
was built and tested entirely against synthetic fixtures (see
`docs/DATASETS.md`).

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

## Project layout

See `docs/ARCHITECTURE.md` for the full repository layout and design.

## Configuration

Non-secret defaults live in `configs/*.yaml`. Machine-specific paths and
secrets go in a local `.env` file (copy `.env.example` to `.env` — `.env` is
never committed).
