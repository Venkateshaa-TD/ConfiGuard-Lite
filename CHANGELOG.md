# Changelog

All notable changes to this project are documented here.

## [Unreleased]

### Phase 0 — Environment and repository foundation (2026-09-29)

- Initialized Git repository and project scaffolding.
- Created `src/configuard` package with `config.py` (YAML config loading)
  and `env_check.py` (Python/Git/FFmpeg/CUDA/GPU device detection).
- Added `scripts/verify_environment.py` environment verification script.
- Added Phase 0 test suite (`tests/test_imports.py`, `tests/test_device.py`,
  `tests/test_config.py`).
- Added project documentation set (README, CLAUDE.md, docs/*).
- Added `.gitignore`, `.env.example`, `pyproject.toml`,
  `requirements.txt`, `requirements-dev.txt`.
- Created Python 3.11 virtual environment (`.venv`) with PyTorch (CUDA 12.1
  build), NumPy, PyYAML, pytest.

### Pre-Phase-1 (2026-09-29)

- Installed FFmpeg 9.0.2 via `winget` (user-approved), verified with
  `ffmpeg -version` / `ffprobe -version`. No CUDA Toolkit installed.

### Phase 1 — Architecture contracts and minimal vertical slice (2026-09-29)

- Added `src/configuard/io_types.py`: typed contracts for image/video
  input, validated media, preprocessing/model/provenance output, and the
  final `DetectionResult` (+ `result_to_dict` JSON serialization).
- Added `src/configuard/validation.py`: secure file validation via
  magic-byte content sniffing, extension cross-check, size limits, and
  `ffprobe`-based video duration/corruption checks.
- Added `src/configuard/pipeline.py`: end-to-end orchestration
  (`run_pipeline`) with a deterministic dummy predictor and the adaptive
  4/8/16 frame-count contract (`select_frame_count`).
- Extended `src/configuard/config.py` with `environment` and
  `ValidationLimits`; added `configs/{development,training,testing,production}.yaml`.
- Added `tests/conftest.py` (hand-built PNG + ffmpeg-generated tiny video
  fixtures, no downloads) and 36 new unit/integration tests across
  `test_io_types.py`, `test_validation.py`, `test_pipeline.py`,
  `test_config_environments.py`. Full suite: 48/48 passing.
- Updated `docs/ARCHITECTURE.md` with the full data flow and JSON schema.

### Phase 2 — Face and media preprocessing (2026-09-29)

- Added `src/configuard/media/` subsystem: `decode.py` (image decoding,
  ffprobe-preferred video metadata), `sampling.py` (deterministic nested
  4/8/16 frame sampling), `face_detector.py` (pluggable `FaceDetector`
  interface, `YuNetFaceDetector` CPU backend, `MockFaceDetector` test
  double), `alignment.py` (eye-leveling + margin + crop), `tracking.py`
  (IoU + landmark video face tracking, primary-track selection),
  `cache.py` (versioned, atomic face-crop cache), `hashing.py`,
  `preprocess.py` (orchestration for images and video).
- Downloaded the official YuNet face detector ONNX
  (`face_detection_yunet_2026may.onnx`, MIT license, 229,738 bytes,
  SHA-256 `ebafce4e3...22f0f0`) from opencv/opencv_zoo — the only asset
  this phase downloaded; full provenance in `docs/DATASETS.md`.
- Added `opencv-python-headless` to `requirements.txt`.
- Added `scripts/benchmark_preprocessing.py` (decode/sampling/detection/
  alignment/cache-hit-vs-miss timings).
- Added 76 new unit/integration tests under `tests/media/` (mock-detector
  unit tests + a small number of real-YuNet integration tests). Full
  suite: 124/124 passing.
- Fixed a Phase 0 `.gitignore` bug (`dir/` vs `dir/*`) that silently
  dropped `data/.gitkeep`, `checkpoints/.gitkeep`, `cache/.gitkeep`,
  `outputs/.gitkeep` from version control; added `models/*` +
  `!models/.gitkeep` for the new external-asset directory.
- Updated `docs/ARCHITECTURE.md`, `docs/DECISIONS.md`, `docs/DATASETS.md`,
  `docs/KNOWN_ISSUES.md`, `docs/PROJECT_PLAN.md` (reordered: preprocessing
  moved ahead of the dataset registry).
