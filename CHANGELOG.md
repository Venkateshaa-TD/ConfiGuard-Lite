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

### Phase 3 — Dataset registry and leakage-safe data splits (2026-09-29)

- Added `src/configuard/datasets/` subsystem: `schema.py` (canonical
  `Sample`), `registry.py` (typed `DatasetRegistry`), `manifest.py`
  (JSONL read/write, lenient + strict validation), `splitting.py`
  (union-find leakage grouping, deterministic hash-bucketed split
  assignment, JSON audit report), `duplicates.py` (SHA-256 exact +
  OpenCV average-hash near-duplicate detection), `storage.py`
  (storage-path safety checks, `UnsafeStoragePathError`).
- Added `adapters/` with two reusable engines
  (`FolderConventionAdapter`, `MetadataSidecarAdapter`) and
  `known_datasets.py` registering adapters for FaceForensics++,
  Celeb-DF-v2, DFDC, DF40, and DeeperForensics-1.0, plus
  `make_generic_metadata_adapter()` for future datasets.
- Added `scripts/check_storage.py` (CLI for the storage-check utility).
- Added 91 new unit/integration tests under `tests/datasets/`, including
  a full synthetic multi-scenario fixture (multiple sources, derivative
  fakes, paired real/fake, repeated identities, exact + near duplicates)
  and a deliberately-leaked split assignment proving the leakage detector
  actually catches leakage. Full suite: 215/215 passing.
- No dataset, checkpoint, or large file downloaded this phase.
- Updated `docs/DATASETS.md` (access/licensing per dataset, structure
  confidence table, "verification required" markers),
  `docs/ARCHITECTURE.md`, `docs/DECISIONS.md` (7 new entries),
  `docs/KNOWN_ISSUES.md`, `docs/PROJECT_PLAN.md`.

### Pre-Phase-4 storage configuration (2026-09-29)

- Ran a read-only storage audit (all drives, repo/.venv/pip-cache/HF-
  cache/torch-cache/project-dir/temp-dir sizes, explained the 21.8 GB ->
  12.9 GB drop, identified `D:` as a safe secondary volume).
- Created `D:\ConfiGuard-Data\{datasets,cache,checkpoints,outputs}` and
  configured local `.env` (never committed) to point
  `CONFIGUARD_DATA_DIR`/`CONFIGUARD_CACHE_DIR`/`CONFIGUARD_CHECKPOINT_DIR`/
  `CONFIGUARD_OUTPUT_DIR` there.
- Added `CONFIGUARD_OUTPUT_DIR` as a fourth storage-checked variable
  (`configuard.datasets.storage.STORAGE_ENV_VARS`, `.env.example`) - the
  project previously had an `outputs/` directory but no corresponding
  configurable external path.
- Verified every directory exists, is actually writable (real write
  test), is outside the repo, is outside OneDrive, and is recognized by
  `scripts/check_storage.py`.
- Purged the pip download cache only (`pip cache purge`), reclaiming
  3,365.8 MB on `C:` (12.92 GB -> 16.05 GB free); no other files touched.
- Updated `docs/ARCHITECTURE.md` (storage layout), `docs/DECISIONS.md`
  (2 new entries), `docs/DATASETS.md`, `docs/KNOWN_ISSUES.md` (disk-space
  entry marked RESOLVED), `README.md`, `.env.example`.

### Phase 4 — Pretrained baseline models and ONNX verification (2026-09-29)

- Configured `HF_HOME`/`HF_HUB_CACHE`/`TORCH_HOME` under
  `D:\ConfiGuard-Data\cache\{huggingface,torch}` via `.env` and a new
  `configuard.env_loader.load_dotenv()` (dependency-free, must run
  before any `import timm`/`torch`/`huggingface_hub`).
- Downloaded the two authorized pretrained backbones only -
  `timm/mobilenetv4_conv_small.e1200_r224_in1k` and
  `timm/tf_efficientnet_b0.in1k` (Apache-2.0, safetensors) - via
  `scripts/download_baseline_models.py`. Full provenance (revision,
  SHA-256, file size) in `docs/DATASETS.md`.
- Added `src/configuard/models/`: `encoder.py` (`DeepfakeVisualEncoder`,
  `EncoderSpec`, `PreprocessConfig`), `registry.py`, `preprocess.py`
  (shared 224×224 contract), `inference.py` (image + fixed-frame
  mean-aggregated video inference), `real_pipeline.py` (optional,
  dependency-injected wiring of Phase 2 preprocessing into a Phase 4
  encoder), `onnx_export.py` (FP32 export + parity verification),
  `benchmark.py` (warm-up + P50/P95 latency, GPU peak memory, FLOPs),
  `device.py`.
- Added `scripts/benchmark_models.py` and `scripts/export_onnx_models.py`.
- Every prediction carries `PREDICTION_DISCLAIMER` and `is_finetuned=False`
  - see the warning added to `docs/MODEL_CARD.md`.
- Found and fixed two real bugs (empirical `num_features` probing;
  eval-mode probe for BatchNorm batch-size-1 safety) and one environment
  regression (`timm`/`onnx`/`onnxruntime` install silently downgrading
  `torch` and mismatching `torchvision` - reinstalled the correct paired
  versions). All documented in `docs/DECISIONS.md` / `docs/KNOWN_ISSUES.md`.
- Added 90 new tests under `tests/models/` + 6 new
  `tests/test_env_loader.py` tests. Full suite: 311/311 passing.
- Benchmarked both models (parameters, FLOPs, CPU/GPU P50/P95 latency,
  peak GPU memory ~134 MB / 2.2% of the 6 GB budget, image-batch and
  4/8/16-frame video timings) and verified ONNX FP32 export + parity
  (max abs diff ~3-4e-07) for both - full numbers in
  `docs/EXPERIMENT_LOG.md`. Provisional only - no deepfake accuracy
  result exists yet.
- No dataset and no unauthorized model checkpoint downloaded.
