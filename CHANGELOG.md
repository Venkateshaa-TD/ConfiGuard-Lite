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

### Phase 5 — Reproducible training pipeline (2026-09-30)

- Dependency safety: `constraints-cuda.txt` pins torch 2.5.1 /
  torchvision 0.20.1 (cu121 index documented);
  `configuard.dependency_safety` hard-fails on CPU-only torch, a
  torchvision import/compiled-op (ABI) failure, a torch↔torchvision
  release mismatch, or lost CUDA; wired into `scripts/verify_environment.py`
  and both CLIs. README/requirements install steps now use the pin. No
  package was installed or upgraded.
- Added `src/configuard/training/`: `config.py` (strict YAML config),
  `paths.py` (D-drive checkpoint/output/cache dirs, refuses in-repo
  paths), `splits.py` (cross-split leakage guard), `datasets.py`
  (manifest-backed image and video-frame datasets over Phase 2 crops),
  `sampling.py`/`dataloader.py` (seeded, source+class-balanced,
  worker-seeded), `optim.py` (AdamW + warm-up/cosine), `metrics.py`
  (threshold-free AUROC/AP vs threshold-dependent confusion matrix,
  sensitivity, specificity, balanced accuracy, F1), `checkpoint.py`
  (atomic, full-state + provenance, mismatch refusal), `logging_utils.py`
  (JSONL + per-split CSV), `trainer.py` (AMP, accumulation, clipping,
  freezing, early stopping, best/latest, exact resume, non-finite-loss and
  CUDA-OOM handling), `runner.py` (wiring + CPU/RTX 4050 smoke mode),
  `synthetic.py` (engineering-only synthetic data).
- Added `scripts/train.py` (config-driven training, `--resume-from`,
  `--smoke cpu|cuda`) and `scripts/evaluate.py` (evaluation-only).
  Added `configs/train/{mobilenetv4_conv_small,efficientnet_b0}.yaml`.
- `EncoderSpec.revision` records the HF commit of each authorized
  backbone; `models.inference.compute_logits_for_batch` provides the
  gradient-friendly image/video forward.
- Fixed while resuming the interrupted draft: GPU resume crashed (RNG
  state loaded onto CUDA); resume wasn't exact (sampler RNG and
  early-stop/best state not restored); OOM only caught in the forward
  pass; unknown config keys silently ignored; single CSV silently dropping
  epoch columns; evaluation crop cache could be written next to
  `media_root`; synthetic data not obviously separable; AMP silently
  skipping every step of a short run (now counted, `amp_init_scale=1024`);
  saturated-AUROC ties keeping a worse "best" checkpoint.
- Measured (synthetic, engineering only): RTX 4050 AMP peak VRAM 154 MB
  (MobileNetV4) / 460 MB (EfficientNet-B0); resume bit-exact on CPU and
  GPU; checkpoints 28.9 / 46.4 MiB. All in `docs/EXPERIMENT_LOG.md`.
- 111 new tests. Full suite: 422/422 passing.
- No dataset, GenD, DINOv2, or additional model downloaded. No accuracy claim.

### Phase 5b — Official FaceForensics++ c23 acquisition and validation (2026-09-30)

- Downloaded the official `faceforensics_download_v4.py` (approved URL →
  HTTPS redirect, same host) to the D: dataset area; read it in full and
  recorded URL, time, and SHA-256 (`5d0b220a…`).
- Added `scripts/download_faceforensics_c23.py`: a hash-pinned wrapper
  that allows only `original`/`Deepfakes`/`Face2Face`/`FaceSwap`/`NeuralTextures`,
  with `-c c23 -t videos --server EU2` fixed. It stops before < 40 GB
  free, runs a stall watchdog (the official script can hang forever),
  kills the whole Windows process tree, and logs JSONL on D:.
- Downloaded 5000 c23 videos (1000 per class, 8.42 GiB) to
  `D:\ConfiGuard-Data\datasets\FaceForensics++`. Nothing else was
  downloaded; D: never went below 83.78 GB free.
- Added `configuard.datasets.faceforensics` + `scripts/validate_faceforensics.py`:
  - structure and per-class counts against the official pair list;
  - zero-byte and partial-download detection;
  - ffprobe header + full packet demux of every video;
  - `<target>_<source>` relationships;
  - manifest, duplicates, leakage groups, and a JSON/Markdown report.
  Result: VALID.
- Fixed the FF++ adapter (`FaceForensicsAdapter`). Each fake now links to
  both originals it was built from (`parent` = target, `paired` =
  source), so leakage groups are the correct 500 × 10. Previously only
  the target was linked. Identity labels are never invented.
- Built the canonical manifest (5000 samples, 0 validation issues, 0
  duplicates) on D:. **No split applied**: the approved source provides
  none, and none was invented.
- 13 new tests; full suite 435/435. No face crops, no training, no GenD,
  and no model code changed.

### Phase 5c — Official FaceForensics++ split integration (2026-09-30)

- Committed Phase 5b as `12e2585`, after moving the FF++ download-script
  URL out of the repository (access information; now only in
  `PROVENANCE.md` on D:).
- Located the official `dataset/splits/{train,val,test}.json` in
  `ondyari/FaceForensics`. Pinned commit `b952e41c`, recorded URLs,
  sizes, SHA-256, and git blob SHAs, and stored the copies on D: only
  (data is under the FF++ ToS, so the files are not committed).
- Added `configuard.datasets.faceforensics_splits`:
  - hash-pinned loader and strict parser;
  - reconciliation against the official 500 pairs;
  - atomic assignment that refuses on any cross-split original,
    parent/pair link, or leakage group.
- Added `scripts/apply_faceforensics_splits.py`:
  - writes the train/val/test manifests on D:;
  - re-checks them with the trainer's leakage guard and Phase 3
    validation;
  - audits counts, group balance, duration, and native resolution per
    split.
  Result: 720/140/140 originals (3600/700/700 videos), 0 leakage.
- Measured FF++ shortcut cues:
  - FaceSwap/NeuralTextures clips are shorter.
  - Face2Face/NeuralTextures round frame width down to a multiple of 16
    (282/1000 videos).
  Binding rules recorded: fixed frame budgets, aligned 224×224 crops
  only, and no metadata as model input.
- 24 new deterministic tests; full suite 459/459. No crops, no training,
  no distillation, no additional downloads.

### Phase 5d — Matched face-crop extraction and shortcut audit (2026-10-01)

- Verified the official FF++ convention from the authors' README
  (`<target sequence>_<source sequence>`) and the paper appendix
  (target = frames; source = face/expressions; per-method clip lengths).
- Added `configuard.crops`:
  - `families`: content families with content and donor parents;
  - `matching`: shared-range nested 16/8/4 sampling, half-gap-bounded
    recovery, joint → individual → failed slot resolution;
  - `alignment`: 5-point Umeyama similarity, margin 0.25,
    reflect-101;
  - `store`: config-tag keyed, atomic, `StaleCropError`;
  - `extract`: sequential decode; sparse face linking re-tracked over
    recovery frames; whole-family quarantine;
  - `manifests`: whitelisted model rows, matched pairs, quarantine,
    audit sidecar, leakage re-validation, detection stats;
  - `audit_stats`.
- Added `configuard.storage_guard.FreeSpaceGuard`, and a YuNet SHA-256
  pin (`verify_yunet_model`).
- Added `scripts/extract_ffpp_face_crops.py`: preflight, estimate,
  trial or `--families`, bounded 12-worker pool, resume, progress/ETA,
  storage floor, manifests, contact sheets.
- Added `scripts/audit_ffpp_crop_shortcuts.py`: crop-or-squeeze
  registration, geometry probes, correlation audit, crop QA sheets.
- Full run: 991/1000 families accepted, 79,280 crops, 63,424/63,424
  exact matched slots, 0 leakage, 4.91 GB on D:. The rerun was
  byte-identical.
- Finding: the F2F/NT width change is a centred crop (562/562), not a
  squeeze.
- 44 new tests; full suite 503/503. No training, no GenD, no raw
  videos deleted, official splits unchanged.

### Phase 6a — Frozen GenD teacher setup and logit caching (2026-10-01)

- Added `configuard.teacher`:
  - `gend`: GenD CLIP-L/14 rebuilt locally and strict-loaded from
    hash-pinned safetensors; `assert_frozen`; BGR→RGB tensor helper;
  - `cache`: resumable, stale-rejecting shard cache; test split
    refused.
- Added `scripts/download_gend_teacher.py` (pinned snapshot + SHA-256
  record) and `scripts/cache_teacher_logits.py` (fp16 bs 64,
  progress/ETA, free-space floor, trial root, per-split teacher
  metrics).
- Added `transformers`, `safetensors` and `huggingface_hub` to
  `requirements.txt`.
- Cached teacher logits for 57,040 train and 11,120 val crops. Val
  frame AUC is 0.960 and video AUC 0.979. Test was never opened.
- 10 new tests; full suite 513/513. No student training.

### Phase 6b — MobileNetV4 student distillation (2026-10-01)

- Added `configuard.distill`:
  - `data`: hash-checked crop rows, aligned cached teacher margins,
    class × method balanced `EpochSampler`, `CropDataset`;
  - `augment`: mild class-independent blur + horizontal jitter;
  - `losses`: BCE + T²-scaled binary logit KD;
  - `evaluate`: frame/video AUROC, AUPRC, ECE, Brier, NLL,
    per-manipulation;
  - `train`: `DistillConfig`, `StudentTrainer` with AMP, early
    stopping, `weights_only` checkpoints and epoch resume.
- Added `configs/distill/mobilenetv4_student.yaml`,
  `scripts/train_distill_student.py` (train/pilot) and
  `scripts/compare_students.py`.
- Pilot (7 runs) chose α 0.5, T 2.
- Full runs (val): baseline video AUROC 0.981 and frame NLL 0.504;
  distilled 0.979 and 0.216. Size, latency and VRAM are identical.
  Test split untouched; GenD not loaded.
- 15 new tests; full suite 528/528. No robustness/GRU/test evaluation.

### Phase 6c — Calibration and the "uncertain" output (2026-10-01)

- Added `configuard.calibration`:
  - `partitions`: deterministic 80/10/10 split of TRAIN families by
    donor-linked component;
  - `core`: temperature scaling, mondrian/marginal split conformal,
    verdict mapping, ECE/NLL/Brier/coverage/abstention/selective
    accuracy/risk-coverage;
  - `artifact`: hash-bound `calibration.json`, `Calibrator`,
    `CalibrationMismatchError`.
- Added `configuard.distill.infer` and `DistillConfig.train_partition`
  (omitted when empty, so 6b configs are unchanged), plus a
  `--train-partition` CLI flag.
- Added `scripts/calibrate_student.py` (`split` / `fit`).
- Retrained the distilled student on 80% of train families. Fitted
  frame and video calibration (T 0.95 / 0.62).
- Default video verdicts on dev: 5.8% uncertain, 2.2% of reals
  flagged. Coverage 0.927 vs nominal 0.95 (calibration/dev shift,
  documented).
- 10 new tests; full suite 538/538. Test split sealed; no adaptive
  inference, robustness training or GRU.

### Phase 6d — Adaptive 4/8/16-frame video inference (2026-10-01)

- Added `configuard.adaptive`:
  - `policy`: nested stage slots, `StagePolicy` with α spending, stage
    decisions;
  - `analyzer`: `AdaptiveVideoAnalyzer` with no re-scoring, stopping
    reasons and evidence timeline; `ArrayScorer` / `StudentCropScorer`.
- `build_artifact` gains `required_levels` / `extra`; 6c artifacts
  unchanged.
- Added `scripts/adaptive_video_eval.py` and `adaptive_calibration.json`
  (per-stage T and mondrian thresholds).
- Dev vs fixed-16:
  - 61% fewer frames;
  - FPR 1.44% vs 2.16%, coverage 0.964 vs 0.927;
  - 13.2% vs 5.8% uncertain;
  - P50 33 vs 81 ms on GPU.
- 8 new tests; full suite 546/546. Test split sealed; no GRU, robustness training or test
  evaluation.

### Phase 6e — Compression-robust student training (2026-10-01)

- Added `configuard.robust`:
  - `degrade`: label-free JPEG / H.264-style / resize / blur / noise /
    gamma with a mild → moderate curriculum;
  - `stress`: deterministic 17-condition val stress suite with real
    libx264;
  - `scoring`: sequential, per-condition-saved, RAM-floored scoring.
- Added `configuard.memory_guard` and `DistillConfig.robust_augment`
  (omitted when empty), plus single-thread DataLoader workers, a
  `--robust` training flag and `scripts/robust_eval.py`.
- Trained `student_distilled_robust_p80` (fixed 6b setup, 80%
  partition).
- Results: worst-case video AUROC 0.825 vs 0.692, but clean 0.927 vs
  0.974. **Not selected**; the current model stays the default.
- Found a blur/downscale → "fake" shortcut in the current model.
  Calibration artifacts refuse the robust checkpoint.
- 7 new tests; full suite 553/553. Test split sealed; no recalibration.

### Phase 7 — Efficient temporal video head (2026-10-01)

- Added `configuard.temporal`:
  - `embeddings`: ordered, hash-checked frame-embedding cache and
    nested-k `VideoSet`; refuses test rows;
  - `gru`: residual 1-layer GRU head (hidden 128), training and
    prediction.
- Added `memory_guard.process_rss_mb` and `scripts/temporal_gru.py`
  (extract / train / evaluate).
- Cached embeddings for train, val and 8 stress conditions (385 MB on
  D:).
- GRU vs mean aggregation (val, k16): AUROC 0.9749 vs 0.9735 (CI
  includes 0), stress −0.0024, FPR@0.5 0.273 vs 0.165. **Rejected**;
  the mean aggregation stays.
- 5 new tests; full suite 558/558. No recalibration, test access or ONNX export.
