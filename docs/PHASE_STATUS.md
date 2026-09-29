# Phase Status

| Phase | Title | Status | Date |
|---|---|---|---|
| 0 | Environment and repository foundation | PASS | 2026-09-29 |
| 1 | Architecture contracts and minimal vertical slice | PASS | 2026-09-29 |
| 2 | Face and media preprocessing | PASS | 2026-09-29 |
| 3 | Dataset registry and leakage-safe data splits | PASS | 2026-09-29 |

Full per-phase results are recorded below as they complete.

---

## Phase 0 — Environment and repository foundation

**Status:** PASS

**Summary:** Repository scaffolding, virtual environment, minimal
dependencies, environment verification, and Phase 0 test suite are all in
place and verified. See `docs/EXPERIMENT_LOG.md` for exact commands/output
and `docs/DECISIONS.md` for the reasoning behind key choices.

**Verified:**
- Git repository initialized.
- Python 3.11.9 virtual environment (`.venv`) created and used for all
  installs/tests (default system Python 3.14.6 is not compatible with the
  ML dependency stack yet).
- PyTorch 2.5.1+cu121, NumPy 2.4.6, PyYAML 6.0.3, pytest 8.4.2 installed.
- `scripts/verify_environment.py` runs and correctly reports CUDA
  availability (RTX 4050 Laptop GPU, 6140 MB, CUDA 12.1).
- `tests/` suite (imports, device detection, config loading): 12/12 passed.

**Open items carried to later phases (see `docs/KNOWN_ISSUES.md`):**
- ~~FFmpeg not installed~~ — RESOLVED 2026-09-29 (pre-Phase-1, user-approved
  `winget` install, verified — see `docs/KNOWN_ISSUES.md`).
- Free disk space (~21.8 GB at scan time) is limited relative to expected
  dataset/checkpoint sizes — needs a decision before Phase 1 downloads
  begin. Phase 1 uses only small generated fixtures, so this remains open
  but non-blocking for now.

---

## Phase 1 — Architecture contracts and minimal vertical slice

**Status:** PASS

**Summary:** Typed data contracts, environment-aware config schemas,
secure validation, and a deterministic dummy end-to-end pipeline are all
implemented and tested. FFmpeg was installed (user-approved) before this
phase and is verified working. See `docs/ARCHITECTURE.md` for the data
flow/schema and `docs/EXPERIMENT_LOG.md` for exact commands/output.

**Verified:**
- `src/configuard/io_types.py`: `ImageInput`, `VideoInput`, `ValidatedMedia`,
  `PreprocessingOutput`, `ModelOutput`, `ProvenanceOutput`, `DetectionResult`
  (+ `result_to_dict`), all frozen dataclasses with `str`-Enum fields.
- `src/configuard/config.py`: `ValidationLimits` + `environment` field on
  `ProjectConfig`; four env configs (`configs/{development,training,testing,production}.yaml`).
- `src/configuard/validation.py`: magic-byte content sniffing, extension
  cross-check, size limits, `ffprobe`-based video duration/corruption
  check — all failures returned as data, never raised.
- `src/configuard/pipeline.py`: `run_pipeline()` end-to-end orchestration
  and `select_frame_count()` adaptive 4/8/16 contract; `PipelineRejectedError`
  for invalid input.
- Manual smoke test: a generated tiny PNG and a generated tiny MP4 (via
  `ffmpeg lavfi`) both produced valid `DetectionResult` JSON; a hand-crafted
  corrupted `.mp4` was rejected safely (no crash) — see `docs/EXPERIMENT_LOG.md`.
- `pytest` suite: **48/48 passed** (12 Phase 0 + 36 new Phase 1 unit/integration
  tests), including corrupted/oversized/mismatched-signature rejection cases.

**Open items carried to later phases (see `docs/KNOWN_ISSUES.md`):**
- Free disk space remains limited — still non-blocking since Phase 1/no
  datasets were downloaded.
- All preprocessing/model/provenance logic is placeholder-only by design;
  Phase 2+ replaces each piece behind the same typed contracts.

---

## Phase 2 — Face and media preprocessing

**Status:** PASS

**Summary:** Independent `src/configuard/media/` subsystem: reliable image
decoding + video metadata (ffprobe-preferred), deterministic nested 4/8/16
frame sampling, a pluggable `FaceDetector` interface with a real
CPU-only OpenCV YuNet backend and a `MockFaceDetector` test double, face
alignment/crop with a configurable margin, simple IoU+landmark video face
tracking with primary-track selection, and a versioned, atomic face-crop
cache. Not yet wired into `configuard.pipeline` (see
`docs/ARCHITECTURE.md`). No dataset, no deepfake model checkpoint, and no
unrelated asset was downloaded — only the YuNet detector ONNX explicitly
named in the task instructions (provenance in `docs/DATASETS.md`).

**Verified:**
- `configuard.media.decode`: valid image decode; corrupted image/video
  raise `DecodeError`; video metadata (fps/duration/frame_count) matches
  the known ffmpeg-generated clip; rotation-tag parsing unit-tested
  against synthetic ffprobe JSON (see `docs/KNOWN_ISSUES.md` for why not
  against a real rotated fixture in this environment).
- `configuard.media.sampling`: nesting (`indices(4) ⊆ indices(8) ⊆
  indices(16)`) verified for frame counts 1–1000, including short videos
  where sets are deduplicated below the requested count.
- `configuard.media.face_detector`: `MockFaceDetector` fully deterministic;
  real `YuNetFaceDetector` loads on CPU (no CUDA) and safely returns no
  detections on a non-face image and on a blank image.
- `configuard.media.alignment`: output shape matches `output_size`; margin
  expansion verifiably pulls in more background; out-of-bounds boxes are
  clipped (partial) or rejected via `AlignmentError` (fully outside).
- `configuard.media.tracking`: single continuous track, gap tolerance,
  track split on a large gap, two simultaneous faces kept as two separate
  tracks, frame order preserved, primary-track selection with and without
  a "multiple tracks" warning.
- `configuard.media.cache`: hit/miss, atomic writes (no leftover temp
  files), different config version/frame index/track ID are distinct
  entries (no stale-hit risk), byte-limit eviction, and cache-reuse (no
  recompute, unchanged mtime) verified both directly and through the full
  `preprocess_image`/`preprocess_video` orchestration.
- `pytest` suite: **124/124 passed** (48 from Phases 0–1 + 76 new Phase 2
  unit/integration tests).
- `scripts/benchmark_preprocessing.py` run once against a synthesized
  640x480/5s/15fps clip — see `docs/EXPERIMENT_LOG.md` for full numbers
  (decode ~68ms/16 frames, YuNet detection ~13ms/frame on CPU, alignment
  ~1.5ms/crop, cache write ~4ms, cache hit read ~15ms, cache miss ~0.2ms).

**Open items carried to later phases (see `docs/KNOWN_ISSUES.md`):**
- Rotation metadata extraction is implemented but unverified against a
  real rotated video file in this environment (this ffmpeg build didn't
  attach rotation metadata to a synthetic test clip); not yet used to
  auto-correct crops.
- `configuard.media` is not yet called from `configuard.pipeline` —
  intentional, deferred until a real encoder exists to consume face crops.
- Free disk space remains limited; still non-blocking (only ~224 KB
  downloaded this phase).

---

## Phase 3 — Dataset registry and leakage-safe data splits

**Status:** PASS

**Summary:** New `src/configuard/datasets/` subsystem: a canonical
`Sample` schema (every field docs/PROJECT_PLAN.md requirement 2 lists),
a typed registry with adapters for FaceForensics++, Celeb-DF-v2, DFDC,
DF40, and DeeperForensics-1.0 (two reusable engines - folder-convention
and metadata-sidecar - the latter doubling as the generic adapter for
future datasets), JSONL manifest read/write with lenient + strict
validation, deterministic leakage-safe splitting (union-find over
source/identity/parent/pair links, hash-bucketed assignment), exact +
near-duplicate detection, and a storage-path safety checker. No dataset,
checkpoint, or large file was downloaded - every adapter was built and
tested against synthetic, generated fixtures only.

**Verified:**
- `configuard.datasets.schema`: round-trip JSON serialization; unknown
  manifest fields tolerated on load; demographic fields default to
  `None` and are never set by any adapter.
- `configuard.datasets.manifest`: write/read round-trip; lenient parser
  isolates malformed JSON, missing fields, invalid labels, and
  unsupported media types as per-line issues without aborting the whole
  read; semantic validation catches duplicate IDs, missing files, broken
  parent/pair references, and cross-split source/identity/pair leakage.
- `configuard.datasets.splitting`: same seed -> bit-identical
  assignments, independent of input ordering; different seed usually
  differs; fractions must sum to 1.0; every leakage group (shared source,
  shared identity, parent link, or pair link) stays in one split, on both
  a synthetic 60-pair stress test and the full multi-scenario fixture;
  audit report written to JSON.
- `configuard.datasets.duplicates`: exact duplicates found via SHA-256
  grouping; near-duplicates found via average-hash + configurable Hamming
  threshold; video samples correctly skipped by the near-duplicate check
  (documented limitation, not a bug - see docs/KNOWN_ISSUES.md); missing
  files handled without raising; nothing ever deleted, report only.
- `configuard.datasets.adapters`: both engines tested against synthetic
  dataset trees (missing root, no buckets present, partial buckets
  tolerated, official-split-file parsing, fake→real pairing,
  JSON/JSONL/CSV metadata formats, DFDC's real `{filename: {...}}`
  metadata shape); all five known-dataset factories registered and
  fail with `DatasetAccessError` (never crash, never attempt any
  network access) when pointed at a nonexistent local root.
- `configuard.datasets.storage`: unset/missing/existing paths;
  writability; free-space warning at a deliberately huge threshold;
  in-repo detection both warns (`check_storage_path`) and hard-refuses
  (`assert_safe_storage_path` raises `UnsafeStoragePathError`) - see
  `scripts/check_storage.py`'s real output in `docs/EXPERIMENT_LOG.md`,
  including the refusal case with a real (data)/(cache) split.
- End-to-end demo: a synthetic FF++-shaped fixture (3 real + 2 fake
  videos, 2 manipulation methods) built into a manifest, validated
  (0 issues), split deterministically (pairs stayed together, no
  leakage), and duplicate-checked - full output in
  `docs/EXPERIMENT_LOG.md`.
- `pytest` suite: **215/215 passed** (124 from Phases 0–2 + 91 new Phase 3
  unit/integration tests).

**Open items carried to later phases (see `docs/KNOWN_ISSUES.md`):**
- DF40's adapter structure is low-confidence ("verification required") -
  no independently-confirmed public documentation of its real on-disk
  layout was available; DeeperForensics-1.0's real-video bucket name is
  a best-effort guess. Both must be adjusted once real access exists.
- Access/licensing details for all five datasets are marked
  "verification required" in `docs/DATASETS.md` where not independently
  confirmable - treat as a starting point, not ground truth.
- Near-duplicate detection only covers IMAGE-type samples; video
  near-duplicate detection would need representative-frame extraction
  (available via `configuard.media`, not yet wired in here).
- `configuard.datasets` is not yet wired into any training loop -
  intentional, per task scope (task 15: "do not wire dataset processing
  into model training yet").
- Free disk space is down to ~12.9 GB at last check (`scripts/check_storage.py`
  output in `docs/EXPERIMENT_LOG.md`) - still non-blocking since no real
  dataset was downloaded, but leaves less room before Phase 3b.
