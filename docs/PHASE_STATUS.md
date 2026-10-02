# Phase Status

| Phase | Title | Status | Date |
|---|---|---|---|
| 0 | Environment and repository foundation | PASS | 2026-09-29 |
| 1 | Architecture contracts and minimal vertical slice | PASS | 2026-09-29 |
| 2 | Face and media preprocessing | PASS | 2026-09-29 |
| 3 | Dataset registry and leakage-safe data splits | PASS | 2026-09-29 |
| 4 | Pretrained baseline models and ONNX verification | PASS | 2026-09-29 |
| 5 | Reproducible training pipeline | PASS | 2026-09-30 |
| 5b | Official FaceForensics++ c23 acquisition and validation | PASS | 2026-09-30 |
| 5c | Official FaceForensics++ split integration | PASS | 2026-09-30 |
| 5d | Matched face-crop extraction and shortcut audit | PASS | 2026-10-01 |
| 6a | Frozen GenD teacher setup and logit caching | PASS | 2026-10-01 |
| 6b | MobileNetV4 student distillation (baseline vs distilled) | PASS | 2026-10-01 |
| 6c | Calibration and the "uncertain" output | PASS | 2026-10-01 |
| 6d | Adaptive 4/8/16-frame video inference | PASS | 2026-10-01 |
| 6e | Compression-robust student training (experiment; not selected) | PASS | 2026-10-01 |
| 7 | Efficient temporal video head (GRU evaluated; rejected) | PASS | 2026-10-01 |
| 8 | Production ONNX export and optimization | PASS | 2026-10-01 |
| 9 | Media-quality safety gate | PASS | 2026-10-01 |
| 9b | Quality-gate hardening (v2 signals) | REJECTED (Phase 9 gate kept) | 2026-10-01 |
| 9c | Final hybrid quality gate | REJECTED (Phase 9 gate kept; gate experimentation ended) | 2026-10-02 |
| 10 | Production inference API/service | PASS | 2026-10-02 |
| 11 | Explainability and lightweight web UI | PASS | 2026-10-02 |
| 12 | C2PA provenance verification | PASS | 2026-10-02 |

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
  dataset was downloaded, but leaves less room before Phase 3b. (Since
  resolved by the storage-configuration step - see `docs/ARCHITECTURE.md`.)

---

## Phase 4 — Pretrained baseline models and ONNX verification

**Status:** PASS

**Summary:** New `src/configuard/models/` subsystem: a pluggable
`DeepfakeVisualEncoder` wrapping either of the two authorized
ImageNet-pretrained timm backbones (MobileNetV4-Conv-Small,
EfficientNet-B0) with a fresh, randomly-initialized binary head; a
shared 224×224 preprocessing contract compatible with Phase 2's aligned
face crops; image and fixed-frame (mean-of-embeddings) video inference;
an optional real-model pipeline wiring Phase 2 face preprocessing into
these encoders (fully dependency-injected); FP32 ONNX export with
verified PyTorch-vs-ONNX parity; and reproducible P50/P95 latency/memory
benchmarking. **Every prediction is explicitly marked untrained and
uncalibrated** (`PREDICTION_DISCLAIMER`, `is_finetuned=False`). Only the
two explicitly authorized models were downloaded, from their official
Hugging Face repos, cached under `D:\ConfiGuard-Data\cache\huggingface`
(never `C:\Users\...\.cache` or the repo) - see `docs/DATASETS.md` for
full provenance.

**Verified:**
- Both models load pretrained weights from the configured D-drive cache
  (`test_pretrained_integration.py`, not skipped - cache was populated).
- Both run on CPU and on the RTX 4050 (GPU) - `resolve_device("auto")`
  correctly selects `cuda`; per-device latency measured for both.
- Both complete a forward+backward smoke step (task 13) without OOM -
  peak GPU memory measured at ~134 MB reserved (2.2% of the 6 GB
  budget) for a single-image forward pass; the full training-mode
  smoke step (batch=2) completes without error on both CPU and GPU.
- Image inference (single + batch) and fixed 4/8/16-frame video
  inference (ordered frame embeddings, mean-aggregated before the
  head) both work and are covered by dedicated tests.
- ONNX CPU inference works for both (verified via
  `run_onnx_cpu_inference` and a dedicated test); PyTorch-vs-ONNX
  parity max abs diff ~3-4e-07 for both models, far inside the
  documented `atol=1e-3, rtol=1e-3` tolerance.
- `pytest` suite: **311/311 passed** (215 from Phases 0-3 + 90 new
  Phase 4 model tests + 6 new `tests/test_env_loader.py` tests).

**Two real bugs found and fixed during this phase** (both covered by
regression tests, both explained in `docs/DECISIONS.md`): (1)
`backbone.num_features` doesn't reliably match the real pooled output
dimension for every timm architecture (MobileNetV4 specifically - fixed
via an empirical probe forward pass); (2) that probe itself first failed
because BatchNorm rejects a batch of size 1 in `train()` mode (fixed by
probing in `eval()` mode). Also **one environment regression found and
fixed**: installing `timm`/`onnx`/`onnxruntime` silently downgraded
`torch` to a CPU-only build and left `torchvision` mismatched - both
reinstalled to the correct paired versions before any benchmark number
was recorded (`docs/KNOWN_ISSUES.md`).

**Performance measurements** (parameter counts, approximate FLOPs,
warm-up + P50/P95 CPU and GPU latency, peak GPU memory, image-batch and
4/8/16-frame video-inference timings, ONNX file sizes and parity) are
fully documented in `docs/EXPERIMENT_LOG.md`, reproducible via
`scripts/benchmark_models.py` and `scripts/export_onnx_models.py`. This
is a **provisional efficiency comparison only** - MobileNetV4-Conv-Small
is faster/smaller on every measured axis, but no deepfake-detection
accuracy result exists yet, so the final model choice is explicitly not
decided by this phase.

**Open items carried to later phases (see `docs/KNOWN_ISSUES.md`):**
- `configuard.models` is not yet wired into `configuard.pipeline` or any
  real training loop - intentional, deferred until a fine-tuning phase.
- No quantization performed yet (FP32 ONNX only) - explicit task scope.
- No temporal model (GRU) yet - video aggregation is mean-pooled
  embeddings only, explicit task scope.
- The pip dependency-resolution fragility that caused the torch/
  torchvision regression this phase is now documented with a mitigation
  step (verify build after every install), but not structurally
  prevented (e.g. via a lockfile) - worth revisiting if it recurs.

---

## Phase 5 — Reproducible training pipeline

**Status:** PASS

**Summary:** New `src/configuard/training/` subsystem plus
`scripts/train.py` / `scripts/evaluate.py`: config-driven training of
MobileNetV4-Conv-Small (default) or EfficientNet-B0 from Phase 3
manifests through Phase 2 aligned face crops, with AMP, accumulation,
clipping, optional backbone freezing, warm-up + cosine AdamW, early
stopping, best/latest atomic checkpoints with full provenance, exact
resume that refuses mismatched checkpoints, JSONL/CSV logging, and
threshold-free vs threshold-dependent validation metrics. Pre-phase
dependency safety pins and guards the torch/torchvision CUDA pair.
**Everything was trained on synthetic data only. No accuracy claim is
made.** The phase was interrupted once and resumed from the uncommitted
working tree on 2026-09-30. The draft's ad-hoc checks were treated as
preliminary, and every result below was re-run on the final code.

**Acceptance criteria:**
- MobileNetV4 synthetic smoke training on the RTX 4050 with mixed
  precision: **met** (`scripts/train.py --smoke cuda`, 5 epochs, AMP on).
- Peak VRAM safely below 6 GB: **met**. 154 MB reserved (MobileNetV4),
  460 MB (EfficientNet-B0), at batch 8.
- EfficientNet-B0 forward/backward: **met**. Full 5-epoch AMP smoke on
  CUDA plus CPU tests.
- Tiny synthetic set overfitted: **met**. Loss 0.678 → 0.0002, eval AUROC 1.0.
- Interrupted training resumes correctly: **met**. Max parameter diff
  0.0 vs uninterrupted, on CPU and RTX 4050, both backbones.
- Checkpoints only on D: **met**. `D:\ConfiGuard-Data\checkpoints\...`;
  the repo and `~/.cache` are untouched.
- Mismatched checkpoints rejected: **met**. Unit tests per field, plus a
  real CLI run listing all 8 mismatches.
- CUDA available after dependency work: **met**. No installs were made;
  `verify_environment.py` is OK before and after.
- Validation metrics and structured logs generated: **met** (JSONL,
  train/epoch CSV, summary JSON, eval JSON on D:).
- No real accuracy claim: **met**. Disclaimers are in outputs, MODEL_CARD
  and the experiment log.
- No unauthorized model/dataset downloaded: **met**. HF cache still holds
  only the two Phase 4 models, and `HF_HUB_OFFLINE=1` is forced.
- Complete test suite passes: **met**, **422/422** (311 + 111 new), 0 skipped.

**Real bugs found and fixed** (all have regression tests; details in
`docs/DECISIONS.md` / `docs/EXPERIMENT_LOG.md`):
- GPU resume always crashed: RNG states were loaded onto CUDA.
- Resume wasn't exact: sampler RNG and best/early-stop state were not
  saved.
- A short AMP run silently skipped every optimizer step (GradScaler
  overflow calibration). Skipped steps are now counted, and
  `amp_init_scale` defaults to 1024.
- A saturated-AUROC tie kept a worse "best" checkpoint (now broken by
  val loss).
- CUDA OOM was only caught in the forward pass.
- Unknown config keys were silently ignored.
- A frozen-header CSV silently dropped every epoch/validation column.
- Mixed image+video manifests silently dropped the images.
- The evaluation crop cache could be written next to the media root.

**Open items carried forward (see `docs/KNOWN_ISSUES.md`):**
- Balanced with-replacement sampling can form single-class batches at
  tiny batch sizes, which BatchNorm then destabilizes. Negligible at the
  configured batch sizes.
- Threshold-dependent metrics are unreliable while BatchNorm running
  statistics settle.
- Resume granularity is one epoch.
- GPU bit-exactness was observed but is not guaranteed by CUDA.
- `torch.load(weights_only=False)`: only load trusted checkpoints.
- Mixed image+video training is not supported yet.
- No real-data training has happened (Phase 3b dependency). VRAM at the
  real configs' batch sizes is not yet measured.

---

## Phase 5b — Official FaceForensics++ c23 acquisition and validation

**Status:** PASS

**Summary:** FaceForensics++ c23 videos were acquired: original and
Deepfakes/Face2Face/FaceSwap/NeuralTextures, 1000 each, 5000 files,
8.42 GiB. Everything came from server EU2 through the official script
URL from the user's approval email, via a hash-pinned, allow-listed,
free-space- and stall-guarded wrapper, into
`D:\ConfiGuard-Data\datasets\FaceForensics++` (outside Git). The
download was validated in full, a canonical manifest was built with the
existing registry, and leakage grouping was verified. No split was
applied, because the approved source provides none; none was invented.

**Requirements:**
1-2. Script downloaded only to `...\FaceForensics++\_official_script\`.
     URL, redirect, UTC time, and SHA-256 are recorded in `docs/DATASETS.md`. **Met.**
3. `-h` run first; the argument syntax was inspected and the whole
   script read before execution. **Met.**
4-6. Server `EU2`; only c23 videos of the 5 classes; no raw, c40, masks,
     models, images, DFD, FaceShifter, or benchmark data (validator
     found 0 unexpected paths). **Met.**
7-8. Stored only under `D:\ConfiGuard-Data\datasets`; `git status`
     shows no data, manifests, or logs. **Met.**
9. Free space monitored throughout, with a watchdog at 40 GB; the
   minimum observed was 83.78 GB. **Met.**
10. Structure, per-class counts (1000/1000 ×5), ffprobe readability
    (5000/5000, full packet demux), 0 zero-byte, 0 partial, and 0
    relationship problems. **Met.**
11. Canonical manifest built via `DEFAULT_REGISTRY` (5000 samples, 0
    issues, 0 duplicates). **Met.**
12. Official split: not available from the approved source, so not
    applied and not invented. **Met, as specified.**
13. Leakage validation: 500 groups × 10 members, 2 originals each, 0
    lineage problems. Acquisition report written. **Met.**
14. No face crops, no training, no GenD, no model code changed. **Met.**
15. Documentation updated. Tests: **435/435 passed**, 0 skipped. **Met.**

**Real problems found and fixed:**
- **Leakage-unsafe FF++ lineage.** The adapter linked each fake only to
  its target original, so a split could have leaked the source identity.
- **Hung downloads.** The official script's `urlretrieve()` has no
  timeout; a stall watchdog was added and proved itself on a later
  Face2Face stall.
- **Orphaned downloader.** The low-space stop would have killed only the
  Windows venv launcher and left the real downloader running.
- **Redirect page saved as the script.** The approved http URL returns a
  301 page, which the first fetch saved; the https target was fetched
  instead.

**Open items (see `docs/KNOWN_ISSUES.md`):**
- No FF++ split yet. Training on FF++ needs either user approval to
  fetch the official split JSONs from the authors' GitHub repo, or an
  explicit decision to generate a leakage-safe split.
- Manipulated classes differ from originals in clip length and
  resolution distribution. Preprocessing must not let these become
  shortcut cues.
- The download wrapper's stall/low-space logic was verified live, not by
  unit tests.

---

## Phase 5c — Official FaceForensics++ split integration

**Status:** PASS

**Summary:** Phase 5b was finalized and committed (`12e2585`) after
removing FF++ access URLs from the repository. The authors' official
split files were fetched only from `ondyari/FaceForensics`, pinned to
commit `b952e41cba017eb37593c39e12bd884a934791e1`, verified against
GitHub's blob SHAs, and kept on D: (not committed, because the data is
under the FF++ ToS). They were reconciled against all 1000 originals,
the 500 official pairs, all 4 methods, and the 500 leakage groups, and
applied without changing membership. Canonical train/val/test manifests
were written and audited.

**Requirements:**
1-3. Split files located; revision pinned; URLs, sizes, SHA-256, and git
     blob SHAs recorded (`docs/DATASETS.md`). **Met.**
4. Stored on D: only; the decision not to commit is documented
   (`docs/DECISIONS.md`). **Met.**
5. Parsed without modifying membership. The loader refuses altered files.
   **Met.**
6. Reconciled: 1000/1000 originals, each in exactly one split; split
   pairs equal the official 500; all 4 methods present in every split;
   recomputed leakage groups equal the 5b groups. **Met.**
7-8. Both source IDs of every fake are in its split; no source, parent,
     pair, or leakage group crosses partitions. The assignment is refused
     otherwise (tested for each case). **Met.**
9. Manifests written: train 3600, val 700, test 700. **Met.**
10-11. Official scale confirmed from the files: 720/140/140 originals
       (360/70/70 pairs); 720/140/140 videos per method. **Met.**
12. Class balance (fake:real 4.0 in every split), group balance (360/70/70
    groups × 10), and duration and resolution distributions per split
    are audited. **Met.**
13. Shortcut risks documented, with binding mitigations
    (`docs/KNOWN_ISSUES.md`, `docs/DECISIONS.md`). **Met.**
14-15. 24 new deterministic tests; full suite **459/459**, 0 skipped. **Met.**
16. Documentation updated. **Met.**

**Open items (see `docs/KNOWN_ISSUES.md`):**
- Face2Face/NeuralTextures width rounding: whether it is a crop or a
  squeeze (≤ ~3.3%) must be checked on aligned frames in the face-crop
  phase.
- Future evaluation must report per-method results and
  duration/resolution-stratified breakdowns.

---

## Phase 5d — Matched face-crop extraction and shortcut audit

**Status:** PASS

**Summary:** All 5000 official-split FF++ c23 videos were processed as
1000 content families (target original + its 4 fakes). Each family
shares one set of 16 nested frame indices, chosen over its shared
frame range and matched by frame index. Faces were detected with the
hash-pinned YuNet, tracked, recovered jointly when missing, aligned with
5 landmarks, and written as 224×224 RGB PNGs into a config-keyed,
atomic, stale-refusing store on D:.
- 991 families accepted (4955 videos, **79,280 crops**, 63,424/63,424
  matched slots exact); 9 quarantined (45 videos, exactly 9 per class).
- 0 leakage, and a byte-identical rerun.
- The F2F/NT width change is a **centred crop, not a squeeze**
  (562/562). No geometry or nuisance factor is meaningfully
  class-predictive after alignment.

**Requirements:**
- *Preflight 1–5:* tree clean at `c17049c`; 5 manifest pins, split
  pins, membership and leakage re-validated; all 5000 videos re-hashed
  (0 mismatches); D: 83.8 GB free → 77.2 GB at end (floor 40 GB, stop at
  42); estimate +4.92 GB / ~55 min measured on trials; everything
  written to D:. **Met.**
- *Sampling 1–8:* exactly 16 ordered frames via the existing nested
  contract (8-set = even slots, 4-set = slots % 4 == 0, tested). No
  clip facts reach the model: the shared range equalises the span, and
  the model manifest is whitelisted. Content and donor originals and
  both leakage parents are recorded for every fake. The convention was
  verified from the official README and paper. Matching is by equal
  frame index inside the family's shared range, which excludes F2F's
  rewound tail. The rule is the same for all classes. **Met.**
- *Face processing 1–8:* pinned YuNet (`verify_yunet_model`);
  primary-face temporal linking; 5-point Umeyama similarity; margin 0.25
  (configurable); identical 224×224 RGB PNG (level 3, no metadata
  chunks) for every class. PNG was kept after measuring 62–65 KB/crop.
  Paths are keyed by input SHA-256 and frame index only. **Met.**
- *Failure handling 1–7:* deterministic ±1…±6 recovery, joint across
  the family first; no placeholders, never short. Rates are reported by
  split, label, method, resolution and duration. Whole-family quarantine
  keeps pairs intact (9 per class). Shortcut risk was assessed: recovery
  is identical across classes; quarantine is higher at 1080p but
  label-neutral. **Met.**
- *Shortcut audit 1–7:* crop-or-squeeze answered with registration and
  landmarks, uncertainty stated. The post-alignment geometry probe is
  weak (DF AUC 0.62, others ≈ 0.52). No method-specific correction.
  Class-independent geometric and blur jitter is recommended for the
  augmentation phase. Correlations with resolution, duration, method,
  face size, confidence and recovery are all reported (AUC 0.49–0.51
  except sharpness). **Met.**
- *Engineering 1–17:* resumable and idempotent (rerun byte-identical,
  80,000 crop hashes verified); atomic writes; config-tag keys; stale
  refusal (tested, and demonstrated live twice); 12-worker bounded pool;
  sequential decode; progress, ETA, quarantine and storage-floor
  logging; frame-level JSONL linked to the video manifest SHA-256;
  lineage only as metadata; leakage re-run on the crop manifests;
  seeded contact sheets on D:; 44 new tests; trial before full; full
  suite run; docs updated; staging audited; commit of code, tests and
  docs only. **Met.**

**Real problems found and fixed during the phase:**
- **Trial validator scope bug.** It reported 612 false leakage problems
  because donors outside the trial subset were looked up in the
  subset. Parents are now always checked against the full official
  manifest.
- **IoU-only sparse linking** split a drifting face (family 682). Linking
  now also accepts a ≤ 1 face-width centre shift at ≤ 1.5× size.
- **Planned-only tracking** ignored recovery frames (family 158, a
  walking presenter). Tracks are now rebuilt over all decoded frames,
  with up to 2 recovery rounds. The full run was stopped at ~20% and
  restarted under the new tag.
- **Missed progress log lines** (the `% 25` check) and a cp1252 console
  crash in the audit printout were both fixed.

**Open items (see `docs/KNOWN_ISSUES.md`):**
- Reflect-101 mirroring at frame edges (equal across classes).
- Fakes are blurrier (NT AUC 0.415): a genuine artifact, but it needs
  class-independent blur/resize augmentation.
- Quarantine slightly under-represents 1080p and removes 5 visibly
  broken fakes.
- Residual DF landmark-geometry signal (AUC 0.62).
- `check_storage.py` doesn't load `.env`.

---

## Phase 6a — Frozen GenD teacher setup and logit caching

**Status:** PASS

**Summary:**
- Only `yermandy/GenD_CLIP_L_14` was downloaded, at a pinned revision,
  to D:. Its license (MIT), version and hashes are recorded.
- The training-data check passed: official FF++ train only.
- The teacher is rebuilt locally and strict-loaded from the pinned
  weights.
- It is fully frozen: 0 trainable parameters, eval mode, and parameter
  grads stay None after backward.
- Logits are cached for the train and val crops (68,160) with fp16
  autocast at bs 64.
- The cache resumes and rejects stale or mismatched entries.
- The test split was never opened.
- The previous session was interrupted during the 2-shard trial. The
  trial resumed cleanly and the phase was completed in this session.

**Requirements:**
1. Only the official model + code was downloaded to the configured D:
   HF cache (`openai/clip-vit-large-patch14` deliberately not
   fetched). **Met.**
2. Version, license and hashes are in `docs/DATASETS.md`. **Met.**
3. Not trained on FF++ val/test: confirmed from the paper and code
   (`docs/DECISIONS.md`). The per-frame lists are gated and could not
   be diffed (`docs/KNOWN_ISSUES.md`). **Met (with a documented
   residual).**
4. Frozen: `load_gend_teacher` runs `requires_grad_(False)` + `eval()`.
   `assert_frozen` runs at load and after every batch. Test with the
   real weights passed. **Met.**
5. Batch-size benchmark: fp16 bs 64 gives 90 img/s at 1.84 GiB peak.
   **Met.**
6. Cached train (57,040) and val (11,120) logits, consolidated
   `teacher_logits.jsonl` per split. **Met.**
7. Test untouched: `ProtectedSplitError` is raised up front and
   per-row; the report has `test_split_touched: false`. **Met.**
8. Resumable (rerun resumed 67/67, byte-identical) and stale-rejecting
   (tag, crop-manifest SHA-256, per-shard crop SHA-256 lists and crop
   bytes; tested). **Met.**
9. Targeted tests (`tests/teacher`, 10) and the full suite (513/513).
   **Met.**
10. Docs updated; commit contains code, tests and docs only. **Met.**

**Not done (by instruction):** no student training.

**Open items:** gated GenD frame lists; crop-distribution mismatch
with GenD's own detector; caching is I/O-bound (`docs/KNOWN_ISSUES.md`).

---

## Phase 6b — MobileNetV4 student distillation

**Status:** PASS

**Summary:** two MobileNetV4-Conv-Small students were trained on the
Phase 5d FF++ train crops under identical settings. One uses
ground-truth BCE only; the other adds GenD logit distillation from the
cached logits (α 0.5, T 2, chosen by a 7-run pilot).
- Val video AUROC is tied: 0.981 vs 0.979, bootstrap Δ −0.002, CI
  [−0.008, 0.004].
- Frame AUROC: 0.963 baseline vs 0.956 distilled.
- The distilled student is much better calibrated: frame NLL 0.504 →
  0.216, ECE 0.071 → 0.033; video ECE 0.052 → 0.044.
- Size, latency and VRAM are identical: 9.7 MiB, 23 ms CPU bs 1,
  3,450 img/s GPU fp16, 604 MB train VRAM.
- GenD was not loaded and the test split was not opened.

**Requirements:**
1. Two comparable MobileNetV4 models (baseline BCE; BCE + logit KD).
   **Met.**
2. Identical seed, splits, sampling and settings. Only α/T differ;
   identical init, draws and augmentation are tested. **Met.**
3. Class- and source-balanced sampling (real 1/2, each method 1/8).
   **Met.**
4. Only mild class-independent blur and horizontal geometry jitter;
   val is not augmented (tested). **Met.**
5. Pilot over α ∈ {0.5, 0.9} × T ∈ {1, 2, 4}; best trained with early
   stopping (patience 3). **Met.**
6. Frame and video AUROC/AUPRC, calibration (ECE, Brier, NLL) and
   per-manipulation val results (`docs/EXPERIMENT_LOG.md`). **Met.**
7. Accuracy, checkpoint size, CPU/GPU latency and VRAM compared.
   **Met.**
8. Test split untouched: the loaders raise `ProtectedSplitError`.
   **Met.**
9. Targeted tests (15) during development; full suite 528/528 (215 s).
   **Met.**
10. Docs updated; commit contains code, configs, tests and docs only.
    **Met.**

**Not done (by instruction):** robustness training, GRU, and test
evaluation.

**Open items:** the val split is used for selection and reporting;
NeuralTextures is the weakest method (frame AUROC 0.90–0.92); fixed
0.5-threshold metrics are uncalibrated; the training step is
data-loader-bound (`docs/KNOWN_ISSUES.md`).

---

## Phase 6c — Calibration and the "uncertain" output

**Status:** PASS

**Summary:**
- Official FF++ train families were split 80/10/10 (570/72/71 families),
  keeping donor pairs whole.
- The distilled student was retrained on the 80% with the fixed 6b
  settings (dev frame/video AUROC 0.949/0.974).
- Temperature was fitted on temp_cal only and conformal thresholds on
  conformal_cal only, separately for frame/image and video.
- A hash-bound `calibration.json` maps logits to calibrated P(fake) and
  a `Verdict` (likely real / likely manipulated / uncertain).
- Default (video, mondrian α 0.05, dev): 5.8% uncertain, real coverage
  0.978, overall coverage 0.927 (nominal 0.95; reached on the
  calibration partition but not on the harder val split).
- Test split sealed.

**Requirements:**
1. Deterministic 80/10/10 split of TRAIN families; complete families
   and donor partners kept together (tested). **Met.**
2. Retrained on 80% with the fixed 6b config; no tuning. **Met.**
3. Temperature fitted only on temp_cal. **Met.**
4. Split-conformal thresholds fitted only on conformal_cal (mondrian +
   marginal; α 0.01/0.05/0.10). **Met.**
5. Three outputs via `configuard.io_types.Verdict`. **Met.**
6. Frame/image and video calibrated separately (own T and thresholds).
   **Met.**
7. ECE, NLL, Brier, coverage, abstention, selective accuracy and
   risk-coverage curves on dev. **Met.**
8. Raw vs temperature-scaled vs conformal compared (plus a
   confidence-abstention baseline). **Met.**
9. Artifact bound to the checkpoint SHA-256, model config/provenance
   and its own content hash; mismatches refused (tested). The trainer
   and the fit refuse a different partitions file. **Met.**
10. Test sealed; full suite 538/538; docs and commit. **Met.**

**Not done (by instruction):** adaptive 4/8/16 inference, robustness
training, test evaluation.

**Open items:** val is harder than held-out train families, so
coverage falls short on dev; mondrian costs selective accuracy;
video α 0.01 is unsupported at n = 71 reals; frames within a video are
not exchangeable (`docs/KNOWN_ISSUES.md`).

---

## Phase 6d — Adaptive 4/8/16-frame video inference

**Status:** PASS

**Summary:** the calibrated distilled student now scores 4 → 8 → 16
nested frames, scoring each frame once.
- Each stage has its own temperature and mondrian conformal thresholds
  (from the 6c partitions), with α spent 0.015/0.015/0.02.
- It stops early only on a singleton set. At 16 frames it returns
  likely real, likely manipulated or uncertain.
- Every result records the stopping reason, frames used, confidence,
  per-stage decisions and a per-frame evidence timeline.
- Dev (val, 695 videos) vs fixed-16:
  - 61% fewer frames (6.19 vs 16);
  - FPR 1.44% vs 2.16%; coverage 0.964 vs 0.927; decided accuracy
    0.959 vs 0.922;
  - in exchange, 13.2% vs 5.8% uncertain;
  - P50 latency 33 vs 81 ms (GPU) and 50 vs 140 ms (CPU); P95 is about
    level.
- Test split sealed.

**Requirements:**
1. Nested sets reused; 4 → 8 → 16 without re-scoring (tested: 16
   distinct slots over 3 calls; the simulator asserts frames scored ==
   frames used). **Met.**
2. Separate temperature + conformal per 4/8/16 stage on the 6c
   partitions. **Met.**
3. Conservative α spending (sum 0.05, each ≥ 1/72). The empirical,
   not guaranteed, nature of coverage under shift is documented in the
   policy docstring, DECISIONS and KNOWN_ISSUES. **Met.**
4. Early stop only on a confident singleton; otherwise escalate.
   **Met.**
5. Three verdicts at 16 frames. **Met.**
6. Stopping reason, frames used, confidence and evidence timeline
   recorded. **Met.**
7. Fixed 4/8/16 vs adaptive (plus an unspent ablation) on dev. **Met.**
8. AUROC, decided accuracy, FPR, coverage, abstention, average frames,
   P50/P95 latency (GPU + CPU) and per-manipulation results. **Met.**
9. ≥ 40% fewer frames (61%) without worse FPR (1.44% ≤ 2.16% + 1 pp).
   **Met.**
10. Targeted tests (17), full suite 546/546 (213 s), docs, commit. **Met.**

**Not done (by instruction):** GRU, robustness training, test
evaluation.

**Open items:** higher abstention (originals 17%, NeuralTextures 30%);
P95 latency is not reduced; coverage is still empirical under shift
(`docs/KNOWN_ISSUES.md`).

---

## Phase 6e — Compression-robust student training

**Status:** PASS. The phase is complete. The robust model is recorded
as an experiment and was **not** selected.

**Summary:**
- Added class-independent JPEG, H.264-style, resize, blur, noise and
  gamma augmentation with a mild → moderate curriculum. GenD targets
  stay clean.
- Built a deterministic 17-condition stress suite of the val crops
  (real libx264; 12.2 GiB on D:).
- Trained one robust distilled MobileNetV4 with the fixed 6b setup on
  the same 80% partition.
- Robust vs current: worst-case video AUROC 0.825 vs 0.692, but clean
  0.927 vs 0.974 (−0.046, CI [−0.062, −0.032]) and mean degraded 0.894
  vs 0.900.
- The current model stays the production default.
- The stress test exposed a blur/downscale → "fake" shortcut in the
  current model (FPR@0.5 up to 100% on strongly blurred real videos).

**Requirements:**
1. JPEG, H.264-style, downscale/upscale, blur, noise and gamma
   augmentations. **Met.**
2. Identical probabilities and severities for every class and method
   (label-free function, tested). **Met.**
3. Mild → moderate curriculum; severe settings kept out of training.
   **Met.**
4. Clean cached GenD logits for distillation; student sees the
   degraded views. **Met.**
5. Deterministic clean/degraded dev stress suite with mild, moderate
   and severe levels (rebuild is byte-identical, tested). **Met.**
6. One robust model, fixed 6b setup, no search. **Met.**
7. Current vs robust compared on clean and degraded dev data. **Met.**
8. Clean AUROC, worst-case AUROC, ΔAUROC, FPR, per-method results,
   latency and training cost reported. **Met.**
9. Decision rule applied: robust not preferred (clean loss 0.046 >
   0.01; mean degraded not improved). **Met.**
10. Caches on D: (suite 12.2 GiB, per-condition logits); D: 64 GB free
    (floor 40). **Met.**
11. Targeted tests (40), full suite 553/553 (272 s), docs, commit. **Met.**

**Incidents:**
- The first robust training attempt was CPU-oversubscribed. It was
  stopped before any epoch was saved and fixed with single-thread
  workers.
- The first evaluation was killed for low system RAM. It was resumed
  as a sequential, per-condition-saved, RAM-floored evaluation with 4
  workers (minimum RAM seen 5.6 GB).

**Not done (by instruction):** recalibration, adaptive production
inference, test-split access, Phase 6f/7.

**Open items:** the blur/downscale shortcut; early stopping
confounded by the curriculum; calibration artifacts apply only to the
current model (`docs/KNOWN_ISSUES.md`).

---

## Phase 7 — Efficient temporal video head

**Status:** PASS. The phase is complete. The GRU was evaluated and
**rejected**, and the current mean-frame-logit aggregation is kept.

**Summary:**
- Ordered 1280-d frame embeddings from the frozen p80 student are
  cached on D: (train/val + 8 stress conditions; 385 MB; RAM-floored
  and resumable).
- A residual 1-layer GRU (hidden 128, 157k params) was trained on
  `final_train` with nested k = 4/8/16.
- Clean val video AUROC is 0.9749 vs 0.9735 (Δ +0.0014, CI
  [−0.0015, 0.0051]); stress mean Δ −0.0024.
- FPR@0.5 is worse (0.165 → 0.273), and balanced accuracy drops 0.894
  → 0.849.
- NeuralTextures is 0.931 vs 0.927. Latency overhead is 2–5% of the
  backbone.

**Requirements:**
1. Ordered embeddings cached on D: for train (all partitions,
   including calibration), val and the stress subset; never test.
   **Met.**
2. 1-layer GRU, hidden ≤ 128 (enforced in code). **Met.**
3. GRU trained with MobileNetV4 frozen (cached embeddings), no
   hyperparameter search. **Met.**
4. Slot order preserved and checked; nested 4/8/16 views from one
   cache; GRU trained on `final_train` only with its partition hash
   checked. **Met.**
5. GRU vs current aggregation on the same 695 videos, at k = 4, 8 and
   16. **Met.**
6. Clean AUROC, FPR, balanced accuracy, NeuralTextures, params, RAM,
   CPU/GPU latency. **Met.**
7. Stress subset (blur, downscale, noise, H.264 × 2 severities).
   **Met.**
8. Pre-registered selection rule applied; the GRU was not selected.
   **Met.**
9. Current aggregation kept and the reason documented. **Met.**
10. No recalibration, test access, ONNX export or Phase 8. **Met.**
11. Targeted tests (12), then one full suite: 558/558 (301 s). **Met.**
12. Docs updated and committed. **Met.**

**Open items:** the GRU had to be trained on in-sample student
features (stacking limitation). A fair temporal test needs out-of-fold
features or end-to-end fine-tuning (`docs/KNOWN_ISSUES.md`).

---

## Phase 8 — Production ONNX export and optimization

**Status:** PASS

**Summary:** `student_distilled_p80` was exported to ONNX FP32 / FP16
plus two static INT8 recipes (train-only calibration), with
normalisation inside the graph.
- FP32 and FP16 match the PyTorch path on the full val split: video
  AUROC 0.9735, verdict agreement 99.9–100%, same false-accusation
  counts.
- Both INT8 recipes fail the agreement bar (76% / 88–90%) and are not
  selected.
- Defaults: **CPU = ONNX FP32** (adaptive P50 8.8 ms vs PyTorch 38 ms)
  and **GPU = ONNX FP32**, which is faster than FP16 here. GPU uses one
  ORT session per batch size, because shape changes cost about 330 ms
  on the CUDA provider.
- A hash-checked package binds the ONNX files to the checkpoint and
  the 6c/6d calibration. The PyTorch checkpoint remains the reference.

**Requirements:**
1. ONNX FP32 + FP16. **Met.**
2. Static INT8 calibrated on `final_train` crops only (val/test
   refused, tested). **Met.**
3. PyTorch↔ONNX parity on clean + 4 degraded conditions. **Met.**
4. FP32 / FP16 / INT8 on the full official val. **Met.**
5. Frame/video AUROC, per-method results, calibrated / adaptive
   verdict agreement, numerical error. **Met.**
6. CPU batch 1 and 4; GPU batch 1/4/16 via the ORT CUDA provider
   (enabled by `onnxruntime-gpu` 1.23.2). **Met.**
7. Model size, model RAM, GPU memory, adaptive 4/8/16 and fixed-16
   per-video model latency. **Met.**
8. Separate CPU and GPU defaults; FP16 on CPU measured and not chosen.
   **Met.**
9. INT8 only if AUROC loss ≤ 0.01 and agreement is acceptable:
   rejected. **Met.**
10. Package with checkpoint / config / calibration hashes; mismatches
    refused (tested; robust checkpoint refused live). **Met.**
11. PyTorch path untouched and kept as reference. **Met.**
12. Targeted tests (22), then the full suite: 563/563 (re-run once to
    capture the summary); docs; commit. **Met.**
13. No API/UI, no test data, no Phase 9. **Met.**

**Open items:**
- INT8 needs its own recalibration to be usable.
- The CUDA provider needs one session per shape.
- The GPU-provider runtime pins `onnxruntime-gpu` 1.23.2 because of
  CUDA 12 (`docs/KNOWN_ISSUES.md`).

---

## Phase 9 — Media-quality safety gate

**Status:** PASS. Targets met and the gate is enabled by default. Known
bypasses and over-triggering are documented.

**Summary:**
- A downgrade-only gate sits on top of ONNX FP32 adaptive 4/8/16
  inference. It uses sharpness, effective resolution and blockiness
  from the same decoded crops, plus the detector's face width.
- Reason codes: LOW_SHARPNESS, LOW_RESOLUTION, HEAVY_COMPRESSION,
  SMALL_FACE, QUALITY_DEPENDENT_VERDICT.
- Thresholds come from TRAIN data, with the percentile chosen on the
  calibration partitions; no val or test tuning.
- Clean val coverage loss is 0.58 pp. Blur/downscale false accusations
  fall from 36.0% to 9.0% (blur σ2 84% → 0%; resize 0.33 46% → 24.5%).

**Requirements:**
1. Lightweight checks for blur, effective resolution, downscaling and
   compression (about 2 ms per crop). **Met.**
2. Signals are kept out of model inputs (scorer interface tested).
   **Met.**
3. Thresholds fitted on train/calibration data only. **Met.**
4. Downgrade to UNCERTAIN only (asserted; 300-case randomised test).
   **Met.**
5. Clear reason codes. **Met.**
6. Clean val plus all 17 stress conditions (and 2 adversarial + 3
   bypass sets). **Met.**
7. Coverage loss, decided accuracy, FA, detection, uncertainty and
   per-method results. **Met.**
8. ≤ 5 pp clean coverage loss (0.58) and a material blur/downscale FA
   reduction (−75% relative). **Met.**
9. Bypass cases:
   - one bad frame: caught when it changes the verdict;
   - mixed and all bad: UNCERTAIN;
   - blur + unsharp: caught;
   - **blur + noise: NOT caught**.

   **Met (tested; one bypass found).**
10. Integrated with adaptive 4/8/16 ONNX via `QualityAwareScorer` /
    `GatedVideoAnalyzer`: frames decoded and scored once, no
    re-scoring. **Met.**
11. Targets met, so the gate is on by default; `enabled=False` keeps
    the previous pipeline. **Met.**
12. Targeted tests (26), full suite 576/576 (233 s), docs, commit. **Met.**
13. No API/UI, retraining, recalibration or test access. **Met.**

**Open items:** the blur + noise bypass; resize 0.75 mislabelled
HEAVY_COMPRESSION with near-total abstention; residual resize 0.33 FA
of 24.5%; noise is not detected; block grid assumed crop-aligned
(`docs/KNOWN_ISSUES.md`).

---

## Phase 9b — Quality-gate hardening (v2 signals)

**Status:** REJECTED by the pre-registered held-out targets. The Phase
9 gate stays in production; the v2 code and artifact are kept as an
experiment.

**Summary:**
- v2 adds a noise estimator, noise-corrected sharpness and
  effective resolution, and an offset-robust blockiness check.
- Developed on `final_train`, tuned on `temp_cal`, verified on the
  held-out `conformal_cal` families; val was read once, labelled
  confirmatory.
- Held-out results:
  - blur+noise false accusations 29% → 0%;
  - benign 0.75× rescale decided 87% (Phase 9: 0%);
  - clean coverage loss 0 pp; 4.2 ms per video.
- **But** severe downscale (0.33×) false accusations regressed from
  33.8% to 47.9%. The limit was +1 pp.

**Requirements:**
1. Developed on train/calibration families only; val not inspected
   while tuning. **Met.**
2. Noise estimator and blur+noise interaction (noise-corrected
   sharpness/resolution, HIGH_NOISE). **Met.**
3. Grid/offset-robust compression check (any offset; a rescaled or
   rotated grid is still not handled). **Met (partially robust).**
4. Train-only quality cases: blur+noise ×3, noise ×3, 0.75 / 0.5 /
   0.33 resize, JPEG ×3 + offset, H.264 ×3, adversarial. **Met.**
5. Separate held-out training families (`conformal_cal`) for
   verification. **Met.**
6. Downgrade-only rule preserved (tested, randomised). **Met.**
7. Targets: three of four met; **severe-downscale non-regression
   failed**. **Not met.**
8. CPU cost 0.84 ms per crop, 4.2 ms per video (< 5 ms). **Met.**
9. One confirmatory val run after freezing, labelled (rerun blocked by
   a marker). **Met.**
10. Mixed-quality and adversarial bypass cases tested (unit tests,
    held-out and confirmatory). **Met.**
11. Targets failed, so Phase 9 is kept and 9b documented as rejected.
    **Met.**
12. Targeted tests (21), full suite 584/584 (236 s), docs, commit. **Met.**
13. No retraining, recalibration, API/UI or test access. **Met.**

**Recommended next step:** combine the v2 noise correction and
offset-robust blockiness with v1's FFT effective-resolution band (or a
downscale-specific detector). Re-verify on fresh held-out training
families (e.g. a new split of `final_train`), because `conformal_cal`
has now been used once for gate verification.

---

## Phase 9c — Final hybrid quality gate

**Status:** REJECTED by the pre-registered targets. The Phase 9 (v1)
gate stays in production and quality-gate experimentation is ended.

**Summary:**
- Hybrid = v2 noise-corrected sharpness + HIGH_NOISE + v2 offset-robust
  blockiness + v1 FFT hf_ratio. Thresholds were copied unchanged from
  the frozen v1/v2 artifacts (no search, no data pass).
- Verified on a fresh final_train challenge split (62 families, 310
  videos; new salt; disjoint from temp_cal/conformal_cal) with new
  corruption seeds, then one confirmatory val run (rerun refused).
- Passed: clean coverage loss 0.97 pp; 0.75× decided 88.1%; blur+noise
  FA 1.6% (v1 24.7%); severe blur FA 0% (= v1).
- **Failed:** 0.33× FA 27.4% vs v1 24.2% (+3.2 pp > 1 pp; val 38.8% vs
  24.5%); cost 6.35 ms/video > 6 ms (pipeline overhead +27%).
- **Root cause (corrects the 9b note):** the FFT check fires the same
  in both gates (val 23 vs 22 videos). v1's 0.33× protection comes from
  its median-denoised sharpness (313 val videos), which v2's
  noise-corrected sharpness replaces (108).

**Requirements:**
1. Only the three named components combined. **Met.**
2. Frozen thresholds reused; no search. **Met.**
3. conformal_cal not used. **Met.**
4. Fresh deterministic train-family challenge split, new seeds. **Met.**
5. Frozen before one confirmatory val run; rerun blocked (verified).
   **Met.**
6. No gate / v1 / hybrid compared (challenge: 8 conditions; val: clean,
   17 stress, 2 adversarial, 3 bypass). **Met.**
7. Targets: 3 of 5 met; **0.33× non-regression failed**. **Not met.**
8. Runtime measured: **6.35 ms/video, target ≤ 6 failed**. **Not met.**
9. Downgrade-only (asserted; 300-case randomised test). **Met.**
10. Targets failed → v1 kept, experimentation ended
    (`PHASE9C_REJECTED.json`). **Met.**
11. Targeted tests (30), one full suite, docs, commit. **Met.**
12. No API/UI, no FF++ test access. **Met.**

**Open items:** v1's known gaps (blur+noise bypass, 0.75×
over-trigger, residual 0.33× FA) remain and are accepted
(`docs/KNOWN_ISSUES.md`).

---

## Phase 10 — Production inference API/service

**Status:** PASS

**Summary:** `configuard.service` is a FastAPI app over the real pipeline:
Phase 5d face extraction → ONNX FP32 → adaptive 4/8/16 (6d) for video or
6c frame calibration for images → Phase 9 v1 gate. It never imports torch.
- Live crops are bit-identical to the stored 5d training crops wherever
  frame indices coincide (8 val videos, max pixel diff 0).
- CPU load test (80 val requests, concurrency 4, 2 workers): 80/80 OK,
  3.78 req/s; video P50 1.41 s / P95 2.86 s; image P50 276 ms / P95
  1.04 s; peak RSS 854 MB.
- The GPU path works, but is slower (3.19 req/s) and uses more memory
  (1.57 GB), so CPU stays the default.

**Requirements:**
1. FastAPI on the real pipeline (the dummy `configuard.pipeline` is not used). **Met.**
2. Validated image + video uploads via `POST /v1/analyze`. **Met.**
3. Verdict, calibrated p_fake / confidence, quality + uncertainty reasons,
   frames used, timeline, model version, timings. **Met.**
4. `/health/live`, `/health/ready` (re-hashes every artifact, every 30 s;
   ONNX sessions + detector must load). **Met.**
5. CPU default; `device: cuda` optional with CPU fallback (tested both). **Met.**
6. Models loaded once; one session per batch size on CUDA (shared on
   CPU); bounded worker pool + admission limit (503 when full). **Met.**
7. Streamed multipart into a private temp dir; magic-byte, size and
   duration validation; upload + analysis timeouts; cleanup on every
   path. **Met.**
8. No uploads retained; filenames/media never logged (verified in tests
   and in 172 load-test log lines); JSON logs with request IDs. **Met.**
9. API-key auth (`X-API-Key`, SHA-256, constant-time); production
   refuses to start without it. **Met.**
10. Structured 4xx/5xx `{"error": {code, message, request_id}}`, no
    traces/paths. **Met.**
11. OpenAPI examples; `service:` config for development/testing/production. **Met.**
12. 26 tests: image, video, corrupt, oversized (per-type and while
    streaming), timeout, concurrency, auth, cleanup, 5 artifact
    tamper cases, safe errors, logs, OpenAPI, GPU fallback, no-torch. **Met.**
13. Real local smoke + load test (uvicorn process, val media only). **Met.**
14. Targeted tests (69) then one full suite; docs; commit. **Met.**
15. No UI, Docker, C2PA or FF++ test access. **Met.**

**Open items:** video latency is dominated by sequential decoding and
YuNet; the timeout is cooperative; image calibration and gate use
video-frame statistics; no TLS or rate limiting
(`docs/KNOWN_ISSUES.md`).

---

## Phase 11 — Explainability and lightweight web UI

**Status:** PASS (with a documented limitation: the CAM hints are only
weakly faithful, so most are withheld).

**Summary:**
- Optional Grad-CAM evidence hints on face crops.
  - Exact for the student's GAP → 1×1 conv → ReLU → Gemm head: the 7×7
    map sums to logit − c with error < 1e-6.
  - Computed torch-free from the hash-verified ONNX, with the feature
    map added as a second output in memory.
- A per-hint occlusion check (top-10 cells vs 3 random sets) withholds
  hints that could mislead.
- A plain HTML/CSS/JS UI is served by FastAPI. Verdict logic,
  calibration, the gate and the model are unchanged; with explain on or
  off the result is identical apart from the explanation field.
- Offline sanity (200 train + 200 val crops):
  - randomization check passes (Spearman −0.14 to −0.18);
  - single-cell occlusion agreement is weak (median Spearman 0.18–0.22);
  - so only 37–40% of slot-0 hints pass; 52.5% of served hints pass in
    the load test.

**Requirements:**
1. Face-crop heatmaps with a faithful CAM for MobileNetV4 (exact Grad-CAM ≡ HiResCAM here). **Met.**
2. Computed after the verdict, never fed back (equality tested for
   image + video); labelled "Visual evidence hint — not proof". **Met.**
3. Perturbation / occlusion / randomization sanity checks;
   failing hints withheld (`failed_occlusion_check`). **Met.**
4. Video: ≤ 4 evidence frames (strongest support for the decision)
   plus the existing timeline. **Met.**
5. Off by default (server `allow_explanations: false` + per-request
   `explain=false`). Cost: +132 ms/video, +53 ms/image server P50;
   +46 MB peak RSS. **Met.**
6. Responsive UI from FastAPI; plain HTML/CSS/JS, no npm/CDN. **Met.**
7. Upload → validation/progress → verdict → confidence → reasons →
   timeline → evidence frames. **Met.**
8. Three visually and textually distinct verdicts; "likely",
   "not proof", "declines to decide" wording. **Met.**
9. Still images marked experimental (API field + UI banner). **Met.**
10. Evidence generated in memory, returned only in the response;
    `Cache-Control: no-store`; no browser storage; temp dir empty. **Met.**
11. Strict CSP + security headers, textContent-only DOM, client- and
    server-side limits, keyboard and mobile support (verified in a
    browser). **Met.**
12. 18 new tests: all verdicts, errors, auth, 5 XSS-like filenames,
    explain off/disabled/on, cleanup, headers, static-code safety. **Met.**
13. Real headless-Chrome smoke test (CDP); normal vs explain latency
    reported. **Met.**
14. Targeted tests (74), one full suite, docs, commit. **Met.**
15. No C2PA, Docker, cloud or FF++ test access. **Met.**

**Open items:** weak CAM faithfulness; evidence JPEGs make explained
video responses about 70 KB; the UI is a local tool (off in the
production config) (`docs/KNOWN_ISSUES.md`).

---

## Phase 12 — C2PA provenance verification

**Status:** PASS

**Summary:** read-only Content Credentials verification with the
official CAI SDK.
- **SDK:** `c2pa-python` 0.38.0 (native c2pa-rs 0.91.0), pinned by
  version + wheel SHA-256.
- **Trust list:** the official C2PA Trust List, pinned to
  `c2pa-org/conformance-public@3573be50`, cached on D: with provenance
  and SHA-256-checked on load.
- **Sandbox:** verification runs in isolated worker processes. Remote
  manifest fetch and OCSP are off, proxies are blackholed and Python
  sockets are blocked. Limits: 5 s timeout (kill + respawn), 512 MB
  per-process memory cap, 50 MB file cap, 2 MB manifest-JSON cap.
- **API:** returns a separate `provenance` object with six statuses
  and a sanitised summary.
- **UI:** a separate Content Credentials card.
- **Isolation:** ML fields are byte-identical with C2PA on vs off
  (tests + benchmark).
- **Cost:** +1.4 ms (unsigned image) to +8.3 ms (signed video) P50.

**Requirements:**
1. Read-only verification for JPEG/PNG/WebP/MP4/MOV/AVI. MKV is
   UNSUPPORTED. No signing code in the service. **Met.**
2. Hard bindings, signatures, chains, validity/timestamps and
   integrity checked by the SDK. Tampered → `assertion.dataHash.mismatch`;
   expired → `signingCredential.expired`. **Met.**
3. Locally cached, hash-pinned official trust list; never fetched in a
   request (refetch only via `scripts/fetch_c2pa_trust_list.py`). **Met.**
4. Manifest URLs never followed: a live local listener received 0
   requests for a remote-manifest asset and an OCSP-URL certificate.
   The positive control (fetch enabled) did hit it. **Met.**
5. ABSENT / VERIFIED_TRUSTED / VERIFIED_UNTRUSTED / INVALID /
   UNSUPPORTED / ERROR. **Met.**
6. Never modifies the verdict, confidence, calibration or gate (tested
   for plain/signed/tampered; benchmark identical). **Met.**
7. Allow-listed summary (signer, generator, actions, IPTC source type
   incl. an AI-generated declaration, timestamps, ingredients,
   validation codes). Control/bidi chars stripped; URL-like strings
   dropped; length caps. **Met.**
8. Separate "Content Credentials" card with ABSENT ≠ fake and
   VERIFIED ≠ factual-truth wording (browser-verified). **Met.**
9. Time, size and memory limits (worker kill on timeout; Job Object /
   RLIMIT_AS cap verified to stop a 600 MB allocation). **Met.**
10. Tests: no manifest, valid-untrusted, trusted test anchor, tampered,
    malformed, expired, timeout (+ crash, memory, SSRF, sanitisation,
    trust-list tamper). **Met.**
11. Temp dirs empty after every path; test keys generated in memory /
    tmp only; a test fails if any tracked file contains a private key. **Met.**
12. Benchmarked with and without credentials (real server). **Met.**
13. Targeted tests (92), one full suite, docs, commit. **Met.**
14. No FF++ test access, Docker/cloud or model changes. **Met.**

**Open items:** TSA list cached but not wired as a separate SDK anchor
set; no real-world trusted-signer sample tested; Windows-only wheel
hash pinned (`docs/KNOWN_ISSUES.md`).
