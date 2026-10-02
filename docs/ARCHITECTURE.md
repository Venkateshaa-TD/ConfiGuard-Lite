# Architecture

Status: Phase 4 complete. The Phase 1 pipeline shape below is real and
tested; its preprocessing/model/provenance stages are still deliberate
placeholders (see "Phase 1 scope" below). Phase 2 added a real,
independently-tested face/media preprocessing subsystem
(`configuard.media`). Phase 3 added a real, independently-tested dataset
registry (`configuard.datasets`). Phase 4 adds two real, ImageNet
-pretrained visual-encoder backbones with an **untrained, uncalibrated**
binary head (`configuard.models`, documented in its own section below).
None of these are yet wired into `configuard.pipeline.run_pipeline` or
into a real training loop - that happens once deepfake fine-tuning
exists (see docs/PROJECT_PLAN.md).

## Data flow

```
   file path (image or video)
            |
            v
  +-------------------+
  |  validation        |  configuard.validation.validate_media_file()
  |  - exists / is a   |  - content-sniffed media type (magic bytes),
  |    regular file    |    NOT trusted from the extension alone
  |  - size limit      |  - extension cross-checked against sniffed type
  |  - (video) duration|  - video duration via ffprobe
  |    via ffprobe     |
  +-------------------+
            |
      invalid/corrupted -----> PipelineRejectedError(ValidationResult)
            |
          valid
            v
  +-------------------+
  |  preprocessing      |  configuard.pipeline._preprocess_placeholder()
  |  (placeholder)      |  - image -> 1 "frame"
  |                     |  - video -> adaptive frame count (4/8/16) via
  |                     |    select_frame_count(duration_seconds)
  |                     |  - no real decoding/tensors yet
  +-------------------+
            v
  +-------------------+
  |  prediction          |  configuard.pipeline._dummy_predict()
  |  (dummy, deterministic)|  - sha256(filename:size) -> pseudo-score
  |                       |  - NOT a trained model
  +-------------------+
            v
  +-------------------+
  |  provenance           |  configuard.pipeline._provenance_placeholder()
  |  (placeholder)        |  - always "not_checked" - C2PA arrives in its
  |                       |    own phase; kept as a separate result, never
  |                       |    merged into the model verdict/score
  +-------------------+
            v
     DetectionResult (see schema below)
```

## Typed contracts (`src/configuard/io_types.py`)

| Type | Produced by | Purpose |
|---|---|---|
| `ImageInput` / `VideoInput` | caller | Pre-validation request for one file |
| `ValidatedMedia` | `validation.validate_media_file` | Confirmed media type + file facts |
| `PreprocessingOutput` | `pipeline._preprocess_placeholder` | Frame count / shape contract |
| `ModelOutput` | `pipeline._dummy_predict` | Verdict, confidence, raw score |
| `ProvenanceOutput` | `pipeline._provenance_placeholder` | C2PA status, always separate |
| `DetectionResult` | `pipeline.run_pipeline` | The one documented final schema |

These dataclasses are frozen (immutable) and the enums (`MediaType`,
`Verdict`, `ProvenanceStatus`) are `str` subclasses, so `result_to_dict()`
/ `json.dumps` produce plain, stable JSON without a custom encoder.

## `DetectionResult` JSON schema (example)

```json
{
  "source_filename": "smoke.mp4",
  "media_type": "video",
  "validated": {
    "source_path": "C:\\path\\to\\smoke.mp4",
    "media_type": "video",
    "file_size_bytes": 3298,
    "duration_seconds": 2.0
  },
  "preprocessing": {
    "media_type": "video",
    "frame_count": 4,
    "frame_shape": [224, 224, 3],
    "notes": "placeholder: no real preprocessing implemented yet"
  },
  "model_output": {
    "verdict": "likely_real",
    "confidence": 0.3516,
    "manipulation_score": 0.3242,
    "model_name": "dummy-placeholder-v0"
  },
  "provenance": {
    "status": "not_checked",
    "details": "C2PA provenance checking is not implemented until its dedicated phase."
  },
  "pipeline_version": "0.1.0-phase1-dummy",
  "warnings": []
}
```

`verdict` is always one of `"likely_real"`, `"likely_manipulated"`,
`"uncertain"` (project requirement 10). `provenance.status` is always one
of `"present_valid"`, `"present_invalid"`, `"absent"`, `"not_checked"`, and
is never combined into `model_output` (requirement 15).

## Validation security model (`src/configuard/validation.py`)

File-type trust comes from content, not filename:

1. **Extension allowlist** - checked, but only used to (a) pick the size
   limit and (b) cross-check against the sniffed type. Never trusted alone.
2. **Magic-byte sniffing** - the first ~64 bytes are inspected for known
   signatures (JPEG `FFD8FF`, PNG `89504E47...`, WEBP `RIFF....WEBP`,
   MP4/MOV `....ftyp`, Matroska/WebM `1A45DFA3`, AVI `RIFF....AVI `). An
   unrecognized signature is rejected outright, even with a "correct"
   extension.
3. **Extension/signature mismatch** is its own rejection reason (e.g. a
   PNG renamed to `.mp4`).
4. **Size limits** - separate `max_image_size_mb` / `max_video_size_mb`
   from the active `ValidationLimits` (per-environment, see below).
5. **Video duration** - read via `ffprobe -show_entries format=duration`
   with a timeout; a corrupted/unreadable video (ffprobe fails or times
   out) is rejected as `VIDEO_UNREADABLE`, never allowed to crash the
   pipeline. Duration over `max_video_duration_seconds` is rejected as
   `VIDEO_TOO_LONG`.

Every rejection path returns a `ValidationResult` (never raises for bad
*input*); `pipeline.run_pipeline` turns an invalid result into a
`PipelineRejectedError` that carries the full result for the caller to
inspect.

## Configuration schemas (`src/configuard/config.py`, `configs/*.yaml`)

`ProjectConfig` (frozen dataclass) now carries an `environment` field
(`"development" | "training" | "testing" | "production"`) and a nested
`ValidationLimits` (frozen dataclass: size/duration limits + allowed
extensions). Four environment files exist under `configs/`:

| File | Environment | Notable limits |
|---|---|---|
| `development.yaml` | development | Generous (20 MB image / 200 MB video / 120 s), `DEBUG` logging |
| `training.yaml` | training | Larger video size (500 MB) for curated samples, fixed seed |
| `testing.yaml` | testing | Tiny (1 MB image / 5 MB video / 5 s) so fixtures stay fast |
| `production.yaml` | production | Strict (15 MB image / 150 MB video / 90 s), quiet logging |

`configs/base.yaml` remains as a generic default profile (used by Phase 0
tests) and picks up the same `ValidationLimits` defaults when no
`validation:` block is present.

## Phase 1 scope (what is and isn't real)

**Real and tested:** typed contracts, config schemas, file-type/size/
duration validation (magic-byte sniffing + ffprobe), end-to-end
orchestration, deterministic dummy prediction, JSON-serializable structured
result.

**Explicitly placeholder (later phases):** actual frame/face decoding,
MobileNetV4-Conv-Small / EfficientNet-B0 encoders, GenD CLIP-L/14
distillation, the temporal GRU, calibration/conformal prediction, ONNX
export, heatmaps/evidence timelines, and real C2PA provenance checking.

## Face and media preprocessing (Phase 2, `src/configuard/media/`)

Independent, separately-tested subsystem. Not yet called from
`configuard.pipeline` - see the note at the top of this document.

```
   image path                          video path
      |                                    |
      v                                    v
  decode_image()                  extract_video_metadata()
  (configuard.media.decode)        (ffprobe, falls back to OpenCV;
      |                             reports fps/duration/rotation/VFR)
      |                                    |
      |                          compute_sampling_plan(frame_count, N)
      |                             (configuard.media.sampling)
      |                             nested nesting: indices(4) subset
      |                             indices(8) subset indices(16),
      |                             deduplicated for short videos
      |                                    |
      |                          decode_sampled_frames()
      |                             sequential decode of just the
      |                             requested indices; missing/corrupt
      |                             frames -> warnings, never a crash
      |                                    |
      v                                    v
  detector.detect(image)            detector.detect(frame.image) per frame
  (FaceDetector protocol:                  |
   YuNetFaceDetector or                    v
   MockFaceDetector for tests)      track_faces() (configuard.media.tracking)
      |                             greedy IoU + landmark-distance
      |                             tie-break, temporal-gap tolerance
      |                             (adapted to the sampling stride -
      |                             see docs/DECISIONS.md), frame order
      |                             preserved
      |                                    |
      |                          select_primary_track()
      |                             longest track wins; other tracks
      |                             with a meaningful length produce a
      |                             warning instead of being silently
      |                             dropped
      v                                    v
  align_and_crop() per face          align_and_crop() per face in the
  (configuard.media.alignment:        primary track (frame order)
   levels the eyes, expands by
   margin_ratio, clips to image
   bounds, resizes to output_size)
      |                                    |
      v                                    v
  FaceCropCache.get()/.put()          FaceCropCache.get()/.put()
  (configuard.media.cache: keyed by (input SHA-256, frame_index,
   track_id, PreprocessingConfig.version_tag); atomic writes via
   temp-file + os.replace; optional oldest-first byte-limit eviction)
      |                                    |
      v                                    v
  ImagePreprocessingResult          VideoPreprocessingResult
```

**Typed contracts** (`src/configuard/media/types.py`): `BoundingBox` (with
`.iou`, `.expanded`, `.clipped`), `FaceLandmarks` (5-point, YuNet
convention: right eye, left eye, nose tip, right mouth corner, left mouth
corner), `DetectedFace`, `FaceTrack`, `VideoMetadata`, `SampledFrame`,
`FrameSamplingPlan`, `PreprocessingConfig`, `ImagePreprocessingResult`,
`VideoPreprocessingResult`.

**FaceDetector is a `Protocol`** (`src/configuard/media/face_detector.py`):
`YuNetFaceDetector` wraps `cv2.FaceDetectorYN_create` (CPU-only, no CUDA
required - see docs/EXPERIMENT_LOG.md for the load/verification run);
`MockFaceDetector` returns pre-configured detections (fixed, or
per-detector-call via a `{call_index: [...]}` map) for fully deterministic
unit tests. `configuard.media.preprocess` takes a `FaceDetector` and a
`FaceCropCache` via dependency injection - nothing in that module imports
`cv2.FaceDetectorYN_create` directly.

**Nested frame sampling** (`src/configuard/media/sampling.py`): the
16-frame set is a uniform sample of the video; the 8-frame set is every
second element of the 16-frame set; the 4-frame set is every second
element of the 8-frame set. This guarantees `indices(4) ⊆ indices(8) ⊆
indices(16)` by construction - see docs/DECISIONS.md for why.

**Handling required edge cases** (docs/PROJECT_PLAN.md / Phase 2 task
list item 13):

| Case | Handling |
|---|---|
| No face | `faces=()` / no primary track; `"no_face_detected"` warning, no crash |
| Multiple faces (image) | Highest-confidence one used; `"multiple_faces_detected"` warning |
| Multiple faces (video) | Longest track used; `"multiple_face_tracks"` warning if another track is a meaningful fraction of that length |
| Rotated media | `VideoMetadata.rotation_degrees` extracted best-effort from ffprobe (legacy `rotate` tag or Display Matrix side data); not yet auto-corrected in the crop (see docs/KNOWN_ISSUES.md) |
| Very small faces | Detections below `PreprocessingConfig.min_face_size_px` are excluded before tracking/alignment; `"small_faces_excluded"` warning |
| Corrupted frames | `decode_sampled_frames` returns whatever frames it could decode + a warning per missing index; a fully unopenable file raises `DecodeError` (already filtered out earlier by `configuard.validation` in the full pipeline) |
| Short videos | `compute_nested_sampling_plans` deduplicates and never requests more frames than exist |
| Variable frame rate | `extract_video_metadata` prefers ffprobe's `avg_frame_rate` vs. `r_frame_rate` comparison to flag `is_variable_frame_rate` and estimate `frame_count` for containers where `nb_frames` is absent |

**Cache** (`src/configuard/media/cache.py`): key = `(input_sha256,
frame_index, track_id, config_version)`; `PreprocessingConfig.version_tag`
is a hash of every alignment/detector setting, so any config change
automatically lands in a new cache namespace rather than silently reusing
stale crops. Writes go to a temp file in the same directory, then
`os.replace()` (atomic on both POSIX and Windows for a same-volume
rename). `max_bytes` triggers oldest-first (mtime) eviction.

## Dataset registry (Phase 3, `src/configuard/datasets/`)

Independent, separately-tested subsystem. Not yet wired into any training
loop (none exists yet - see docs/PROJECT_PLAN.md).

```
   adapter.build_manifest(local_root)
        |
        v
  DatasetAdapter (Protocol: validate_structure(), build_manifest())
    |                                    |
    v                                    v
  FolderConventionAdapter          MetadataSidecarAdapter
  (real/fake in known                (a JSON/JSONL/CSV sidecar at
   subdirectories; identity           the dataset root declares
   from filename)                     path/label/... per row - no
    |                                  folder assumptions at all;
    | backs FaceForensics++,          also the officially "generic"
    | Celeb-DF-v2,                    adapter for future datasets)
    | DeeperForensics-1.0              |
    |                                  | backs DFDC, DF40, and any
    |                                  | future dataset via
    |                                  | make_generic_metadata_adapter()
    +------------------+---------------+
                        v
              list[Sample]  (canonical schema, configuard.datasets.schema)
                        |
          +-------------+-------------+
          v                           v
   write_manifest()             validate_samples() /
   (JSONL, one Sample            validate_manifest_file()
    per line)                    missing files, duplicate IDs,
          |                      invalid labels, broken parent/
          v                      pair refs, unsupported media
   read_manifest() /             types, cross-split source/
   parse_manifest_lenient()      identity/pair leakage
   (strict / issue-collecting)          |
          |                             v
          |                   ManifestValidationReport
          v
   split_samples(samples, SplitConfig(seed, fractions))
     1. compute_leakage_groups(): union-find over shared
        source_id, shared identity_id, parent_sample_id
        links, and paired_sample_id links
     2. each group -> one split, via
        hash(seed, group_key) bucketed against cumulative
        fractions (deterministic, order-independent)
          |
          v
   SplitAuditReport (assignments + per-group audit trail)
   -> write_split_audit_report() (JSON)

   build_duplicate_report(samples, media_root)
     - find_exact_duplicates(): group by Sample.checksum_sha256
     - find_near_duplicate_images(): average-hash (aHash) computed
       via OpenCV, pairwise Hamming distance <= configurable threshold
          |
          v
   DuplicateReport (report only - never deletes anything)
```

**Canonical schema** (`src/configuard/datasets/schema.py`): `Sample` is a
frozen dataclass with every field docs/PROJECT_PLAN.md requirement 2
lists (sample/dataset identity, media type/path, label, source/identity/
parent/paired IDs, manipulation family/method/compression, official
split, license status, optional demographic attributes, checksum,
preprocessing version). `demographic_attrs` defaults to `None` and is
never computed/inferred by any adapter - only ever set from a field an
adapter's underlying metadata source explicitly provides (none of the
five built-in adapters currently populate it, since no verified official
source of per-sample demographic labels was available to this phase).

**Leakage-safe splitting** (`src/configuard/datasets/splitting.py`): see
docs/DECISIONS.md for why grouping is done as connected components
(union-find) rather than simpler pairwise rules, and why split assignment
is a deterministic hash rather than a seeded shuffle.

**Storage-path checking** (`src/configuard/datasets/storage.py`):
`check_storage_path`/`check_all_storage_paths` report configured path,
free/total space, writability, and whether the path resolves inside the
git repository, for `CONFIGUARD_DATA_DIR` / `CONFIGUARD_CACHE_DIR` /
`CONFIGUARD_CHECKPOINT_DIR` / `CONFIGUARD_OUTPUT_DIR`.
`assert_safe_storage_path` raises `UnsafeStoragePathError` instead of
just warning when a path is inside the repo (docs/PROJECT_PLAN.md
requirement 11's "must warn or refuse"). See `scripts/check_storage.py`
for the CLI and docs/EXPERIMENT_LOG.md for real output, including the
refusal case.

**This machine's configured storage layout** (local `.env`, never
committed - see `.env.example` for the template):

```
CONFIGUARD_DATA_DIR       = D:\ConfiGuard-Data\datasets
CONFIGUARD_CACHE_DIR      = D:\ConfiGuard-Data\cache
CONFIGUARD_CHECKPOINT_DIR = D:\ConfiGuard-Data\checkpoints
CONFIGUARD_OUTPUT_DIR     = D:\ConfiGuard-Data\outputs
```

All four sit on `D:` (92.9 GB free, a separate local fixed volume - not
the repo's `C:` drive, not OneDrive-synced), per the approved storage
audit - see `docs/EXPERIMENT_LOG.md` for the full verification (exists /
writable / not-in-repo / not-in-OneDrive / recognized by
`scripts/check_storage.py`) and `docs/KNOWN_ISSUES.md` for the disk-space
history this resolves.

Pretrained model weight caches (Phase 4) are configured the same way,
via the *standard* env var names `huggingface_hub`/`torch` themselves
read - not `CONFIGUARD_*` names, since these libraries own that
contract:

```
HF_HOME       = D:\ConfiGuard-Data\cache\huggingface
HF_HUB_CACHE  = D:\ConfiGuard-Data\cache\huggingface\hub
TORCH_HOME    = D:\ConfiGuard-Data\cache\torch
```

**Critical ordering constraint:** `huggingface_hub` and `torch` read
these as module-level constants at *import* time - setting them after
`import timm`/`import torch` has already run anywhere in the process has
no effect. `configuard.env_loader.load_dotenv()` (a tiny dependency-free
`.env` parser) must be called before any such import; every Phase 4
entrypoint script does this first, and the root `tests/conftest.py` does
it before test collection so the whole suite is covered - see
docs/DECISIONS.md.

## Pretrained visual encoders (Phase 4, `src/configuard/models/`)

Independent, separately-tested subsystem. **Every prediction from this
subsystem is untrained and uncalibrated** - see the big warning in
docs/MODEL_CARD.md. Not yet wired into `configuard.pipeline`; trained
only on synthetic data so far by the Phase 5 training pipeline (below).

```
   EncoderSpec (name, timm_model_name, hf_repo_id, license)
        |
        v
   DeepfakeVisualEncoder(spec, pretrained=True|False)
     - self.backbone = timm.create_model(timm_model_name, num_classes=0)
     - self.num_features probed empirically via one eval()-mode forward
       pass (NOT trusted from backbone.num_features - see docs/DECISIONS.md)
     - self.head = nn.Linear(num_features, 1)   <- randomly initialized,
       NEVER trained on deepfake data in this phase
        |
        +-- forward_features(x) -> (B, num_features)      pooled embedding
        +-- forward_logits(x)   -> (B,)                    pre-sigmoid logit
        +-- forward_probs(x)    -> (B,)                     sigmoid probability
        +-- forward(x) = forward_logits(x)                   <- ONNX export entry point
        |
        v
   configuard.models.preprocess
     preprocess_bgr_image()/preprocess_batch(): BGR uint8 HxWx3
     (the same format configuard.media.alignment.align_and_crop
     produces) -> normalized (3,224,224) float tensor, using the
     ENCODER'S OWN resolved mean/std (timm.data.resolve_data_config),
     not a hardcoded ImageNet constant
        |
        v
   configuard.models.inference
     infer_image() / infer_image_batch(): independent per-image predictions
     infer_video_fixed_frames(): ordered frame list -> forward_features
       per frame -> MEAN of the (N, num_features) embeddings -> one
       head pass -> one PredictionResult per clip (no GRU yet - task
       scope; see docs/DECISIONS.md for why aggregation happens in
       feature space, not on per-frame probabilities)
        |
        v
   configuard.models.real_pipeline  (optional; NOT configuard.pipeline)
     run_real_image_pipeline() / run_real_video_pipeline():
     configuard.media (Phase 2 face crop) -> configuard.models
     (Phase 4 encoder), fully dependency-injected (detector/cache/
     encoder) so tests substitute MockFaceDetector + a pretrained=False
     encoder - no network access needed for unit tests
        |
        v
   configuard.models.onnx_export
     export_to_onnx(): FP32, dynamic batch axis, legacy TorchScript
       exporter (dynamo=False - see docs/DECISIONS.md)
     verify_onnx_parity(): seeded random inputs, PyTorch vs ONNX Runtime
       (CPU), max/mean abs diff vs documented (atol, rtol)
     run_onnx_cpu_inference(): direct ONNX Runtime CPU inference

   configuard.models.benchmark
     measure_latency(): warm-up iterations + P50/P95 (never a single
       timing) measure_gpu_peak_memory() / approximate_flops()
       (torch.utils.flop_counter, no new dependency)
```

**Registry** (`registry.py`): `ENCODER_SPECS` holds exactly the two
authorized models - `mobilenetv4_conv_small` (preferred candidate) and
`efficientnet_b0` (comparison baseline). `create_encoder(name,
pretrained=True|False)` is the single construction entry point.

**Two real bugs found and fixed while building this** (both covered by
regression tests, both explained in docs/DECISIONS.md): (1)
`backbone.num_features` doesn't reliably match the actual pooled output
dimension for every timm architecture (MobileNetV4 specifically), fixed
by probing empirically; (2) that empirical probe itself first failed
because BatchNorm rejects a batch of size 1 in `train()` mode, fixed by
probing in `eval()` mode and restoring the original mode afterward.

**Performance measurements** (parameter counts, checkpoint sizes,
approximate FLOPs, CPU/GPU P50/P95 latency, peak GPU memory, image-batch
and 4/8/16-frame video-inference timings, ONNX parity numbers) are all in
`docs/EXPERIMENT_LOG.md` - reproducible via `scripts/benchmark_models.py`
and `scripts/export_onnx_models.py`. These are a **provisional efficiency
comparison only**; no deepfake-detection accuracy result exists yet, so
the final model choice is explicitly deferred, not decided by latency
alone (docs/DECISIONS.md).

## Reproducible training pipeline (Phase 5, `src/configuard/training/`)

Trains either Phase 4 encoder (MobileNetV4-Conv-Small by default,
EfficientNet-B0 as the comparison) as a binary real(0)/fake(1)
classifier from Phase 3 manifests, through Phase 2 aligned face crops.
So far it has only been run on synthetic data (see `docs/MODEL_CARD.md`).

```
 configs/train/*.yaml ──> TrainingConfig (frozen; unknown keys rejected)
        │
 scripts/train.py ── assert_dependency_safety()  (CPU-only torch / broken torchvision / lost CUDA = hard stop)
        │            HF_HUB_OFFLINE=1            (cached Phase 4 weights only)
        v
 runner.build_trainer(config, detector, face_config, checkpoint_dir, output_dir, face_cache_dir)
   ├─ set_global_seed(seed)          python / numpy / torch / cuda, cuDNN deterministic
   ├─ read_manifest(train, val)      Phase 3 canonical Sample schema
   ├─ assert_no_cross_split_leakage  source / identity / real-fake pair / sample_id
   ├─ sha256(train & val manifests)  -> checkpoint provenance
   ├─ build_manifest_dataset         IMAGE -> ManifestImageDataset      item (3,224,224)
   │                                 VIDEO -> ManifestVideoFrameDataset item (N,3,224,224)
   │                                 both via configuard.media.preprocess_{image,video}
   │                                 (detect -> align -> FaceCropCache) -> models.preprocess
   ├─ build_dataloader               source+class-balanced WeightedRandomSampler (train),
   │                                 sequential (val); seeded generator + per-worker seeding
   └─ Trainer
        per epoch E:  reseed sampler/loader generators with seed+E   (exact-resume basis)
          train:  autocast(fp16 on CUDA) -> compute_logits_for_batch (5D video = mean of frame
                  embeddings) -> BCE on fp32 logits -> NaN/Inf check -> GradScaler.scale/backward
                  every grad_accum_steps: unscale -> clip_grad_norm -> scaler.step/update
                  -> scheduler.step() only if the step was not AMP-skipped
                  CUDA OOM anywhere -> TrainingOutOfMemoryError (batch size never changed)
          validate: loss + ValidationMetrics{threshold_free: AUROC, AP |
                                             threshold_dependent @0.5: confusion matrix,
                                             sensitivity, specificity, balanced acc, precision, F1}
          select:   val AUROC -> (undefined) lowest val loss -> (no val set) lowest train loss;
                    exact ties -> lower val loss
          log:      <output>/<run>.jsonl + <run>_{train,epoch}.csv
          save:     <ckpt>/<run>_latest.pt every epoch, <run>_best.pt on improvement
          early stop after `early_stopping_patience` non-improving epochs
```

**Checkpoint contents** (atomic temp-file + `os.replace`, format v2):
model, optimizer, scheduler, and AMP scaler states; epoch and global step;
Python/NumPy/torch-CPU/torch-CUDA RNG states; `TrainState` (best metric,
tie-break, best epoch, patience counter, AMP-skipped steps); metrics;
provenance (encoder name, HF model id **and revision**, face-preprocessing
`version_tag`, train and val manifest SHA-256, git commit, full config).

**Resume** loads on CPU, then `verify_checkpoint_compatible` refuses any
mismatch in encoder, model id, preprocessing version, either manifest
checksum, or a `RESUME_CRITICAL_CONFIG_KEYS` field, and lists every
mismatch. Resumed and uninterrupted runs are identical (max parameter
diff 0.0 measured on CPU and RTX 4050). Granularity is one epoch.

**Storage:** checkpoint dir = `CONFIGUARD_CHECKPOINT_DIR`, logs/outputs
= `CONFIGUARD_OUTPUT_DIR`, face crops = `CONFIGUARD_CACHE_DIR/face_crops`
(`configuard.training.paths`, which refuses paths inside the repo). On
this machine all three are on `D:\ConfiGuard-Data\`.

**Smoke mode** (`scripts/train.py --smoke cpu|cuda [--encoder ...]`,
`runner.run_smoke`): generates synthetic tinted-checkerboard data under
`<output>/smoke/<run>/`, trains with a full-frame mock face detector,
re-runs as interrupted + resumed to measure resume consistency, and
writes `<run>_summary.json` (peak VRAM, step/validation time, checkpoint
size, resume diff). **Evaluation-only:** `scripts/evaluate.py` loads a
checkpoint's weights (encoder taken from its provenance) and writes the
metrics JSON to `<output>/eval/`.

**Dependency safety** (`configuard.dependency_safety`, pinned pair in
`constraints-cuda.txt`): checks the torch CUDA build, torchvision import
plus the first compiled op, torch↔torchvision release pairing, and CUDA
availability. Runs in `scripts/verify_environment.py` and at CLI start.

## FaceForensics++ c23 acquisition (Phase 5b)

The first real dataset. It lives entirely under `CONFIGUARD_DATA_DIR`
(on this machine `D:\ConfiGuard-Data\datasets\FaceForensics++`) and
never enters Git.

```
D:\ConfiGuard-Data\datasets\FaceForensics++\
├── _official_script\faceforensics_download_v4.py   official script (SHA-256 pinned in scripts/download_faceforensics_c23.py)
│                    filelist.json                  official 500-pair list (expected file set)
├── original_sequences\youtube\c23\videos\NNN.mp4               1000 real
├── manipulated_sequences\{Deepfakes,Face2Face,FaceSwap,NeuralTextures}\c23\videos\TTT_SSS.mp4
│                                                               1000 fake each; TTT = target original, SSS = source original
├── _official_splits\<revision>\{train,val,test}.json          official split, pinned (Phase 5c; not in Git)
└── _manifests\faceforensics++_c23.jsonl                        canonical Phase 3 manifest (no split)
               faceforensics++_c23_leakage_groups.json          union-find groups any split must respect
               faceforensics++_c23_{train,val,test}.jsonl       official-split manifests (Phase 5c)
               faceforensics++_c23_official_split.jsonl         all 5000 samples, split-labelled (Phase 5c)
```

Pipeline:
1. `scripts/download_faceforensics_c23.py` runs the hash-pinned official
   script with an allow-list (5 datasets, `c23`, `videos`, `EU2`) and a
   free-space watchdog (stops before < 40 GB free).
2. `scripts/validate_faceforensics.py` performs these checks:
   - `configuard.datasets.faceforensics.validate_ffpp_c23`: structure,
     no raw/c40/masks/models/DFD/FaceShifter, per-class counts against
     the official pair list, zero-byte files, `tmp*` partials, ffprobe
     header + full packet demux of every video, and the
     `<target>_<source>` relationships.
   - `FaceForensicsAdapter.build_manifest`: SHA-256 per file; each fake
     gets `parent` = target original and `paired` = source original.
   - Phase 3 `validate_samples`, `find_exact_duplicates` and
     `compute_leakage_groups`.
   - It writes a JSON + Markdown acquisition report to
     `CONFIGUARD_OUTPUT_DIR/acquisition/faceforensics/`.

Face crops are extracted in Phase 5d (next section); no model consumes this data yet.

**Official split (Phase 5c).** `scripts/apply_faceforensics_splits.py`
does the following:
- Loads the pinned official split files
  (`_official_splits/<revision>/`) through
  `configuard.datasets.faceforensics_splits.load_official_splits`, which
  refuses on any size or SHA-256 mismatch.
- `reconcile_pairs` checks them against the official 500 pairs.
- `assign_official_splits` labels each Phase 5b manifest sample by its
  originals' split. It refuses the whole assignment on any cross-split
  original, parent, pair, or leakage group.
- It writes `faceforensics++_c23_{train,val,test}.jsonl` and re-checks
  them with the Phase 5 trainer's own `find_cross_split_leakage`.
- It audits counts, group balance, duration, and native resolution per
  split.

## Matched face-crop extraction (Phase 5d, `src/configuard/crops/`)

Turns the 5000 official-split FF++ c23 videos into exactly 16 aligned
224×224 RGB PNG crops per accepted video, with real/fake temporal
matching and leakage lineage preserved. Entry point:
`scripts/extract_ffpp_face_crops.py`; shortcut audit:
`scripts/audit_ffpp_crop_shortcuts.py`.

```
official-split manifest ──► families.build_content_families
   (1000 families = target original TTT + {DF,F2F,FS,NT}/TTT_SSS; donor SSS = 2nd leakage parent)
        │  one family per worker (bounded ProcessPool, cv2 single-threaded per worker)
        ▼
extract.extract_family
   ffprobe frame counts ─► matching.shared_frame_count = min over the 5 videos
   matching.planned_indices = nested 16/8/4 contract over [0, shared-1]   (same indices for all 5)
   read_frames_sequential (grab() past unwanted frames; never loads a whole video)
   YuNet (hash-pinned) ─► build_primary_track (IoU OR ≤1 face-width centre shift, ≤1.5× size)
   missing face? ─► matching.recovery_offsets (+1,-1,+2,…,±6, half-gap bounded), ≤2 rounds;
                    tracks rebuilt over planned+recovery frames after each round
                    resolve_slot: joint (all 5 same index) ► individual (approximate) ► failed
   any failed slot ─► whole family QUARANTINED (crops kept for review, never padded)
   alignment.align_face: 5-point Umeyama similarity → template, margin 0.25, reflect-101
        ▼
store.CropStore  crops/<config_tag>/<sha[:2]>/<input_sha256>/f<frame>.png   (atomic)
                 families/<config_tag>/<family_id>.json                     (written last)
        ▼
manifests.*  crops_{train,val,test}.jsonl (model-facing, whitelisted fields)
             matched_pairs.jsonl · quarantine.jsonl · crop_audit.jsonl (audit only)
             extraction_summary.json (deterministic; leakage re-validation; stats)
```

Contracts:
- **Temporal matching is by frame index**, not timestamp: fake frame *i*
  shows target frame *i* (verified by registration; 64 fakes have a
  different fps header but still align by index).
- **One rule for every class.** Real and fake members of a family use
  the same planned indices and the same detector/linking/recovery/alignment
  code; nothing branches on label or method.
- **Model-facing manifest** rows contain only `crop_path`, `crop_sha256`,
  `sample_id`, `label`, `slot`, `nested_levels` and a `metadata` block
  (split, method, source/leakage group, content and donor parents,
  family, config tag, link to the video manifest + its SHA-256). Source
  resolution, duration, fps, frame count, codec, file size and frame
  indices are excluded by construction (`MODEL_ROW_FIELDS`), and live
  only in `crop_audit.jsonl` / family records.
- **Nested selection:** slot *s* is in the 8-set iff *s* is even, and in
  the 4-set iff *s* % 4 == 0 (`nested_levels`), mirroring
  `configuard.media.sampling`.
- **Staleness is refused** (`StaleCropError`): a store root accepts one
  config (`store_config.json`); records must match the config tag, the
  member input SHA-256s, and the crop sizes (full crop SHA-256 on demand).
- **Resumable/idempotent:** completed families are verified and
  skipped; manifests are rebuilt from records only, sorted, without
  timestamps, so reruns are byte-identical.
- **Storage floor:** `configuard.storage_guard.FreeSpaceGuard` stops
  scheduling at floor + margin (40 + 2 GB) and the run refuses up front
  if the projected size would cross it.

On this machine the store root is `D:\ConfiGuard-Data\cache\ffpp_face_crops\store`.

## Frozen GenD teacher and logit cache (Phase 6a, `src/configuard/teacher/`)

- **`gend.py`** rebuilds GenD CLIP-L/14 (`CLIPVisionTransformer`, ViT-L/14,
  224 px, `sdpa`) plus the unused `visual_projection` and a
  `Linear(1024, 2)` head. It strict-loads the hash-pinned
  `model.safetensors` from the HF cache on D:. `load_gend_teacher`
  verifies the SHA-256, sets `requires_grad_(False)` and `eval()`, and
  calls `assert_frozen`. Forward: CLIP mean/std normalisation →
  `pooler_output` → L2 normalise → linear. Logits are `[real, fake]`.
- **`cache.py`** (`TeacherLogitCache`):
  `<cache>/teacher_logits/gend_clip_l14/<tag>/<split>/`.
  - `meta.json` holds the teacher config, the crop-manifest SHA-256 and
    the shard size.
  - `shard_NNNNN.npz` covers fixed manifest rows and stores the tag,
    the crop SHA-256s and float32 (n, 2) logits.
  - `teacher_logits.jsonl` is consolidated in manifest order and keyed
    by `crop_sha256`.
  - The tag is a SHA-256 over the repo, revision, weights hash, head,
    preprocessing, attention implementation, autocast dtype, batch
    size, torch/transformers versions, shard size and schema.
  - Allowed splits are `train` and `val`. `test` raises
    `ProtectedSplitError`, and so does any row whose split differs from
    the requested one.
- **`scripts/cache_teacher_logits.py`** reads the Phase 5d store as-is.
  It checks each crop manifest's SHA-256 against
  `extraction_summary.json`, re-hashes crop bytes as they are decoded,
  runs fp16 autocast at bs 64, logs progress/ETA/free space, and writes
  a JSON report with teacher AUC/accuracy per split.
  `--limit-shards N` writes to a separate `_trial` root.
- The teacher is only a frozen, offline distillation target. It is
  never trained, exported or used for serving.

## Student training and distillation (Phase 6b, `src/configuard/distill/`)

- **`data.py`**:
  - `load_crop_rows` reads `crops_{train,val}.jsonl` and checks its
    SHA-256 against the Phase 5d summary. `test` raises
    `ProtectedSplitError`.
  - `load_teacher_margins` aligns the Phase 6a consolidated logits row
    by row (crop SHA-256, sample id, slot, tag, manifest SHA-256) and
    returns `m = logit_fake − logit_real`. GenD itself is never loaded.
  - `balanced_weights`: real 1/2, each method 1/8.
  - `EpochSampler` yields `(index, epoch)` pairs, so augmentation is
    seeded per item while workers stay persistent.
  - `CropDataset` returns uint8 RGB CHW tensors, the label, the margin
    and the index.
- **`augment.py`**: class-independent mild Gaussian blur plus horizontal
  x-scale/shift, seeded by (seed, epoch, index). Train only.
- **`losses.py`**: `(1−α)·BCE + α·T²·BCE(σ(z/T), σ(m/T))`; α = 0 is
  the baseline.
- **`evaluate.py`**:
  - Frame and video metrics; video score = mean frame logit per
    `sample_id`.
  - Threshold-free: AUROC, AUPRC. Calibration: ECE (15 bins), Brier,
    NLL. At 0.5: accuracy, balanced accuracy.
  - Per-manipulation metrics vs all originals.
- **`train.py`**: `DistillConfig` (YAML, unknown keys rejected) and
  `StudentTrainer`.
  - Normalisation on the GPU; fp16 AMP; AdamW + warm-up-cosine; grad
    clip.
  - Early stopping on val frame AUROC (tiebreak NLL).
  - Writes `best.pt` and `last.pt` (`weights_only`-loadable), plus
    `epochs.jsonl`, `val_best_logits.npy` and `summary.json`.
  - Epoch-level resume; stale config or provenance is refused.
- **Scripts:** `scripts/train_distill_student.py` (`train` / `pilot`)
  and `scripts/compare_students.py` (val metrics, paired bootstrap,
  size/latency/VRAM). Config: `configs/distill/mobilenetv4_student.yaml`.
  Runs are written to `CONFIGUARD_CHECKPOINT_DIR\distill\`.

## Calibration and the "uncertain" output (Phase 6c, `src/configuard/calibration/`)

- **`partitions.py`**: deterministic 80/10/10 split of official TRAIN
  families (`final_train`, `temp_cal`, `conformal_cal`).
  - Union-find joins each family with its donor family; components are
    ordered by SHA-256(seed:key).
  - Written once to `<cache>\calibration_splits\<crop_tag>\partitions_seed42.json`.
    A different file at the same path, or a different train manifest,
    is refused.
  - `DistillConfig.train_partition` trains on one partition and pins
    the file's SHA-256 into the checkpoint provenance.
- **`core.py`**:
  - `fit_temperature` (NLL, log-T grid + golden section);
  - `fit_conformal` (mondrian/marginal split conformal),
    `prediction_sets`, `verdicts_from_sets`;
  - metrics: ECE/NLL/Brier, coverage, abstention, selective accuracy,
    risk-coverage/AURC, and a confidence-abstention baseline.
- **`artifact.py`**: `build_artifact` / `save_artifact` /
  `load_calibration` with checkpoint SHA-256, config/provenance and
  content-hash checks (`CalibrationMismatchError`).
  `Calibrator.predict(logits, level, alpha, mode)` returns calibrated
  P(fake) plus `Verdict`s.
  - `level="frame"` is for single images/frames.
  - `level="video"` takes the mean frame logit.
- **`configuard.distill.infer`**: `load_student` / `predict_rows`
  (checkpoint → logits for crop rows).
- **`scripts/calibrate_student.py`**: `split` builds the partitions;
  `fit` computes logits (cached, keyed by checkpoint hash), fits per
  level, writes `calibration.json` and `calibration_report.json`, and
  refuses models not trained on `final_train`.

## Adaptive 4/8/16-frame video inference (Phase 6d, `src/configuard/adaptive/`)

- **`policy.py`**:
  - `StagePolicy`: stages (4, 8, 16), per-stage α spending (default
    0.015/0.015/0.02), mondrian mode.
  - `stage_slots(k)`: the nested slot sets.
  - `decide_stage`: temperature + conformal set for one stage.
  - `min_supported_alpha(n) = 1/(n+1)`.
- **`analyzer.py`**:
  - `AdaptiveVideoAnalyzer.analyze(scorer, available=None)` escalates
    4 → 8 → 16 and asks the scorer only for slots not yet scored.
  - It returns an `AdaptiveResult`: verdict, `stopping_reason`
    (`confident_singleton_k4|k8`, `final_k16_singleton`,
    `final_k16_uncertain_both|empty`, `insufficient_frames_for_kN`),
    `frames_used`, `p_fake`, `confidence`, per-stage decisions, the
    evidence `timeline` (slot, position, logit, raw P(fake), stage
    added) and latency.
  - `fixed(scorer, k, alpha)` is the non-adaptive baseline.
  - Scorers: `ArrayScorer` (precomputed logits) and `StudentCropScorer`
    (reads and decodes crops, then runs the student per stage batch).
- **Calibration:** `adaptive_calibration.json` (levels
  `video_k4/k8/k16`, policy in the content hash), loaded with
  `load_calibration` and the same checkpoint/config/hash checks as
  Phase 6c.
- **`scripts/adaptive_video_eval.py`**: fits the per-stage calibration,
  simulates fixed vs adaptive on dev, runs the live GPU/CPU latency and
  agreement checks, and writes `adaptive_report.json`.

## Robust training and stress suite (Phase 6e, `src/configuard/robust/`)

- **`degrade.py`**: primitives (`jpeg`, `h264_style`, `resize_down_up`,
  `gaussian_blur`, `add_noise`, `gamma`) and `RobustAugmentConfig` +
  `robust_degrade(img, cfg, rng, epoch)`.
  - The function is label-free; severity is capped by a curriculum.
  - Enabled by `DistillConfig.robust_augment` (omitted when empty, so
    older configs are unchanged) and applied in `CropDataset` after the
    base augmentation, only for training items (epoch ≥ 0).
- **`stress.py`**: `StressSuite` holds 17 deterministic conditions
  (`CONDITIONS`) over official val crops.
  - Real libx264 for H.264; atomic, resumable, tag-versioned;
    `StaleStressSuiteError`.
  - Accepts val rows only (`ProtectedSplitError` otherwise).
  - `condition_rows(name)` returns rows with absolute paths.
- **`scoring.py`**: `score_conditions` scores conditions sequentially
  through one loader, saves each condition's logits when complete,
  resumes from saved files, and stops on `LowMemoryError`.
- **`configuard.memory_guard`**: `available_ram_gb` (Windows
  `GlobalMemoryStatusEx`, Linux `/proc/meminfo`) and `RamGuard`.
- **`configuard.distill.data`**: `single_thread_workers()` /
  `worker_init` keep DataLoader workers single-threaded.
- **`scripts/robust_eval.py`**: `build` (suite, free-space floor) and
  `evaluate` (per-condition metrics, decision rule, calibration
  invalidation check).

## Temporal head experiment (Phase 7, `src/configuard/temporal/`; not in the production path)

- **`embeddings.py`**: `extract` gives frozen-student features + frame
  logits in row order and refuses test rows.
  - `EmbeddingCache` (`<cache>\temporal_embeddings\<ckpt sha12>\<set>.npz`
    + JSON sidecar of checkpoint SHA-256, row-SHA list hash and feature
    dim) raises `StaleEmbeddingCacheError` on mismatch.
  - `to_videos` builds a `VideoSet` (V, 16, D) in slot order and checks
    `nested_levels`.
  - `VideoSet.stage(k)` returns the nested 4/8/16 views.
- **`gru.py`**: `ResidualGRUHead` (Linear D→64 → GELU → 1-layer GRU
  128 → zero-init Linear, added to the mean frame logit),
  `GRUTrainConfig`, `train_gru` (nested-k batches, class-weighted BCE,
  early stopping) and `predict`.
- **`configuard.memory_guard.process_rss_mb`**: process working set
  for RAM reporting.
- **`scripts/temporal_gru.py`**: `extract` / `train` / `evaluate`,
  with the pre-registered selection rule in `RULE`.
- **Decision:** rejected. Production video scoring stays as the mean
  frame logit (Phase 4/6d).

## ONNX export and deployment package (Phase 8, `src/configuard/export/`)

- **`onnx_student.py`**:
  - `StudentGraph` (normalisation inside the graph) and `export`
    (FP32; FP16 traced on CUDA with fp32 I/O).
  - `CropReader` (train-only INT8 calibration data) and
    `quantize_int8` (static QDQ; MinMax or Percentile).
  - `session` (CPU or CUDA provider), `run`, and `ShapePinnedRunner`
    (one CUDA session per batch size).
- **`package.py`**: `build_package` / `load_package` / `variant_path`,
  using `export_manifest.json` with per-file SHA-256s, the checkpoint,
  config and calibration hashes, and the CPU/GPU default selection.
  Mismatches raise `ExportMismatchError`.
- **`scripts/export_student_onnx.py`**: `build`, `parity`, `evaluate`,
  `bench` (isolated subprocess per runtime) and `package`.
- **Package directory:** `CONFIGUARD_CHECKPOINT_DIR\export\student_p80\`
  holds the ONNX files, copies of `calibration.json` /
  `adaptive_calibration.json`, the reports and the manifest.
- **Production inference contract:** frame logit from ONNX (CPU and
  GPU default: FP32), then the 6c/6d calibration on the mean frame
  logit over the nested 4/8/16 slots. `configuard.distill` +
  `best.pt` stay the PyTorch reference.

## Media-quality safety gate (Phase 9, `src/configuard/quality/`)

- **`signals.py`**: `crop_signals(img)` returns sharpness, effective
  resolution and blockiness. Never a model input.
- **`gate.py`**:
  - `GateThresholds`; `apply_gate(result, quality_by_slot, face_px,
    thresholds, calibrator, policy)` returns a `GatedResult` (verdict,
    base verdict, reason codes, failing fractions). Downgrade-only.
  - `QualityAwareScorer` decodes each requested crop once, computes
    quality and runs the detector runner.
  - `GatedVideoAnalyzer(calibrator, policy, thresholds, enabled=True)`
    combines adaptive 4/8/16 analysis and the gate in one call.
  - `save_thresholds` / `load_thresholds` handle the hash-bound
    artifact (`QualityGateMismatchError`).
- **`scripts/quality_gate.py`**: `compute` / `fit` / `evaluate`.
  Artifact at
  `CONFIGUARD_CHECKPOINT_DIR\quality_gate\p80\quality_gate.json`.
- **Production video path:** crops → `QualityAwareScorer`
  (`ShapePinnedRunner`, ONNX FP32) → `AdaptiveVideoAnalyzer` (6d
  calibration) → quality gate → verdict + reason codes.

## Inference service (Phase 10)

```
POST /v1/analyze (multipart 'file')
  -> request ID + JSON log context -> API key (if enabled) -> engine ready? (503)
  -> admission: running < max_concurrent_inference + max_queue (else 503 server_busy)
  -> uploads.receive_upload: streamed into <temp>/req-*/ with byte cap (413), upload timeout (408)
  -> validation.validate_media_file: magic bytes, extension cross-check, size, ffprobe duration
  -> worker pool (max_concurrent_inference threads), CancelToken deadline (504):
       image: decode (pixel cap) -> YuNet -> align (5d) -> ONNX FP32 -> 6c 'frame' calibration -> v1 checks + SMALL_FACE
       video: ffprobe packets -> 5d planned 16 indices -> sequential decode -> YuNet -> primary track -> recovery
              -> align -> AdaptiveVideoAnalyzer (6d) scoring only requested slots -> apply_gate (Phase 9 v1)
  -> temp dir removed when the worker finishes; JSON result with timings
GET /health/live   -> always 200 while the process serves
GET /health/ready  -> 200 only if verify_bundle passes (re-checked every ready_recheck_s) and sessions loaded
```

- **`service/config.py`:** `ServiceConfig` from the `service:` YAML block
  plus env paths and keys; production fails closed.
- **`service/artifacts.py`:** `verify_bundle` / `ModelBundle`
  (torch-free).
- **`service/engine.py`:** `OnnxRunner` (CPU shared session, CUDA one
  per batch size, fallback) and `InferenceEngine` (per-thread YuNet).
- **`service/extract.py`, `uploads.py`, `app.py`, `schemas.py`,
  `logs.py`:** extraction, streamed uploads, routes, response models and
  structured logs.

### Website routes and hero (12c)

```
src/App.tsx         route switch: "/" LandingPage (entry chunk) | "/detect" DetectPage (lazy) | "/about" AboutPage (lazy) | 404
src/router.tsx      History-API router: Link, navigate, useLocation, scroll restore, hash targets, H1 focus per [data-page]
src/brand/          primitives (Brackets, TechLabel, Wordmark, CtaLink, Reveal, Contours, SectionHeading), SiteNav, SiteFooter
src/landing/        Hero, Problem (scroll-linked checker dissolve), Stages (sticky panels), Sections (Performance, Trust,
                    Privacy, FinalCta)
src/hero/           HeroVisual (poster -> lazy scene; one WebGL context; IO pause; frame guard; fallbacks), heroScene.ts
                    (procedural head, clipping-plane scan), HeroFallback (static SVG)
src/lib/raf.ts      the single shared requestAnimationFrame loop (runs only with subscribers and a visible tab)
src/facts.ts        the only model statistics shown on the site, with dataset/split labels
backend             app.py: /assets (immutable), /fonts (1 week), SPA fallback registered after all API routes
```

### React frontend (12b)

```
frontend/ (Vite)
  src/api/client.ts      same-origin fetch (/v1/limits) + XHR upload (progress, abort, timeout, response shape check)
  src/App.tsx            state machine idle -> uploading -> analyzing -> done | error; API key in React state only
  src/components/        UploadPanel, ProgressView/EmptyState/ErrorView, ResultView, Charts (SVG), EvidenceGrid,
                         CredentialsPanel, ErrorBoundary, ui primitives
  dist/                  build output (not committed) -> served by FastAPI: "/" index.html (no-store), /assets/* (immutable)
```

- **Rendering:** React text rendering only; server images are accepted
  only as validated base64 data URIs.
- **Assets:** no external origins; fonts are self-hosted.
- **Serving:** the backend selects React when `ServiceConfig.ui_dist_dir`
  contains `index.html` (default `frontend/dist`, override
  `CONFIGUARD_UI_DIST`), otherwise the Phase 11 static UI.

### Content Credentials (Phase 12)

```
engine.analyze: ML result final -> (explanation) -> C2paVerifier.verify(upload path, ext, remaining deadline)
  verifier: ext/size gate -> idle worker (spawned lazily, warmed at startup) -> JSON line {path, mime}
  worker (real python -I, Job Object / RLIMIT_AS cap, sockets disabled, proxies blackholed):
      c2pa.Reader(context: trust_anchors = pinned C2PA-TRUST-LIST.pem, remote_manifest_fetch=false, ocsp_fetch=false)
      -> validation_state + sanitised summary (summary.py) -> JSON line
  timeout -> worker killed and respawned -> ERROR; crash / memory cap -> ERROR
response.provenance = {status, reason, summary, notice, trust_list, sdk, elapsed_ms}   (ML fields untouched)
```

- **`provenance/trust.py`:** the pin plus the SHA-256-verified loader.
- **`provenance/worker.py`, `provenance/verifier.py`,
  `provenance/summary.py`:** the sandboxed SDK worker, the pool and
  limits, and the safe summary.
- **`service/static`:** the separate Content Credentials card.

### Evidence hints and UI (Phase 11)

- **`service/explain.py`:** `CamExplainer`. It reuses the verified ONNX
  with the GAP input added as an output, in memory. `cell_maps` gives
  the signed 7×7 Grad-CAM (exact decomposition), `faithfulness` runs
  the top-10 vs random-10 cell occlusion, and `explain` builds the
  `explanation` payload (crop + overlay JPEGs, cells, check result).
- **Engine flow:** `InferenceEngine.analyze(..., explain)` produces the
  final result first. Only then does it explain ≤ 4 strongest-evidence
  crops (video) or the single crop (image). The `explanation_ms` timing
  is separate.
- **`service/static/`:** `index.html`, `app.css`, `app.js`, served at
  `/` and `/static/*`.
- **Middleware:** applies CSP + security headers to every response and
  `no-store` to non-static ones. `/v1/limits` feeds the UI.

## Quality gate hybrid experiment (Phase 9c; rejected, not in production)

- **`quality/signals_hybrid.py`**: `crop_signals_hybrid` reuses
  `signals_v2` sharpness / noise σ / blockiness and `signals.hf_ratio`
  (FFT), each with its own ROI. Same (4,) layout as v2.
- **`quality/gate.py`**: `GateThresholdsHybrid`, schema
  `p9c-quality-gate-hybrid-1`; `load_thresholds` maps all three
  schemas to their classes.
- **`scripts/quality_gate_hybrid.py`**: `assemble`, `challenge-split`,
  `verify`, `bench`, `confirm-val` (once; marker
  `CONFIRMATORY_VAL_RUN_9C.json`) and `decide`.
- **Production:** unchanged, Phase 9 v1 `quality_gate.json`.

## Quality gate v2 experiment (Phase 9b; rejected, not in production)

- **`quality/signals_v2.py`**: `crop_signals_v2` returns noise-corrected
  sharpness, noise-corrected effective resolution (spatial filters),
  offset-robust blockiness and Immerkær noise σ (about 0.8 ms per
  crop).
- **`quality/gate.py`**: `GateThresholdsV2` (adds HIGH_NOISE) with a
  `signal_fn` hook, plus schema `p9b-quality-gate-2`.
  - `QualityAwareScorer` / `GatedVideoAnalyzer` take the signal
    function from the thresholds, so v1 and v2 share the same
    downgrade-only gate.
  - `load_thresholds` returns the matching class.
- **`scripts/quality_gate_v2.py`**: `quantiles`, `cases`, `fit`,
  `verify`, `bench` and `confirm-val` (once; blocked by a marker).
- **Production:** `GatedVideoAnalyzer` with
  `load_thresholds(.../quality_gate.json)` (Phase 9 v1).

## Repository layout

```
ConfiGuard-Lite/
├── src/configuard/
│   ├── __init__.py
│   ├── config.py          Environment-aware config loading (YAML -> dataclasses)
│   ├── env_check.py         Python/Git/FFmpeg/CUDA/GPU device detection
│   ├── io_types.py           Typed pipeline data contracts (Phase 1)
│   ├── validation.py         Secure file-type/size/duration validation (Phase 1)
│   ├── pipeline.py           End-to-end vertical slice orchestration (Phase 1)
│   ├── env_loader.py          Tiny dependency-free .env parser (Phase 4)
│   ├── media/                Face/media preprocessing (Phase 2)
│   │   ├── types.py, decode.py, sampling.py, face_detector.py,
│   │   │   alignment.py, tracking.py, cache.py, hashing.py, preprocess.py
│   ├── datasets/              Dataset registry (Phase 3)
│   │   ├── schema.py, registry.py, manifest.py, splitting.py,
│   │   │   duplicates.py, storage.py, faceforensics.py (Phase 5b validation),
│   │   │   faceforensics_splits.py (Phase 5c official split)
│   │   └── adapters/
│   │       ├── __init__.py (DatasetAdapter protocol, DatasetAccessError)
│   │       ├── folder_convention.py, metadata_sidecar.py (the two engines)
│   │       └── known_datasets.py (FF++/Celeb-DF-v2/DFDC/DF40/DeeperForensics-1.0 + generic)
│   ├── models/                 Pretrained visual encoders (Phase 4)
│   │   ├── encoder.py (DeepfakeVisualEncoder, EncoderSpec, PreprocessConfig)
│   │   ├── registry.py (the two authorized models), preprocess.py,
│   │   │   inference.py, real_pipeline.py, onnx_export.py, benchmark.py,
│   │   │   device.py
│   ├── dependency_safety.py     torch/torchvision/CUDA regression guard (Phase 5)
│   ├── storage_guard.py         free-space floor guard (Phase 5d)
│   ├── crops/                   Matched FF++ face-crop extraction (Phase 5d)
│   │   ├── families.py, matching.py, alignment.py, store.py, extract.py,
│   │   │   manifests.py, audit_stats.py
│   ├── teacher/                 Frozen GenD CLIP-L/14 teacher + logit cache (Phase 6a)
│   │   ├── gend.py, cache.py
│   ├── distill/                 Student training + logit distillation on crops (Phase 6b)
│   │   ├── data.py, augment.py, losses.py, evaluate.py, train.py, infer.py
│   ├── calibration/             Partitions, temperature, conformal, artifacts (Phase 6c)
│   │   ├── partitions.py, core.py, artifact.py
│   ├── adaptive/                Adaptive 4/8/16-frame video inference (Phase 6d)
│   │   ├── policy.py, analyzer.py
│   ├── robust/                  Robust augmentation, stress suite, safe scoring (Phase 6e)
│   │   ├── degrade.py, stress.py, scoring.py
│   ├── temporal/                Frame-embedding cache + GRU head experiment (Phase 7; rejected)
│   │   ├── embeddings.py, gru.py
│   ├── export/                  Production ONNX export + hash-checked package (Phase 8)
│   │   ├── onnx_student.py, package.py
│   ├── quality/                 Downgrade-only media-quality safety gate (Phase 9; v2 experiment 9b; hybrid 9c)
│   ├── service/                 FastAPI inference service (Phase 10)
│   ├── provenance/              C2PA Content Credentials verification (Phase 12)
│   │   ├── signals.py, signals_v2.py, gate.py
│   ├── memory_guard.py          available-RAM floor (Phase 6e)
│   └── training/                Reproducible training pipeline (Phase 5)
│       ├── config.py, paths.py, splits.py, datasets.py, sampling.py,
│       │   dataloader.py, optim.py, metrics.py, checkpoint.py,
│       │   logging_utils.py, trainer.py, runner.py, synthetic.py
├── scripts/                 Operational scripts (env verification, preprocessing
│                              benchmark, storage check, baseline model download,
│                              model benchmark, ONNX export, train, evaluate,
│                              FF++ c23 download wrapper + validation, official split,
│                              matched face-crop extraction + shortcut audit,
│                              GenD teacher download + teacher-logit caching,
│                              student training/pilot + student comparison,
│                              calibration split/fit, adaptive video evaluation,
│                              robust stress suite + evaluation,
│                              temporal GRU experiment, production ONNX export,
│                              quality gate compute/fit/evaluate,
│                              quality gate v2 and hybrid experiments)
├── tests/                    pytest suite (unit + integration), tests/conftest.py +
│                              tests/media/conftest.py + tests/datasets/conftest.py +
│                              tests/models/conftest.py generate all fixtures at
│                              test time (hand-built PNG, ffmpeg lavfi clips,
│                              synthetic dataset trees/manifests) - nothing checked in
├── configs/                  base.yaml + development/training/testing/production.yaml
│                              + train/{mobilenetv4_conv_small,efficientnet_b0}.yaml (Phase 5)
├── constraints-cuda.txt       pinned torch/torchvision CUDA pair (Phase 5)
├── models/                    (gitignored) external model assets, e.g. YuNet ONNX -
│                              see docs/DATASETS.md for provenance
├── docs/                      Living project documentation
├── data/ checkpoints/ cache/ outputs/   (all gitignored)
└── .venv/                    (gitignored) Python 3.11 virtual environment
```
