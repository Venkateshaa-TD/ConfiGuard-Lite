# Architecture

Status: Phase 3 complete. The Phase 1 pipeline shape below is real and
tested; the preprocessing, model, and provenance stages it uses are still
deliberate placeholders (see "Phase 1 scope" below). Phase 2 added a real,
independently-tested face/media preprocessing subsystem
(`configuard.media`). Phase 3 adds a real, independently-tested dataset
registry (`configuard.datasets`, documented in its own section below).
Neither is yet wired into `configuard.pipeline.run_pipeline` or into any
training loop - that happens once a real encoder/model exists.

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
`CONFIGUARD_CHECKPOINT_DIR`. `assert_safe_storage_path` raises
`UnsafeStoragePathError` instead of just warning when a path is inside
the repo (docs/PROJECT_PLAN.md requirement 11's "must warn or refuse").
See `scripts/check_storage.py` for the CLI and docs/EXPERIMENT_LOG.md for
real output, including the refusal case.

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
│   ├── media/                Face/media preprocessing (Phase 2)
│   │   ├── types.py, decode.py, sampling.py, face_detector.py,
│   │   │   alignment.py, tracking.py, cache.py, hashing.py, preprocess.py
│   └── datasets/              Dataset registry (Phase 3)
│       ├── schema.py, registry.py, manifest.py, splitting.py,
│       │   duplicates.py, storage.py
│       └── adapters/
│           ├── __init__.py (DatasetAdapter protocol, DatasetAccessError)
│           ├── folder_convention.py, metadata_sidecar.py (the two engines)
│           └── known_datasets.py (FF++/Celeb-DF-v2/DFDC/DF40/DeeperForensics-1.0 + generic)
├── scripts/                 Operational scripts (env verification, preprocessing
│                              benchmark, storage check)
├── tests/                    pytest suite (unit + integration), tests/conftest.py +
│                              tests/media/conftest.py + tests/datasets/conftest.py
│                              generate all fixtures at test time (hand-built PNG,
│                              ffmpeg lavfi clips, synthetic dataset trees/manifests)
│                              - nothing checked in
├── configs/                  base.yaml + development/training/testing/production.yaml
├── models/                    (gitignored) external model assets, e.g. YuNet ONNX -
│                              see docs/DATASETS.md for provenance
├── docs/                      Living project documentation
├── data/ checkpoints/ cache/ outputs/   (all gitignored)
└── .venv/                    (gitignored) Python 3.11 virtual environment
```
