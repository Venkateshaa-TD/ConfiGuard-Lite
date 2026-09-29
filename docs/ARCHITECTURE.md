# Architecture

Status: Phase 1 complete. The pipeline shape below is real and tested; the
preprocessing, model, and provenance stages are deliberate placeholders
(deterministic dummy logic, not trained models) - see "Phase 1 scope" below.

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

## Repository layout

```
ConfiGuard-Lite/
├── src/configuard/
│   ├── __init__.py
│   ├── config.py          Environment-aware config loading (YAML -> dataclasses)
│   ├── env_check.py         Python/Git/FFmpeg/CUDA/GPU device detection
│   ├── io_types.py           Typed pipeline data contracts (Phase 1)
│   ├── validation.py         Secure file-type/size/duration validation (Phase 1)
│   └── pipeline.py           End-to-end vertical slice orchestration (Phase 1)
├── scripts/                 Operational scripts (env verification)
├── tests/                    pytest suite (unit + integration), tests/conftest.py
│                              generates all fixtures at test time - nothing checked in
├── configs/                  base.yaml + development/training/testing/production.yaml
├── docs/                      Living project documentation
├── data/ checkpoints/ cache/ outputs/   (all gitignored)
└── .venv/                    (gitignored) Python 3.11 virtual environment
```
