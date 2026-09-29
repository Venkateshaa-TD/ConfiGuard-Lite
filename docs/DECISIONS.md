# Architectural Decisions

Format: one entry per decision, newest first.

---

## 2026-09-29 — Nested 4/8/16 frame sampling is derived, not independently sampled

**Context:** Requirement: the 4-frame selection must be a subset of the
8-frame selection, which must be a subset of the 16-frame selection.

**Decision:** `compute_nested_sampling_plans` computes one uniform
16-frame sample of the video, then derives the 8-frame set as every
second element of that 16-set, and the 4-frame set as every second
element of the 8-set (`indices_16[::2][::2]`).

**Why:** An independently-computed uniform 8-frame sample of a video is
generally *not* a subset of an independently-computed uniform 16-frame
sample - the nesting requirement can only be satisfied by construction.
This also means frames already decoded/analyzed at a smaller frame count
are automatically reusable if a later stage escalates to a larger one
(e.g. an "uncertain" 4-frame result triggering an 8-frame re-analysis),
without any extra bookkeeping.

---

## 2026-09-29 — Video frame decoding is sequential, not seek-based

**Context:** Need to decode a sparse set of target frame indices from a
video reliably.

**Decision:** `decode_sampled_frames` reads frames sequentially from the
start and keeps the ones at wanted indices, rather than seeking via
`cv2.CAP_PROP_POS_FRAMES`.

**Why:** `CAP_PROP_POS_FRAMES` seeking is well-documented as unreliable
across codecs/containers with irregular keyframe intervals (can land on
the wrong frame silently). Sequential decode is slower per frame reached
but correct, and this project's validated video length limits (tens of
seconds - see `configs/*.yaml`) keep the cost acceptable for laptop-scale
preprocessing.

---

## 2026-09-29 — Video metadata prefers ffprobe over OpenCV VideoCapture properties

**Context:** Requirement: reliable video metadata extraction, including
variable-frame-rate video and rotation.

**Decision:** `extract_video_metadata` tries `ffprobe` first (comparing
`r_frame_rate` vs. `avg_frame_rate` to flag VFR, reading `nb_frames` or
estimating from duration, and reading rotation from the legacy `rotate`
tag or Display Matrix side data), falling back to `cv2.VideoCapture`
properties only if ffprobe is unavailable or fails to parse.

**Why:** OpenCV's own `CAP_PROP_FRAME_COUNT`/`CAP_PROP_FPS` are known to
be unreliable for VFR content and don't expose container rotation at all.
ffprobe (already a project dependency since Phase 1) is more accurate for
both. Rotation extraction is best-effort: this ffmpeg build did not
attach rotation metadata to a synthetic `lavfi`-generated test clip via
either convention (verified by direct `ffprobe` inspection), so the
parsing logic is unit-tested directly against synthetic ffprobe JSON
shapes rather than an actual rotated fixture - see `docs/KNOWN_ISSUES.md`.

---

## 2026-09-29 — Face tracking's temporal-gap tolerance is stride-aware

**Context:** `configuard.media.tracking.track_faces` closes a track if a
face isn't matched for more than `max_frame_gap` *video* frames. But
`preprocess_video` only ever runs detection on the sparse sampled frames
(e.g. indices 0, 4, 8, 12, ... for a 30-frame clip), so a face present in
every sampled frame still has a real-frame gap of ~4 between detections.

**Decision:** `preprocess_video` computes the widest stride actually
present in the sampling plan's indices and uses
`max(config.max_track_frame_gap, sample_stride)` as the tracker's gap
tolerance, instead of passing the raw config value straight through.

**Why:** With a fixed small `max_frame_gap` (e.g. the config default of
2), every track would appear to "expire" between consecutive sparse
samples, fragmenting a single continuous face into many spurious
single-frame tracks. `track_faces` itself keeps `max_frame_gap` as a
literal frame-index parameter (unit-tested that way in
`tests/media/test_tracking.py`, matching a caller that tracks over dense/
consecutive frames) - the adaptation lives in the caller that knows its
own sampling density.

---

## 2026-09-29 — Cache key/version derived from a config hash, not a manual bump

**Context:** Requirement: cache keyed by input hash, frame index, track
ID, and preprocessing configuration/version.

**Decision:** `PreprocessingConfig.version_tag` is `f"v1-{sha256(repr of
every alignment/detector field)[:12]}"`, computed automatically, and used
as the cache key's `config_version` component (`FaceCropCache` /
`CacheKey`).

**Why:** A manually-maintained version string is easy to forget to bump
when a margin ratio or detector threshold changes, silently serving stale
crops. Deriving it from the config's own field values makes any change
automatically land in a new cache namespace with no extra step.

---

## 2026-09-29 — YuNet `2026may` (dynamic input shape) over `2023mar`

**Context:** opencv_zoo publishes several YuNet ONNX exports; task
instructions permitted downloading only this official asset.

**Decision:** Downloaded `face_detection_yunet_2026may.onnx` (dynamic
input-shape re-export), not `face_detection_yunet_2023mar.onnx` (fixed
input shape).

**Why:** This project installed `opencv-python-headless` 5.0.0, and
opencv_zoo's own documentation states the `2026may` export is the one
compatible with OpenCV 5.x's ONNX graph engine, while `2023mar` targets
OpenCV 4.x's DNN module. Verified working end-to-end (loads on CPU,
correctly returns no detections on a non-face image) - see
`docs/EXPERIMENT_LOG.md`. Full provenance (source URL, license, size,
SHA-256) recorded in `docs/DATASETS.md`.

---

## 2026-09-29 — `opencv-python-headless` over `opencv-python`

**Context:** Need OpenCV for image/video decoding and `FaceDetectorYN`.

**Decision:** Added `opencv-python-headless` to `requirements.txt`, not
the GUI-enabled `opencv-python`.

**Why:** Requirement 14 is "run on CPU-only servers" - the headless build
skips Qt/GUI dependencies this project never uses (`cv2.imshow`, etc.),
which matters for a lean server deployment and avoids pulling in system
GUI libraries on Linux later.

---

## 2026-09-29 — Fixed a Phase 0 `.gitignore` bug: `dir/` excludes its own `!dir/.gitkeep` exception

**Context:** While adding `models/` (for the YuNet asset) to `.gitignore`,
noticed `data/.gitkeep`, `checkpoints/.gitkeep`, `cache/.gitkeep`, and
`outputs/.gitkeep` were never actually committed in Phase 0 despite the
`!dir/.gitkeep` exception lines - only `configs/.gitkeep` was (because
`configs/` was never itself ignored).

**Decision:** Changed the ignore patterns from `dir/` to `dir/*` for
`data/`, `checkpoints/`, `models/`, `cache/`, and `outputs/`.

**Why:** Git cannot re-include a file whose *parent directory* is itself
excluded - `!dir/.gitkeep` only works if the directory match is `dir/*`
(contents excluded, directory itself not excluded), not `dir/` (the
directory itself excluded). Verified with `git check-ignore -v` on every
affected `.gitkeep` path after the fix.

---

## 2026-09-29 — Media type is decided by content sniffing, not file extension

**Context:** Phase 1 needs secure file-type validation for uploaded
images/video.

**Decision:** `configuard.validation` reads the first ~64 bytes of the file
and matches known magic-byte signatures (JPEG/PNG/WEBP/MP4/MOV/MKV/AVI) to
decide the real media type. The extension is still checked, but only to
pick the right size limit and as a consistency cross-check - a mismatch
between sniffed content and extension is its own rejection reason.

**Why:** Extensions are trivially spoofable (rename `payload.mp4` to
`image.png`); trusting them for type-based branching is an OWASP-flagged
file-upload weakness. Content sniffing is the standard mitigation and adds
no new dependency (stdlib only).

---

## 2026-09-29 — Video duration/corruption check uses `ffprobe` with a timeout, never raises

**Context:** Requirement: "invalid and corrupted files are rejected
safely."

**Decision:** `_ffprobe_duration_seconds()` runs `ffprobe` with a 10s
timeout and `check=False`; any non-zero exit, timeout, or unparsable JSON
output returns `None`, which `validate_media_file` turns into a
`VIDEO_UNREADABLE` `ValidationResult` entry - never an unhandled exception.

**Why:** A corrupted or adversarial video file must not be able to hang or
crash the service. Bounding subprocess time and treating every failure
mode as "reject, don't raise" keeps that guarantee at the validation
boundary, before any decoding is attempted.

---

## 2026-09-29 — Phase 1 preprocessing/model/provenance are explicit deterministic placeholders

**Context:** Phase 1 must prove the full pipeline shape end-to-end without
implementing real models, face detection, or C2PA (per phase scope).

**Decision:** `_dummy_predict()` derives a pseudo-score from
`sha256(filename:size)` rather than random numbers, so every run on the
same file is reproducible and testable. `_preprocess_placeholder()` only
decides frame *count* (via the real adaptive 4/8/16 rule) without decoding
any pixels. `_provenance_placeholder()` always returns `NOT_CHECKED`.

**Why:** Deterministic placeholders let the test suite assert exact
behavior (`test_pipeline_is_deterministic`) instead of tolerating
randomness, while keeping a clear, greppable seam (`_dummy_predict`,
`placeholder`) for where later phases plug in real implementations.

---

## 2026-09-29 — `ProjectConfig` gains `environment` + `validation`, stays backward compatible

**Context:** Requirement: config schemas for development/training/testing/
production.

**Decision:** Extended the existing `ProjectConfig` (rather than
introducing a parallel config type) with `environment: Literal[...]`
(default `"development"`) and `validation: ValidationLimits` (a new frozen
dataclass, default-constructed if the YAML has no `validation:` block).
Added `configs/{development,training,testing,production}.yaml`;
`configs/base.yaml` from Phase 0 is kept as a generic default and still
passes its original tests unchanged.

**Why:** Avoids a second, parallel config type while still giving each
environment (especially `testing`, with deliberately tiny limits) its own
file. Backward compatible: existing Phase 0 config-loading tests pass
without modification.

---

## 2026-09-29 — Hand-built PNG fixture instead of adding Pillow

**Context:** Tests need a real, valid tiny image to exercise the full
validate → preprocess → predict path.

**Decision:** `tests/conftest.py` constructs a minimal valid PNG (IHDR +
zlib-compressed IDAT + IEND chunks) using only `struct`/`zlib`
(stdlib), rather than adding Pillow as a test dependency. Video fixtures
use `ffmpeg`'s `lavfi testsrc` generator (already installed, no download).

**Why:** Keeps dependencies minimal per project rules; a hand-built PNG is
~15 lines of stdlib code and needs no image library at all.

---

## 2026-09-29 — Python 3.11 for the project virtual environment

**Context:** The machine's default `python`/`py` resolves to Python 3.14.6,
but a `py -3.11` interpreter (3.11.9) is also installed.

**Decision:** Use Python 3.11 for `.venv`, pinned via `pyproject.toml`
(`requires-python = ">=3.11,<3.13"`).

**Why:** PyTorch, ONNX Runtime, and common CV libraries (OpenCV, timm) do not
yet reliably support Python 3.14 at the time of this phase. 3.11 is a
mature, broadly-supported version across the full expected dependency set
(PyTorch, ONNX Runtime, timm, OpenCV, face-detection libraries).

---

## 2026-09-29 — PyTorch installed via explicit CUDA index (cu121)

**Context:** The laptop has an NVIDIA RTX 4050 Laptop GPU (6 GB VRAM,
driver 591.66) but no system CUDA Toolkit (`nvcc`) installed.

**Decision:** Install PyTorch from `https://download.pytorch.org/whl/cu121`
rather than the bare PyPI default, and do not require a system CUDA Toolkit
install. PyTorch's CUDA wheels bundle the CUDA runtime they need.

**Why:** Avoids requiring a system-level CUDA Toolkit install (which the
project rules require asking about first), while still enabling GPU
acceleration through the NVIDIA driver already present.

---

## 2026-09-29 — Config loading uses stdlib dataclasses + PyYAML, not pydantic

**Context:** Phase 0 needs a "configuration loading" mechanism and test.

**Decision:** Implement `ProjectConfig` as a plain `dataclasses.dataclass`
loaded from YAML via PyYAML, rather than adding `pydantic` as a dependency.

**Why:** Keeps Phase 0 dependencies minimal, per instructions. Revisit if a
later phase needs richer validation (nested schemas, custom validators).

---

## 2026-09-29 — src-layout package (`src/configuard`)

**Context:** Need an importable, testable Python package from the start.

**Decision:** Use a `src/` layout with package name `configuard`, added to
`pythonpath` in `pyproject.toml`'s `[tool.pytest.ini_options]` rather than
installing the package in editable mode.

**Why:** Avoids requiring an editable install step for Phase 0 while keeping
the package importable by both `scripts/` and `tests/`. Revisit if
packaging/distribution becomes a requirement.
