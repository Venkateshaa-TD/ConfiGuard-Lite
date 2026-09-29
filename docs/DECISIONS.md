# Architectural Decisions

Format: one entry per decision, newest first.

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
