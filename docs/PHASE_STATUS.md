# Phase Status

| Phase | Title | Status | Date |
|---|---|---|---|
| 0 | Environment and repository foundation | PASS | 2026-09-29 |
| 1 | Architecture contracts and minimal vertical slice | PASS | 2026-09-29 |

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
