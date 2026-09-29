# CLAUDE.md — Operating rules for ConfiGuard-Lite

This file governs how Claude (or any agent) works in this repository. It is
the source of truth for process; project facts live in `docs/`.

## Project

ConfiGuard-Lite: efficient, uncertainty-aware deepfake image and video
detection, designed to train within a 6 GB VRAM laptop GPU budget and deploy
on CPU-only servers with optional GPU acceleration.

## Hardware target

- Windows laptop, AMD Ryzen 7000-series CPU, NVIDIA RTX 4050 Laptop GPU (6 GB VRAM)
- Training and development must fit this machine. See `docs/PROJECT_PLAN.md`
  for the full requirements list.

## Working rules

- Work on only the requested phase. Stop and wait for approval after each one.
- Before editing, inspect the existing repository and current Git status.
- Preserve existing files and user changes. Never delete or overwrite
  unrelated work.
- Do not download large datasets or model checkpoints without asking first.
- Do not install system-level software (drivers, CUDA toolkit, FFmpeg, etc.)
  without asking first. Installing Python packages into the project's `.venv`
  to satisfy an explicitly requested phase is in scope.
- Prefer configuration files and reusable Python modules over notebooks.
  Notebooks are for exploration only, never the source of truth.
- Use type hints, clear interfaces, deterministic seeds, and structured
  logging.
- Keep dataset paths, secrets, and machine-specific paths out of source code
  (use `.env`, never commit it — see `.env.example`).
- Never commit datasets, checkpoints, uploaded media, secrets, or generated
  caches (see `.gitignore`).
- Build Windows-compatible code first; keep future Linux/Docker deployment
  compatibility in mind.
- Optimize for 6 GB GPU memory: mixed precision, gradient accumulation, and
  cached face crops/features.
- Do not claim a test passed unless it was actually executed. Report exact
  commands and results.
- If a dependency or hardware requirement is missing, report it clearly
  rather than silently working around it.

## Architecture constraints (do not violate without a recorded decision)

- One shared lightweight visual encoder for images and video frames.
- MobileNetV4-Conv-Small is the preferred student model.
- EfficientNet-B0 is the comparison baseline.
- GenD CLIP-L/14 is used only as a **frozen, offline teacher** for
  distillation signal. It is never fine-tuned/trained on this laptop.
- Temporal video reasoning uses a small GRU head over frame embeddings.
- Video analysis adaptively samples 4, 8, or 16 frames.
- Output is one of three classes: "likely real", "likely manipulated",
  "uncertain" — driven by confidence calibration + conformal prediction.
- Training includes compression-robust augmentation (re-encoding, JPEG/H.264
  artifacts, resolution downscaling).
- Models are exported to ONNX in FP32, FP16, and INT8.
- Inference must run on CPU-only servers, with optional NVIDIA GPU
  acceleration.
- Image explanations use heatmaps; video explanations use an evidence
  timeline; C2PA provenance is reported as a separate, distinct signal (never
  merged into the model's manipulation score).

## Required documentation (create and keep current every phase)

`README.md`, `CLAUDE.md`, `docs/PROJECT_PLAN.md`, `docs/ARCHITECTURE.md`,
`docs/DECISIONS.md`, `docs/EXPERIMENT_LOG.md`, `docs/PHASE_STATUS.md`,
`docs/KNOWN_ISSUES.md`, `docs/DATASETS.md`, `docs/MODEL_CARD.md`,
`CHANGELOG.md`, `.env.example`, `.gitignore`.

## End-of-phase checklist

1. Update `docs/PHASE_STATUS.md`.
2. Add architectural decisions to `docs/DECISIONS.md`.
3. Record commands, configs, and results in `docs/EXPERIMENT_LOG.md`.
4. Update `README.md` if setup or usage changed.
5. Run the relevant tests.
6. Show the exact tests executed and their results.
7. List files created or changed.
8. List unresolved risks or blockers.
9. Recommend the next phase.
10. Stop and wait for approval.

## Completion format

```
PHASE RESULT
- Phase:
- Status: PASS / PARTIAL / BLOCKED
- Implemented:
- Verification performed:
- Test results:
- Performance measurements:
- Files changed:
- Documentation updated:
- Known issues:
- Next recommended phase:
```
