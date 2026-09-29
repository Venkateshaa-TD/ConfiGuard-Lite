# Known Issues / Blockers

Format: one entry per issue. Mark resolved issues rather than deleting them.

---

## OPEN — FFmpeg not installed

**Detected:** Phase 0 environment scan (2026-09-29).

**Impact:** Video frame extraction (needed starting Phase 1/5) will not work
without FFmpeg on PATH. Not required for Phase 0.

**Action needed:** User should confirm before we install FFmpeg (e.g. via
`winget install Gyan.FFmpeg` or a static build), since installing system
software requires prior approval per project rules.

---

## OPEN — Limited free disk space

**Detected:** Phase 0 environment scan (2026-09-29).

**Impact:** `C:` drive had ~20 GB free out of 200 GB at time of scan. Face
manipulation / deepfake datasets and multiple model checkpoints can easily
exceed this. Will block Phase 1 (data pipeline) if not addressed.

**Action needed:** Confirm target dataset sizes with the user before
downloading, and consider an external/secondary drive path (configurable via
`.env`, e.g. `CONFIGUARD_DATA_DIR`) before Phase 1 begins.

---

## OPEN — No system CUDA Toolkit (`nvcc`) installed

**Detected:** Phase 0 environment scan (2026-09-29).

**Impact:** None currently — PyTorch's CUDA wheel (cu121) bundles its own
CUDA runtime and does not require a system Toolkit for training/inference.
Would only matter if a future phase needs to compile custom CUDA kernels.

**Action needed:** None for now. Revisit only if a custom-kernel dependency
is introduced.
