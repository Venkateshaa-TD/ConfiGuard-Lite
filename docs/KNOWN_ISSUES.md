# Known Issues / Blockers

Format: one entry per issue. Mark resolved issues rather than deleting them.

---

## RESOLVED — FFmpeg not installed

**Detected:** Phase 0 environment scan (2026-09-29).

**Resolved:** 2026-09-29, with explicit user approval, via
`winget install --id Gyan.FFmpeg -e --source winget`. Installed
FFmpeg 9.0.2 (full build, gyan.dev) to
`%LOCALAPPDATA%\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-9.0.2-full_build\bin`,
added to the user `PATH` environment variable by winget. Verified with
`ffmpeg -version` and `ffprobe -version` (both report version 9.0.2). See
`docs/EXPERIMENT_LOG.md` for full output.

**Note:** Shell processes already running at install time (including this
agent session's Bash/PowerShell tool shells) do not see the updated `PATH`
until restarted — a new terminal window picks it up automatically. Not an
environment defect, just a live-session caveat.

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
