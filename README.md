# ConfiGuard-Lite

Efficient and uncertainty-aware deepfake image and video detection, designed
to train within a 6 GB VRAM laptop GPU budget and deploy on CPU-only
servers with optional GPU acceleration.

See `docs/PROJECT_PLAN.md` for the full requirements and phase roadmap,
`docs/ARCHITECTURE.md` for the system design, and `CLAUDE.md` for the
operating rules this project is developed under.

## Status

Phase 0 (environment and repository foundation) — see
`docs/PHASE_STATUS.md` for current status.

## Requirements

- Windows 10/11 (Linux/Docker compatibility is a design goal, not yet built)
- Python 3.11 (3.14 is not yet supported by the ML dependency stack — see
  `docs/DECISIONS.md`)
- Git
- NVIDIA GPU + recent driver, optional (CPU-only inference is supported)
- FFmpeg, required starting the video-pipeline phase (not installed by
  default — see `docs/KNOWN_ISSUES.md`)

## Setup

```powershell
# 1. Create the virtual environment (Python 3.11)
py -3.11 -m venv .venv

# 2. Install PyTorch matching your hardware
#    GPU (NVIDIA, CUDA 12.1-compatible driver):
.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu121
#    CPU-only:
.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu

# 3. Install the remaining dependencies
.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
```

## Verify your environment

```powershell
.venv\Scripts\python.exe scripts\verify_environment.py
```

This reports Python version, Git/FFmpeg availability, and PyTorch/CUDA/GPU
detection without exposing any usernames, tokens, or secrets.

## Run tests

```powershell
.venv\Scripts\python.exe -m pytest -v
```

## Project layout

See `docs/ARCHITECTURE.md` for the full repository layout and design.

## Configuration

Non-secret defaults live in `configs/*.yaml`. Machine-specific paths and
secrets go in a local `.env` file (copy `.env.example` to `.env` — `.env` is
never committed).
