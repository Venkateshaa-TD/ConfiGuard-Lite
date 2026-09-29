# Changelog

All notable changes to this project are documented here.

## [Unreleased]

### Phase 0 — Environment and repository foundation (2026-09-29)

- Initialized Git repository and project scaffolding.
- Created `src/configuard` package with `config.py` (YAML config loading)
  and `env_check.py` (Python/Git/FFmpeg/CUDA/GPU device detection).
- Added `scripts/verify_environment.py` environment verification script.
- Added Phase 0 test suite (`tests/test_imports.py`, `tests/test_device.py`,
  `tests/test_config.py`).
- Added project documentation set (README, CLAUDE.md, docs/*).
- Added `.gitignore`, `.env.example`, `pyproject.toml`,
  `requirements.txt`, `requirements-dev.txt`.
- Created Python 3.11 virtual environment (`.venv`) with PyTorch (CUDA 12.1
  build), NumPy, PyYAML, pytest.
