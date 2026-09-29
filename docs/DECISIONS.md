# Architectural Decisions

Format: one entry per decision, newest first.

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
