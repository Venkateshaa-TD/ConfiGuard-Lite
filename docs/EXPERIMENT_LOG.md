# Experiment Log

Format: one entry per meaningful run, newest first. This log records exact
commands and results, not narrative summaries.

---

## 2026-09-29 — Phase 0 environment scan

Commands:
```
python --version
git --version
ffmpeg -version
nvidia-smi --query-gpu=name,memory.total,driver_version,memory.free --format=csv
nvcc --version
```

Results (raw hardware/software detection, no user-identifying data):
- Python (default `python`/`py`): 3.14.6
- Python (via `py -3.11`): 3.11.9 — selected for this project's `.venv`
- Git: 2.55.0.windows.3
- FFmpeg: not found on PATH
- GPU: NVIDIA GeForce RTX 4050 Laptop GPU, 6141 MiB total VRAM, driver 591.66
- CUDA Toolkit (`nvcc`): not found (not required — see `docs/DECISIONS.md`)
- CPU: AMD Ryzen 7 7435HS, 8 cores / 16 logical processors
- RAM: 23.69 GB total
- Disk (`C:`): ~200 GB total, ~21.8 GB free at time of scan

---

## 2026-09-29 — Phase 0 dependency install

Commands:
```
py -3.11 -m venv .venv
.venv/Scripts/python.exe -m pip install --upgrade pip
.venv/Scripts/python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu121
.venv/Scripts/python.exe -m pip install -r requirements-dev.txt
```

Results:
- pip upgraded 24.0 -> 26.2.1
- torch 2.5.1+cu121 installed (with sympy, networkx, jinja2, fsspec, filelock)
- numpy 2.4.6, PyYAML 6.0.3, pytest 8.4.2 installed

---

## 2026-09-29 — Phase 0 environment verification script

Command:
```
.venv/Scripts/python.exe scripts/verify_environment.py
```

Output:
```
=== ConfiGuard-Lite Environment Report ===
Python version : 3.11.9
Platform       : Windows-10-10.0.26200-SP0
Git available  : True
FFmpeg available: False

--- Device ---
torch installed : True (version=2.5.1+cu121)
CUDA available  : True (cuda=12.1)
GPU             : NVIDIA GeForce RTX 4050 Laptop GPU
GPU memory (MB) : 6140
Selected device : cuda

[WARNING] FFmpeg not found on PATH. Required for video frame extraction in a
later phase, not needed for Phase 0.
```

---

## 2026-09-29 — Phase 0 test suite run

Command:
```
.venv/Scripts/python.exe -m pytest -v
```

Result: **12 passed in 3.50s**
```
tests/test_config.py::test_load_base_config PASSED
tests/test_config.py::test_load_config_missing_file_raises PASSED
tests/test_config.py::test_load_config_rejects_non_mapping PASSED
tests/test_config.py::test_config_from_dict_separates_extra_fields PASSED
tests/test_device.py::test_probe_device_returns_valid_report PASSED
tests/test_device.py::test_probe_device_cuda_fields_consistent PASSED
tests/test_device.py::test_probe_environment_runs PASSED
tests/test_imports.py::test_import_numpy PASSED
tests/test_imports.py::test_import_yaml PASSED
tests/test_imports.py::test_import_torch PASSED
tests/test_imports.py::test_import_configuard_package PASSED
tests/test_imports.py::test_import_configuard_submodules PASSED
```

---

## 2026-09-29 — FFmpeg install (approved by user, pre-Phase-1)

Command:
```
winget install --id Gyan.FFmpeg -e --source winget --accept-package-agreements --accept-source-agreements
```

Result: Installed FFmpeg 9.0.2 (gyan.dev full build). winget added
`%LOCALAPPDATA%\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-9.0.2-full_build\bin`
to the user `PATH`. No system CUDA Toolkit was installed (per instruction).

Verification commands:
```
ffmpeg -version
ffprobe -version
```

Output (truncated to version lines):
```
ffmpeg version 9.0.2-full_build-www.gyan.dev Copyright (c) 2000-2026 the FFmpeg developers
libavutil      61.  1.102 / 61.  1.102
libavcodec     63.  1.102 / 63.  1.102
ffprobe version 9.0.2-full_build-www.gyan.dev Copyright (c) 2007-2026 the FFmpeg developers
```

Both commands succeeded (exit code 0).

---

## 2026-09-29 — Phase 1 test suite run

Command:
```
.venv/Scripts/python.exe -m pytest -v
```
(run with FFmpeg's bin directory added to PATH for this shell — see
"session PATH caveat" note in `docs/KNOWN_ISSUES.md`)

Result: **48 passed in 4.46s** (12 from Phase 0 + 36 new: `test_io_types.py`,
`test_validation.py`, `test_pipeline.py`, `test_config_environments.py`).
Full pass/fail list recorded; no skips, no failures.

---

## 2026-09-29 — Phase 1 manual smoke test (acceptance criteria)

Generated a tiny real image and tiny real video with `ffmpeg` (no
downloads), then ran `configuard.pipeline.run_pipeline()` directly:

```
ffmpeg -y -f lavfi -i "color=c=blue:size=32x32" -frames:v 1 smoke.png
ffmpeg -y -f lavfi -i "testsrc=duration=2:size=64x64:rate=5" -pix_fmt yuv420p smoke.mp4
```

`smoke.png` (134 bytes) -> valid `DetectionResult`:
```json
{
  "media_type": "image",
  "validated": {"file_size_bytes": 134, "duration_seconds": null},
  "preprocessing": {"frame_count": 1, "frame_shape": [224, 224, 3]},
  "model_output": {"verdict": "likely_real", "confidence": 0.5237, "manipulation_score": 0.2381, "model_name": "dummy-placeholder-v0"},
  "provenance": {"status": "not_checked"},
  "pipeline_version": "0.1.0-phase1-dummy"
}
```

`smoke.mp4` (3298 bytes, 2.0s) -> valid `DetectionResult`:
```json
{
  "media_type": "video",
  "validated": {"file_size_bytes": 3298, "duration_seconds": 2.0},
  "preprocessing": {"frame_count": 4, "frame_shape": [224, 224, 3]},
  "model_output": {"verdict": "likely_real", "confidence": 0.3516, "manipulation_score": 0.3242, "model_name": "dummy-placeholder-v0"},
  "provenance": {"status": "not_checked"},
  "pipeline_version": "0.1.0-phase1-dummy"
}
```

`corrupt.mp4` (17 bytes of literal text, `.mp4` extension) ->
`PipelineRejectedError: File content does not match a supported
image/video format.` — rejected safely, no crash, no traceback surfaced to
the caller.

All three outcomes match the Phase 1 acceptance criteria.

---

## 2026-09-29 — YuNet ONNX asset download and verification (Phase 2)

Commands:
```
curl -sL -o model.onnx "https://raw.githubusercontent.com/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2026may.onnx"
```
Result: 131-byte Git LFS pointer file, not the binary:
```
version https://git-lfs.github.com/spec/v1
oid sha256:ebafce4e3c118d6554634be5c27ab333b4c047a9a8c3faf1d7cf93101c22f0f0
size 229738
```

Actual binary fetched from the LFS media endpoint instead:
```
curl -sL -o model_real.onnx "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2026may.onnx"
sha256sum model_real.onnx
```
Result: 229,738 bytes; `ebafce4e3c118d6554634be5c27ab333b4c047a9a8c3faf1d7cf93101c22f0f0` —
**matches the LFS pointer's declared hash exactly**. First bytes
(`0806 1207 7079 746f 7263 681a 0331 2e37` = `...pytorch..1.7`) consistent
with an ONNX export from PyTorch 1.7, as expected.

Placed at `models/face_detection/face_detection_yunet_2026may.onnx`
(gitignored). Full provenance table in `docs/DATASETS.md`.

Load + safety verification:
```python
import cv2
det = cv2.FaceDetectorYN_create('models/face_detection/face_detection_yunet_2026may.onnx', '', (320,320), 0.6, 0.3, 5000)
```
Output: `cv2 version: 5.0.0`; detector created successfully (CPU, no CUDA
requested or required). Running `.detect()` on a blank 320x320 black image
returned `faces: None` (i.e. zero detections) — no crash, no exception.

---

## 2026-09-29 — Phase 2 test suite run

Command (ffmpeg's bin dir added to PATH for this shell, as in prior runs):
```
.venv/Scripts/python.exe -m pytest tests/media -v
```
Result: **76 passed in 2.94s** across `test_alignment.py`, `test_cache.py`,
`test_decode.py`, `test_face_detector.py`, `test_preprocess.py`,
`test_sampling.py`, `test_tracking.py`. Two YuNet-backed tests
(`test_yunet_loads_on_cpu_and_satisfies_protocol`,
`test_yunet_safely_returns_no_faces_on_non_face_image`, plus one more) ran
for real (not skipped) since the model file is present locally.

Full combined suite:
```
.venv/Scripts/python.exe -m pytest -v
```
Result: **124 passed in 7.11s** (48 from Phases 0–1 + 76 new).

---

## 2026-09-29 — Phase 2 preprocessing benchmark

Command:
```
.venv/Scripts/python.exe scripts/benchmark_preprocessing.py
```
Clip: synthesized via `ffmpeg lavfi testsrc`, 640x480, 5s @ 15fps (75
frames), no real face present (expected — this is a geometry/timing
benchmark, not a detection-accuracy one).

| Stage | Time |
|---|---|
| hash input file (SHA-256) | 0.87 ms |
| `extract_video_metadata` (ffprobe) | 66.09 ms |
| `compute_sampling_plan` (16 frames) | 0.27 ms |
| `decode_sampled_frames` (16 frames, sequential decode) | 68.19 ms |
| `YuNetFaceDetector()` load (cold) | 16.07 ms |
| face detection, 16 frames (CPU) | 213.04 ms (13.31 ms/frame) |
| `align_and_crop` (1 crop, geometry only) | 1.53 ms |
| `cache.put` (write, cold) | 3.65 ms |
| `cache.get` (hit) | 15.14 ms |
| `cache.get` (miss) | 0.17 ms |

Notes: `frame_count=75, fps=15.0` correctly extracted;
`indices=(0, 5, 10, 15, 20, 25, 30, 35, 39, 44, 49, 54, 59, 64, 69, 74)`
for the 16-frame plan. Zero faces found (expected: synthetic test pattern,
not a real face). Console also showed a benign OpenCV warning
(`Targets are not supported by the new graph engine for now`) — informational
only, detection still ran correctly on CPU.
