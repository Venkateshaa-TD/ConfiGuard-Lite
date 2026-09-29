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

---

## 2026-09-29 — Phase 3 test suite run

Command:
```
.venv/Scripts/python.exe -m pytest tests/datasets -v
```
Result: **91 passed in 1.01s** across `test_schema.py`, `test_manifest.py`,
`test_splitting.py`, `test_duplicates.py`, `test_storage.py`,
`test_adapters.py`, `test_fixture_scenarios.py`.

Full combined suite:
```
.venv/Scripts/python.exe -m pytest -q
```
Result: **215 passed in 7.83s** (124 from Phases 0–2 + 91 new). Two benign
`moov atom not found` stderr lines appeared from ffmpeg/OpenCV probing a
Phase-1 fixture (a deliberately truncated video, used to test corrupted-
file handling) - expected noise, not a failure.

---

## 2026-09-29 — Phase 3 storage-check utility run

Command (no env vars set):
```
.venv/Scripts/python.exe scripts/check_storage.py
```
Output: all three of `CONFIGUARD_DATA_DIR`/`CONFIGUARD_CACHE_DIR`/
`CONFIGUARD_CHECKPOINT_DIR` reported as not set (exit code 0 - unset is a
warning, not a refusal).

Command (deliberately misconfigured, to exercise the refusal path):
```powershell
$env:CONFIGUARD_DATA_DIR = "C:\Users\balag\ConfiGuard-Data-Outside-Repo"
$env:CONFIGUARD_CACHE_DIR = "C:\Users\balag\Projects\ConfiGuard-Lite\data\bad_location"
.venv/Scripts/python.exe scripts/check_storage.py
```
Output (abbreviated):
```
--- CONFIGUARD_DATA_DIR ---
  configured path : C:\Users\balag\ConfiGuard-Data-Outside-Repo
  exists          : False
  writable        : True
  inside repo     : False
  total           : 199.6 GB
  free            : 12.9 GB
  [WARNING] CONFIGUARD_DATA_DIR path does not exist yet: ...

--- CONFIGUARD_CACHE_DIR ---
  configured path : C:\Users\balag\Projects\ConfiGuard-Lite\data\bad_location
  inside repo     : True
  [WARNING] ... is INSIDE the git repository ... Point it at a location outside the repo.

[REFUSE] One or more paths are inside the git repository - reconfigure before storing any real data.
```
Exit code: **1** (refusal, as intended for the in-repo case). Confirms the
utility correctly distinguishes "not set" / "outside repo, just doesn't
exist yet" / "inside repo - refuse" without creating or downloading
anything. Free disk space at time of this check: **12.9 GB** (down from
~21.8 GB at Phase 0 - consumed by `.venv` packages: torch, opencv, etc. -
see `docs/KNOWN_ISSUES.md`).

---

## 2026-09-29 — Phase 3 end-to-end dataset registry demo

Built a synthetic, FaceForensics++-shaped fixture (3 real videos under
`original_sequences/youtube/c23/videos/`, 2 fake videos under
`manipulated_sequences/{Deepfakes,Face2Face}/c23/videos/`, all tiny
placeholder byte content - not real video), then ran the full workflow:

```python
adapter = make_faceforensics_adapter()
samples = adapter.build_manifest(root)          # -> 5 Sample objects
write_manifest(samples, manifest_path)           # JSONL
report = validate_manifest_file(manifest_path, media_root=root)
split_report = split_samples(samples, SplitConfig(seed=42))
write_split_audit_report(split_report, audit_path)
dup_report = build_duplicate_report(samples, root)
```

Results:
- **5 samples built**, correctly labeled real/fake, `source_id` extracted
  from filenames (`001`, `002`, `003`), each fake's `parent_sample_id`
  correctly linked to its matching real sample by shared `source_id`
  (`001_002.mp4` -> parent `001.mp4`; `002_003.mp4` -> parent `002.mp4`),
  `generator_method` set to `Deepfakes`/`Face2Face` per bucket.
- **Manifest validation: `is_valid=True`, 0 issues.**
- **Split (seed=42):** `{001-real, 001_002-fake}` -> `validation`;
  `{002-real, 002_003-fake, 003-real}` -> `train`. The real/fake pairs for
  sources 001 and 002 stayed together in their respective splits (no
  leakage), matching `parent_sample_id`-based grouping.
- Audit report written successfully (1560 bytes JSON).
- Duplicate report: 0 exact, 0 near (expected - all 5 fixture files have
  distinct placeholder byte content).

---

## 2026-09-29 — Storage audit (read-only)

Full drive listing, directory-size measurements (repo, `.venv`, pip
cache, HF/torch caches, project dirs, user temp), and the 21.8 GB -> 12.9
GB explanation are recorded in the conversation transcript (storage-audit
report). Key figures, reused below: `C:` 199.56 GB total / 12.92 GB free;
`D:` 276.38 GB total / 92.93 GB free (separate local fixed NTFS volume,
not OneDrive); `.venv` 4,671.5 MB (torch alone 4,372.1 MB); pip cache
3,209.9 MB; FFmpeg (WinGet) 664.2 MB.

---

## 2026-09-29 — Storage configuration (approved: create dirs, configure `.env`, purge pip cache only)

Commands:
```powershell
New-Item -ItemType Directory -Force -Path "D:\ConfiGuard-Data\datasets"
New-Item -ItemType Directory -Force -Path "D:\ConfiGuard-Data\cache"
New-Item -ItemType Directory -Force -Path "D:\ConfiGuard-Data\checkpoints"
New-Item -ItemType Directory -Force -Path "D:\ConfiGuard-Data\outputs"
```
Result: all four created successfully.

Code change: added `CONFIGUARD_OUTPUT_DIR` to
`configuard.datasets.storage.STORAGE_ENV_VARS` (was data/cache/checkpoint
only) and to `.env.example`, so the project actually supports a fourth,
configurable output directory - see `docs/DECISIONS.md`.

Local `.env` created (untracked; verified below that it stays gitignored)
with:
```
CONFIGUARD_DATA_DIR=D:\ConfiGuard-Data\datasets
CONFIGUARD_CHECKPOINT_DIR=D:\ConfiGuard-Data\checkpoints
CONFIGUARD_CACHE_DIR=D:\ConfiGuard-Data\cache
CONFIGUARD_OUTPUT_DIR=D:\ConfiGuard-Data\outputs
```

Verification 1 - `git check-ignore -v .env`:
```
.gitignore:16:.env	.env
```
`.env` matched by the ignore rule; `git status --porcelain` showed no
`.env` entry. Confirmed never staged/committed.

Verification 2 - `scripts/check_storage.py` (env vars set to the D:
paths): all four report `exists=True`, `writable=True`,
`inside repo=False`, no warnings, `total=276.4 GB`, `free=92.9 GB`.

Verification 3 - explicit OneDrive-exclusion + real write test (not just
a permission-bit check):
```
OneDrive root: C:\Users\balag\OneDrive
Repo root: C:\Users\balag\Projects\ConfiGuard-Lite
CONFIGUARD_DATA_DIR: exists=True writable(actual write test)=True inside_onedrive=False inside_repo=False
CONFIGUARD_CACHE_DIR: exists=True writable(actual write test)=True inside_onedrive=False inside_repo=False
CONFIGUARD_CHECKPOINT_DIR: exists=True writable(actual write test)=True inside_onedrive=False inside_repo=False
CONFIGUARD_OUTPUT_DIR: exists=True writable(actual write test)=True inside_onedrive=False inside_repo=False
```
(Each check wrote a `.write_test.tmp` file and deleted it immediately -
no files left behind.)

Pip cache purge (the only cleanup authorized):
```
.venv/Scripts/python.exe -m pip cache purge
```
Output: `Files removed: 1598 (3365.8 MB)`, `Directories removed: 2602`.
No other temporary files or application data were touched.

Free space recheck:
| Drive | Before | After |
|---|---|---|
| `C:` | 12.92 GB | **16.05 GB** |
| `D:` | 92.93 GB | 92.93 GB (unchanged - new dirs are empty) |

Test run:
```
.venv/Scripts/python.exe -m pytest tests/datasets/test_storage.py -v
```
Result: **9 passed** (all storage tests, including the renamed
`test_check_all_storage_paths_covers_every_configured_var`, which now
also asserts `CONFIGUARD_OUTPUT_DIR` is in `STORAGE_ENV_VARS`).

Full combined suite:
```
.venv/Scripts/python.exe -m pytest -q
```
Result: **215 passed** (unchanged pass count - only a storage-module
extension + one test rename, no new tests added for this configuration
step).
