# Experiment Log

Format: one entry per meaningful run, in chronological order (oldest first). This log records exact
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

---

## 2026-09-29 — Phase 4 cache configuration and baseline model download

Cache directories created (`New-Item -ItemType Directory`):
```
D:\ConfiGuard-Data\cache\huggingface
D:\ConfiGuard-Data\cache\torch
```

`.env` updated with:
```
HF_HOME=D:\ConfiGuard-Data\cache\huggingface
HF_HUB_CACHE=D:\ConfiGuard-Data\cache\huggingface\hub
TORCH_HOME=D:\ConfiGuard-Data\cache\torch
```

Command:
```
.venv/Scripts/python.exe -m pip install "timm>=1.0" onnx onnxruntime
```
Result: `timm 1.0.30`, `onnx 1.23.0`, `onnxruntime 1.30.0` installed.
**Side effect discovered afterward:** this silently downgraded `torch`
from `2.5.1+cu121` to `2.14.0+cpu` (see `docs/KNOWN_ISSUES.md` -
RESOLVED by reinstalling `torch==2.5.1` with the `cu121` index
immediately after discovery, before any benchmark numbers were recorded).

Command (confirmed both exact authorized model names are available):
```python
timm.list_models('mobilenetv4_conv_small*', pretrained=True)
# ['mobilenetv4_conv_small.e1200_r224_in1k', 'mobilenetv4_conv_small.e2400_r224_in1k',
#  'mobilenetv4_conv_small.e3600_r256_in1k', 'mobilenetv4_conv_small_050.e3000_r224_in1k']
timm.list_models('tf_efficientnet_b0*', pretrained=True)
# ['tf_efficientnet_b0.aa_in1k', 'tf_efficientnet_b0.ap_in1k', 'tf_efficientnet_b0.in1k', 'tf_efficientnet_b0.ns_jft_in1k']
```

Command (the only download this phase performs):
```
.venv/Scripts/python.exe scripts/download_baseline_models.py
```
Output (abbreviated): both models constructed successfully via
`timm.create_model(..., pretrained=True, num_classes=0)`; cache env vars
confirmed applied from `.env`.

Cache location verification (from the script's own output):
```
HF hub cache resolved to : D:\ConfiGuard-Data\cache\huggingface\hub
  under default ~/.cache : False
  under repo              : False
Torch home resolved to   : D:\ConfiGuard-Data\cache\torch
  under default ~/.cache : False
  under repo              : False
```

Downloaded files (verified via `find`/`sha256sum` on the actual cache
directory):
```
D:\ConfiGuard-Data\cache\huggingface\hub\models--timm--mobilenetv4_conv_small.e1200_r224_in1k\
  refs\main                                    -> c9f31ac64483d7f0590db9edccb4418392a96eea
  snapshots\c9f31ac.../model.safetensors        -> 15,223,016 bytes
                                                    sha256=5a2ef04d419ce6d1bf27bfa735bb200d3f8d8997c3ac36320f5bf30382f6b43c

D:\ConfiGuard-Data\cache\huggingface\hub\models--timm--tf_efficientnet_b0.in1k\
  refs\main                                    -> 8186ca4217f9c67824ebe7566008bdc69976d15a
  snapshots\8186ca4.../model.safetensors        -> 21,355,344 bytes
                                                    sha256=276dfe076f3fca30c2f7bf1e44039e395e6de50248caaa159d16530699f16995
```
Both `safetensors` format (preferred, as instructed). `D:\ConfiGuard-Data\cache\torch`
received no files - both models were fetched entirely via the HF hub
mechanism, `TORCH_HOME` is configured but unused by this phase (kept for
completeness/future use). `huggingface_hub` warned about Windows
symlinks being unavailable (informational only - see
`docs/KNOWN_ISSUES.md`).

No dataset and no other checkpoint (GenD, DINOv2, or anything else) was
downloaded.

---

## 2026-09-29 — Phase 4 environment regression, found and fixed (torch + torchvision)

Two cascading environment issues, both caused by the `pip install timm
onnx onnxruntime` command above re-resolving `torch` from the default
index. Full detail and root cause in `docs/KNOWN_ISSUES.md`; commands and
final verification here:

```
pip install "torch==2.5.1" --index-url https://download.pytorch.org/whl/cu121
pip install "torchvision==0.20.1" --index-url https://download.pytorch.org/whl/cu121
```

Final verification:
```
python -c "import torch, torchvision; print(torch.__version__, torch.cuda.is_available(), torchvision.__version__)"
# 2.5.1+cu121 True 0.20.1+cu121
python -c "import timm; print('timm import OK')"
# timm import OK
```

---

## 2026-09-29 — Phase 4 full smoke test (both models, CUDA)

Command: ad-hoc script constructing each registered encoder
(`pretrained=True`), running image inference, 8-frame video inference, a
forward+backward step, `approximate_flops`, ONNX export, and ONNX parity
- device resolved via `resolve_device("auto")` -> `cuda` (RTX 4050).

Result (abbreviated; full numbers in the benchmark/export runs below):
both models produced valid `PredictionResult`s (`is_finetuned=False`,
`PREDICTION_DISCLAIMER` present) for image and 8-frame video inference;
`loss.backward()` populated `head.weight.grad` for both; ONNX parity
`max_abs_diff` on the order of 1e-7 for both, well inside the documented
1e-3 tolerance.

---

## 2026-09-29 — Phase 4 test suite run

Two real bugs found and fixed while getting these to pass (both are also
documented as regression tests, see `docs/DECISIONS.md`):
`test_infer_video_uses_mean_of_embeddings_not_just_first_frame` initially
compared `probability` (sigmoid-saturated to the same float32 value for
two inputs with different logits) - fixed to compare `logit` instead;
`test_real_image_pipeline_with_face_found` /
`test_real_video_pipeline_with_face_found` initially used a 5x5 mock face
box against Phase 2's default `min_face_size_px=20`, which silently
excluded it - fixed by setting `min_face_size_px=1` in that test file's
config (test-only, not a production default). Also found and fixed a
test-isolation bug in `tests/test_env_loader.py`
(`load_dotenv` mutates `os.environ` directly, not via
`monkeypatch.setenv`, so a later test needed its own `monkeypatch.delenv`
to avoid a leaked value from an earlier test in the same file).

Command:
```
.venv/Scripts/python.exe -m pytest tests/models -v
```
Result: **90 passed in 67.98s**, across `test_encoder.py`,
`test_inference.py`, `test_onnx_export.py`, `test_preprocess.py`,
`test_pretrained_integration.py` (real checkpoints, not skipped - the
cache was populated), `test_real_pipeline.py`, `test_registry.py`,
`test_smoke_backward.py`, `test_device.py`.

Full combined suite:
```
.venv/Scripts/python.exe -m pytest -q
```
Result: **311 passed in 80.08s** (215 from Phases 0-3 + 90 new Phase 4
model tests + 6 new `tests/test_env_loader.py` tests).

---

## 2026-09-29 — Phase 4 benchmark (real numbers, warm-up + P50/P95)

Command:
```
.venv/Scripts/python.exe scripts/benchmark_models.py
```

| Metric | EfficientNet-B0 | MobileNetV4-Conv-Small |
|---|---|---|
| Parameters (total = trainable) | 4,008,829 | 2,494,305 |
| Approx. FLOPs @ 224×224 (`torch.utils.flop_counter`) | 769,072,064 | 369,453,696 |
| CPU latency, batch=1 (P50 / P95, ms) | 99.82 / 198.57 | 46.73 / 69.05 |
| CPU image batch, n=8 (P50 / P95, ms) | 396.71 / 467.38 | 115.42 / 159.87 |
| CPU video, 4 frames (P50 / P95, ms) | 238.38 / 323.10 | 83.13 / 118.43 |
| CPU video, 8 frames (P50 / P95, ms) | 351.64 / 448.42 | 118.48 / 186.54 |
| CPU video, 16 frames (P50 / P95, ms) | 689.87 / 778.61 | 186.35 / 261.53 |
| GPU (RTX 4050) latency, batch=1 (P50 / P95, ms) | 39.26 / 49.71 | 22.19 / 22.85 |
| GPU peak memory (allocated / reserved, MB) | 110.6 / 134.0 | 33.5 / 134.0 |
| GPU peak memory vs. 6 GB budget | 2.2% (reserved) | 2.2% (reserved) |

Warm-up: 5 iterations (CPU bs=1), 3 (CPU batch/video), 10 (GPU) before
timing; 20/10/30 measured iterations respectively - see
`configuard.models.benchmark.measure_latency`. **Both models fit
comfortably within the RTX 4050's 6 GB VRAM budget** (peak reserved
~134 MB, ~2.2%).

**MobileNetV4-Conv-Small is faster and smaller on every measured axis**
(fewer parameters, fewer FLOPs, lower CPU and GPU latency at every batch/
frame-count tested). This is a **provisional efficiency comparison
only** - no deepfake-detection accuracy result exists yet (see
docs/DECISIONS.md and docs/MODEL_CARD.md); the final model choice is not
decided by this data alone.

---

## 2026-09-29 — Phase 4 ONNX export and parity verification

Command:
```
.venv/Scripts/python.exe scripts/export_onnx_models.py
```

| Model | ONNX file size | Parity (max abs diff) | Parity (mean abs diff) | Within tolerance? |
|---|---|---|---|---|
| EfficientNet-B0 | 16,100,693 bytes (~15.4 MB) | 4.359e-07 | 2.517e-07 | Yes (atol=1e-3, rtol=1e-3, n=8 seeded samples) |
| MobileNetV4-Conv-Small | 9,952,742 bytes (~9.5 MB) | 3.320e-07 | 2.081e-07 | Yes (atol=1e-3, rtol=1e-3, n=8 seeded samples) |

Both exports: FP32, opset 17, dynamic batch axis, legacy TorchScript
exporter (`dynamo=False` where supported - see docs/DECISIONS.md). ONNX
CPU inference verified for both (output shape `(4,)` for a 4-sample
batch, via ONNX Runtime `CPUExecutionProvider`). No quantization
performed (Phase 4 scope). Exported files land in `outputs/onnx/`
(gitignored - never committed).

---

## 2026-09-30 — Phase 5 environment verification (before and after; no installs)

No package was installed, upgraded, or reinstalled in Phase 5, so the
"before and after any installation" check reduces to verifying the
environment at the start and end of the phase.

```
.venv/Scripts/python.exe scripts/verify_environment.py
```
Result (both times): Python 3.11.9; torch 2.5.1+cu121, CUDA available
(12.1), NVIDIA GeForce RTX 4050 Laptop GPU, 6140 MB; torchvision
0.20.1+cu121 imports **and** runs `torchvision.ops.nms`; cuDNN 90100;
driver 591.66; timm 1.0.30; `Dependency safety: OK`, no warnings; exit 0.
These are the versions pinned in `constraints-cuda.txt`.

---

## 2026-09-30 — Phase 5 test runs (formal)

Baseline, before any Phase 5 resume work (preliminary, includes the
interrupted draft's own tests):
```
.venv/Scripts/python.exe -m pytest -q -p no:cacheprovider
```
Result: 372 passed in 85.03s.

Targeted Phase 5 tests, final code:
```
.venv/Scripts/python.exe -m pytest tests/training tests/test_dependency_safety.py -v -p no:cacheprovider
```
Result: **109 passed, 0 skipped in 105.53s**: test_trainer 35,
test_checkpoint 13, test_metrics 13, test_dependency_safety 9,
test_splits_and_logging 8, test_config 6, test_datasets 6,
test_sampling 6, test_optim 5, test_paths 5, test_dataloader 3. The
CUDA-only tests (AMP smoke for both backbones, CUDA resume, AMP-skip
accounting, RNG restore after a CUDA load) **ran** on the RTX 4050; they
were not skipped.

Afterwards, 2 more tests were added to close coverage gaps (video-frame training end-to-end; `num_workers=2` spawned workers bit-identical to single-process). Both passed, bringing the Phase 5 total to **111**. Full suite: see the final entry below.

Failures found and fixed while getting here (each now a regression test):
1. `test_tiny_synthetic_set_can_be_overfitted` initially failed: loss
   0.41 → 3.11 over 12 epochs. Probe (three lr/pretrained settings)
   showed loss reaching ~0 and then spiking at the **same epochs (8, 11)**
   in every setting. Printing per-epoch batch labels showed those epochs
   contain an all-real batch: the balanced sampler draws with
   replacement, and train-mode BatchNorm normalizes away the colour cue
   inside a single-class batch. With full batch and no replacement, loss
   goes 0.678 → 0.0002, monotonically. The test now uses that setup
   (`docs/KNOWN_ISSUES.md`).
2. `test_cuda_resume_matches_uninterrupted_run` failed with `TypeError:
   RNG state must be a torch.ByteTensor`: a **pre-existing draft bug**
   (checkpoint loaded with `map_location="cuda"` moved the RNG states to
   the GPU). Fixed; GPU resume never worked before.
3. One test-code kwarg collision (`epochs` passed twice); fixed.

---

## 2026-09-30 — Phase 5 smoke training (CLI, D: drive) — ENGINEERING TESTS ONLY

> All numbers below come from synthetic blue- vs. red-tinted
> checkerboards with a deliberately obvious signal. They verify pipeline
> mechanics. **They are not deepfake-detection accuracy.**

Smoke config (`configuard.training.runner.run_smoke`): pretrained
ImageNet weights from `D:\ConfiGuard-Data\cache\huggingface`
(`HF_HUB_OFFLINE=1`), 64 train (32/class) + 16 val (8/class) synthetic
images, full-frame mock face detector through Phase 2 crop/cache, batch
8, grad accumulation 2 (4 optimizer steps/epoch), 5 epochs, AdamW lr
1e-3, warm-up 2 + cosine, grad clip 1.0, seed 1234, source+class-balanced
sampling. Each run also repeats training as "interrupted after epoch 4 +
resumed from latest" and compares the final weights.

Commands:
```
.venv/Scripts/python.exe scripts/train.py --smoke cuda
.venv/Scripts/python.exe scripts/train.py --smoke cuda --encoder efficientnet_b0
.venv/Scripts/python.exe scripts/train.py --smoke cpu
```
All three exited 0.

| | MobileNetV4 — RTX 4050, AMP | EfficientNet-B0 — RTX 4050, AMP | MobileNetV4 — CPU (fp32) |
|---|---|---|---|
| Run | `smoke_mobilenetv4_conv_small_cuda_20260930-212901` | `smoke_efficientnet_b0_cuda_20260930-212938` | `smoke_mobilenetv4_conv_small_cpu_20260930-213025` |
| Parameters | 2,494,305 | 4,008,829 | 2,494,305 |
| Peak VRAM allocated / reserved | 132.1 / **154.0 MB** (2.5% of 6 GB) | 437.8 / **460.0 MB** (7.5% of 6 GB) | n/a |
| Mean training step (batch 8), epochs 1-4 | 76.6-92.0 ms | 164.5-183.1 ms | 244.4-254.5 ms |
| Epoch-0 step (cold: crop cache + cuDNN warm-up) | 255.9 ms | 379.6 ms | 251.7 ms |
| Validation pass (16 images), epochs 1-4 | 0.20-0.21 s | 0.23-0.24 s | 0.26-0.27 s |
| Optimizer steps attempted / skipped by AMP | 20 / 3 | 20 / 0 | 20 / 0 |
| Checkpoint size (latest = best, incl. AdamW state) | 30,278,650 B (28.9 MiB) | 48,607,919 B (46.4 MiB) | 30,270,202 B (28.9 MiB) |
| Resume: max abs param diff vs uninterrupted | **0.0** | **0.0** | **0.0** |
| Resume: final train loss / global step equal | yes / yes (20) | yes / yes (20) | yes / yes (20) |
| Train loss epoch 0 → 4 | 0.690 → 0.063 | 0.582 → 0.024 | 0.621 → 0.048 |
| Val AUROC / balanced acc, final epoch (synthetic) | 1.0 / 1.0 | 1.0 / 1.0 | 1.0 / 1.0 |
| Best epoch (AUROC, val-loss tie-break) | 3 | 2 | 4 |

Written (and nothing else):
- checkpoints: `D:\ConfiGuard-Data\checkpoints\smoke\<run>\<run>_{latest,best}.pt`
  and `<run>_resume_{latest,best}.pt`
- logs: `D:\ConfiGuard-Data\outputs\smoke\<run>\<run>.jsonl`,
  `<run>_train.csv`, `<run>_epoch.csv`, `<run>_summary.json`
- synthetic media + manifests + face-crop cache:
  `D:\ConfiGuard-Data\outputs\smoke\<run>\synthetic_data\`, `...\face_crop_cache\`

Preliminary, superseded smoke runs, kept on D: for the record and
**not** used as results:
- `..._cuda_20260930-212438` (3 epochs, 6 steps): train loss stuck
  ~0.69. The checkpoint was 10.2 MB with **empty optimizer state**:
  GradScaler skipped all 6 steps (scale 65536 → 1024). This led to
  skip counting and `amp_init_scale=1024` (`docs/DECISIONS.md`).
- `..._cuda_20260930-212525` (5 epochs, default init scale): 9 of 20
  steps skipped (scale → 128).
- `..._cuda_20260930-212728` (init scale 1024, before the val-loss
  tie-break): best checkpoint = epoch 1 (AUROC 1.0, val loss 0.51)
  instead of epoch 3 (AUROC 1.0, val loss 0.028). This led to the
  tie-break.

---

## 2026-09-30 — Phase 5 evaluation-only CLI

```
.venv/Scripts/python.exe scripts/evaluate.py --synthetic \
  --checkpoint D:/ConfiGuard-Data/checkpoints/smoke/smoke_mobilenetv4_conv_small_cuda_20260930-212901/smoke_mobilenetv4_conv_small_cuda_20260930-212901_best.pt \
  --manifest   D:/ConfiGuard-Data/outputs/smoke/smoke_mobilenetv4_conv_small_cuda_20260930-212901/synthetic_data/val_manifest.jsonl \
  --media-root D:/ConfiGuard-Data/outputs/smoke/smoke_mobilenetv4_conv_small_cuda_20260930-212901/synthetic_data/media
```
Result (exit 0; ENGINEERING TEST ONLY, disclaimer included in the
output): cuda, 16 samples, 1.38 s. Threshold-free: AUROC 1.0, AP 1.0.
Threshold-dependent (@0.5): sensitivity 1.0, specificity 1.0, balanced
accuracy 1.0, F1 1.0, TP 8 / FP 0 / TN 8 / FN 0. Written to
`D:\ConfiGuard-Data\outputs\eval\eval_smoke_mobilenetv4_conv_small_cuda_20260930-212901_best_20260930-213124.json`.

---

## 2026-09-30 — Phase 5 mismatched-checkpoint rejection through the real CLI

Resumed the MobileNetV4 smoke checkpoint under the production config
(same encoder, same manifests):
```
.venv/Scripts/python.exe scripts/train.py --config configs/train/mobilenetv4_conv_small.yaml \
  --train-manifest .../synthetic_data/train_manifest.jsonl --val-manifest .../val_manifest.jsonl \
  --media-root .../synthetic_data/media --resume-from .../smoke_mobilenetv4_conv_small_cuda_20260930-212901_latest.pt
```
Result: exit 1, `CheckpointMismatchError` listing all 8 mismatches
(preprocessing_version: mock vs. YuNet detector; seed, batch_size,
epochs, lr, warmup_steps, grad_accum_steps, early_stopping_patience).
No training step was taken.

---

## 2026-09-30 — Phase 5 output-location audit

- `D:\ConfiGuard-Data\checkpoints`: 687 MB (6 smoke runs x {main, resume}
  x {latest, best}). `D:\ConfiGuard-Data\outputs`: 5.4 MB.
  `D:\ConfiGuard-Data\cache\face_crops`: 204 KB (the CLI-rejection run).
- `D:\ConfiGuard-Data\cache\huggingface\hub`: still exactly the two
  Phase 4 models. No new model was downloaded.
- `%USERPROFILE%\.cache\huggingface` and `\torch` do not exist.
- The repo gained no generated files: `git status --ignored` shows only
  source/test/config/doc changes. The ignored `outputs/onnx/` (Phase 4,
  2026-09-29) and `.pytest_cache/` (2026-09-29) predate this phase.
- pytest runs write their checkpoints/logs under pytest's `tmp_path`
  (`%TEMP%\pytest-of-<user>\...`), by design:
  tests must not depend on or pollute the configured D: directories.

---

## 2026-09-30 — Phase 5 final verification

```
.venv/Scripts/python.exe -m pytest -p no:cacheprovider -q -rs
```
Result: **422 passed, 0 skipped, 0 failed in 191.97s** (311 from Phases
0-4 + 111 new Phase 5 tests). The CUDA tests ran on the RTX 4050.

```
.venv/Scripts/python.exe scripts/verify_environment.py
```
Result: torch 2.5.1+cu121 CUDA build, torchvision 0.20.1+cu121 (import +
`ops.nms` OK), CUDA available, `Dependency safety: OK`, exit 0. No
package changed during the phase.

---

## 2026-09-30 — Phase 5b FaceForensics++ c23 acquisition

Free space on D: before starting: 92.22 GB (floor: 40 GB).

Official script (`docs/DATASETS.md` has full provenance):
```
curl -sS -f -L --proto-redir =https -o D:/ConfiGuard-Data/datasets/FaceForensics++/_official_script/faceforensics_download_v4.py \
  <approved script URL from the approval email - see _official_script/PROVENANCE.md on D:>
# 301 -> same host over HTTPS, 200, 10727 bytes
# sha256 5d0b220ad0c88bba9d80f45426aef48a89d182e88956ded85c8a9d310f8d04d0
python faceforensics_download_v4.py -h     # read in full before running; defaults are -c raw -d all -> never used
```
(The first fetch without `-L` saved the 353-byte 301 page; it was replaced.)

Trial: 5 files per class, then full download (5 detached single-stream instances):
```
.venv/Scripts/python.exe scripts/download_faceforensics_c23.py --num-videos 5
.venv/Scripts/python.exe scripts/download_faceforensics_c23.py --datasets <each of the 5>
```
- Trial: 25 files, 43.55 MB (~1.75 MB/file) → projected ~8.7 GB for 5000
  files, leaving ~83 GB. Proceeded.
- Full: ~1.1-1.5 files/s across 5 streams. `original` hung at 456 files
  (dead connection, 17:06:39 → noticed 17:14). It was killed and all
  streams were moved onto the patched wrapper (stall watchdog +
  process-tree kill). Face2Face later auto-recovered from a stall at 787
  files (17:35:14).
- Completed 17:52:52 UTC: 5 × 1000 files, 0 `tmp*` partials,
  9,041,543,739 bytes. Free space afterwards: 83.78 GB (minimum
  observed; never near the 40 GB floor).
- Logs: `D:\ConfiGuard-Data\outputs\acquisition\faceforensics\` (`download_*.jsonl`,
  per-dataset `*.stderr.log`, `wrapper_*.out.log`, `wrapper_*_restart.out.log`).

## 2026-09-30 — Phase 5b FaceForensics++ c23 validation, manifest, leakage

```
.venv/Scripts/python.exe scripts/validate_faceforensics.py
```
Result: **VALID**, exit 0, 96 s total (ffprobe of 5000 files: 70 s with 8 workers).
- Structure: exactly `original_sequences/youtube/c23/videos` and
  `manipulated_sequences/{Deepfakes,Face2Face,FaceSwap,NeuralTextures}/c23/videos`;
  no raw/c40/masks/models/DFD/FaceShifter/other paths.
- Counts vs official pair list: 1000/1000 for each of the 5 classes; 0
  missing, 0 unexpected, 0 zero-byte, 0 partial.
- ffprobe header + full packet demux: 5000/5000 readable, all h264.
- Relationships: 0 problems. Every fake is `<target>_<source>`, an
  official pair, with both originals present.
- Manifest (existing registry, `faceforensics++` adapter): 5000 samples
  (1000 real / 4000 fake), 0 Phase 3 validation issues, 0 exact SHA-256
  duplicates.
- Leakage: no official split available from the approved source, so
  none was applied or invented. 500 leakage groups, every one with 10
  members and exactly 2 originals; 0 lineage problems.
- Report: `D:\ConfiGuard-Data\outputs\acquisition\faceforensics\acquisition_report_20260930-232450.{json,md}`.

Tests:
```
.venv/Scripts/python.exe -m pytest tests/datasets -q -p no:cacheprovider       # 104 passed (91 Phase 3 + 13 new)
.venv/Scripts/python.exe -m pytest -q -p no:cacheprovider -rs                   # 435 passed, 0 skipped, 192.49s
```
Final Phase 5b verification (after all code and doc changes):
`pytest -q -p no:cacheprovider -rs` → **435 passed, 0 skipped, 0 failed
in 192.49s** (422 through Phase 5 + 13 new in
`tests/datasets/test_faceforensics.py`).

---

## 2026-09-30 — Phase 5b commit

Audit before commit: redacted the FF++ download-script URL from 4 files
(moved to `_official_script/PROVENANCE.md` on D:). Staged 15 text files;
0 binaries, 0 data/manifest/report/log files, 0 access hosts, 0 secrets.
`pytest tests/datasets` → 104 passed. Commit **`12e2585`**.

## 2026-09-30 — Phase 5c official split fetch

```
# locate + pin (GitHub API, read-only)
GET https://api.github.com/repos/ondyari/FaceForensics                      # default_branch=master, license=NOASSERTION
GET .../commits/master                                                        # b952e41cba017eb37593c39e12bd884a934791e1 (2020-07-15)
GET .../contents/dataset/splits?ref=b952e41cba017eb37593c39e12bd884a934791e1                                     # test 2102, train 10802, val 2102 (+ blob SHAs)
# fetch pinned copies to D:
curl -sS -f -o <D:>/_official_splits/b952e41cba017eb37593c39e12bd884a934791e1/<split>.json \
  https://raw.githubusercontent.com/ondyari/FaceForensics/b952e41cba017eb37593c39e12bd884a934791e1/dataset/splits/<split>.json
git hash-object <split>.json   # == GitHub blob SHA for all three
```
Fetched 2026-09-30T18:05:57Z. LICENSE = MIT (code); README says data is under the FF++ ToS.

## 2026-09-30 — Phase 5c split application, leakage validation, audit

```
.venv/Scripts/python.exe scripts/apply_faceforensics_splits.py
```
Result: **VALID - split applied**, exit 0.
- Reconciliation: 360/70/70 pairs, 720/140/140 originals, 0 problems.
  Leakage groups recomputed from the manifest are identical to the Phase
  5b file.
- Post-write re-check: the Phase 5 trainer's `find_cross_split_leakage`
  over the three written manifests reports 0 problems; Phase 3
  `validate_samples` (with media root) reports 0 issues.
- Counts: train 3600 (720 per class), val 700, test 700 (140 per class);
  fake:real = 4.0 in every split; leakage groups 360/70/70, all of size 10.
- Duration (s), median [p10-p90], max:

  | Split | original / DF / F2F | FaceSwap / NeuralTextures |
  |---|---|---|
  | train | 16.62 [11.26-28.64], 53.56 | 13.80 [10.60-20.80], 30.04 |
  | val | 16.15 [10.89-27.12], 47.37 | 13.34 [10.35-20.93], 25.88 |
  | test | 16.92 [11.20-30.58], 72.56 | 14.63 [11.02-20.63], 41.52 |

- Resolution share (%) of 1280×720 / 640×480 / 1920×1080 / 854×480 / other:
  train originals 33.1/24.4/12.4/9.7/20.4 vs F2F/NT 33.1/34.4/12.4/0.0/20.1;
  val 37.9/28.6/8.6/10.0/15.0 vs 37.9/35.7/8.6/0.0/17.9;
  test 24.3/29.3/15.7/10.0/20.7 vs 24.3/35.7/15.7/0.0/24.3.
- Follow-up metadata check (ffprobe width/height/SAR/DAR only, no frames
  decoded or extracted): F2F/NT round width down to a multiple of 16 in
  282/1000 videos each; DF/FS unchanged (1000/1000 identical to their
  target original). No SAR/DAR set.
- Report: `D:\ConfiGuard-Data\outputs\acquisition\faceforensics\split_report_20260930-233957.{json,md}`.
Tests:
```
.venv/Scripts/python.exe -m pytest tests/datasets/test_faceforensics_splits.py -v -p no:cacheprovider -rs   # 24 passed (incl. real pinned-file integration)
.venv/Scripts/python.exe -m pytest -q -p no:cacheprovider -rs                                               # 459 passed, 0 skipped, 175.98s
```

## 2026-10-01 — Phase 5d preflight and official-convention verification

```
git status --short                                         # clean at c17049c
sha256sum <D:>/FaceForensics++/_manifests/*.jsonl          # all 5 == pins in docs/DATASETS.md
.venv/Scripts/python.exe scripts/extract_ffpp_face_crops.py --preflight-only
```
Preflight OK:
- the 5 manifest pins match;
- the official split files are pinned, and the recomputed membership is
  unchanged (3600/700/700, 360/70/70 pairs);
- 0 leakage problems;
- 5000/5000 videos present and re-hashed against the manifest
  (0 mismatches);
- YuNet SHA-256 `ebafce4e…` matches its pin;
- D: free 83.8 GB.

Official sources read for the naming convention: `dataset/README.md`
@ `b952e41c` and arXiv:1901.08971v3 (appendix), quoted in
`docs/DATASETS.md`.

Exploration (outputs on D: `_exploration/` only):
- **ffprobe packet counts for all 5000 videos (62 s):**
  - DF = target length (1000/1000);
  - F2F = source length (992/1000);
  - FS = min(target, source) (1000/1000); NT the same (999/1000);
  - 16 fakes per method have an fps header different from the target;
  - per-family shared range: 287–1038 frames (median 375).
- **Index correspondence:** registration of 54 seeded fakes against
  target frames i−4…i+4 at 3 positions gave offset 0 in 47/54. The
  remaining ±1–2 cases are near-static scenes. The fps-mismatched fakes
  still align by index.
- **Crop vs squeeze:** 12 width-changed pairs, then the full audit below.

## 2026-10-01 — Phase 5d trials

| Run | Config tag | Families | Result |
|---|---|---|---|
| trial 10 (seed 0) | `p5d-2ce67d23…` | 10 | 9 accepted; family 682 quarantined (IoU-only sparse linking split a drifting face); validator reported 612 false "leakage" problems: scope bug, parents of unselected families looked up in the subset |
| rerun ×2 after the validator fix | same | 10 (all resumed) | 0 leakage; JSONL byte-identical across 3 runs |
| config change (size-normalised linking) | `p5d-c02fd00a…` | — | old store **refused as stale**, moved to `superseded_trial1_*` |
| trial 10 | `p5d-c02fd00a…` | 10 | 10/10 accepted, 640/640 exact, 0 leakage; ~62 KB/crop; 38 s/family |
| full run 1 | `p5d-c02fd00a…` | stopped at ~200/990 | family 158 quarantined although the face was detected everywhere: tracking ignored recovery frames. Stopped (`taskkill /T`), store moved to `superseded_run2_*` |
| families 158,682 (targeted) | `p5d-b451b5ca…` | 2 | 2/2 accepted, 128/128 exact |
| trial 10 | `p5d-b451b5ca…` | 10 | 10/10 accepted, 640/640 exact, 0 leakage |

## 2026-10-01 — Phase 5d full extraction

```
.venv/Scripts/python.exe scripts/extract_ffpp_face_crops.py            # 12 workers, floor 40+2 GB
```
- Preflight OK (all 5000 videos re-hashed).
- Estimate: 65.2 KB/crop → +4.92 GB, ~77.5 GB free after.
- Elapsed 3414.5 s (57 min) for 989 families (11 resumed from the
  trials). 39.9 s/family mean, 921 s max. ~18 families/min.
- **Result:**
  - 1000 families: 991 accepted, 9 quarantined (45 videos, 9 per class);
  - 79,280 crops; every accepted video has exactly 16 ordered slots;
  - matched slots 63,424/63,424 exact;
  - leakage re-validation: 0 problems;
  - 4.91 GB; D: free 77.17 GB at end.

Detection / recovery (planned-slot detection rate; recovered-slot rate; failed slots; quarantined videos):

| Group | Videos | Quarantined | Detection | Recovered | Failed slots |
|---|---|---|---|---|---|
| overall | 5000 | 45 | 99.853% | 0.094% | 95 |
| train / val / test | 3600 / 700 / 700 | 35 / 5 / 5 | 99.846 / 99.759 / 99.982% | 0.104 / 0.089 / 0.045% | 69 / 25 / 1 |
| real / fake | 1000 / 4000 | 9 / 36 | 99.869 / 99.848% | 0.094 / 0.094% | 17 / 78 |
| original / DF / F2F / FS / NT | 1000 each | 9 each | 99.869 / 99.781 / 99.881 / 99.863 / 99.869% | 0.094% each (joint) | 17 / 26 / 17 / 18 / 17 |
| 1280×720 / 1920×1080 / 640×480 / 854×480 / other-480p | 1625 / 615 / 1467 / 294 / 919 | 10 / 25 / 5 / 3 / 2 | 99.92 / 99.56 / 99.91 / 99.62 / 99.89% | 0.115 / 0.356 / 0.021 / 0.000 / 0.034% | — |
| <10 s / 10–20 / 20–30 / 30–45 / ≥45 s | 145 / 3589 / 995 / 220 / 51 | 3 / 38 / 2 / 2 / 0 | 99.91 / 99.82 / 99.97 / 99.77 / 100% | 0.431 / 0.092 / 0.069 / 0.028 / 0% | — |

Individual (approximate) recovery was never needed: 0 slots. Every
recovery was joint, so recovery counts are identical across classes by
construction.

## 2026-10-01 — Phase 5d rerun / idempotency verification

```
.venv/Scripts/python.exe scripts/extract_ffpp_face_crops.py --verify-crop-hashes
```
- 1000/1000 families resumed, 0 re-extracted.
- All 80,000 crop SHA-256s verified against their records.
- The 6 JSONL manifests and `extraction_summary.json` are
  **byte-identical** to the first build (sha256 diff empty).
- Contact sheets now cover all 9 quarantined families. All were
  reviewed: 4 content-wide (cutaway or clip end), 5 fake-only
  (broken manipulation output).

## 2026-10-01 — Phase 5d shortcut audit

```
.venv/Scripts/python.exe scripts/audit_ffpp_crop_shortcuts.py   # 252 s for A+B; report shortcut_audit_20261001-114343.{json,md}
```
- **A. Crop or squeeze:**
  - 562 width-changed fakes + 200 controls registered;
  - verdict 562/562 centred crop;
  - ECC sx 1.0000 ± 0.0001 (squeeze would be 0.9730);
  - squeeze/crop residual ratio median 9.57 (min 2.03);
  - landmark interocular ratio F2F 0.9985 [0.9968, 1.0003], NT 1.0008
    [0.9992, 1.0025].
- **B. Geometry probe** (train→test, 14 features):
  - 5-class acc 0.237 vs 0.200;
  - real-vs-fake AUC 0.537;
  - DF/F2F/FS/NT vs original AUC 0.617 / 0.526 / 0.534 / 0.524;
  - width/height ratio alone 0.513 overall.
- **C. Factor AUC fake-vs-real:**
  - face width 0.498; detection confidence 0.501; recovered slots
    0.500; out-of-frame 0.494; alignment scale 0.503;
    `shared_frame_count` 0.500;
  - source duration in the raw files 0.423 (FS/NT 0.346), but it does
    not reach sampling;
  - crop sharpness 0.467 (NT 0.415);
  - nuisance-factor probe real-vs-fake AUC 0.503.
- **D. Crop quality:**
  - confidence p0.1% 0.70, median 0.94; residual p99 6.9 px;
  - confidence < 0.7: 82 crops (orig 10, DF 32, F2F 4, FS 24, NT 12);
  - review sheets of the 64 lowest-confidence and 64 highest-residual
    crops show only faces (profiles, hand occlusion, broken DF
    renders);
  - out-of-frame share equal across classes (any 26–28%, >25% 0.7–0.8%).

## 2026-10-01 — Phase 5d tests

```
.venv/Scripts/python.exe -m pytest tests/crops -q        # 44 passed
.venv/Scripts/python.exe -m pytest -q -p no:cacheprovider -rs   # 503 passed, 0 skipped, 408.87 s (ran concurrently with the audit)
```

---

## 2026-10-01 — Phase 6a teacher download and provenance

```
.venv/Scripts/python.exe scripts/download_gend_teacher.py
```
- Downloaded `yermandy/GenD_CLIP_L_14` @ `891ce014a0308386c4d7d25b3dcf436a22db5504`
  (6 files, 1.216 GB) into `D:\ConfiGuard-Data\cache\huggingface\hub`.
- Recorded the hashes in `docs/DATASETS.md`. `problems: []`.
- `model.safetensors` SHA-256 is `d76f0bdf…6833`.
- Training-data review: paper arXiv 2508.06248 plus code @ `387a422`.
  Result: FF++ c23 official train only; FF++ val unused; FF++ test used
  for evaluation only (`docs/DECISIONS.md`).
- The gated HF dataset `yermandy/GenD` returned 401, so the per-frame
  lists were not diffed.

## 2026-10-01 — Phase 6a freezing and batch-size benchmark (RTX 4050 Laptop, 6 GB)

Benchmark on 256 seeded train crops (scratch script; numbers only).
- Load + SHA-256 check: 15.6 s. Params: 303,968,258, of which 0 are
  trainable.
- Outside `inference_mode` the output has `requires_grad=False`.
- Accuracy at 0.5: 0.910.
- fp16 vs fp32: max |Δlogit| 0.0116, max |Δprob| 0.0057.

| mode | bs 16 | 32 | 64 | 128 | 256 |
|---|---|---|---|---|---|
| fp32 img/s (peak GiB) | 29 (1.41) | 29 (1.68) | 28 (2.22) | 27 (3.29) | 8 (5.45) |
| fp16 autocast img/s (peak GiB) | 87 (1.32) | 90 (1.49) | **90 (1.84)** | 89 (2.54) | 86 (3.94) |

Chosen setting: **fp16 autocast, batch size 64**. It is at peak
throughput with more than 4 GiB of headroom.

## 2026-10-01 — Phase 6a teacher-logit caching (train + val only)

```
.venv/Scripts/python.exe scripts/cache_teacher_logits.py --limit-shards 2 --splits train   # trial: interrupted at 1/2, resumed to 2/2
.venv/Scripts/python.exe scripts/cache_teacher_logits.py                                   # full
.venv/Scripts/python.exe scripts/cache_teacher_logits.py                                   # rerun: 67/67 shards resumed
.venv/Scripts/python.exe scripts/cache_teacher_logits.py --splits test                     # ProtectedSplitError
```
- Tag `t6a-f87ebb7553a64e99`; crop tag `p5d-b451b5ca770c8923`; shard size 1024.
- Cache root: `D:\ConfiGuard-Data\cache\teacher_logits\gend_clip_l14`
  (43 MB). The reports are `teacher_cache_full_20261001-133330.json`
  and `teacher_cache_full_20261001-140456.json` (rerun).
- The trial logged a metrics crash: its first 2048 rows are all fake,
  so AUC is undefined. Logging now prints `n/a` instead.

| split | frames / videos | time | frame AUC | video AUC | acc@0.5 | mean P(fake) real / fake | AUC DF / F2F / FS / NT | consolidated SHA-256 |
|---|---|---|---|---|---|---|---|---|
| train | 57,040 / 3,565 | 1540 s | 0.9781 | 0.9933 | 0.9395 | 0.165 / 0.906 | 0.995 / 0.983 / 0.990 / 0.945 | `eab3dfbe35897af37b473f36d1348c093b04a55bd031eef00c76ec7c56bb8f69` |
| val | 11,120 / 695 | 301 s | 0.9598 | 0.9792 | 0.9174 | 0.197 / 0.891 | 0.985 / 0.965 / 0.983 / 0.906 | `97ff4ddba7ae1b1740ca8ef3d8fc688ef917f1187269d66fae940266e7637355` |

- Crop manifests were verified against the Phase 5d summary: train
  `38635dd3…`, val `5faf4a3d…`.
- The rerun was byte-identical. `test_split_touched: false`.
- Teacher fits train better than val (expected, since it was trained on
  FF++ train frames). Its weakest method is NeuralTextures. These are
  distillation targets, not results for our model.

## 2026-10-01 — Phase 6a tests

```
.venv/Scripts/python.exe -m pytest tests/teacher -q                 # 10 passed (real-teacher test ran, not skipped)
.venv/Scripts/python.exe -m pytest -q -p no:cacheprovider -rs       # 513 passed, 0 skipped, 214.35 s
```

---

## 2026-10-01 — Phase 6b setup measurements

- Crop loading (one process): reading takes 2,652 files/s, and
  read + PNG decode runs at 358 img/s.
- DataLoader steady state: 974 img/s with 8 workers and 1,349 img/s
  with 12. Windows spawn start-up costs 44 s and 67 s respectively,
  paid once per persistent loader.
- Train-step micro-benchmark (MobileNetV4-Conv-Small, bs 64, fp16
  autocast, synthetic tensors):

  | format | cudnn.benchmark off | on |
  |---|---|---|
  | channels_last | 370 img/s | 368 img/s |
  | NCHW | 1,063 img/s | 1,033 img/s |

  `channels_last` is about 3× slower here, so it was removed after the
  pilot.
- The pilot ran with channels_last. Memory format does not change the
  algorithm, so the α/T choice still holds.
- Both full runs used NCHW. Steady state was about 750 img/s, limited
  by data loading.

## 2026-10-01 — Phase 6b pilot (α / T selection)

```
.venv/Scripts/python.exe scripts/train_distill_student.py pilot --epochs 4 --samples-per-epoch 25000
```
- Budget: 4 epochs × 25,000 balanced draws, with a full cosine schedule.
- Grid: α ∈ {0.5, 0.9} × T ∈ {1, 2, 4}, plus a baseline. Same seed,
  data and augmentation for every run.
- Selection rule: highest val frame AUROC among the distilled runs;
  ties go to lower val NLL.
- Report: `D:\ConfiGuard-Data\checkpoints\distill\pilot\pilot_report_20261001-142414.json`.

| run | α | T | frame AUROC | video AUROC | frame NLL | frame ECE | frame AUROC DF/F2F/FS/NT |
|---|---|---|---|---|---|---|---|
| pilot_baseline | 0 | – | 0.9496 | 0.9724 | 0.3012 | 0.0614 | 0.968/0.964/0.973/0.893 |
| pilot_a0.5_t1 | 0.5 | 1 | 0.9421 | 0.9660 | 0.2684 | 0.0542 | 0.965/0.960/0.965/0.877 |
| **pilot_a0.5_t2** | 0.5 | 2 | **0.9468** | 0.9718 | **0.2533** | **0.0493** | 0.970/0.962/0.970/0.885 |
| pilot_a0.5_t4 | 0.5 | 4 | 0.9427 | 0.9657 | 0.2681 | 0.0589 | 0.965/0.958/0.971/0.876 |
| pilot_a0.9_t1 | 0.9 | 1 | 0.9388 | 0.9642 | 0.2737 | 0.0627 | 0.964/0.958/0.970/0.863 |
| pilot_a0.9_t2 | 0.9 | 2 | 0.9357 | 0.9596 | 0.2778 | 0.0593 | 0.962/0.954/0.972/0.855 |
| pilot_a0.9_t4 | 0.9 | 4 | 0.9355 | 0.9615 | 0.2770 | 0.0550 | 0.961/0.956/0.972/0.853 |

- Chosen: **α 0.5, T 2**.
- α 0.9 (teacher-dominated) is worse on every method.
- At the pilot budget the baseline already edges the best distilled run
  on AUROC, while distillation wins on NLL and ECE.

## 2026-10-01 — Phase 6b full runs (identical settings; only α/T differ)

```
.venv/Scripts/python.exe scripts/train_distill_student.py train --run-name student_baseline --alpha 0 --temperature 1
.venv/Scripts/python.exe scripts/train_distill_student.py train --run-name student_distilled --alpha 0.5 --temperature 2
.venv/Scripts/python.exe scripts/compare_students.py --runs student_baseline,student_distilled
```
- Config: `configs/distill/mobilenetv4_student.yaml`.
  - Seed 42; MobileNetV4-Conv-Small (ImageNet, 2,494,305 params).
  - bs 64; AdamW lr 3e-4, wd 0.05; 300 warm-up steps, then cosine over
    20 epochs.
  - fp16 AMP (init scale 1024); grad clip 1.0.
  - 57,040 balanced draws per epoch: real 1/2, each method 1/8.
  - Augmentation (train only, every class): blur p 0.3, σ 0.3–1.0;
    horizontal jitter p 0.5, x-scale 0.95–1.05, x-shift ±4 px.
  - Early stopping on val frame AUROC (tiebreak lower NLL), patience 3.
- Verified identical across the two runs (tested): initial weights, the
  per-epoch draws and the augmented pixels.
- Baseline: best epoch 13 of 17 run, 25.3 min. Distilled: best epoch 11
  of 15 run, 22.6 min. 0 skipped AMP steps.
- Peak train VRAM 604 MB for both.
- Data: Phase 5d train crops (57,040) and val crops (11,120 frames /
  695 videos). Teacher margins come from cache `t6a-f87ebb7553a64e99`.
  GenD was never loaded, and test was never opened.
- Run dirs: `D:\ConfiGuard-Data\checkpoints\distill\student_{baseline,distilled}`.
  Comparison report: `...\distill\reports\compare_20261001-160341.json`.

**Validation, frame level (11,120 frames):**

| model | AUROC | AUPRC | acc@0.5 | bal-acc@0.5 | ECE | Brier | NLL | AUROC DF/F2F/FS/NT | AUPRC DF/F2F/FS/NT |
|---|---|---|---|---|---|---|---|---|---|
| baseline (α 0) | **0.9631** | **0.9905** | 0.9169 | 0.8847 | 0.0713 | 0.0751 | 0.5039 | 0.976/**0.974**/0.978/**0.924** | 0.977/**0.978**/0.978/**0.937** |
| distilled (α 0.5, T 2) | 0.9562 | 0.9881 | **0.9237** | **0.8930** | **0.0329** | **0.0610** | **0.2157** | **0.978**/0.967/**0.980**/0.900 | **0.980**/0.966/**0.981**/0.883 |
| GenD teacher (cached, ref.) | 0.9598 | 0.9891 | 0.9174 | 0.9005 | 0.0479 | 0.0639 | 0.2201 | 0.985/0.965/0.983/0.906 | 0.984/0.966/0.982/0.909 |

**Validation, video level (mean frame logit, 695 videos):**

| model | AUROC | AUPRC | acc@0.5 | bal-acc@0.5 | ECE | Brier | NLL | AUROC DF/F2F/FS/NT | AUPRC DF/F2F/FS/NT |
|---|---|---|---|---|---|---|---|---|---|
| baseline (α 0) | **0.9808** | **0.9955** | 0.9396 | 0.9137 | 0.0520 | 0.0528 | 0.2887 | 0.991/0.988/0.994/**0.950** | 0.993/0.991/0.995/**0.962** |
| distilled (α 0.5, T 2) | 0.9791 | 0.9950 | **0.9468** | **0.9263** | **0.0443** | **0.0432** | **0.1632** | **0.992**/**0.989**/**0.995**/0.941 | 0.993/0.991/**0.996**/0.950 |
| GenD teacher (cached, ref.) | 0.9792 | 0.9950 | 0.9338 | 0.9263 | 0.0500 | 0.0471 | 0.1692 | 0.994/0.983/0.998/0.941 | 0.996/0.987/0.998/0.949 |

- **Paired video bootstrap** (2,000 stratified resamples): Δ AUROC
  (distilled − baseline) = −0.0016, 95% CI [−0.0079, +0.0040],
  P(Δ ≤ 0) = 0.71. Not significant.
- **Efficiency:** both students are the same architecture.

  | | baseline | distilled |
  |---|---|---|
  | `best.pt` (fp32 weights + metadata) | 9.7 MiB | 9.7 MiB |
  | CPU bs 1 (8 threads) | 22.7 ms | 23.2 ms |
  | CPU bs 32 | 179 ms | 178 ms |
  | GPU fp32 bs 1 | 16.0 ms | 16.5 ms |
  | GPU fp16 bs 64 | 3,449 img/s | 3,450 img/s |
  | Peak inference VRAM (bs 64) | 179 MB | 179 MB |
  | Peak train VRAM | 604 MB | 604 MB |

  For reference, the GenD teacher is 1.216 GB and runs at 90 img/s
  (fp16, 1.84 GiB peak; Phase 6a).
- **Reading:**
  - Distillation does not improve ranking. Video AUROC is tied and
    frame AUROC is 0.007 lower, mostly on NeuralTextures and Face2Face.
  - It roughly halves NLL and ECE: frame NLL 0.50 → 0.22, ECE 0.071 →
    0.033.
  - It improves balanced accuracy at 0.5 (frame +0.008, video +0.013).
  - The baseline keeps sharpening after AUROC plateaus, and its val NLL
    rose from 0.30 to 0.60 over training.
  - The distilled student matches or slightly beats the teacher on
    every aggregate except NeuralTextures frame AUPRC.
- **Caveat:** the same val split chose α/T, did early stopping, and is
  reported here. These are optimistic, model-selection numbers, not
  held-out estimates. The test split stays untouched for a later
  phase.

## 2026-10-01 — Phase 6b tests

```
.venv/Scripts/python.exe -m pytest tests/distill -q -p no:cacheprovider    # 15 passed
.venv/Scripts/python.exe -m pytest -q -p no:cacheprovider -rs          # 528 passed, 0 skipped, 215.00 s
```

---

## 2026-10-01 — Phase 6c partitions (official TRAIN families only)

```
.venv/Scripts/python.exe scripts/calibrate_student.py split
```
- File: `D:\ConfiGuard-Data\cache\calibration_splits\p5d-b451b5ca770c8923\partitions_seed42.json`,
  SHA-256 `e2deca855599e93d629fb3cb197b2cb69bec4af486ec773b70c063f04cc763a0`.
- Built from `crops_train.jsonl` `38635dd3…`; seed 42.
- Unit of assignment: a donor-linked family component. FF++ donors are
  reciprocal pairs, giving 360 components of 1–2 families. Components
  are ordered by SHA-256(seed:key) and cut 288/36/36.

| partition | families | videos | frames | per class (real / each method) |
|---|---|---|---|---|
| final_train | 570 | 2,850 | 45,600 | 9,120 |
| temp_cal | 72 | 360 | 5,760 | 1,152 |
| conformal_cal | 71 | 355 | 5,680 | 1,136 |

## 2026-10-01 — Phase 6c retrain on final_train (fixed Phase 6b settings)

```
.venv/Scripts/python.exe scripts/train_distill_student.py train --run-name student_distilled_p80 --alpha 0.5 --temperature 2 --train-partition final_train
```
- Unchanged from 6b: `configs/distill/mobilenetv4_student.yaml` with
  α 0.5, T 2. Only the train rows differ (45,600; one draw per row per
  epoch, as in 6b).
- Early stopping on official val, as in 6b. Best epoch 18 of 20 run;
  24.1 min.
- `best.pt` SHA-256 `03f648b166135ff78a300f2c48daf88cd76c966880198f63865f4be9b0798957`.
- Dev (official val) discrimination: frame AUROC 0.9488 / AUPRC 0.9866;
  video AUROC 0.9735 / AUPRC 0.9937.
- For comparison, the 6b distilled model trained on 100% of train
  scored 0.9562 / 0.9791.
- On held-out train families the same model scores higher:
  - temp_cal: frame 0.9609, video 0.9895;
  - conformal_cal: frame 0.9647, video 0.9876.

## 2026-10-01 — Phase 6c calibration fit and development results

```
.venv/Scripts/python.exe scripts/calibrate_student.py fit --run student_distilled_p80
```
- Fitting data:
  - temperature on `temp_cal` only (5,760 frames / 360 videos);
  - conformal thresholds on `conformal_cal` only (5,680 frames / 355
    videos; 71 real + 284 fake videos), using temperature-scaled
    probabilities.
- Frame/image and video levels are fitted separately; video score is
  the mean frame logit.
- Development data: official val (11,120 frames / 695 videos), never
  used for fitting. Test is sealed.
- Artifact: `...\student_distilled_p80\calibration.json`.
  - File SHA-256 `86b355c0491ed0b919aedbd9c273b76a87fe3283bb4c2881c2859288f3ffd6be`.
  - Content SHA-256 `8df62834…c03c`.
  - Model config SHA-256 `d8e7c086…26ba`.
  - It was round-trip loaded with all checks.
- Report: `...\calibration_report.json`.

**Raw vs temperature-scaled (dev).** Temperature scaling never changes
decisions at 0.5 or the ranking, so accuracy and AURC are unchanged.

| level | T | | ECE | NLL | Brier | acc@0.5 | AURC |
|---|---|---|---|---|---|---|---|
| frame | 0.947 | raw | 0.0354 | 0.2427 | 0.0706 | 0.9100 | 0.0220 |
| | | temp-scaled | 0.0363 | 0.2448 | 0.0710 | 0.9100 | 0.0219 |
| video | 0.623 | raw | 0.0428 | 0.1826 | 0.0524 | 0.9295 | 0.0109 |
| | | temp-scaled | 0.0348 | 0.1883 | 0.0552 | 0.9295 | 0.0109 |

- On `temp_cal` itself, video TS improves ECE 0.061 → 0.037 and NLL
  0.132 → 0.110. The frame level is already near-calibrated
  (T ≈ 0.95).
- The video gain does not fully transfer to the harder val split:
  ECE improves, but NLL and Brier get slightly worse.

**Risk-coverage (dev, abstain on least confident first; selective risk
at coverage):**

| coverage | 0.24 | 0.43 | 0.62 | 0.81 | 0.90 | 0.95 | 1.00 |
|---|---|---|---|---|---|---|---|
| frame | 0.0042 | 0.0090 | 0.0209 | 0.0429 | 0.0612 | 0.0729 | 0.0900 |
| video | 0.0000 | 0.0034 | 0.0023 | 0.0213 | 0.0366 | 0.0544 | 0.0705 |

**Conformal (dev).** Coverage = true label in the set. Abstain = the
"uncertain" verdict. The last column is plain confidence-ranked
abstention at the same abstention rate.

| level | mode | α | coverage | real cov | fake cov | abstain | selective acc | confidence-abstain acc |
|---|---|---|---|---|---|---|---|---|
| frame | mondrian | 0.01 | 0.983 | 0.989 | 0.982 | 0.449 | 0.969 | 0.985 |
| frame | **mondrian** | **0.05** | 0.929 | 0.950 | 0.924 | 0.125 | 0.919 | 0.946 |
| frame | mondrian | 0.10 | 0.871 | 0.888 | 0.867 | 0.026 | 0.895 | 0.920 |
| frame | marginal | 0.05 | 0.932 | 0.860 | 0.950 | 0.051 | 0.928 | 0.928 |
| video | mondrian | 0.01 | 0.981 | 1.000 | 0.977 | 0.850 | 0.875 | 1.000 |
| video | **mondrian** | **0.05** | 0.927 | 0.978 | 0.914 | 0.058 | 0.922 | 0.950 |
| video | mondrian | 0.10 | 0.866 | 0.863 | 0.867 | 0.084 | 0.945 | 0.959 |
| video | marginal | 0.05 | 0.919 | 0.820 | 0.944 | 0.016 | 0.934 | 0.934 |

- On `conformal_cal` itself, coverage is nominal: mondrian α 0.05 gives
  frame 0.951 and video 0.958.
- On dev it is 2–3 points short. Val is harder than held-out train
  families, so exchangeability between them does not hold (see
  `KNOWN_ISSUES.md`).
- **Default verdicts (video, mondrian α 0.05) on 695 dev videos:**
  180 likely real, 475 likely manipulated, 40 uncertain.
  - 2.2% of real videos are wrongly called "likely manipulated".
  - Marginal α 0.05 would wrongly flag 18% of real videos.
- **Mondrian trade-off:**
  - Per-class coverage protects reals.
  - Decided cases are less accurate than with confidence-ranked
    abstention at the same rate (video 0.922 vs 0.950). They are even
    below no abstention (0.930).
  - Cause: with q_fake 0.12, any p < 0.88 that does not fall in the
    uncertain band gets a real-only set. Fakes with moderate scores
    become "likely real" (missed detections, not false accusations).
- Video α 0.01 is not supported: 71 real calibration videos is below
  the 99 needed, so q_real = 1 and 85% of videos become uncertain.

## 2026-10-01 — Phase 6c tests

```
.venv/Scripts/python.exe -m pytest tests/calibration tests/distill -q -p no:cacheprovider   # 9 + 16 passed
.venv/Scripts/python.exe -m pytest -q -p no:cacheprovider -rs                               # 538 passed, 0 skipped, 213.56 s
```
- New tests: `tests/calibration/test_calibration.py` (9 tests) and 1
  trainer-partition test in `tests/distill` (16 there now).

---

## 2026-10-01 — Phase 6d adaptive 4/8/16-frame video inference

```
.venv/Scripts/python.exe scripts/adaptive_video_eval.py --run student_distilled_p80
```
- **Model:** `student_distilled_p80` (`best.pt` `03f648b1…8957`).
- **Frame logits:** the per-frame logits cached in Phase 6c were reused
  for temp_cal, conformal_cal and val. Nothing was re-scored.
- **Stage score:** stage k uses the mean logit over the nested k-set
  (4 = slots 0/4/8/12, 8 = even slots, 16 = all), checked against
  every row's `nested_levels`.
- **Artifact:** `adaptive_calibration.json`.
  - File SHA-256 `790c12166e583aaeef9fba7d96b60fbd7d9fc9802c9da8c76fb7571341c7f2a2`.
  - Content SHA-256 `1bae6e40…74ab`.
  - Bound to `best.pt`. Loading it against the 6b checkpoint raised
    `CalibrationMismatchError`, and the 6c `calibration.json` still
    loads.
- **Report:** `...\student_distilled_p80\adaptive_report.json`.

**Per-stage calibration.** Temperature is fitted on temp_cal (360
videos) and mondrian thresholds on conformal_cal (355 videos: 71 real,
284 fake).

| stage | T | temp_cal ECE raw → scaled | q_real / q_fake at spent α | q_real / q_fake at α 0.05 |
|---|---|---|---|---|
| 4 frames | 0.723 | 0.053 → 0.034 | 0.970 / 0.783 (α 0.015) | 0.915 / 0.241 |
| 8 frames | 0.675 | 0.052 → 0.034 | 0.977 / 0.850 (α 0.015) | 0.950 / 0.147 |
| 16 frames | 0.623 | 0.061 → 0.037 | 0.974 / 0.674 (α 0.02) | 0.963 / 0.123 |

**Policy:**
- Stop at 4 or 8 frames only on a singleton set; otherwise escalate.
- At 16 frames, return the singleton verdict or "uncertain".
- α spending is 0.015 + 0.015 + 0.02 = 0.05 (union bound). The minimum
  supported α is 1/72 ≈ 0.0139, set by 71 real calibration videos.

**Development results (official val, 695 videos).** FPR is the share
of real videos called "likely manipulated"; miss is the share of fakes
called "likely real"; coverage means the true label is in the final
set.

| inference | AUROC | avg frames | uncertain | decided acc | FPR | miss | detection | coverage |
|---|---|---|---|---|---|---|---|---|
| fixed 4 (α 0.05) | 0.9721 | 4.00 | 0.047 | 0.9335 | 0.0360 | 0.0701 | 0.8831 | 0.9367 |
| fixed 8 (α 0.05) | 0.9730 | 8.00 | 0.060 | 0.9296 | 0.0288 | 0.0755 | 0.8597 | 0.9338 |
| fixed 16 (α 0.05) | **0.9735** | 16.00 | 0.058 | 0.9221 | 0.0216 | 0.0863 | 0.8489 | 0.9266 |
| **adaptive (spent)** | 0.9731 | **6.19** | 0.132 | **0.9585** | **0.0144** | **0.0414** | 0.8363 | **0.9640** |
| adaptive, unspent ablation (α 0.05 every stage) | 0.9717 | 4.44 | 0.022 | 0.9206 | 0.0432 | 0.0863 | 0.8903 | 0.9223 |

- **Targets:**
  - Frame reduction is 61.3% vs fixed-16 (target ≥ 40%).
  - FPR is 1.44% vs 2.16% for fixed-16 (target: no more than +1 pp).
    **Both met.**
- **Stopping:**
  - 556 videos stopped at 4 frames and 18 at 8 (confident singleton).
  - 29 reached 16 frames and got a singleton.
  - 92 reached 16 frames and ended uncertain (both labels in the set).
- **Unspent ablation:** without α spending, early stopping doubles the
  FPR (4.3%). The spending is what keeps false accusations down.
- **Cost:** adaptive abstains more (13.2% vs 5.8%) and detects slightly
  fewer fakes outright (83.6% vs 84.9%). It halves outright misses
  (4.1% vs 8.6%), and the extra uncertain verdicts absorb the hard
  cases.
- **On conformal_cal (in-sample sanity check):** fixed-k coverage is
  0.958 by construction. Adaptive coverage is 0.986 with FPR 0, which
  shows how conservative the union bound is.

**Per manipulation (dev videos, 139 each):**

| class | fixed-16: uncertain / decided acc / detection (FPR for originals) / AUROC | adaptive: frames / uncertain / decided acc / detection (FPR) / AUROC |
|---|---|---|
| original | 0.029 / 0.978 / FPR 0.022 / – | 6.99 / 0.173 / 0.983 / FPR 0.014 / – |
| Deepfakes | 0.065 / 0.946 / 0.885 / 0.988 | 5.41 / 0.086 / 0.976 / 0.892 / 0.989 |
| Face2Face | 0.014 / 0.949 / 0.935 / 0.985 | 5.12 / 0.065 / 0.977 / 0.914 / 0.983 |
| FaceSwap | 0.014 / 0.964 / 0.950 / 0.994 | 4.75 / 0.043 / 0.993 / 0.950 / 0.995 |
| NeuralTextures | 0.165 / 0.750 / 0.626 / 0.927 | 8.69 / 0.295 / 0.837 / 0.590 / 0.926 |

**Latency (live, all 695 dev videos).** Scope: read + PNG decode of the
needed crops, student forward, and the decision. Face detection and
alignment are excluded. Live verdicts and frame counts match the
simulation: 100% on GPU and 99.86% on CPU (1 borderline video;
fp32 vs fp16 logits).

| device | adaptive P50 / P95 / mean | fixed-16 P50 / P95 / mean |
|---|---|---|
| RTX 4050 (fp16, batch = new frames per stage) | 32.7 / 111.3 / 47.0 ms | 81.2 / 84.8 / 80.3 ms |
| CPU (8 threads, fp32) | 50.2 / 183.2 / 74.6 ms | 139.6 / 177.1 / 146.3 ms |

- Median and mean latency drop by about 40–60%.
- P95 is slightly worse than fixed-16: an escalated video makes three
  sequential calls (4 + 4 + 8 frames) instead of one batch of 16.

## 2026-10-01 — Phase 6d tests

```
.venv/Scripts/python.exe -m pytest tests/adaptive tests/calibration -q -p no:cacheprovider   # 17 passed
.venv/Scripts/python.exe -m pytest -q -p no:cacheprovider -rs                               # 546 passed, 0 skipped, 212.52 s
```

---

## 2026-10-01 — Phase 6e robust augmentation and stress suite

**Training augmentation** (`configuard.robust.degrade.RobustAugmentConfig`
defaults). Applied after the unchanged 6b blur/jitter, with the same
probabilities and severities for every class and method (the function
never sees the label).

| op | probability | mild → moderate range |
|---|---|---|
| gamma | 0.3 | γ ≈ 0.95–1.05 → 0.80–1.25 |
| down/up-scale | 0.3 | 0.9× → 0.45× |
| blur | 0.2 | σ 0.3 → 1.5 |
| Gaussian noise | 0.2 | σ 1 → 8 (uint8 levels) |
| JPEG | 0.35 | quality 90 → 40 |
| H.264-style emulation | 0.3 | QP 22 → 36 |

- JPEG and H.264 are mutually exclusive.
- The H.264 emulation is 4:2:0 chroma + 4×4 DCT quantisation + light
  deblocking (PSNR 39.6 / 34.9 / 30.7 dB at QP 22 / 30 / 36).
- Curriculum: the severity cap is 0.5 at epoch 0 and rises linearly to
  1.0 at epoch 4.
- GenD targets stay the CLEAN cached logits; only the student's view is
  degraded.
- Cost: about 3.4 ms/crop average. Measured by micro-benchmark after
  replacing einsum (16 ms) with a block-diagonal DCT (5.7 ms for the
  H.264 op).

**Stress suite** (`scripts/robust_eval.py build`):
- Tag `p6e-2974c51936f46856`, at `D:\ConfiGuard-Data\cache\robust_stress\`.
- 17 deterministic conditions × 11,120 official-val crops, 12.2 GiB in
  about 6.5 min. D: free went 75.6 → 63.3 GB (floor 40).
- H.264 conditions use **real libx264** (preset medium, yuv420p,
  threads 1) over each video's 16 crops; a round-trip is
  byte-deterministic.
- Seven conditions are outside the training range (marked *).

**Robust student:**

```
.venv/Scripts/python.exe scripts/train_distill_student.py train --run-name student_distilled_robust_p80 --alpha 0.5 --temperature 2 --train-partition final_train --robust
```
- Setup: the fixed 6b config (α 0.5, T 2, early stopping on clean val,
  patience 3) on the same 80% `final_train` partition as the current
  model `student_distilled_p80`.
- `best.pt` SHA-256 `5919c0a1…1c40`.
- **First attempt stopped and restarted.** With the new numpy/OpenCV
  degradations, 12 spawn workers × 16 BLAS/OpenCV threads
  oversubscribed the CPU: 236 img/s, GPU 6–38%. It was stopped before
  any epoch was written. Workers are now single-threaded
  (`single_thread_workers` + `worker_init`), giving 1,037 img/s. The
  results do not depend on this.
- **Training cost:**

  | model | epochs | best epoch | minutes (train+val) | median img/s | peak VRAM |
  |---|---|---|---|---|---|
  | current p80 | 20 | 18 | 23.8 | 779 | 604 MB |
  | robust p80 | 7 (early stop) | 3 | 12.9 | 566 | 604 MB |

- The robust run stopped at epoch 6, two epochs after the curriculum
  reached full severity. That is likely premature: clean-val early
  stopping was not designed for a curriculum.

**Evaluation:**

```
.venv/Scripts/python.exe scripts/robust_eval.py evaluate --runs student_distilled_p80,student_distilled_robust_p80 --workers 4 --ram-floor-gb 4
.venv/Scripts/python.exe scripts/compare_students.py --runs student_distilled_p80,student_distilled_robust_p80
```
- **A first attempt was killed by Claude Code for low system RAM.** It
  used 12 workers and scored all conditions in one pass, and saved
  nothing.
- It was resumed with 4 workers, sequential per-condition scoring,
  per-condition `.npy` saved immediately (`<run>\stress\<suite>\<ckpt12>\`)
  and a 4 GB available-RAM floor (`configuard.memory_guard`).
- Available RAM stayed at or above 5.6 GB. The two models took 889 s
  and 924 s.
- Report: `...\distill\reports\robust_eval_20261001-184550.json`;
  latency in `compare_20261001-184647.json`.

**Video / frame AUROC and video FPR** (real videos with raw P(fake) ≥
0.5; both models are uncalibrated here):

| condition | severity | current vAUC | robust vAUC | current fAUC | robust fAUC | current FPR@0.5 | robust FPR@0.5 |
|---|---|---|---|---|---|---|---|
| **clean** | – | **0.9735** | 0.9274 | **0.9488** | 0.8863 | 0.166 | 0.166 |
| jpeg_q75 | mild | 0.9454 | 0.9216 | 0.9119 | 0.8786 | 0.050 | 0.180 |
| jpeg_q50 | moderate | 0.9037 | 0.9140 | 0.8581 | 0.8670 | 0.000 | 0.180 |
| jpeg_q30 | severe* | 0.8641 | 0.9005 | 0.8055 | 0.8497 | 0.000 | 0.187 |
| x264_crf23 | mild | 0.9427 | 0.9085 | 0.9096 | 0.8665 | 0.151 | 0.259 |
| x264_crf30 | moderate | 0.9037 | 0.8911 | 0.8660 | 0.8471 | 0.122 | 0.273 |
| x264_crf37 | severe* | 0.8548 | 0.8630 | 0.8083 | 0.8211 | 0.144 | 0.317 |
| resize_0.75 | mild | 0.9619 | 0.9147 | 0.9302 | 0.8713 | 0.237 | 0.281 |
| resize_0.5 | moderate | 0.9560 | 0.9022 | 0.9168 | 0.8568 | 0.345 | 0.302 |
| resize_0.33 | severe* | 0.8883 | 0.8620 | 0.8362 | 0.8137 | **0.849** | 0.410 |
| blur_s1.0 | moderate | 0.9560 | 0.9029 | 0.9219 | 0.8570 | 0.237 | 0.281 |
| blur_s2.0 | severe* | 0.8212 | 0.8250 | 0.7729 | 0.7810 | **1.000** | 0.540 |
| noise_s4 | moderate | 0.8945 | 0.9174 | 0.8461 | 0.8730 | 0.007 | 0.223 |
| noise_s10 | severe* | **0.6922** | 0.8841 | 0.6650 | 0.8345 | 0.000 | 0.259 |
| gamma_0.7 | severe* | 0.9720 | 0.9271 | 0.9455 | 0.8855 | 0.187 | 0.209 |
| gamma_1.4 | severe* | 0.9628 | 0.9056 | 0.9374 | 0.8659 | 0.259 | 0.252 |
| social_resize0.5_jpeg60 | moderate | 0.8955 | 0.8945 | 0.8432 | 0.8454 | 0.094 | 0.245 |
| stream_resize0.5_x264crf30 | moderate | 0.8890 | 0.8698 | 0.8325 | 0.8247 | 0.252 | 0.317 |

\* outside the training augmentation range.

**Summary (video AUROC):**

| | current p80 | robust p80 |
|---|---|---|
| clean | **0.9735** | 0.9274 |
| worst case | 0.6922 (noise_s10) | **0.8250** (blur_s2.0) |
| mean over 17 degraded | **0.9002** | 0.8943 |
| mild / moderate / severe mean | **0.950 / 0.914** / 0.865 | 0.915 / 0.899 / **0.881** |
| worst-case ΔAUROC vs own clean | −0.281 | −0.102 |
| max / mean degraded FPR@0.5 | 1.000 / 0.231 | 0.540 / 0.277 |

- Paired video bootstrap, clean (robust − current): Δ −0.046, 95% CI
  [−0.062, −0.032].
- **Per manipulation, video AUROC:**

| condition | current DF / F2F / FS / NT | robust DF / F2F / FS / NT |
|---|---|---|
| clean | 0.988 / 0.985 / 0.994 / 0.927 | 0.962 / 0.963 / 0.961 / 0.824 |
| jpeg_q50 | 0.963 / 0.940 / 0.987 / 0.724 | 0.958 / 0.954 / 0.953 / 0.791 |
| x264_crf30 | 0.945 / 0.930 / 0.988 / 0.752 | 0.943 / 0.936 / 0.947 / 0.738 |
| x264_crf37 | 0.920 / 0.865 / 0.961 / 0.672 | 0.930 / 0.891 / 0.924 / 0.707 |
| resize_0.5 | 0.982 / 0.966 / 0.987 / 0.890 | 0.939 / 0.945 / 0.941 / 0.783 |
| blur_s2.0 | 0.907 / 0.821 / 0.861 / 0.696 | 0.883 / 0.847 / 0.873 / 0.696 |
| noise_s10 | 0.801 / 0.720 / 0.671 / 0.578 | 0.948 / 0.934 / 0.935 / 0.720 |
| social_resize0.5_jpeg60 | 0.969 / 0.923 / 0.929 / 0.761 | 0.943 / 0.938 / 0.942 / 0.755 |
| **worst over 17** | 0.801 / 0.720 / 0.671 / 0.578 | 0.883 / 0.847 / 0.873 / 0.696 |
| **mean over 17** | 0.950 / 0.918 / 0.944 / 0.790 | 0.941 / 0.933 / 0.939 / 0.764 |

- **Latency, size and VRAM:** the architecture is identical.
  - `best.pt` 9.7 MiB for both.
  - CPU bs 1: 27.5 vs 38.5 ms, and GPU fp16 bs 64: 2,602 vs 2,871
    img/s. Both differences are run-to-run noise on a busy laptop,
    since the networks are the same.
  - Inference VRAM is the same as Phase 6b.

**Decision rule:** prefer robust only if worst-case AND mean degraded
video AUROC improve, and clean video AUROC loss ≤ 0.01. Result:
worst-case improves, mean does not, and the clean loss is 0.046.
**Keep the current model** (`student_distilled_p80`) as the production
default. The robust model is recorded as an experiment only.

**Calibration:** `calibration.json` and `adaptive_calibration.json`
(bound to the current checkpoint) both refuse the robust checkpoint
with `CalibrationMismatchError`. Any retrained model must be
recalibrated before use.

**Findings:**
1. The current model has a blur/downscale → "fake" shortcut. With
   σ 2 blur, all real videos score ≥ 0.5, and at 0.33× downscale, 85%
   do. Fakes are blurrier in FF++ (Phase 5d). Additive noise pushes it
   the other way (FPR 0, AUROC 0.69).
2. Robust training halves those failure FPRs and lifts the worst case
   by 0.13. It wins on strong JPEG and noise, but costs about 0.05 clean
   and mild-condition AUROC. NeuralTextures suffers most (clean 0.927 →
   0.824).
3. The comparison is confounded by the early stop at epoch 6. A fair
   robust run needs early stopping that waits for the curriculum (or
   selects on a degraded dev set), not a new hyperparameter search.

## 2026-10-01 — Phase 6e tests

```
.venv/Scripts/python.exe -m pytest tests/robust tests/distill tests/calibration tests/adaptive -q -p no:cacheprovider   # 40 passed
.venv/Scripts/python.exe -m pytest -q -p no:cacheprovider -rs     # 553 passed, 0 skipped, 271.94 s
```

---

## 2026-10-01 — Phase 7 GRU temporal head (frozen student_distilled_p80)

```
.venv/Scripts/python.exe scripts/temporal_gru.py extract     # 4 workers, 4 GB RAM floor, resumable per set
.venv/Scripts/python.exe scripts/temporal_gru.py train
.venv/Scripts/python.exe scripts/temporal_gru.py evaluate
```

**Embeddings:**
- The frozen p80 student (`best.pt` `03f648b1…`) produced 1280-d
  pooled features plus frame logits, in manifest order. They are cached
  at `D:\ConfiGuard-Data\cache\temporal_embeddings\03f648b16613\`
  (385 MB, float16 features).
- Sets: train (all 57,040 crops, every partition), val (11,120) and 8
  val stress conditions (blur σ1/σ2, resize 0.5/0.33, noise σ4/σ10,
  x264 CRF 30/37).
- Each set took 80–300 s; minimum available RAM was 5.7 GB.
- The cached val logits equal the Phase 6c val logits exactly (max
  |Δ| 0.0), so the baseline uses identical frame scores.
- Test split: never read (`extract` refuses test rows).

**Head:**
- Residual: `mean(frame logits) + Linear(128→1)(GRU(Linear 1280→64 →
  GELU))`, with one GRU layer, hidden 128 and 156,609 parameters
  (0.60 MB fp32).
- The output layer is zero-initialised, so the untrained head equals
  the current aggregation.
- Training: `final_train` videos only (2,850; the student's own
  training partition; calibration partitions untouched) with nested
  k ∈ {4, 8, 16} drawn per batch.
- Class-weighted BCE; AdamW lr 1e-3, wd 1e-2; bs 64; dropout 0.2;
  seed 42. Early stopping on val video AUROC (k16), patience 5.
- 6 epochs in 0.39 min. Best epoch 0. Train loss went 0.052 → 0.012,
  because the student's features on its own training videos are
  near-separable.
- Head: `D:\ConfiGuard-Data\checkpoints\temporal\gru_p80\best.pt`
  (`51b299e7…`); report `gru_eval_20261001-205208.json`.

**Clean val (695 videos, same videos for both):**

| aggregation | AUROC | AUPRC | FPR@0.5 | TPR@0.5 | balanced acc@0.5 | FPR@TPR 0.90 | AUROC DF / F2F / FS / NT |
|---|---|---|---|---|---|---|---|
| mean logit k4 | 0.9721 | 0.9934 | 0.144 | 0.944 | 0.9002 | 0.050 | 0.989 / 0.984 / 0.995 / 0.921 |
| GRU k4 | 0.9736 | 0.9937 | 0.288 | 0.969 | 0.8408 | 0.050 | 0.988 / 0.985 / 0.993 / 0.929 |
| mean logit k8 | 0.9730 | 0.9936 | 0.158 | 0.950 | 0.8957 | 0.036 | 0.988 / 0.986 / 0.994 / 0.925 |
| GRU k8 | 0.9738 | 0.9937 | 0.266 | 0.969 | 0.8516 | 0.050 | 0.987 / 0.986 / 0.990 / 0.932 |
| **mean logit k16 (current)** | 0.9735 | 0.9937 | **0.165** | 0.953 | **0.8939** | 0.029 | 0.988 / 0.985 / 0.994 / 0.927 |
| GRU k16 | 0.9749 | 0.9939 | 0.273 | 0.971 | 0.8489 | 0.029 | 0.988 / 0.987 / 0.993 / 0.931 |

- Paired video bootstrap, k16 (GRU − mean): ΔAUROC +0.0014, 95% CI
  [−0.0015, +0.0051].
- At the fixed 0.5 threshold, the GRU shifts scores towards "fake":
  more detections, but FPR 0.165 → 0.273. Both models are uncalibrated
  here.

**Stress subset (video AUROC at k16; FPR@0.5 mean → GRU):**

| condition | mean | GRU | FPR@0.5 | NT AUROC mean → GRU |
|---|---|---|---|---|
| blur σ1.0 | 0.9560 | 0.9565 | 0.237 → 0.417 | 0.896 → 0.900 |
| blur σ2.0 | 0.8212 | 0.8211 | 1.000 → 1.000 | 0.696 → 0.696 |
| resize 0.5 | 0.9560 | 0.9561 | 0.345 → 0.540 | 0.890 → 0.891 |
| resize 0.33 | 0.8883 | 0.8883 | 0.849 → 0.921 | 0.824 → 0.826 |
| noise σ4 | 0.8945 | 0.8902 | 0.007 → 0.036 | 0.787 → 0.775 |
| noise σ10 | 0.6922 | 0.6811 | 0.000 → 0.050 | 0.578 → 0.571 |
| x264 CRF30 | 0.9037 | 0.9032 | 0.122 → 0.281 | 0.752 → 0.751 |
| x264 CRF37 | 0.8548 | 0.8511 | 0.144 → 0.266 | 0.672 → 0.669 |

Mean stress ΔAUROC: −0.0024.

**Cost (head only, k16, one video, 8 CPU threads):**
- Parameters: 156,609 (0.60 MB).
- CPU: GRU 1.98 / 2.24 ms (P50/P95) vs mean 0.02 ms. The CPU backbone
  for 16 frames takes 106 / 134 ms, so the overhead is 1.9%.
- GPU: GRU 1.24 / 1.68 ms vs fp16 backbone 22.8 / 27.1 ms, an
  overhead of 5.5%. Peak GPU memory for the head is 17 MB.
- RAM: CPU process RSS rose 2 MB during GRU inference.

**Pre-registered rule** (in `scripts/temporal_gru.py`, fixed before
results): select the GRU iff the improvement is meaningful AND clean
AUROC loss ≤ 10% AND head latency ≤ 10% of the backbone.
- "Meaningful" means clean Δ ≥ +0.005 with a CI lower bound > 0, or
  mean stress Δ ≥ +0.02 with clean loss ≤ 0.005.
- Result: clean +0.0014 (CI includes 0) and stress −0.0024, so it is
  not meaningful. The clean-loss and latency checks pass.
- **Rejected; the mean-frame-logit aggregation stays.**

## 2026-10-01 — Phase 7 tests

```
.venv/Scripts/python.exe -m pytest tests/temporal tests/robust -q -p no:cacheprovider    # 12 passed
.venv/Scripts/python.exe -m pytest -q -p no:cacheprovider -rs                         # 558 passed, 0 skipped, 300.56 s
```

---

## 2026-10-01 — Phase 8 ONNX export of student_distilled_p80

**Runtime change:**
- `onnxruntime` 1.30.0 (CPU) was replaced by `onnxruntime-gpu`
  1.23.2, which includes the CPU provider.
- ORT 1.30's CUDA provider needs CUDA 13 and failed here. 1.23.2 is
  built for CUDA 12 + cuDNN 9 and uses the DLLs bundled with torch
  2.5.1+cu121, provided torch is imported first. No system install.
- Checks: `verify_environment.py` gives Dependency safety OK (torch
  2.5.1+cu121, CUDA available), and `tests/models` passes 90/90.

```
.venv/Scripts/python.exe scripts/export_student_onnx.py build      # FP32, FP16, INT8 (MinMax) + INT8 (Percentile 99.999)
.venv/Scripts/python.exe scripts/export_student_onnx.py parity
.venv/Scripts/python.exe scripts/export_student_onnx.py evaluate
.venv/Scripts/python.exe scripts/export_student_onnx.py bench --videos 200
.venv/Scripts/python.exe scripts/export_student_onnx.py package
```

**Graph:**
- Input `pixels` float32 (B, 3, 224, 224), RGB 0..255; output `logit`
  (B,). ImageNet normalisation is inside the graph. Opset 17, dynamic
  batch.
- FP16 keeps fp32 I/O and fp32 normalisation, then casts to half.
- Package: `D:\ConfiGuard-Data\checkpoints\export\student_p80\`.

| file | size | SHA-256 |
|---|---|---|
| student_fp32.onnx | 9.49 MiB | `4e365f0d…0279` |
| student_fp16.onnx | 4.76 MiB | `244d317c…fb38` |
| student_int8.onnx (MinMax) | 2.67 MiB | `45f1f1c7…0a3f` |
| student_int8_percentile.onnx | 2.67 MiB | `6dc78740…d212` |

**INT8 calibration:**
- 510 official TRAIN crops from `final_train` (102 per class/method,
  seed 42; list hash `3650f206…`); no val or test.
- Static QDQ, per-channel int8 weights, uint8 activations, equal
  batches of 30.
- ORT 1.23's percentile collector fails on a short final batch, so
  the batch size divides the sample.

**Parity vs PyTorch FP32 eager (CPU), 256 val crops × {clean, blur σ1,
resize 0.5, noise σ4, x264 CRF30}** (max |Δlogit| / sign agreement,
range over conditions):

| variant | max \|Δlogit\| | sign agreement |
|---|---|---|
| ONNX FP32 CPU | 0.0000 | 100% |
| ONNX FP32 CUDA | 0.021–0.024 | 100% |
| ONNX FP16 CPU | 0.044–0.080 | 99.6–100% |
| ONNX FP16 CUDA | 0.068–0.090 | 99.2–100% |
| INT8 MinMax CPU | 4.6–6.1 | 86.3–91.4% |
| INT8 Percentile CPU / CUDA | 3.6–5.2 / 2.3–5.2 | 90.6–96.9% / 92.2–97.3% |

**Full official val (11,120 frames / 695 videos).**
- Reference: the PyTorch path whose logits the 6c/6d calibration was
  fitted on (video AUROC 0.9735, adaptive average 6.19 frames).
- Verdicts: 6c calibrated video and frame verdicts at mondrian α 0.05,
  and 6d adaptive verdicts.
- FA = real videos called "likely manipulated" (reference: 3 video /
  2 adaptive).

| variant | frame AUROC | video AUROC | video DF / F2F / FS / NT | NT frame | \|Δlogit\| mean / p99 / max | verdict agree video / frame / adaptive | FA video / adaptive | adaptive frames |
|---|---|---|---|---|---|---|---|---|
| PyTorch FP32 CPU | 0.9488 | 0.9735 | 0.988 / 0.985 / 0.994 / 0.927 | 0.886 | 0.004 / 0.036 / 0.095 | 99.86 / 99.89 / 100 % | 3 / 2 | 6.18 |
| **ONNX FP32 CPU** | 0.9488 | 0.9735 | 0.988 / 0.985 / 0.994 / 0.927 | 0.886 | 0.004 / 0.036 / 0.095 | 99.86 / 99.89 / 100 % | 3 / 2 | 6.18 |
| ONNX FP16 CUDA | 0.9488 | 0.9735 | 0.988 / 0.985 / 0.994 / 0.927 | 0.886 | 0.007 / 0.049 / 0.122 | 100 / 99.87 / 99.86 % | 3 / 2 | 6.19 |
| ONNX FP16 CPU | 0.9488 | 0.9735 | same | 0.886 | 0.007 / 0.047 / 0.132 | 99.86 / 99.84 / 99.86 % | 3 / 2 | 6.19 |
| INT8 MinMax CPU | 0.9326 | 0.9615 | 0.987 / 0.976 / 0.994 / 0.889 | 0.841 | 1.107 / 5.02 / 6.48 | 75.8 / 69.2 / 72.5 % | 0 / 0 | 7.71 |
| INT8 Percentile CPU | 0.9413 | 0.9678 | 0.983 / 0.983 / 0.995 / 0.910 | 0.868 | 0.514 / 3.67 / 6.29 | 88.4 / 87.5 / 89.6 % | 0 / 0 | 6.57 |

The PyTorch reference here is CUDA fp16 autocast, so the PyTorch FP32
CPU row differs from it by the same 0.004 mean as ONNX FP32.

**Benchmark** (isolated process per row; 8 CPU threads; crops
pre-decoded, so this is model + decision latency; 200 val videos; model
RAM = process RSS growth after creating and warming the runtime,
before test data):

| runtime | bs1 P50/P95 | bs4 P50/P95 | adaptive video P50/P95 (avg frames) | fixed-16 video P50/P95 | model RAM | GPU memory |
|---|---|---|---|---|---|---|
| PyTorch CPU | 21.6 / 44.0 | 35.6 / 49.7 | 38.1 / 133.1 (5.32) | 96.7 / 133.8 | 234 MB | – |
| **ONNX FP32 CPU** | 2.28 / 2.86 | 7.13 / 9.49 | 8.8 / 35.7 (5.32) | 33.7 / 40.1 | 91 MB | – |
| ONNX FP16 CPU | 2.39 / 4.40 | 7.17 / 8.65 | 9.1 / 33.6 (5.32) | 34.2 / 42.1 | 94 MB | – |
| INT8 MinMax CPU | 2.21 / 2.87 | 6.01 / 8.63 | 8.0 / 30.3 (8.36) | 24.8 / 28.5 | 59 MB | – |
| INT8 Percentile CPU | 2.27 / 2.95 | 5.82 / 7.21 | 7.4 / 29.4 (5.60) | 24.4 / 28.5 | 60 MB | – |
| PyTorch CUDA (fp16 autocast) | 20.2 / 23.9 | 20.7 / 23.9 | 22.5 / 69.1 (5.32) | 26.6 / 28.5 | 698 MB | +196 MB |
| **ONNX FP32 CUDA** (per-shape sessions) | 4.99 / 8.98 | 5.08 / 5.83 | 7.1 / 22.1 (5.32) | 10.6 / 11.5 | 627 MB | +418 MB |
| ONNX FP16 CUDA (per-shape sessions) | 6.13 / 7.08 | 6.50 / 8.39 | 8.0 / 25.1 (5.32) | 12.4 / 13.5 | 793 MB | +348 MB |

Times are in ms. GPU memory is the device-wide `nvidia-smi` delta
(per-process numbers are N/A under Windows WDDM).

- **CUDA shape finding:** with one ORT CUDA session, alternating batch
  sizes (4, 4, 8, 16) cost about 333 ms per cycle; a fixed shape costs
  6.5 ms. HEURISTIC conv search and the arena setting did not help.
- A session per batch size cuts the 4 + 4 + 8 escalation to 16 ms,
  against 35 ms when padding everything to 16. A first benchmark
  without this showed 110–150 ms per video on CUDA and was discarded.

**Selection (pre-registered rule in the script):**
- INT8 needs video AUROC loss ≤ 0.01 vs ONNX FP32, verdict agreement
  ≥ 98% (video and adaptive), and at most 1 extra false accusation.
  - MinMax fails all three (loss 0.012, agreement 76%).
  - Percentile passes AUROC (loss 0.0057) but fails agreement (88–90%).
  - Both shift scores towards "real" relative to the calibration (fewer
    false accusations, more misses), so INT8 would need its own
    recalibration (out of scope).
- **CPU default: ONNX FP32.** FP16 on CPU is no faster (bs4 7.17 vs
  7.13 ms) and is not chosen.
- **GPU default: ONNX FP32.** FP16 passes parity but is slower on this
  GPU (bs4 6.50 vs 5.08 ms; adaptive 8.0 vs 7.1 ms), so the rule
  (eligible and faster) keeps FP32.
- `export_manifest.json` (content `819f5866…`) verifies against the p80
  checkpoint and refuses the robust checkpoint.

## 2026-10-01 — Phase 8 tests

```
.venv/Scripts/python.exe -m pytest tests/export tests/calibration tests/adaptive -q -p no:cacheprovider   # 22 passed
.venv/Scripts/python.exe -m pytest -q -p no:cacheprovider -rs                                            # 563 passed, 0 skipped, 223.36 s
```
- The full suite was run twice. The first run's summary line was not
  captured (the output tail showed only ffmpeg stderr from an existing
  corrupt-video test), so it was re-run to record the result.
