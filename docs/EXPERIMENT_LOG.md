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
