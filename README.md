# ConfiGuard-Lite

Efficient and uncertainty-aware deepfake image and video detection, designed
to train within a 6 GB VRAM laptop GPU budget and deploy on CPU-only
servers with optional GPU acceleration.

See `docs/PROJECT_PLAN.md` for the full requirements and phase roadmap,
`docs/ARCHITECTURE.md` for the system design, and `CLAUDE.md` for the
operating rules this project is developed under.

## Status

Phase 11 (optional Grad-CAM evidence hints, gated by an occlusion check, and a local web UI at `/`). Phase 10 (production inference API: FastAPI over ONNX FP32 adaptive inference + the Phase 9 v1 quality gate; CPU default, development data only, test sealed). Phase 9c (final hybrid quality-gate experiment, rejected; quality-gate experimentation ended). The Phase 9 downgrade-only media-quality gate over ONNX FP32 adaptive inference remains enabled by default (development data only, test sealed). Phase 9b (quality-gate hardening) was also rejected. Phase 8 exported the calibrated `student_distilled_p80` to ONNX (FP32 default on CPU and GPU). Phase 7 evaluated and rejected a GRU temporal head. Phase 6e ran the compression-robust training experiment (not selected). Phase 6d added adaptive 4/8/16-frame video inference. Phase 6c added calibration and the uncertain output. Phase 6b trained the MobileNetV4 students (BCE baseline vs GenD-distilled). Phase 6a cached frozen GenD CLIP-L/14 teacher logits for train/val. Before that, Phase 5d (matched face-crop extraction and shortcut audit) — see
`docs/PHASE_STATUS.md` for current status. FF++ c23 (5000 videos,
8.4 GiB) is downloaded to `D:`, validated, and split with the authors'
official train/val/test files (720/140/140 originals, leakage-checked).
79,280 aligned 224×224 face crops (16 per video, real/fake matched by
frame index, 9/1000 families quarantined) are on `D:`, and the shortcut
audit is documented. Nothing has been trained on them yet. Earlier, Phase 5 built the reproducible
training pipeline. The training pipeline
(`configuard.training`) is real and verified on CPU and the RTX 4050,
but **has only ever trained on synthetic engineering data** — no
deepfake detector exists yet. The end-to-end pipeline
(`configuard.pipeline`) runs but uses a **deterministic dummy predictor**,
not a trained model. The face preprocessing subsystem (`configuard.media`),
the dataset registry (`configuard.datasets`), and two pretrained visual
encoders (`configuard.models`) are all real and independently tested, but
none of them are wired into the main pipeline or a real training loop yet
— see `docs/ARCHITECTURE.md` for what's real vs. placeholder. **No
training dataset has been downloaded.** Two small ImageNet-pretrained
backbones (MobileNetV4-Conv-Small, EfficientNet-B0) *have* been
downloaded, per explicit authorization — see `docs/DATASETS.md` for full
provenance. **Every prediction `configuard.models` produces is untrained
and uncalibrated** — see the warning in `docs/MODEL_CARD.md`.

## Requirements

- Windows 10/11 (Linux/Docker compatibility is a design goal, not yet built)
- Python 3.11 (3.14 is not yet supported by the ML dependency stack — see
  `docs/DECISIONS.md`)
- Git
- NVIDIA GPU + recent driver, optional (CPU-only inference is supported)
- FFmpeg (install with `winget install --id Gyan.FFmpeg -e`, then open a new
  terminal so the updated `PATH` takes effect)
- The YuNet face detector ONNX model, for real face detection (optional -
  `configuard.media.face_detector.MockFaceDetector` works without it, e.g.
  for tests). See `docs/DATASETS.md` for the official download source,
  license, and SHA-256 to verify against; place it at
  `models/face_detection/face_detection_yunet_2026may.onnx` (gitignored)
  or point `CONFIGUARD_YUNET_MODEL_PATH` at it.
- The two pretrained backbone weights (MobileNetV4-Conv-Small,
  EfficientNet-B0), for real encoder construction (optional - tests use
  `pretrained=False`, no download needed). Set `HF_HOME`/`HF_HUB_CACHE`/
  `TORCH_HOME` in `.env` first (see `.env.example`), then run
  `scripts\download_baseline_models.py`. See `docs/DATASETS.md` for full
  provenance.

## Setup

```powershell
# 1. Create the virtual environment (Python 3.11)
py -3.11 -m venv .venv

# 2. Install PyTorch matching your hardware
#    GPU (NVIDIA, CUDA 12.1-compatible driver):
.venv\Scripts\python.exe -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
#    CPU-only:
.venv\Scripts\python.exe -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu

# 3. Install the remaining dependencies, constrained to the pinned torch
#    pair so pip cannot swap in a different (e.g. CPU-only) build
.venv\Scripts\python.exe -m pip install -r requirements-dev.txt -c constraints-cuda.txt

# 4. Verify PyTorch still has the GPU build after step 3 (installing new
#    packages can silently re-resolve/downgrade torch - see docs/KNOWN_ISSUES.md)
.venv\Scripts\python.exe scripts\verify_environment.py
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

## Run the pipeline on a file (Phase 1: dummy prediction)

```python
import json
from configuard.config import load_config
from configuard.pipeline import run_pipeline
from configuard.io_types import result_to_dict

config = load_config("configs/development.yaml")
result = run_pipeline("path/to/image_or_video.mp4", config.validation)
print(json.dumps(result_to_dict(result), indent=2))
```

Invalid, corrupted, oversized, or wrong-type files raise
`configuard.pipeline.PipelineRejectedError` (carries the full
`ValidationResult` with human-readable `error_messages()`), rather than
producing a prediction.

## Run face preprocessing on a file (Phase 2)

```python
from configuard.media.cache import FaceCropCache
from configuard.media.face_detector import YuNetFaceDetector
from configuard.media.preprocess import preprocess_image, preprocess_video
from configuard.media.types import PreprocessingConfig

detector = YuNetFaceDetector()  # requires the model file - see Requirements above
cache = FaceCropCache("cache/face_crops", max_bytes=2_000_000_000)
config = PreprocessingConfig(detector_name=detector.name, detector_version=detector.version)

image_result = preprocess_image("photo.jpg", detector, cache, config)
video_result = preprocess_video("clip.mp4", requested_frame_count=8, detector=detector, cache=cache, config=config)
```

Benchmark each stage separately:

```powershell
.venv\Scripts\python.exe scripts\benchmark_preprocessing.py
```

## Build a dataset manifest (Phase 3)

```python
from configuard.datasets.registry import DEFAULT_REGISTRY
from configuard.datasets.manifest import write_manifest, validate_manifest_file
from configuard.datasets.splitting import SplitConfig, split_samples, write_split_audit_report
from configuard.datasets.duplicates import build_duplicate_report

adapter = DEFAULT_REGISTRY.get_adapter("faceforensics++")  # or celeb-df-v2 / dfdc / df40 / deeperforensics-1.0
samples = adapter.build_manifest("/path/to/your/local/FaceForensics++")  # never downloaded for you - see docs/DATASETS.md
write_manifest(samples, "manifest.jsonl")

report = validate_manifest_file("manifest.jsonl", media_root="/path/to/your/local/FaceForensics++")
split_report = split_samples(samples, SplitConfig(seed=42))
write_split_audit_report(split_report, "split_audit.json")
dup_report = build_duplicate_report(samples, media_root="/path/to/your/local/FaceForensics++")
```

A missing/incomplete local dataset root raises
`configuard.datasets.adapters.DatasetAccessError` with a clear message —
adapters never download anything.

Check your configured storage paths before pointing anything real at them:

```powershell
.venv\Scripts\python.exe scripts\check_storage.py
```

## Run a pretrained encoder (Phase 4 — UNTRAINED/uncalibrated output)

```python
import numpy as np
from configuard.models.registry import create_encoder
from configuard.models.inference import infer_image, infer_video_fixed_frames

encoder = create_encoder("mobilenetv4_conv_small", pretrained=True)  # or "efficientnet_b0"
config = encoder.resolve_preprocess_config()

image = np.zeros((224, 224, 3), dtype=np.uint8)  # a real aligned face crop, e.g. from configuard.media
result = infer_image(encoder, image, config)
print(result.probability, result.disclaimer)  # ALWAYS UNTRAINED_UNCALIBRATED right now

frames = [image] * 8  # ordered 4/8/16-frame sample, e.g. from configuard.media.sampling
video_result = infer_video_fixed_frames(encoder, frames, config)
```

Download the two authorized pretrained backbones (requires `HF_HOME`/
`HF_HUB_CACHE`/`TORCH_HOME` set in `.env` first):

```powershell
.venv\Scripts\python.exe scripts\download_baseline_models.py
.venv\Scripts\python.exe scripts\export_onnx_models.py     # FP32 ONNX + parity check
.venv\Scripts\python.exe scripts\benchmark_models.py         # latency/memory/FLOPs
```

## Train and evaluate (Phase 5)

Pinned CUDA build — reinstall/verify only with the pinned pair (the
default PyPI index serves a CPU-only torch on Windows):

```powershell
.venv\Scripts\pip.exe install -c constraints-cuda.txt torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
.venv\Scripts\python.exe scripts\verify_environment.py   # fails loudly on CPU-only torch / broken torchvision / lost CUDA
```

Synthetic smoke training (**engineering test only — not deepfake
accuracy**); checkpoints go to `CONFIGUARD_CHECKPOINT_DIR`, logs to
`CONFIGUARD_OUTPUT_DIR`:

```powershell
.venv\Scripts\python.exe scripts\train.py --smoke cuda                          # MobileNetV4, RTX 4050, AMP
.venv\Scripts\python.exe scripts\train.py --smoke cuda --encoder efficientnet_b0
.venv\Scripts\python.exe scripts\train.py --smoke cpu
```

Real training, once you have leakage-safe manifests from
`configuard.datasets` (Phase 3) over your own local data:

```powershell
.venv\Scripts\python.exe scripts\train.py --config configs\train\mobilenetv4_conv_small.yaml `
    --train-manifest D:\...\train.jsonl --val-manifest D:\...\val.jsonl --media-root D:\...\media
# exact resume from the last completed epoch (refused if data/preprocessing/config changed):
.venv\Scripts\python.exe scripts\train.py --config ... --resume-from D:\...\<run>_latest.pt
# evaluation only:
.venv\Scripts\python.exe scripts\evaluate.py --checkpoint D:\...\<run>_best.pt --manifest D:\...\test.jsonl --media-root D:\...\media
```

Both CLIs run offline (`HF_HUB_OFFLINE=1`) and never download a model or
dataset. See `docs/ARCHITECTURE.md` for checkpoint contents, resume
rules, and metrics.

## Acquire and validate FaceForensics++ c23 (Phase 5b)

Requires your own official FaceForensics++ access. Place the official
script at `%CONFIGUARD_DATA_DIR%\FaceForensics++\_official_script\`
(its SHA-256 is pinned), then:

```powershell
.venv\Scripts\python.exe scripts\download_faceforensics_c23.py --num-videos 5   # trial
.venv\Scripts\python.exe scripts\download_faceforensics_c23.py                  # c23 videos, EU2, 5 classes only;
                                                                              # stops before < 40 GB free; restarts stalls
.venv\Scripts\python.exe scripts\validate_faceforensics.py                      # counts, ffprobe, relationships,
                                                                              # manifest, leakage groups, report
```

Then apply the official split (Phase 5c). This needs the pinned split
files in `...\FaceForensics++\_official_splits\<revision>\`; see
`docs/DATASETS.md` for URLs and hashes:

```powershell
.venv\Scripts\python.exe scripts\apply_faceforensics_splits.py   # refuses on any leakage; writes train/val/test manifests
```

## Extract matched face crops and audit shortcuts (Phase 5d)

Needs the Phase 5c manifests and the hash-pinned YuNet model. Everything
is written under `%CONFIGUARD_CACHE_DIR%fpp_face_crops\store` (D:).

```powershell
.venv\Scripts\python.exe scripts\extract_ffpp_face_crops.py --preflight-only   # pins, membership, leakage, video hashes, storage
.venv\Scripts\python.exe scripts\extract_ffpp_face_crops.py --trial 10         # seeded trial (stratified by split)
.venv\Scripts\python.exe scripts\extract_ffpp_face_crops.py                    # full; resumable; stops at 40+2 GB free
.venv\Scripts\python.exe scripts\extract_ffpp_face_crops.py --verify-crop-hashes   # rerun: verify every crop, rebuild manifests
.venv\Scripts\python.exe scriptsudit_ffpp_crop_shortcuts.py                  # crop-or-squeeze, geometry, correlations, crop QA
```

A store root accepts exactly one config. Changing any detector,
sampling or alignment parameter is refused as stale: use a new
`--store-root`.

## Download the frozen GenD teacher and cache its logits (Phase 6a)

Needs the Phase 5d crop store and `HF_HOME`/`HF_HUB_CACHE` on D: (see
`.env.example`). It downloads only `yermandy/GenD_CLIP_L_14` at the
pinned revision (~1.2 GB) and verifies its hashes. Logits are cached
for **train and val only**; `test` is refused.

```powershell
.venv\Scripts\python.exe scripts\download_gend_teacher.py                  # pinned snapshot + SHA-256 record
.venv\Scripts\python.exe scripts\cache_teacher_logits.py --limit-shards 2  # trial (separate _trial root)
.venv\Scripts\python.exe scripts\cache_teacher_logits.py                   # full train+val; resumable
```

Rerunning resumes from the completed shards. A cache built with a
different teacher config, crop manifest or shard size is refused as
stale.

## Train the MobileNetV4 students (Phase 6b)

Needs the Phase 5d crop store and the Phase 6a teacher-logit cache on D:
(GenD itself is not loaded). It trains on train crops and early-stops on
val crops. The test split is refused.

```powershell
.venv\Scripts\python.exe scripts\train_distill_student.py pilot --epochs 4 --samples-per-epoch 25000   # alpha/T grid
.venv\Scripts\python.exe scripts\train_distill_student.py train --run-name student_baseline --alpha 0
.venv\Scripts\python.exe scripts\train_distill_student.py train --run-name student_distilled --alpha 0.5 --temperature 2
.venv\Scripts\python.exe scripts\compare_students.py --runs student_baseline,student_distilled
```

Runs resume from `last.pt`. Reusing a `run_name` with a different config
is refused.

## Calibrate the student and get the "uncertain" output (Phase 6c)

```powershell
.venv\Scripts\python.exe scripts\calibrate_student.py split                                   # 80/10/10 train families
.venv\Scripts\python.exe scripts\train_distill_student.py train --run-name student_distilled_p80 --alpha 0.5 --temperature 2 --train-partition final_train
.venv\Scripts\python.exe scripts\calibrate_student.py fit --run student_distilled_p80         # writes calibration.json
```

In code:

```python
from configuard.calibration.artifact import load_calibration
cal = load_calibration(run_dir / "calibration.json", run_dir / "best.pt")  # refuses mismatches
out = cal.predict(video_mean_logits, level="video")   # out.p_fake, out.verdicts
```

## Adaptive 4/8/16-frame video inference (Phase 6d)

```powershell
.venv\Scripts\python.exe scripts\adaptive_video_eval.py --run student_distilled_p80   # per-stage calibration + dev comparison + latency
```

```python
from configuard.adaptive.analyzer import AdaptiveVideoAnalyzer, StudentCropScorer
from configuard.adaptive.policy import StagePolicy
from configuard.calibration.artifact import load_calibration
cal = load_calibration(run_dir / "adaptive_calibration.json", run_dir / "best.pt")
result = AdaptiveVideoAnalyzer(cal, StagePolicy()).analyze(StudentCropScorer(model, norm, crop_paths_by_slot))
result.verdict, result.stopping_reason, result.frames_used, result.timeline
```

## Robustness stress suite (Phase 6e)

```powershell
.venv\Scripts\python.exe scripts\robust_eval.py build                                   # 17 degraded val conditions on D: (~12 GiB)
.venv\Scripts\python.exe scripts\train_distill_student.py train --run-name student_distilled_robust_p80 --alpha 0.5 --temperature 2 --train-partition final_train --robust
.venv\Scripts\python.exe scripts\robust_eval.py evaluate --workers 4 --ram-floor-gb 4     # resumable; stops safely on low RAM
```

## Temporal GRU experiment (Phase 7; rejected)

```powershell
.venv\Scripts\python.exe scripts\temporal_gru.py extract    # frame embeddings on D: (resumable, RAM floor)
.venv\Scripts\python.exe scripts\temporal_gru.py train
.venv\Scripts\python.exe scripts\temporal_gru.py evaluate   # GRU vs mean-logit aggregation + decision
```

## Export to ONNX (Phase 8)

```powershell
.venv\Scripts\python.exe scripts\export_student_onnx.py build      # FP32 / FP16 / INT8 (train-only calibration)
.venv\Scripts\python.exe scripts\export_student_onnx.py parity     # PyTorch vs ONNX on clean + degraded val crops
.venv\Scripts\python.exe scripts\export_student_onnx.py evaluate   # full val: AUROC, verdict agreement
.venv\Scripts\python.exe scripts\export_student_onnx.py bench      # CPU/GPU latency, RAM/VRAM, adaptive 4/8/16
.venv\Scripts\python.exe scripts\export_student_onnx.py package    # hash-checked export_manifest.json + defaults
```

GPU inference with ONNX Runtime uses `onnxruntime-gpu==1.23.2`, which
works with the CUDA 12 / cuDNN 9 DLLs bundled with torch. Import torch
first. CPU-only deployments can use plain `onnxruntime`.

## Media-quality safety gate (Phase 9)

```powershell
.venv\Scripts\python.exe scripts\quality_gate.py compute    # quality signals + ONNX FP32 logits (resumable, RAM floor)
.venv\Scripts\python.exe scripts\quality_gate.py fit        # thresholds from TRAIN / calibration partitions only
.venv\Scripts\python.exe scripts\quality_gate.py evaluate   # clean + 17 stress + adversarial/bypass sets
```

The gate only turns confident verdicts into UNCERTAIN, with reason
codes (LOW_SHARPNESS, LOW_RESOLUTION, HEAVY_COMPRESSION, SMALL_FACE,
QUALITY_DEPENDENT_VERDICT).

## Quality-gate v2 experiment (Phase 9b; rejected)

```powershell
.venv\Scripts\python.exe scripts\quality_gate_v2.py quantiles   # then: cases, fit, bench, verify, confirm-val (once)
```

## Inference API (Phase 10)

```powershell
.venv\Scripts\python.exe scripts\serve.py --env development            # http://127.0.0.1:8000/docs
curl.exe -F "file=@clip.mp4" http://127.0.0.1:8000/v1/analyze
curl.exe http://127.0.0.1:8000/health/ready
.venv\Scripts\python.exe scripts\service_load_test.py --videos 40 --images 40 --concurrency 4
```

- **Production** (`--env production`) requires `CONFIGUARD_API_KEYS` and
  the `X-API-Key` header. It refuses to start without keys.
- **GPU:** `--device cuda` (falls back to CPU automatically).
- **Paths:** see `.env.example`. Responses carry the verdict (`likely_real`
  / `likely_manipulated` / `uncertain`), calibrated `p_fake` and
  `confidence`, quality and uncertainty reasons, frames used, the
  evidence timeline, model version and timings.

### Web UI and evidence hints (Phase 11)

- **UI:** open `http://127.0.0.1:8000/` after `scripts\serve.py --env
  development`. Upload a file and read the verdict, confidence,
  reasons, timeline and (optionally) evidence frames.
- **Evidence hints:** `POST /v1/analyze?explain=true` (needs
  `allow_explanations: true`; on in development, off in production).
  Hints are labelled "Visual evidence hint — not proof" and are withheld
  when they fail an occlusion check.
- **Checks:** `scripts\explain_sanity.py` (offline) and
  `scripts\browser_smoke.py` (headless Chrome).

## Quality-gate hybrid experiment (Phase 9c; rejected, final)

```powershell
.venv\Scripts\python.exe scripts\quality_gate_hybrid.py assemble   # then: challenge-split, verify, bench, confirm-val (once), decide
```

## Project layout

See `docs/ARCHITECTURE.md` for the full repository layout and design.

## Configuration

Non-secret defaults live in `configs/*.yaml`. Machine-specific paths and
secrets go in a local `.env` file (copy `.env.example` to `.env` — `.env` is
never committed). Point `CONFIGUARD_DATA_DIR` / `CONFIGUARD_CACHE_DIR` /
`CONFIGUARD_CHECKPOINT_DIR` / `CONFIGUARD_OUTPUT_DIR` at a location
outside the repo (and outside any cloud-synced folder like OneDrive) with
enough free space — run `scripts\check_storage.py` to verify before
storing anything real. This machine's configured layout is recorded in
`docs/ARCHITECTURE.md`.
