# Datasets

Status: No training dataset has been downloaded. Phase 3 added a typed
**dataset registry** (`src/configuard/datasets/`) with adapters for five
named deepfake-detection datasets, built and tested entirely against
synthetic fixtures - see docs/PHASE_STATUS.md and docs/ARCHITECTURE.md for
what the registry actually does. This document tracks (a) the one small
non-training model asset downloaded so far (the YuNet face detector) and
(b) access/licensing information for each registered dataset, verified
only where explicitly marked - anything not independently confirmed is
marked **VERIFICATION REQUIRED** rather than presented as fact.

Per project rules, no dataset is downloaded without explicit user
confirmation, and none of the five datasets below were downloaded to
build their adapters - all are access-controlled and require the user to
obtain them independently under each dataset's own terms. Local storage
paths are always configurable via `.env` (`CONFIGUARD_DATA_DIR`,
`CONFIGUARD_CACHE_DIR`, `CONFIGUARD_CHECKPOINT_DIR`,
`CONFIGUARD_OUTPUT_DIR` - see `scripts/check_storage.py` and this
machine's configured layout in `docs/ARCHITECTURE.md`), never hardcoded.
Raw data itself is never
committed (see `.gitignore`).

---

## Registered datasets: access and licensing

**How to read this table:** "Structure confidence" reflects how sure this
project is that `src/configuard/datasets/adapters/known_datasets.py`'s
expected on-disk layout matches the *real* dataset - none of it was
checked against real data (which this project doesn't have access to).
Low-confidence entries should be treated as a starting point to adjust
once real access is obtained, not as ground truth.

| Dataset | Access process | License/terms | Structure confidence |
|---|---|---|---|
| FaceForensics++ | Request access via the official form linked from `github.com/ondyari/FaceForensics` (requires agreeing to the dataset's terms of use; academic/research use). **VERIFICATION REQUIRED**: exact current request URL/process - confirm against the live repo before relying on this. | Restricted - EULA/terms-of-use agreement required before download; redistribution prohibited. **VERIFICATION REQUIRED** for exact current license text. | High for the top-level `original_sequences/`/`manipulated_sequences/{method}/{c}/videos` layout (well-documented, stable across the dataset's public lifetime). Filename source/target semantics treated liberally by design - see docs/DECISIONS.md. |
| Celeb-DF-v2 | Request access via the Google Form linked from `github.com/yuezunli/celeb-deepfakeforensics` (academic/research use, requires institutional affiliation in most public accounts of the process). **VERIFICATION REQUIRED**: exact current form URL/requirements. | Restricted - access-request agreement; research use only per the dataset's own terms. **VERIFICATION REQUIRED** for exact current terms. | High for `Celeb-real/`, `Celeb-synthesis/`, `YouTube-real/`, `List_of_testing_videos.txt` (consistently referenced across public documentation of this dataset). |
| DFDC (Deepfake Detection Challenge) | Was distributed via Kaggle (`kaggle.com/c/deepfake-detection-challenge`) and an AWS-hosted full dataset for competition participants; general public download availability may have changed since the competition ended. **VERIFICATION REQUIRED**: current availability/access path. | Competition rules / dataset license as published by Meta AI (then Facebook AI) - restricted use terms applied historically. **VERIFICATION REQUIRED** for exact current terms. | High for the per-part `metadata.json` = `{"<file>.mp4": {"label": "REAL"\|"FAKE", "split": ..., "original": "<file>.mp4"\|null}}` shape - this is extensively documented in public competition materials. |
| DF40 | A more recent (2024+) multi-forgery-method benchmark. **VERIFICATION REQUIRED**: this project could not independently confirm the official repository, access process, license, or on-disk structure. The adapter's `metadata.jsonl` assumption is a generic placeholder, not a confirmed fact. | **VERIFICATION REQUIRED**. | Low - do not rely on `make_df40_adapter()`'s defaults without first confirming the real structure and adjusting `MetadataSidecarSpec` accordingly. |
| DeeperForensics-1.0 | Request access via the official form linked from the dataset's GitHub repo (academic/research use). **VERIFICATION REQUIRED**: exact current request URL/process. | Restricted - EULA required. **VERIFICATION REQUIRED** for exact current terms. | Moderate for `manipulated_videos/` (fake, end-to-end perturbed videos - stated with moderate confidence from the paper's description). Low for the `source_videos/` (real) bucket name, which is a best-effort guess - the paper describes real source actor recordings distinct from FaceForensics++, but the exact release folder name was not confirmed. |

None of these datasets were downloaded, browsed, or accessed in any way to
write this project's adapters - the structures above come from general,
public documentation knowledge of each dataset, which may be incomplete
or have drifted from the current official release. **Before pointing any
adapter at a real local copy, verify the actual folder/metadata layout
against the dataset's own current documentation and adjust
`known_datasets.py` if it differs.**

---

## External model assets (non-training)

### YuNet face detector (ONNX)

| Field | Value |
|---|---|
| Purpose | Face detection + 5-point landmarks, used by `configuard.media.face_detector.YuNetFaceDetector` |
| File | `face_detection_yunet_2026may.onnx` |
| Source repo | https://github.com/opencv/opencv_zoo |
| Source path | `models/face_detection_yunet/face_detection_yunet_2026may.onnx` |
| Download URL used | `https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2026may.onnx` (resolved via the repo's Git LFS pointer, since the `raw.githubusercontent.com` URL for this file is a 131-byte LFS pointer, not the binary) |
| License | MIT License, Copyright (c) 2020 Shiqi Yu \<shiqi.yu@gmail.com\> (full text at `https://github.com/opencv/opencv_zoo/blob/main/models/face_detection_yunet/LICENSE`) |
| Version | `2026may` — dynamic input-shape re-export of the `2023mar` weights, documented by opencv_zoo as compatible with OpenCV 5.x's ONNX graph engine (this project uses `opencv-python-headless` 5.0.0) |
| File size | 229,738 bytes (~224 KB) |
| SHA-256 | `ebafce4e3c118d6554634be5c27ab333b4c047a9a8c3faf1d7cf93101c22f0f0` (matches the hash declared in the repo's own Git LFS pointer file) |
| Downloaded | 2026-09-29, with explicit user approval (Phase 2 task instructions named this exact asset) |
| Local path | `models/face_detection/face_detection_yunet_2026may.onnx` (gitignored — see `.gitignore`'s `models/` rule; never committed) |
| Verification | Loaded successfully via `cv2.FaceDetectorYN_create(...)` on CPU (no CUDA) and confirmed to return no detections on a blank/non-face image — see `docs/EXPERIMENT_LOG.md` |

No other model checkpoint, dataset, or unrelated asset was downloaded in
that phase.

### Pretrained visual-encoder backbones (Phase 4)

Both authorized by the Phase 4 task instructions, downloaded only from
their official Hugging Face / timm repositories, `safetensors` format
(preferred, as instructed). Cached under `D:\ConfiGuard-Data\cache\huggingface`
(`HF_HOME`/`HF_HUB_CACHE`, configured in the untracked local `.env` - see
`docs/ARCHITECTURE.md`), never under `C:\Users\...\.cache` or inside this
repository. Both are ImageNet-1k classification backbones with the
classifier head removed (`num_classes=0`) - see docs/MODEL_CARD.md for
why their predictions are not meaningful deepfake results.

#### MobileNetV4-Conv-Small

| Field | Value |
|---|---|
| Model ID | `timm/mobilenetv4_conv_small.e1200_r224_in1k` |
| Revision (commit) | `c9f31ac64483d7f0590db9edccb4418392a96eea` |
| Source | https://huggingface.co/timm/mobilenetv4_conv_small.e1200_r224_in1k |
| License | Apache License 2.0 |
| File format | `model.safetensors` |
| File size | 15,223,016 bytes (~14.5 MB) |
| SHA-256 | `5a2ef04d419ce6d1bf27bfa735bb200d3f8d8997c3ac36320f5bf30382f6b43c` |
| Parameters (feature extractor, `num_classes=0`) | 2,494,305 (incl. this project's randomly-initialized binary head: 2,493,024 backbone + 1,281 head) |
| Local cache path | `D:\ConfiGuard-Data\cache\huggingface\hub\models--timm--mobilenetv4_conv_small.e1200_r224_in1k\snapshots\c9f31ac...\model.safetensors` |
| Downloaded | 2026-09-29, via `scripts/download_baseline_models.py`, with explicit user authorization naming this exact model |

#### EfficientNet-B0 (`tf_efficientnet_b0`)

| Field | Value |
|---|---|
| Model ID | `timm/tf_efficientnet_b0.in1k` |
| Revision (commit) | `8186ca4217f9c67824ebe7566008bdc69976d15a` |
| Source | https://huggingface.co/timm/tf_efficientnet_b0.in1k |
| License | Apache License 2.0 |
| File format | `model.safetensors` |
| File size | 21,355,344 bytes (~20.4 MB) |
| SHA-256 | `276dfe076f3fca30c2f7bf1e44039e395e6de50248caaa159d16530699f16995` |
| Parameters (feature extractor, `num_classes=0`) | 4,008,829 (incl. this project's randomly-initialized binary head: 4,007,548 backbone + 1,281 head) |
| Local cache path | `D:\ConfiGuard-Data\cache\huggingface\hub\models--timm--tf_efficientnet_b0.in1k\snapshots\8186ca4...\model.safetensors` |
| Downloaded | 2026-09-29, via `scripts/download_baseline_models.py`, with explicit user authorization naming this exact model |

**Integrity verification method:** each file's SHA-256 was computed
locally after download (`sha256sum`) and cross-checked against
`huggingface_hub`'s own snapshot revision directory naming (the commit
hash embedded in the cache path) - see `docs/EXPERIMENT_LOG.md` for the
full download/verification transcript.

No other checkpoint (GenD, DINOv2, or anything else) and no dataset was
downloaded in this phase.
