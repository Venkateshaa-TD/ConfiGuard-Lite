# Datasets

Status: No training datasets have been downloaded or registered yet. This
document also tracks small, non-training external model assets (e.g. the
face detector below), per project rules that any download must be recorded
with its provenance.

Per project rules, no dataset is downloaded without explicit user
confirmation. When training datasets are added, this document will record,
per dataset: name, source/license, size, download method, local storage
path convention (via `.env`, never hardcoded), and any preprocessing
applied. Raw data itself is never committed (see `.gitignore`).

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
this phase.
