# Project Plan

## Goal

ConfiGuard-Lite: Efficient and Uncertainty-Aware Deepfake Image and Video
Detection — a production-oriented final-year project that must train and run
inference within a 6 GB VRAM laptop GPU budget.

## Final system requirements

1. Accept images and videos as input.
2. Detect AI-generated and face-manipulated media.
3. Use one shared lightweight visual encoder for both images and video frames.
4. Use MobileNetV4-Conv-Small as the preferred student model.
5. Compare it against EfficientNet-B0.
6. Use pretrained GenD CLIP-L/14 only as a frozen, offline teacher (for
   distillation), never fine-tuned locally.
7. Never attempt to train the large GenD teacher on this laptop.
8. Use a small temporal GRU for video sequence reasoning.
9. Analyze video adaptively using 4, 8, or 16 sampled frames.
10. Return one of three outcomes: "likely real", "likely manipulated", or
    "uncertain".
11. Include compression-robust training (re-encoding / JPEG / H.264
    artifact augmentation).
12. Include confidence calibration and conformal prediction.
13. Export FP32, FP16, and INT8 ONNX models.
14. Run on CPU-only servers, with optional NVIDIA GPU acceleration.
15. Include image heatmaps, video evidence timelines, and separate C2PA
    provenance results (reported alongside, not merged into, the model
    score).

## Hardware envelope

- Windows 11 laptop
- AMD Ryzen 7 7435HS (8 cores / 16 threads)
- NVIDIA RTX 4050 Laptop GPU, 6 GB VRAM
- ~24 GB system RAM
- Disk headroom is limited (~20 GB free as of Phase 0) — dataset/checkpoint
  storage plans must account for this; see `docs/KNOWN_ISSUES.md`.

## Phase roadmap (living list — update as phases complete/are added)

| Phase | Title | Status |
|---|---|---|
| 0 | Environment and repository foundation | See `docs/PHASE_STATUS.md` |
| 1 | Architecture contracts and minimal vertical slice | See `docs/PHASE_STATUS.md` |
| 2 | Face and media preprocessing (decode, nested sampling, YuNet detection, tracking, alignment, cache) | See `docs/PHASE_STATUS.md` |
| 3 | Dataset registry and leakage-safe data splits | See `docs/PHASE_STATUS.md` |
| 3b | Wire real dataset(s) into Phase 2 preprocessing (once user provides local dataset access) | Planned |
| 4 | Pretrained baseline models (MobileNetV4-Conv-Small + EfficientNet-B0) and ONNX verification | See `docs/PHASE_STATUS.md` |
| 5 | Reproducible training pipeline (config-driven training, exact resume, checkpoints, metrics, logging, dependency safety) | See `docs/PHASE_STATUS.md` |
| 5b | Official FaceForensics++ c23 acquisition and validation (no split, no crops, no training) | See `docs/PHASE_STATUS.md` |
| 5c | Official FaceForensics++ split integration (pinned official files, leakage refusal, per-split audit) | See `docs/PHASE_STATUS.md` |
| 6 | Distillation from frozen GenD CLIP-L/14 teacher | Planned |
| 7 | Compression-robust augmentation | Planned |
| 8 | Temporal GRU + adaptive frame sampling for video | Planned |
| 9 | Calibration + conformal prediction, tri-class decision rule | Planned |
| 10 | ONNX export (FP32/FP16/INT8) + CPU/GPU inference runtime | Planned |
| 11 | Explainability: heatmaps, video evidence timelines | Planned |
| 12 | C2PA provenance integration (separate signal) | Planned |
| 13 | Serving API + packaging | Planned |

This table is a planning aid, not a commitment — phases may be split,
reordered, or merged as findings from earlier phases dictate (Phase 2 was
reordered ahead of the dataset registry on 2026-09-29 at the user's
direction, since face preprocessing needed to exist before real datasets
are wired in; Phase 5 was redefined as the reproducible training
pipeline on 2026-09-30, shifting later phases by one). Each change should be reflected here and explained in
`docs/DECISIONS.md`.
