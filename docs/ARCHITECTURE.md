# Architecture

Status: Phase 0 — only the repository/environment scaffolding exists.
No model architecture has been implemented yet. This document will be filled
in as each phase lands.

## Planned high-level design

```
                depths of image / video frames
                              |
                 shared lightweight visual encoder
              (MobileNetV4-Conv-Small | EfficientNet-B0)
                              |
              +---------------+---------------+
              |                               |
        image path                      video path
     (single embedding)         (per-frame embeddings -> GRU)
              |                               |
              +---------------+---------------+
                              |
              calibration + conformal prediction
                              |
        "likely real" | "likely manipulated" | "uncertain"

  (offline, frozen) GenD CLIP-L/14 --distillation loss (train-time only)--> student encoder
  C2PA provenance check -------------------------------------------> reported as a separate result
```

## Repository layout (Phase 0)

```
ConfiGuard-Lite/
├── src/configuard/        Python package (importable source of truth)
├── scripts/                One-off / operational scripts (e.g. env verification)
├── tests/                   pytest test suite
├── configs/                 YAML configuration files (non-secret)
├── docs/                    Living project documentation
├── data/                    (gitignored) local datasets
├── checkpoints/             (gitignored) local model checkpoints
├── cache/                   (gitignored) generated caches (face crops, features)
├── outputs/                 (gitignored) generated reports/artifacts
└── .venv/                   (gitignored) Python 3.11 virtual environment
```

## Notes for future phases

- Sections for the encoder, distillation setup, temporal module,
  calibration/conformal layer, export pipeline, and explainability modules
  will each be added when their phase is implemented, not before.
