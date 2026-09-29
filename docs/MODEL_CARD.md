# Model Card

Status: No deepfake-detection model has been trained yet.

This document will be populated starting in the phase where a first model
checkpoint is produced, and will cover: intended use, training data summary
(no raw data, only dataset provenance — see `docs/DATASETS.md`), evaluation
metrics, calibration/conformal guarantees, known failure modes, and export
formats.

## Auxiliary component: YuNet face detector (Phase 2)

Not the deepfake-detection model itself, but a pretrained auxiliary
component used for face localization ahead of it. Provenance (source,
license, version, SHA-256) is tracked in `docs/DATASETS.md` rather than
duplicated here, since it is an external, frozen, off-the-shelf detector —
not something this project trains or fine-tunes.
