# Model Card

Status: **No deepfake-detection model has been trained yet.** Phase 4
added two candidate visual-encoder backbones
(`configuard.models.DeepfakeVisualEncoder`), each an ImageNet-1k-pretrained
timm backbone with its classifier head replaced by a single-logit binary
(real/fake) head. **That head is randomly initialized and has never seen
a single deepfake example.**

> ⚠️ **Every prediction `configuard.models` can currently produce is
> UNTRAINED and UNCALIBRATED.** `is_finetuned` is `False` on every
> encoder and every `PredictionResult` carries
> `PREDICTION_DISCLAIMER`. Do not interpret any probability/logit from
> this phase as a meaningful judgment of whether an image or video is a
> deepfake - it reflects an ImageNet-pretrained feature extractor plus a
> random linear layer, nothing more. This will remain true until a real
> deepfake fine-tuning phase exists (see `docs/PROJECT_PLAN.md`).

This document will be more fully populated starting in the phase where a
first *fine-tuned* checkpoint is produced, and will cover: intended use,
training data summary (no raw data, only dataset provenance — see
`docs/DATASETS.md`), evaluation metrics, calibration/conformal
guarantees, known failure modes, and export formats.

## Candidate visual-encoder backbones (Phase 4, pretrained-only)

| | MobileNetV4-Conv-Small | EfficientNet-B0 |
|---|---|---|
| Role | Preferred candidate | Comparison baseline |
| Source weights | `timm/mobilenetv4_conv_small.e1200_r224_in1k` | `timm/tf_efficientnet_b0.in1k` |
| Pretraining | ImageNet-1k classification (not deepfake data) | ImageNet-1k classification (not deepfake data) |
| Head | Classifier removed, replaced with `nn.Linear(num_features, 1)`, randomly initialized | Same |
| Fine-tuned on deepfake data? | **No** | **No** |
| Parameters (incl. head) | 2,494,305 | 4,008,829 |
| Approx. FLOPs @ 224×224 | 369,453,696 | 769,072,064 |
| CPU latency, P50 (ms, bs=1) | 46.73 | 99.82 |
| RTX 4050 latency, P50 (ms, bs=1) | 22.19 | 39.26 |
| GPU peak memory (reserved) | 134.0 MB (2.2% of 6 GB) | 134.0 MB (2.2% of 6 GB) |
| ONNX FP32 export | Verified, parity max abs diff 3.3e-07 | Verified, parity max abs diff 4.4e-07 |

Full provenance (revision, license, file size, SHA-256) is in
`docs/DATASETS.md`. Performance measurements (latency, memory, ONNX
parity) are in `docs/EXPERIMENT_LOG.md`. Neither model's benchmark
numbers should be read as an accuracy comparison - no deepfake-labeled
evaluation exists yet; see `docs/DECISIONS.md` for why the final model
choice is deferred past this phase.

## Auxiliary component: YuNet face detector (Phase 2)

Not the deepfake-detection model itself, but a pretrained auxiliary
component used for face localization ahead of it. Provenance (source,
license, version, SHA-256) is tracked in `docs/DATASETS.md` rather than
duplicated here, since it is an external, frozen, off-the-shelf detector —
not something this project trains or fine-tunes.
