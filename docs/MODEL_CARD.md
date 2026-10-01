# Model Card

Status: **No deepfake-detection model has been trained yet.** Phase 5
built the training pipeline but has only trained on *synthetic*
engineering data (see "Phase 5 training pipeline" below) - no checkpoint
from it is a deepfake detector. Phase 4 added two candidate visual-encoder backbones
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

### MobileNetV4-Conv-Small: source-model vs. reduced-head parameter count

| | Parameters |
|---|---|
| Source model (ImageNet-1k, 1000-class classifier) | 3,774,024 |
| After removing the 1000-class head (`num_classes=0` backbone only) | 2,493,024 |
| Reduction from removing the original classifier | 1,281,000 (~33.9%) |
| ConfiGuard-Lite final model (backbone + new binary head) | **2,494,305** |

The original classifier is `nn.Linear(1280, 1000)` (1,280,000 weights +
1,000 bias = 1,281,000 params - matches exactly). The new binary head is
`nn.Linear(1280, 1)` (1,280 weights + 1 bias = 1,281 params). Note:
`timm`'s own `model.num_features` attribute reports 960 for this
architecture's full (classified) config, which does **not** match the
backbone's actual 1280-dim pooled output with `num_classes=0` - this is
the same discrepancy `configuard.models.encoder.DeepfakeVisualEncoder`
already works around by probing the real output shape empirically rather
than trusting that attribute (see docs/DECISIONS.md, Phase 4).

## Training data status

FaceForensics++ c23 was acquired and validated in Phase 5b, and the
authors' official split (720/140/140 originals) was applied in Phase 5c
(`docs/DATASETS.md`). In Phase 5d, 79,280 matched 224×224 face crops
were extracted (4955 videos, 16 each; 9/1000 families quarantined).
**No model has been trained on them.** Known shortcut cues in FF++ (clip length,
native resolution) and the binding mitigations are in
`docs/KNOWN_ISSUES.md` / `docs/DECISIONS.md`. Any future FF++ result
must be reported per manipulation method, and stratified by native
resolution and duration. The Phase 5d audit found the F2F/NT width
change to be a centred crop with no residual geometric cue. Fakes are
somewhat blurrier than their matched reals (a genuine artifact).

## Frozen teacher: GenD CLIP-L/14 (Phase 6a, distillation only)

- **What it is:** `yermandy/GenD_CLIP_L_14` @ `891ce014…` (MIT),
  303.97M params, trained by its authors on the official FF++ c23 train
  split.
- **How we use it:** it is loaded frozen and only provides offline soft
  targets. It is not part of the shipped model and is never fine-tuned
  here.
- **Teacher quality on our crops:**
  - val frame AUC 0.960 (DF 0.985, F2F 0.965, FS 0.983, NT 0.906);
  - val video AUC 0.979;
  - train frame AUC 0.978.
- Test-split numbers do not exist by design. Provenance is in
  `docs/DATASETS.md`.

## Phase 5 training pipeline: engineering verification only

> ⚠️ **No accuracy claim.** Every Phase 5 training run used synthetic
> blue- vs. red-tinted checkerboards (`configuard.training.synthetic`)
> with a deliberately obvious signal, so that the training loop,
> checkpointing, resume, metrics, and logging could be verified without
> downloading any dataset. The resulting AUROC/balanced accuracy of 1.0
> only shows the pipeline can learn a trivial cue. It says **nothing**
> about detecting deepfakes. Every such result carries
> `SYNTHETIC_RESULT_DISCLAIMER` ("ENGINEERING_TEST_ONLY ...").

What Phase 5 verified (full numbers in `docs/EXPERIMENT_LOG.md`):

| | MobileNetV4-Conv-Small (default candidate) | EfficientNet-B0 (baseline) |
|---|---|---|
| RTX 4050 mixed-precision smoke training | Completed, 5 epochs | Completed, 5 epochs |
| Peak VRAM reserved (batch 8, AMP) | 154 MB (2.5% of 6 GB) | 460 MB (7.5% of 6 GB) |
| Mean training step, batch 8, GPU / CPU | ~77-92 ms / ~244-255 ms | ~165-183 ms (GPU) |
| Checkpoint size (weights + AdamW state + RNG + provenance) | 28.9 MiB | 46.4 MiB |
| Interrupt + resume vs. uninterrupted | identical (max param diff 0.0) | identical (max param diff 0.0) |

The real training configs (`configs/train/*.yaml`) target batch 32 for
MobileNetV4 and 16 × 2 accumulation for EfficientNet-B0. At the measured
peak those are well within 6 GB, but that is **unmeasured at those batch
sizes on real 224×224 face crops** until real data exists (Phase 3b).

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
