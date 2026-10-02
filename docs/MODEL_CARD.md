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

## Content Credentials (Phase 12)

- **What it is:** a C2PA provenance status reported next to, never
  inside, the model output.
- **How it is verified:** with the official CAI SDK against the
  official C2PA Trust List.
- **What it means:** credentials describe who signed the file and what
  they declared. Their absence is normal and says nothing about
  manipulation.

## Evidence hints (Phase 11)

- **What they are:** optional Grad-CAM heatmaps on the face crops,
  shown only when they pass an occlusion faithfulness check. They are
  labelled "Visual evidence hint — not proof" and never affect the
  verdict.
- **How faithful they are:** offline they only weakly predict
  occlusion effects (median Spearman ~0.2), so roughly half of the
  hints are withheld.
- **What they show:** they indicate where the model's score came from
  on a coarse 7×7 grid, not where a manipulation is.

## Serving (Phase 10)

- **How it is deployed:** a FastAPI service (`configuard.service`)
  running the ONNX FP32 student on CPU by default. Videos use adaptive
  4/8/16 frames; images use frame-level calibration. Both pass through
  the Phase 9 v1 quality gate.
- **What it returns:** every response carries a notice that this is an
  automated estimate validated only on FF++ development data. It is
  not a forensic determination.
- **Out-of-distribution inputs:** still images and non-FF++ video are
  outside what was evaluated.
- **When it declines:** no face, too few frames, an ambiguous
  calibrated prediction, or low quality all yield "uncertain", with a
  reason code.

## Phase 9c (rejected experiment; gate work ended)

- **What was tried:** a hybrid gate (Phase 9b noise handling and
  compression check + Phase 9 resolution check).
- **Result:** it closed the blur+noise bypass and the 0.75×
  over-trigger. However, it protected less against strongly downscaled
  real videos (false accusations 27% vs 24% held-out, 39% vs 25% on
  val) and exceeded the 6 ms budget, so it was not adopted.
- The production gate is unchanged (Phase 9), and its known gaps are
  accepted as final.

## Phase 9b (rejected experiment)

- **What was tried:** a hardened quality gate (noise-aware,
  offset-robust).
- **Result:** it closed the blur+noise bypass, but it protected less
  against strongly downscaled real videos (held-out false accusations
  48% vs 34%), so it was not adopted.
- The production gate is unchanged (Phase 9). Its known gaps still
  apply.

## Phase 9 quality gate (development data only)

- **What it does:** low-quality inputs (blur, low effective resolution,
  heavy blocking, small faces) turn confident verdicts into UNCERTAIN,
  with reason codes. It never changes real ↔ fake.
- **Clean val:** coverage −0.6 pp.
- **False accusations of real videos:** blur σ2 84% → 0%, 0.33×
  downscale 46% → 24.5%.
- **Not protected:** blur + noise (bypass), heavy noise (missed fakes),
  and residual low-resolution false accusations.

## Phase 8 deployment formats (development data only)

- **ONNX FP32** (9.5 MiB) is the default on CPU and GPU. It reproduces
  the PyTorch path on the full val split: video AUROC 0.9735, verdict
  agreement 99.9–100%.
- **Speed (model only):** CPU batch 1 takes 2.3 ms, and an adaptive
  video takes 8.8 ms at P50 (35.7 ms at P95).
- **FP16** (4.8 MiB) is equivalent in accuracy but not faster here.
- **INT8** (2.7 MiB) is not calibrated, has lower agreement, and is not
  for use.
- Package and hashes: `docs/EXPERIMENT_LOG.md` (Phase 8).

## Phase 7 temporal head (development data only)

- **Tested:** a 157k-parameter GRU over frozen frame embeddings.
- **Outcome:** no meaningful gain (val video AUROC +0.0014, CI
  includes 0), with more false positives at 0.5. Not used.
- **Production video score:** still the mean of frame logits over the
  4/8/16 nested frames, with Phase 6c/6d calibration.

## Phase 6e robustness (development data only)

- **Production default:** `student_distilled_p80` (unchanged).
- **Stress results (17 degradations of the val crops):**
  - video AUROC 0.900 on average and 0.692 in the worst case (strong
    noise), against 0.974 clean;
  - **strong blur or downscaling makes it call real videos fake**
    (FPR@0.5 up to 100%).
- **Robust experiment (`student_distilled_robust_p80`, not selected):**
  - worst case 0.825, but clean 0.927;
  - uncalibrated: the existing calibration artifacts refuse it.
- Do not deploy on re-encoded or low-resolution media until the
  shortcut is addressed.

## Phase 6d adaptive video inference (FF++ c23, development data only)

- **How it works:** videos are scored on 4 → 8 → 16 nested frames, and
  the analysis stops early only on a confident conformal singleton.
  Artifact: `adaptive_calibration.json`.
- **Dev (official val, 695 videos):**
  - 6.19 frames on average (−61%);
  - FPR 1.44% (real flagged as manipulated), miss 4.1%;
  - 13.2% uncertain; decided accuracy 95.9%;
  - AUROC 0.973.
- **Weakest class:** NeuralTextures, with 29.5% uncertain and 59%
  detected outright.
- **Coverage is empirical:** not guaranteed under shift.
- No test-split or cross-dataset results. Do not deploy.

## Phase 6c calibrated student (FF++ c23, development data only)

- **Model:** `student_distilled_p80` (`best.pt` SHA-256 `03f648b1…8957`),
  i.e. the 6b distilled config trained on 80% of FF++ train families,
  plus `calibration.json`.
- **Outputs:** calibrated P(fake) and a verdict (likely real / likely
  manipulated / uncertain). Frame/image and video levels are
  calibrated separately.
- **Dev results (official val, 695 videos), video level:**
  - AUROC 0.974; ECE 0.043 raw → 0.035 temperature-scaled.
  - Default mondrian α 0.05: 5.8% uncertain, 2.2% of real videos
    called "likely manipulated", 92.2% of decided videos correct.
  - Coverage 0.927, below the nominal 0.95.
- **Limitations:**
  - The coverage guarantee did not hold on FF++ val, so it should not
    be expected on other data.
  - No test-split, cross-dataset or compression-robustness results.
  - Do not deploy.

## Phase 6b students: MobileNetV4-Conv-Small (FF++ c23, validation only)

- **What they are:** two FF++-trained students, a BCE baseline and a
  GenD-distilled model (α 0.5, T 2).
- **Checkpoints:** on D: only (`checkpoints\distill\student_*\best.pt`,
  9.7 MiB fp32). 2.49M params. Inputs are 224×224 aligned RGB face
  crops with ImageNet mean/std. Output is one logit, P(fake) = σ(z).
- **Val results** (FF++ val, 695 videos):

  | | baseline | distilled |
  |---|---|---|
  | Video AUROC | 0.981 | 0.979 |
  | Frame AUROC | 0.963 | 0.956 |
  | Frame NLL | 0.504 | 0.216 |
  | Frame ECE | 0.071 | 0.033 |

  NeuralTextures is weakest (frame AUROC 0.92 / 0.90).
- **Not established yet:**
  - These are optimistic val numbers, since val also picked α/T and the
    epoch.
  - There is no test-split result, no cross-dataset result and no
    compression robustness.
  - There is no calibrated threshold or conformal "uncertain" class
    yet.
  - Do not deploy.
- `configuard.models` encoders loaded without these checkpoints remain
  untrained (see the warning above).

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
