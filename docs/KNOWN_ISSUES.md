# Known Issues / Blockers

Format: one entry per issue. Mark resolved issues rather than deleting them.

---

## OPEN — Quality gate is bypassed by blur + noise (Phase 9)

Gaussian blur σ2 followed by σ4 noise passes every check: 0 reason
codes, and 14.4% of real val videos are still called "likely
manipulated". The median-denoised sharpness and spectral measures
recover enough high-frequency energy from the noise to clear the
0.5th-percentile thresholds. Blur + unsharp masking IS caught (FA 79%
→ 2%).

A noise-aware sharpness estimate, e.g. edge-width or structure-tensor
based, is needed. It must be designed and checked on train data, then
re-evaluated.

## OPEN — Gate over-triggers on mild 0.75× rescaling with the wrong reason code (Phase 9)

Crop-level 0.75× down/up-scaling (a 4/3 ratio) creates a period-4
interpolation pattern aligned with the block grid. The blockiness
signal reads it as HEAVY_COMPRESSION: 693/695 videos become uncertain,
although the detector is accurate there (accuracy 0.95, FA 0.05).

Real-world rescaling happens before alignment, so the pattern is
unlikely to be grid-aligned in practice. The fix is a blockiness
measure robust to other periodicities, validated on train data.

## OPEN — Residual false accusations after strong downscaling (Phase 9)

At 0.33× down-scaling the gate leaves 32% of real videos decided, and
24.5% of all real videos are still called "likely manipulated"
(ungated 46%). Combined with the Phase 6e blur/downscale shortcut, very
low-resolution genuine media remains a false-accusation risk.

## OPEN — Quality signals assume a crop-aligned block grid; noise is not gated (Phase 9)

- Blockiness measures 8/4-px blocks on the aligned crop. Compression
  applied to source frames before alignment produces a rotated and
  rescaled grid, so real-world HEAVY_COMPRESSION recall is unmeasured.
- Heavy noise lowers detection (noise σ10: 0% of fakes caught) but is
  not gated, because it does not cause false accusations.

## OPEN — INT8 exports are not calibrated (Phase 8)

Both static INT8 recipes shift logits relative to the FP32 model that
the 6c/6d calibration was fitted on:
- verdict agreement is 76% (MinMax) and 88–90% (Percentile);
- video AUROC loss is 0.012 and 0.006.

The INT8 files stay in the package for reference but are not
production-eligible until they are recalibrated on their own logits.

## OPEN — ORT CUDA provider re-plans on every batch-size change (Phase 8)

With a single session, alternating batch sizes cost about 330 ms per
switch. `ShapePinnedRunner` (one session per batch size) is required
for adaptive inference on GPU and adds GPU memory per shape. CPU is
not affected.

## OPEN — GPU runtime pinned to onnxruntime-gpu 1.23.2 (Phase 8)

Newer ORT GPU builds (1.30) need CUDA 13. This venv relies on the CUDA
12 / cuDNN 9 DLLs bundled with torch 2.5.1+cu121, which must be
imported before onnxruntime. Upgrading ORT, torch or the driver
requires re-running `scripts/export_student_onnx.py parity` and
`bench`.

## OPEN — Temporal head trained on in-sample features (Phase 7)

The GRU learned from embeddings of the student's own training videos,
on which the student is nearly perfect. That masks temporal structure
the frozen features might carry on unseen videos. The rejection is
therefore a result about this training setup, not proof that temporal
modelling cannot help.

Using the held-out calibration partitions instead would remove the
bias but consume data reserved for recalibration. Out-of-fold
features or end-to-end training would be needed, at extra cost.

## OPEN — Current model treats blur/downscaling as evidence of manipulation (Phase 6e)

On the stress suite, the production-default student
(`student_distilled_p80`) scores every real val video as P(fake) ≥ 0.5
after σ 2 Gaussian blur, and 85% after 0.33× downscaling. This matches
the Phase 5d finding that FF++ fakes are blurrier than their matched
reals. Heavy noise does the opposite (FPR 0, AUROC 0.69).

Real-world re-encoded or low-resolution genuine videos are therefore at
risk of false accusation. The Phase 6c/6d conformal thresholds were
fitted on clean crops and do not protect against this shift. The
robust experiment roughly halves these FPRs but loses 0.046 clean
AUROC. This needs fixing before any deployment.

## OPEN — Robust run early-stopped during the curriculum (Phase 6e)

The fixed 6b early stopping (clean val frame AUROC, patience 3)
stopped the robust model at epoch 6, two epochs after the severity
curriculum reached full strength (best epoch 3). The clean-AUROC loss
therefore partly reflects under-training, so the robust result is a
lower bound on what robust training can do.

## OPEN — Stress suite degrades aligned crops, not source frames (Phase 6e)

Real re-encoding happens before face detection and alignment and can
move or lose the face. The suite tests the classifier only, so
end-to-end robustness is not yet measured.

## OPEN — Calibration artifacts belong to the current checkpoint only (Phase 6e)

`calibration.json` and `adaptive_calibration.json` refuse any other
checkpoint (verified against the robust one). A retrained model has no
valid calibration until Phase 6c/6d fitting is rerun for it.

## OPEN — Adaptive inference abstains more, especially on originals and NeuralTextures (Phase 6d)

On dev, adaptive inference returns "uncertain" for 13.2% of videos
(fixed-16: 5.8%): 17.3% of originals and 29.5% of NeuralTextures.

The stricter stage α values make two-label sets at 16 frames common
for borderline videos. That is the intended price for the lower FPR
(1.44%) and miss rate (4.1%). A product surface needs to present
"uncertain" well. Any later change to α spending must be re-validated
on dev, and the FPR result must not be carried over without that.

## OPEN — Adaptive P95 latency is not lower than fixed-16 (Phase 6d)

Escalated videos run three sequential student calls (4 + 4 + 8 frames)
instead of one 16-frame batch. GPU P95 is 111 ms vs 85 ms; CPU P95 is
183 vs 177 ms. P50 and mean improve by about 40–60%. Timings exclude
face detection and alignment, which also scale with frames decoded.

## OPEN — Adaptive coverage is empirical under shift (Phase 6d)

The union-bound guarantee needs dev or deployment videos to be
exchangeable with conformal_cal, and Phase 6c showed they are not even
inside FF++. Dev coverage (0.964) happened to exceed nominal, but this
is not guaranteed on other datasets, compressions or
manipulation types.

## OPEN — Conformal coverage falls short on official val (calibration/dev shift) (Phase 6c)

The student does better on held-out TRAIN families than on official val
(video AUROC 0.988–0.990 vs 0.974). Thresholds fitted on conformal_cal
therefore cover at nominal level there (0.951 frame, 0.958 video at
α 0.05) but only 0.929 / 0.927 on val. The shortfall is concentrated in
fakes (0.92 / 0.91). Real coverage holds (0.950 / 0.978).

The guarantee assumes exchangeability between calibration and
deployment data, and that does not hold even inside FF++. Expect
larger gaps on other datasets and compressions. Options for later
phases: calibrate on data closer to deployment, use a smaller α,
or use a shift-aware / risk-controlling method.

## OPEN — Mondrian conformal lowers selective accuracy (Phase 6c)

At the default (video, α 0.05), decided cases are 92.2% correct. That
is below no abstention (93.0%) and below confidence-ranked abstention at
the same 5.8% rate (95.0%). The label-conditional fake threshold is
strict (q_fake 0.12, i.e. p ≥ 0.88 needed for a fake-only set). This is
deliberate, since it keeps false accusations of real videos at 2.2%,
but missed fakes rise. See `docs/DECISIONS.md`.

## OPEN — Video-level α 0.01 is not supported; frame-level sets are approximate (Phase 6c)

- conformal_cal has 71 real videos. α 0.01 needs n ≥ 99 per class, so
  q_real = 1 and 85% of videos are "uncertain".
- Frame-level conformal treats the 16 frames of a video as
  exchangeable units, but they are correlated. The frame-level
  guarantee is approximate; video level is the principled one.

## OPEN — Phase 6b numbers are model-selection estimates on val (Phase 6b)

The FF++ val split (140 originals, 695 videos after quarantine) did
three jobs: it chose α/T in the pilot, picked the early-stopping epoch,
and is the split we report. The reported AUROC/calibration numbers are
therefore optimistic. Only the untouched test split (a later, explicitly
approved phase) gives an unbiased estimate. A fresh val-internal holdout
was not possible without shrinking an already small split.

## OPEN — NeuralTextures is the weakest manipulation for students and teacher (Phase 6b)

Val frame AUROC on NeuralTextures is 0.924 for the baseline, 0.900 for
the distilled student and 0.906 for the teacher. The other methods are
0.97–0.98. NT also has the blurriest fakes (Phase 5d). Distillation
transfers the teacher's NT weakness, so NT may need method-specific
attention in the robustness phase.

## OPEN — Baseline student becomes overconfident as training continues (Phase 6b)

Baseline val NLL rose from 0.30 (epoch 1) to 0.60 (epoch 16) while
AUROC kept creeping up. Selection on AUROC therefore picks a poorly
calibrated checkpoint (ECE 0.071). Any decision threshold or conformal
set built on the baseline needs post-hoc calibration first. The
distilled student is much less affected (NLL 0.22).

## Informational — Student training is data-loader-bound (Phase 6b)

GPU step speed is about 1,060 img/s (NCHW). End-to-end training runs at
about 750 img/s with 12 PNG-decoding workers, and Windows worker spawn
adds about 70 s per run. An epoch takes about 77 s + 6 s val. A
decoded uint8 cache would remove the PNG cost, but it was not needed at
this scale.

## OPEN — GenD per-frame training lists are not independently verifiable (Phase 6a)

The claim that the teacher was trained on the FF++ train split rests on
three sources: the paper (3600 = 720 + 4×720 videos), the training code
(`trn_files = FF.train`) and the HF model card ("the model from
Tab. 2"). The actual path lists are in the gated HF dataset
`yermandy/GenD`, which returned 401 without authentication. If FF++
val/test frames had leaked into that training, teacher logits on our
val crops would look optimistic. Our test split is unaffected because it
is never passed to the teacher. To close this, download the lists with
approved HF access and diff them against the Phase 5c split.

## OPEN — Teacher crops differ from GenD's own preprocessing (Phase 6a)

GenD was trained on its own detector crops (scale 1.3, `detector.py`).
Our Phase 5d crops are 5-point aligned with margin 0.25 and are already
224×224, so CLIP resizing does nothing. The teacher is used on a
slightly different crop distribution. This is acceptable for
distillation targets, but teacher AUCs here are not comparable to the
paper's numbers.

## Informational — Teacher caching is I/O/decode-bound (Phase 6a)

The GPU sustains about 90 img/s (fp16, bs 64), but end-to-end caching
runs at about 30–60 img/s. It reads, re-hashes and decodes each PNG on
D:. The first trial shard ran at about 6 img/s (cold start); the resumed
trial ran at about 60 img/s.

## OPEN — Reflect-101 borders mirror the head when a face is near the frame edge (Phase 5d)

About 28% of accepted crops include some out-of-frame area. 6.6% have
more than 10% of the crop out of frame, and 0.8% have more than 25%.
Reflect-101 fills that area with mirrored content (a doubled hairline,
and occasionally an upside-down partial face above a face at the top
edge). The share is **the same in every class** (any: 26.2–28.4%;
>10%: 6.1–6.8%; >25%: 0.74–0.83%), and real/fake pairs share content,
so it is not a label cue. It is visible noise, however. Phase 6+ may
want to evaluate a different fill or a tighter margin for edge faces,
under a new config tag. Black padding stays excluded.

---

## OPEN — Fakes are slightly blurrier in aligned crops (genuine artifact, resolution-correlated)

Paired Δ Laplacian variance (fake − matched real) is −22.6 for DF, −5.7
for F2F, −0.6 for FS and −51.0 for NT. The AUC vs original is
0.456 / 0.491 / 0.506 / 0.415. Sharpness also correlates with source
height (ρ 0.38). It is a real property of the manipulations, but a
model could over-rely on it, and it is fragile under recompression.
**Mitigation:** class-independent blur, resize and JPEG/H.264 jitter in
the augmentation phase (`docs/DECISIONS.md`), plus resolution-stratified
evaluation.

---

## OPEN — Quarantine removes a few of the most broken fakes; quarantine rate is higher at 1080p

9/1000 families are quarantined (45 videos, exactly 9 per class).
Five of them are fake-only failures on visibly broken manipulation
output, a mild selection toward harder fakes. The quarantine rate by
native resolution is 1920×1080: 25/615 videos (4.1%) vs 1280×720:
10/1625 and 640×480: 5/1467. The rate by duration bucket is ≤1.1%.
Real and fake are always removed together, so this cannot become a
label shortcut. It does make the training data slightly
under-represent 1080p. Recorded in `quarantine.jsonl`; not restored
manually.

---

## OPEN — Residual aligned-geometry signal for Deepfakes (AUC 0.62)

A train→test probe on aligned landmark geometry alone separates
Deepfakes from originals at AUC 0.617. The other methods score
0.52–0.53, and real-vs-fake overall scores 0.537. This is consistent
with Deepfakes rendering the *source* identity's facial proportions,
i.e. the manipulation itself, rather than preprocessing. Mild
class-independent geometric jitter is recommended so that the
detector's evidence is not dominated by landmark geometry.

---

## OPEN — `scripts/check_storage.py` does not load `.env`

Run directly, it reports every `CONFIGUARD_*` path as "not set". The
other entry points call `configuard.env_loader.load_dotenv` first. The
Phase 5d scripts resolve and check the D: paths themselves (store root
must be on D:, outside the repo, above the free-space floor). This is a
one-line fix for a later phase; not changed here to keep Phase 5d scoped.

---

## RESOLVED — FFmpeg not installed

**Detected:** Phase 0 environment scan (2026-09-29).

**Resolved:** 2026-09-29, with explicit user approval, via
`winget install --id Gyan.FFmpeg -e --source winget`. Installed
FFmpeg 9.0.2 (full build, gyan.dev) to
`%LOCALAPPDATA%\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-9.0.2-full_build\bin`,
added to the user `PATH` environment variable by winget. Verified with
`ffmpeg -version` and `ffprobe -version` (both report version 9.0.2). See
`docs/EXPERIMENT_LOG.md` for full output.

**Note:** Shell processes already running at install time (including this
agent session's Bash/PowerShell tool shells) do not see the updated `PATH`
until restarted — a new terminal window picks it up automatically. Not an
environment defect, just a live-session caveat.

---

## RESOLVED — Limited free disk space on `C:`

**Detected:** Phase 0 environment scan (2026-09-29). `C:` drive had ~20 GB
free out of 200 GB at time of scan; dropped further to ~12.9 GB by Phase 3
(`.venv`/pip cache/FFmpeg - see the "Disk space update" entry below).
Face manipulation / deepfake datasets and multiple model checkpoints
would easily exceed this if stored on `C:`.

**Resolved:** 2026-09-29, via a user-approved read-only storage audit
followed by explicit storage configuration:
1. Audit found `D:` (a separate local fixed NTFS volume, not OneDrive-
   synced) with 92.9 GB free - see the storage-audit report in the
   conversation history and `docs/EXPERIMENT_LOG.md`.
2. Created `D:\ConfiGuard-Data\{datasets,cache,checkpoints,outputs}` and
   configured local `.env` (never committed) to point
   `CONFIGUARD_DATA_DIR`/`CONFIGUARD_CACHE_DIR`/
   `CONFIGUARD_CHECKPOINT_DIR`/`CONFIGUARD_OUTPUT_DIR` there - see
   `docs/ARCHITECTURE.md` for the exact layout.
3. Purged the pip download cache (`pip cache purge`, official command;
   nothing else was deleted) - reclaimed 3,365.8 MB on `C:`.
4. Verified every directory exists, is actually writable (real write
   test, not just a permission bit check), is outside both the repo and
   OneDrive, and is recognized by `scripts/check_storage.py` - see
   `docs/EXPERIMENT_LOG.md` for full command output.

`C:` free space: 12.92 GB -> **16.05 GB** after the pip cache purge.
`D:` free space: 92.93 GB, unaffected (new directories are empty).
Real datasets will now be stored on `D:`, not `C:`, avoiding this
constraint going forward.

---

## OPEN — No system CUDA Toolkit (`nvcc`) installed

**Detected:** Phase 0 environment scan (2026-09-29).

**Impact:** None currently — PyTorch's CUDA wheel (cu121) bundles its own
CUDA runtime and does not require a system Toolkit for training/inference.
Would only matter if a future phase needs to compile custom CUDA kernels.

**Action needed:** None for now. Revisit only if a custom-kernel dependency
is introduced.

---

## OPEN — Video rotation metadata not observed from this ffmpeg build's lavfi-synthesized clips

**Detected:** Phase 2, writing `configuard.media.decode._extract_rotation`
and its test (2026-09-29).

**Impact:** `ffmpeg -metadata:s:v:0 rotate=90` and `-display_rotation 90`
did not produce a `rotate` tag or Display Matrix side data on an mp4 muxed
from a `lavfi testsrc` input, verified by direct `ffprobe -show_streams`
inspection, with this project's installed ffmpeg 9.0.2. Real phone-
recorded video (the actual target use case) reliably carries this
metadata in practice, so this is a test-fixture limitation, not
necessarily a production one - but it has not been verified against a
real rotated video file in this environment. `_extract_rotation`'s parsing
logic (both the legacy `rotate` tag and Display Matrix side-data
conventions) is unit-tested directly against synthetic ffprobe JSON
instead - see `docs/DECISIONS.md`.

**Action needed:** When a real rotated sample video becomes available
(e.g. user-provided, not downloaded), re-verify `extract_video_metadata`
end-to-end against it. Not blocking - rotation is reported, defaulting
safely to 0, and is not yet used to auto-correct crops (see
`docs/ARCHITECTURE.md`'s edge-case table).

---

## RESOLVED — `.gitignore` silently dropped `.gitkeep` placeholders in ignored directories

**Detected:** Phase 2, while adding `models/` to `.gitignore` for the
YuNet asset (2026-09-29).

**Resolved:** 2026-09-29. Changed `data/`, `checkpoints/`, `models/`,
`cache/`, `outputs/` patterns to `data/*`, `checkpoints/*`, etc. so the
`!dir/.gitkeep` exceptions actually work (git cannot re-include a file
under an excluded *directory*, only under excluded *contents* of a
non-excluded directory). See `docs/DECISIONS.md` for detail. Verified via
`git check-ignore -v` on all five `.gitkeep` paths.

---

## OPEN — DF40 and DeeperForensics-1.0 adapter structures are low-confidence

**Detected:** Phase 3, writing `configuard/datasets/adapters/known_datasets.py`
(2026-09-29).

**Impact:** `make_df40_adapter()`'s `metadata.jsonl` assumption and
`make_deeperforensics_adapter()`'s `source_videos/` bucket name were not
independently verified against real dataset documentation - see
`docs/DATASETS.md`'s "structure confidence" column. Pointing either
adapter at a real local copy may fail `validate_structure`/raise
`DatasetAccessError` even with legitimate data present, if the real
layout differs.

**Action needed:** Once the user obtains real access to either dataset,
confirm the actual on-disk/metadata layout and adjust the corresponding
`FolderConventionSpec`/`MetadataSidecarSpec` in `known_datasets.py`
accordingly (this is expected, routine adjustment - the specs were
designed to be easy to update).

---

## OPEN — Near-duplicate detection doesn't cover video samples

**Detected:** Phase 3, writing `configuard/datasets/duplicates.py`
(2026-09-29).

**Impact:** `find_near_duplicate_images` only processes `SampleMediaType.IMAGE`
samples. Near-duplicate video clips (e.g. the same source re-encoded at a
different compression level, common across FF++'s `raw`/`c23`/`c40`
tiers) are not detected.

**Action needed:** Extend duplicate detection to videos once
representative-frame extraction is wired in (the machinery already
exists in `configuard.media`, just not connected to
`configuard.datasets` yet - intentional, per this phase's scope).

---

## Disk space update (2026-09-29, Phase 3)

Free space on `C:` is now **~12.9 GB** (down from ~21.8 GB at Phase 0),
consumed by `.venv` packages (PyTorch, OpenCV, etc.) — see the original
"Limited free disk space" entry above, still OPEN and now more pressing
ahead of Phase 3b (wiring in a real dataset). Since resolved by the
approved storage audit + configuration step (see the "Limited free disk
space on C:" entry, now RESOLVED, and `docs/ARCHITECTURE.md`).

---

## RESOLVED — `pip install timm onnx onnxruntime` silently downgraded PyTorch to a CPU-only build

**Detected:** Phase 4, immediately before running the model smoke test
(2026-09-29).

**Impact:** Installing `timm`, `onnx`, and `onnxruntime` without pinning
`torch` caused pip to resolve a *newer* `torch` release from the default
PyPI index (2.14.0+cpu) as a transitive dependency, silently replacing
the Phase 0-installed CUDA build (2.5.1+cu121). `torch.cuda.is_available()`
started returning `False` even though the RTX 4050 and its driver were
untouched (confirmed working via `nvidia-smi`) - this would have made
every Phase 4 "RTX 4050 latency/peak memory" measurement silently
CPU-only and wrong.

**Resolved:** 2026-09-29. Reinstalled the correct build explicitly:
```
pip install "torch==2.5.1" --index-url https://download.pytorch.org/whl/cu121
```
Verified `torch.cuda.is_available()` returns `True` and
`torch.__version__` reports the `+cu121` build again before running any
GPU benchmark - see `docs/EXPERIMENT_LOG.md`.

**Cascading follow-on issue:** the same `pip install timm onnx
onnxruntime` also pulled in `torchvision==0.29.0` (compatible with the
accidental `torch 2.14.0+cpu`, not with `torch 2.5.1`). After reinstalling
`torch==2.5.1+cu121`, `import timm` started failing with `RuntimeError:
operator torchvision::nms does not exist` (a torch/torchvision ABI
mismatch) - `timm/__init__.py` unconditionally imports a submodule that
imports `torchvision`, even though none of this project's code uses it
directly. Fixed by also reinstalling a matching version:
```
pip install "torchvision==0.20.1" --index-url https://download.pytorch.org/whl/cu121
```
`torch==2.5.1` pairs with `torchvision==0.20.x` per PyTorch's own
compatibility matrix.

**Action needed for future dependency installs:** when installing any new
package into this environment, run
`python -c "import torch, torchvision; print(torch.__version__, torch.cuda.is_available(), torchvision.__version__)"`
immediately afterward to catch a silent CUDA-build downgrade *or* a
torch/torchvision version-skew before it contaminates a GPU benchmark or
breaks `import timm` - a plain `pip install <package>` can re-resolve
`torch` (and thus require a matching `torchvision`) from the default
index even when compatible versions are already installed, if the new
package's dependency metadata doesn't pin tightly enough for pip's
resolver to leave them alone.

---

## Informational — HF hub cache uses non-symlinked storage on this Windows machine

**Detected:** Phase 4, `scripts/download_baseline_models.py` run
(2026-09-29).

**Impact:** `huggingface_hub` warned that Windows Developer Mode isn't
enabled (or Python isn't running elevated), so its cache falls back to
full file copies instead of symlinks for deduplication. Purely a
disk-space-efficiency note - functionality is unaffected, and Phase 4's
two small models (~15 MB and ~21 MB) make this immaterial. Would matter
more if many large checkpoint variants of the same model were cached.

**Action needed:** None. If cache size becomes a concern later, enabling
Windows Developer Mode addresses it.

---

## OPEN — Balanced sampling (with replacement) can yield single-class batches; BatchNorm then erases the class cue

**Detected:** Phase 5 (2026-09-30), from the failing overfit test (see
`docs/EXPERIMENT_LOG.md`).

**Impact:** `WeightedRandomSampler` draws with replacement. On a tiny
set with a small batch, a batch can hold only one class. Train-mode
BatchNorm then normalizes away the between-class difference inside that
batch and the loss spikes: measured at batch 4 over 8 samples, loss ~0
→ 2.1-2.9 at exactly the epochs containing an all-real batch. At a real
batch size of 32 with balanced classes, P(one-class batch) ≈ 2 × 0.5^32,
which is negligible, so this mainly affects smoke/test scales.

**Mitigation now:** The overfit test uses full batch without
replacement. Configs default to batch 32 (MobileNetV4) / 16 × accumulation
2 (EfficientNet-B0).
**Possible fix later:** a per-epoch class-stratified sampler without
replacement, or GroupNorm/frozen BN statistics for very small batches.

---

## OPEN — Threshold-dependent metrics are unreliable while BatchNorm running statistics are still settling

**Impact:** In the from-scratch 12-step overfit test, eval-mode AUROC is
1.0 while balanced accuracy at threshold 0.5 is 0.5: BN running
mean/var haven't converged, so eval-mode logits are offset (ranking
intact, threshold wrong). Early-training threshold metrics (sensitivity,
specificity, F1, balanced accuracy) should not be read as meaningful.
Checkpoint selection uses threshold-free AUROC for this reason, and the
planned calibration phase re-derives the operating threshold anyway.

---

## OPEN — Resume granularity is one epoch

Checkpoints are written only at epoch boundaries (the basis of the
bit-exact resume guarantee). An interruption mid-epoch loses that epoch's
progress. Resuming mid-epoch would need DataLoader iterator/sampler
position state and is not implemented.

---

## OPEN — GPU resume was bit-exact here, but CUDA doesn't guarantee it

`set_global_seed` enables `cudnn.deterministic` and disables
`cudnn.benchmark`; every RTX 4050 smoke resume measured max parameter
diff **0.0**. CUDA kernels (e.g. atomics in some backward ops) are not
guaranteed deterministic across driver/library versions, so the CUDA
resume test asserts < 1e-4 rather than exact equality. CPU resume is
asserted exactly equal.

---

## OPEN — AMP still skips a few optimizer steps at start-up

With `amp_init_scale=1024`, MobileNetV4 skipped 3 of its first 20
steps (fp16 gradient overflow while GradScaler calibrates); EfficientNet-B0
skipped 0. Skipped steps are now counted and logged
(`optimizer_steps_skipped_amp`) and don't advance the LR schedule. On
real runs (thousands of steps) this is noise. On very short runs, check
the counter; before this fix, a 6-step run silently skipped every step
(`docs/DECISIONS.md`).

---

## OPEN — Checkpoints load with `torch.load(weights_only=False)`

Needed for the Python/NumPy RNG state tuples stored for exact resume.
Unpickling can execute code, so **only load checkpoints produced by this
project** - never a checkpoint downloaded or received from elsewhere.
A future export path for sharing weights should save a weights-only
`state_dict` (or ONNX, per the roadmap) separately.

---

## OPEN — Mixed image+video manifests are refused

`build_manifest_dataset` raises on a manifest containing both IMAGE and
VIDEO samples (the draft silently dropped the images). Joint image+video
training needs separate loaders or a mixed-shape collate function; to be
decided when real datasets are wired in (Phase 3b).

---

## OPEN — No real-data training has happened; nothing here measures detection ability

Every Phase 5 training run used synthetic tinted checkerboards and the
full-frame mock face detector. The production path (YuNet detector on
real faces, real manifests from `configuard.datasets`) is wired and its
entry point is exercised by the CLI mismatch-rejection check, but it has
not trained on real media, because no dataset has been provided
(Phase 3b). `is_finetuned` remains `False`; `docs/MODEL_CARD.md` makes
no accuracy claim.

---

## Informational — Phase 5 smoke artifacts on D: (≈ 692 MB)

`D:\ConfiGuard-Data\checkpoints\smoke\` (687 MB, 6 runs incl. 3
superseded preliminary runs) and `D:\ConfiGuard-Data\outputs\smoke\`
(5.4 MB) are safe to delete at any time. They are kept only as evidence
for `docs/EXPERIMENT_LOG.md`. Each future `--smoke` run adds about 60-200 MB
(latest + best, main + resume run).

---

## RESOLVED — Official FF++ download script can hang forever on a dead connection

**Detected:** Phase 5b (2026-09-30), live: the `original` stream's partial
file froze at 3,375,104 bytes for about 8 minutes (`urllib.request.urlretrieve`
has no timeout).
**Resolved:** a stall watchdog in `scripts/download_faceforensics_c23.py`
kills and relaunches a stalled stream after 5 minutes without progress
(finished files are skipped). It auto-recovered a later Face2Face stall
(`stalled_restart` at 17:35:14 UTC). The same fix made the low-space stop
kill the whole Windows launcher → interpreter process tree (previously
it would have orphaned the downloader). See `docs/DECISIONS.md`.

---

## RESOLVED (Phase 5c) — FaceForensics++ has no train/val/test split applied

The approved source provides none, so none was applied or invented
(`docs/DATASETS.md`). Any training on FF++ is blocked until either
(a) the user approves fetching the official split JSONs from the
authors' GitHub repo, or (b) a leakage-safe split is generated with
`configuard.datasets.splitting` as an explicit, documented decision.
Either way, the split must keep each of the 500 leakage groups (2
originals + 8 fakes) intact.

**Resolved 2026-09-30 (Phase 5c):** option (a). The official split at
commit `b952e41c` was applied with 0 reconciliation or leakage problems
(`docs/DATASETS.md`).

---

## OPEN — FF++ manipulated classes differ from originals in duration and resolution

Measured on the c23 download (Phase 5b):
- **Duration:** FaceSwap and NeuralTextures clips cap at 41.52 s, while
  originals, Deepfakes and Face2Face reach 72.56 s. The same pair can
  differ, e.g. `033_097` is 32.4 s (Deepfakes) vs 19.1 s (NeuralTextures).
- **Resolution:** Face2Face and NeuralTextures have 14 distinct
  resolutions (348 videos at 640×480), versus 29 for originals, Deepfakes
  and FaceSwap (257 at 640×480).

**Risk:** clip length, frame count, or native resolution could become
shortcut cues. The preprocessing/training phases must sample frames by
position within each clip (Phase 2 nested sampling already does) and
resize crops to the fixed 224×224 contract. They must not feed raw
resolution or length to a model.

---

## Informational — FF++ has no identity labels

FF++ publishes no subject identities; `identity_id` stays `null`
(never inferred). Leakage safety relies on the official filename
lineage instead (`parent_sample_id` = target original, `paired_sample_id`
= source original). Possible same-person reappearances across different
original YouTube IDs are neither documented nor detectable here.

---

## OPEN — FF++ shortcut cues measured per split (Phase 5c audit)

Consistent across train/val/test (full tables in the split report):
- **Duration:** FaceSwap and NeuralTextures are truncated. Train median
  13.8 s vs 16.6 s for originals/Deepfakes/Face2Face; max 30.0 s vs
  53.6 s. The same holds in val (13.3 vs 16.2) and test (14.6 vs 16.9).
- **Native resolution:** Face2Face and NeuralTextures round the frame
  width **down to a multiple of 16** relative to the target original, in
  282 of 1000 videos each (854→832 ×98, 656→640 ×66, 600→576 ×48,
  720→704 ×27, 654→640 ×20, ...; 22 distinct changes). Deepfakes and
  FaceSwap keep the exact native resolution. No aspect-ratio metadata
  is set.
- ~~**Unverified:** whether that width change is a crop or an anisotropic
  squeeze.~~ **RESOLVED in Phase 5d: centred crop.** All 562/562
  width-changed accepted fakes register as a centred crop: ECC sx
  1.0000 vs 0.973 for a squeeze, and the landmark interocular ratio is
  0.9985/1.0008 vs 0.973. No geometric cue survives alignment
  (`docs/DECISIONS.md`).
- **Phase 5d crops:** sampling uses a per-family shared range, so the
  sampled span is identical for real and fake (`shared_frame_count`
  AUC 0.500). Source height/width/duration reach neither the crops nor
  the model-facing manifest.

Mitigations (binding, `docs/DECISIONS.md`): fixed per-video frame budgets
sampled by position; aligned 224×224 crops only; no metadata as input.
**Also required:** evaluation should report per-method results, and a
duration-stratified and resolution-stratified breakdown on test, so any
residual shortcut shows up as a gap.
