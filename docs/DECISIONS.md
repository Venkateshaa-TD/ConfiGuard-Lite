# Architectural Decisions

Format: one entry per decision, newest first.

---

## 2026-10-01 — Phase 7: residual GRU head rejected; mean-frame-logit aggregation kept

- **Design tested:** a residual GRU over frozen embeddings (zero-init
  output, so it starts as the current aggregation) with nested
  4/8/16 sequences from one cache. 157k params and 1–2 ms per video.
- **Training data choice:** the GRU trains on `final_train` only, the
  same families the student saw. temp_cal and conformal_cal stay
  untouched for a future recalibration, and val stays the dev set.
  - Cost: the student's features on its own training videos are
    near-separable (train loss 0.05 → 0.01), so the GRU has almost
    nothing to learn and generalises as a small bias shift.
- **Result vs the pre-registered rule:**
  - clean ΔAUROC +0.0014 (CI includes 0) and stress Δ −0.0024, so
    there is no meaningful improvement;
  - FPR@0.5 0.165 → 0.273 (the scores move toward "fake").
  - Rejected. The Phase 4/6 mean-frame-logit aggregation and the
    Phase 6c/6d calibration artifacts remain the production path.
- **Kept:** the embedding cache and the GRU code/checkpoint as
  experiment artifacts.
- **Possible later options, not started:**
  - out-of-fold student features (k-fold students) for a fair
    temporal head;
  - short end-to-end fine-tuning with temporal augmentation;
  - temporal features that frame averaging cannot see (e.g.
    landmark/flicker statistics).

---

## 2026-10-01 — Phase 6e: robust augmentation recorded as an experiment; current p80 model stays the default

- **Rule:** prefer the robust model only if worst-case and mean
  degraded video AUROC both improve and clean video AUROC drops by at
  most 0.01. Measured: worst case +0.133, mean −0.006, clean −0.046.
  Not preferred. `student_distilled_p80` and its 6c/6d calibration stay
  the default. The robust checkpoint is kept on D: as an experiment.
- **Augmentation design:**
  - It is label-free by construction (same probabilities and severity
    distributions for every class and method), avoiding a new
    class-conditional cue.
  - It runs after the 6b blur/jitter.
  - One compression per sample: JPEG or an H.264-style emulation. The
    emulation is cheap enough for training workers; real libx264 is
    reserved for the stress suite so evaluation does not just test the
    emulation.
  - Severities stop at "moderate", and the suite's "severe" levels are
    outside the training range.
- **Distillation targets** stay the clean cached GenD logits. The
  student learns to reproduce clean-image teacher judgments from
  degraded views. GenD is not re-run on degraded crops.
- **Stress suite:**
  - It degrades aligned crops, not full frames before detection: fast
    and deterministic, but it ignores detector/alignment failures
    under degradation.
  - It is versioned by a tag over conditions + val manifest + ffmpeg
    version + x264 settings.
- **Engineering:** DataLoader workers are single-threaded
  (OMP/OpenBLAS/MKL = 1, `cv2.setNumThreads(1)`).
  - Long evaluations run sequentially per condition, save each result
    at once, and stop at a 4 GB available-RAM floor
    (`configuard.memory_guard`).
  - Both changes came from incidents in this phase.
- **Not adopted, for a later phase if approved:**
  - early stopping that waits for the curriculum or selects on a
    degraded dev split;
  - a blur/downscale-specific fix for the shortcut;
  - recalibration of any retrained model.

---

## 2026-10-01 — Phase 6d: adaptive 4 → 8 → 16 with per-stage calibration and α spending 0.015/0.015/0.02

**Stages:**
- The Phase 5d nested sampling contract is reused: 4 ⊂ 8 ⊂ 16 slots.
- Stage k scores only the slots not already scored, and its video
  score is the mean logit over the k-set. This is the same aggregation
  as Phase 4/6c, so no GRU is involved.
- Per-frame logits are cached per video, so a frame is never re-scored.

**Calibration:**
- Each stage has its own temperature (on temp_cal) and mondrian
  thresholds (on conformal_cal). A 4-frame mean is noisier than a
  16-frame mean: T is 0.72 / 0.68 / 0.62 for 4 / 8 / 16 frames.
- Everything is stored in `adaptive_calibration.json` (`kind:
  adaptive-4-8-16`) with the policy inside the content hash, bound to
  `best.pt` like the 6c artifact.
- `build_artifact` takes `required_levels` and `extra`; Phase 6c
  artifacts are byte-compatible.

**Stopping rule:**
- Stop at 4 or 8 frames only on a singleton set at that stage's α_k;
  otherwise escalate.
- At 16 frames, a singleton gives a verdict; an empty or two-label set
  gives "uncertain".

**α spending:**
- 0.015 + 0.015 + 0.02 = 0.05. Under exchangeability, the union bound
  keeps the probability that the committed label is wrong at most 0.05,
  regardless of which stage stopped.
- Each α_k must be at least 1/(n_min + 1) = 1/72 with 71 real
  calibration videos, which is why the split is not
  0.01/0.01/0.03 (the script refuses unsupported α).
- Equal 0.05 at every stage (no spending) cut frames further (4.4) but
  doubled the dev FPR (4.3%), so it was rejected.
- **Coverage under domain shift is empirical, not guaranteed:**
  - dev coverage 0.964 exceeded nominal here;
  - the single-stage 6c version fell short (0.927) on the same split.

**Measured trade-off:** 61% fewer frames and lower FPR than fixed-16,
paid for with more "uncertain" (13% vs 6%), concentrated on originals
and NeuralTextures.

---

## 2026-10-01 — Phase 6c: family-component partitions, per-level temperature, label-conditional (mondrian) conformal at α 0.05 as default

**Partitions:**
- Calibration data comes from the official TRAIN families, not from
  val, so val stays an independent development set and test stays
  sealed.
- Assignment is by donor-linked component, i.e. FF++ reciprocal pairs:
  a family and the donor whose face appears in its fakes never land in
  different partitions.
- The partition file is pinned by SHA-256 into the model's provenance.

**Model:** the 6b distilled config (α 0.5, T 2) is retrained on
`final_train` with no new tuning; early stopping still uses val. The 6b
checkpoints stay as they are.

**Temperature scaling:** one scalar T per level, fitted by NLL on
temp_cal.
- Frame and video are fitted separately: the video score is a mean of
  16 correlated frame logits and is far less noisy than one frame.
- Results: frame T 0.95, effectively 1; video T 0.62.

**Conformal:** split conformal on conformal_cal, nonconformity
1 − p(true class), over temperature-scaled probabilities.
- Default mode is **mondrian** (one threshold per class). FF++ is 80%
  fake, and marginal conformal spends its error budget on reals: on
  dev, 18% of real videos got a fake-only set. Mondrian gives 2.2%,
  with per-class coverage guarantees.
- Default α is **0.05**. α 0.01 needs at least 99 real calibration
  videos (we have 71).
- Empty or two-label sets become `UNCERTAIN`.
- Accepted cost: mondrian decisions are less accurate than
  confidence-ranked abstention at the same abstention rate. It shifts
  the fake boundary to p ≥ 0.88, trading missed detections for fewer
  false accusations. Revisit with a risk-controlling threshold
  (conformal risk control / LTT on the false-accusation and miss
  rates) in a later phase.

**Artifact:** `calibration.json` beside `best.pt` holds the checkpoint
SHA-256, model config (+ SHA-256), model and calibration provenance,
all fitted (mode, α) thresholds and a content SHA-256.
`load_calibration` refuses any mismatch or edit.

---

## 2026-10-01 — Phase 6b: distil with α 0.5, T 2; keep both students; prefer the distilled one for calibration-driven stages

**Loss:** `(1−α)·BCE(z, y) + α·T²·BCE(σ(z/T), σ(m/T))`.
- `m = logit_fake − logit_real` is the cached GenD margin, so `σ(m)` is
  GenD's softmax P(fake) exactly.
- The student keeps its single-logit head, so the Phase 4 export and
  inference contracts are unchanged.

**Choice:** a 4-epoch × 25k pilot over α ∈ {0.5, 0.9} × T ∈ {1, 2, 4}
picked α 0.5, T 2 (val frame AUROC, tiebreak NLL). α 0.9 hurt every
manipulation.

**Outcome:**
- Distillation does not improve discrimination. Video AUROC is tied
  (CI includes 0) and frame AUROC is 0.007 lower.
- It halves NLL/ECE and raises balanced accuracy at 0.5.
- The ranking is the same as the baseline's, and the later
  calibration/conformal "uncertain" class depends on well-behaved
  probabilities. So the distilled student is the default candidate for
  the next stages.
- The baseline is kept as the control.
- Neither is a final model: there is no robustness training yet and no
  test evaluation.

**Implementation choices:**
- New `configuard.distill` package instead of extending
  `configuard.training`. That trainer is built around raw-media
  `Sample` manifests with on-the-fly face detection; Phase 6b trains on
  the hash-verified Phase 5d crop manifests and needs per-row teacher
  targets.
- Reused pieces: Phase 4 encoder registry, Phase 5 AUROC/AP metrics,
  warm-up-cosine schedule, storage paths, Phase 5d atomic writes and
  Phase 6a split protection.
- Sampling: class × method balancing (real 1/2, each manipulation 1/8),
  with replacement, one draw per train row per epoch. The epoch draw is
  seeded by (seed, epoch); augmentation is seeded by (seed, epoch,
  index). Runs that share a seed therefore see identical pixels
  regardless of worker layout.
- Augmentation is the Phase 5d recommendation only: mild Gaussian blur
  and horizontal x-scale/shift (reflect-101). The functions never
  receive the label. There is no flip, colour, or compression
  augmentation (robustness is a later phase).
- NCHW, not channels_last: 3× faster on this GPU (EXPERIMENT_LOG).
- Checkpoints hold only tensors and primitives and load with
  `torch.load(weights_only=True)`. `best.pt` is weights + config +
  provenance (crop/teacher tags, manifest hashes). `last.pt` adds
  optimizer/scheduler/scaler state for epoch-level resume, refused on
  config or provenance mismatch.

---

## 2026-10-01 — GenD CLIP-L/14 teacher: rebuilt locally, strict-loaded from hash-pinned weights, frozen, fp16 bs 64; logits cached for train/val only

**Source:** Hugging Face `yermandy/GenD_CLIP_L_14` @
`891ce014a0308386c4d7d25b3dcf436a22db5504` (MIT, "the GenD (CLIP)
model from Tab. 2" of Yermakov et al., WACV 2026, arXiv 2508.06248).
Training code reviewed at github.com/yermandy/GenD @ `387a422`.
`model.safetensors` SHA-256 `d76f0bdf…6833` is pinned in
`configuard.teacher.gend` and re-checked on every load.

**Not running the official `modeling_gend.py`:** its constructor calls
`CLIPModel.from_pretrained("openai/clip-vit-large-patch14")`, which
would download a second, unrequested 1.7 GB model just to overwrite
its weights. We rebuild the same `CLIPVisionTransformer` from a fixed
ViT-L/14 config and **strict-load** all GenD tensors (names and shapes
must all match, 303,968,258 params). The forward is the official one:
`pooler_output → L2 normalise → Linear(1024, 2)`, with index 0 = real
and 1 = fake.

**FF++ train-data check (task requirement):**
- The paper says it trains on FF++ c23 "3600 videos, of which 720 are
  real and 4×720 fake", which is exactly the official train split
  (720 real + 4×720 fake). The 115k frames match 3600 × 32.
- The training code (`src/exp/wacv_rebuttal.py`) sets
  `trn_files = files.FF.train`.
- Model selection used a custom validation set (DeepSpeak v1.1/v2,
  CDFv3, FFIW) because "the FF++ validation set is very similar to the
  training set".
- FF++ test appears only in `tst_files`, for evaluation.
- Conclusion: the checkpoint did **not** use FF++ val or test for
  training or model selection.
- Residual uncertainty: the per-frame path lists live in a gated HF
  dataset (`yermandy/GenD`, 401 without auth), so we did not verify
  them file by file. The HF model card states the checkpoint is the
  paper model, not a re-train.

**Inference settings:**
- Batch size 64 with fp16 autocast under `torch.inference_mode`.
  Benchmark: 90 img/s at 1.84 GiB peak; throughput is flat from bs 32
  to 256, and fp32 at bs 256 needs 5.45 GiB.
- fp16 vs fp32 on 256 crops: max |Δlogit| 0.012, max |Δprob| 0.006.
- Both the batch size and the autocast dtype are part of the cache tag.

**Cache design:**
- Fixed-index shards of 1024 rows. Each shard stores the teacher tag
  and its crop SHA-256 list. `meta.json` pins the tag, the crop-manifest
  SHA-256 and the shard size.
- A mismatch raises `StaleTeacherCacheError`; it is never silently
  mixed.
- Crop bytes are re-hashed as they are read. The crop manifests are
  checked against the Phase 5d summary, so no videos are re-hashed and
  no crops are re-extracted.
- `test` raises `ProtectedSplitError` before any file is opened.

---

## 2026-10-01 — F2F/NT width change is a centred crop: no correction; geometry/blur jitter recommended class-independently

**Evidence (Phase 5d audit, `shortcut_audit_20261001-114343`):**
- **Raw-frame registration.** All 562 width-changed accepted fakes
  (281 F2F + 281 NT) were registered on 3 matched frames each against
  their target original, plus 200 seeded same-width controls.
  - The ECC affine horizontal scale is 1.0000 (sd 0.0001). A squeeze
    would give 0.9730.
  - The best crop offset equals `(orig_w − fake_w)/2` in every case.
  - The squeeze hypothesis leaves 2.0–9.6× (median) more border
    residual than the crop.
  - Verdict: 562/562 `centred_crop`; 0 squeeze, 0 inconclusive.
- **Raw-landmark cross-check** over 4,496 exactly matched slots per
  method:
  - The fake/real interocular ratio is 0.9985 [0.9968, 1.0003] for F2F
    and 1.0008 [0.9992, 1.0025] for NT. A squeeze predicts 0.9730.
  - The eye-midpoint shift matches the centred-crop prediction to
    0.09 ± 2.5 px (F2F) and 0.06 ± 2.4 px (NT).
- **After alignment,** the paired Δ width/height ratio of width-changed
  F2F (+0.0034) is indistinguishable from same-width F2F (+0.0029). Both
  are about 0.05 sd of the real distribution (sd 0.058).

**Uncertainty:** the crop removes 8–12 px of *background* at each frame
edge. A sub-0.01% resampling would be below what ECC resolves. The
crop does slightly change where the face sits in the raw frame, but
5-point alignment removes that.

**Decision:** No method-specific correction (none is needed, and none
is allowed). Geometry is only weakly method-predictive after alignment:
- A train→test probe on 14 aligned-geometry features reaches 5-class
  accuracy 0.237 vs 0.200 chance and real-vs-fake AUC 0.537.
- The only notable signal is Deepfakes vs original (AUC 0.617), which
  is consistent with swapped-in identity geometry, i.e. the manipulation
  itself.
- The width/height ratio alone is at chance (AUC 0.51–0.53).

**Recommendation for the augmentation phase (user-numbered Phase 8;
roadmap "Compression-robust augmentation"):** apply class-independent
geometric jitter (±3% anisotropic scale, small rotation and translation)
and blur/resize/JPEG/H.264 jitter to every class. Crop sharpness is
lower in fakes, especially NT: paired Δ Laplacian variance −51, AUC
0.415. That is a genuine manipulation artifact, but it is also
resolution-correlated (ρ 0.38), so the model should not be allowed to
rely on sharpness alone.

---

## 2026-10-01 — Quarantine outcome accepted as-is; no manual restoration

The full run quarantined 9 of 1000 families (45 videos):
- **4 content-wide:** 212, 370, 509, 738. A cutaway or the clip ending
  inside the shared range leaves no face in all 5 members.
- **5 fake-only:** 386, 569 and 894 (Deepfakes), 618 (Face2Face) and
  908 (FaceSwap). The manipulation output is broken in those frames
  (colour-noise faces, blobs, collapsed geometry), so YuNet finds no face.

All 9 contact sheets were reviewed. Because quarantine is whole-family,
**every class loses exactly 9 videos**: per-class counts stay equal
(713/139/139 per class in train/val/test) and every accepted fake keeps
its matched real. The fake-only cases do remove a few of the *most
visibly broken* fakes. That is a mild selection effect toward harder
fakes, not a cue a model can exploit. These families are not restored by
hand. A future detector or tracker change would re-run them under a
new config tag.

---

## 2026-10-01 — FF++ crops are sampled per content family over a shared range, matched by frame index

**Context:** Official sources fix the roles: filenames are
`<target sequence>_<source sequence>` (`dataset/README.md` @ `b952e41c`).
The target supplies the frames; the source supplies the swapped face
(DF/FS) or the driving expressions (F2F/NT) (Rössler et al. 2019,
appendix). The same appendix says Deepfakes manipulates every target
frame, FaceSwap/NeuralTextures only `min(target, source)` frames, and
Face2Face "maps all source expressions to the target sequence and
rewinds the target video if necessary". The Phase 5d probe of all 5000
videos agrees exactly: DF = target length (1000/1000), F2F = source
length (992/1000), FS/NT = min(target, source) (1000 / 999).

**Decision:**
- A content family is one target original plus its 4 fakes (1000
  families × 5 videos; each video belongs to exactly one).
- Every member samples the **same 16 frame indices**, given by the
  existing nested contract applied to `[0, min(frame counts) − 1]`.
  This excludes F2F's rewound tail, and the real member's sampled span
  equals its fakes'.
- Equivalent temporal position = **equal frame index**, not timestamp.
  Registration of a seeded sample showed index offset 0 (the
  residual ±1–2 cases were near-static scenes). 64 fakes carry a
  different fps header from their target but still align by index, so
  timestamp matching would have been wrong.
- The donor original is recorded as the second leakage parent
  (`donor_parent_sample_id`) but is not a temporal partner.

**Why:** This gives real/fake pairs the same content positions, and
leaves clip length unable to change which part of a clip either class
shows.

---

## 2026-10-01 — Missing faces: joint recovery, then individual, else quarantine the whole family

Applied identically to every member:
1. Planned index if every member has a valid face there.
2. Else the first offset in +1, −1, +2, −2, …, ±6 at which **all**
   members are valid (joint recovery; the match stays exact).
3. Else per member: planned if valid, otherwise its own first valid
   offset. This is recorded as `individual_recovery` / `exact_match: false`
   in `matched_pairs.jsonl`.
4. If any member still has no face, the **whole family** is quarantined.

Offsets are bounded by half the gap to the neighbouring planned index,
so recovered frames stay strictly ordered (property-tested).
Quarantining a whole family, rather than one video, keeps paired sampling
intact: an original without crops would orphan 4 fakes, and dropping only
a fake would unbalance the family. Successful crops of quarantined
families are kept for review. Nothing is ever padded, blanked, or
returned short.

---

## 2026-10-01 — Sparse-sample face linking uses IoU OR a size-normalised centre shift

**Context:** In the first trial, family 682 was quarantined even though
YuNet found the face at 0.93 confidence. Samples are ~33 frames apart,
the speaker drifted ~0.6 face widths, so the IoU fell to 0.19 (< 0.3).
The Phase 2 IoU-only tracker then split the track and the shorter segment
was rejected.

**Decision:** Link two detections if IoU ≥ 0.3, **or** if the centre
shift is ≤ 1.0 mean face width **and** the area ratio is ≤ 1.5². The
primary face is the longest linked track (ties: higher mean confidence,
then earliest). Both thresholds are part of the config tag. A different
person 3 face widths away and a 4× zoom cut are not linked (tested).
The Phase 2 `configuard.media.tracking` module is unchanged.

**Follow-up (same day, full run 1, stopped at ~25%):** family 158 (a
weather presenter walking across the frame, detected at 0.93–0.94 in
every frame) was quarantined. One planned frame was faceless, so the
planned-only track jumped 1.37 face widths and split. A recovery
frame at 224 would have bridged the gap (shifts 1.0 and then 0.37),
but tracking ignored recovery frames. **Fix:** each member's primary
track is rebuilt over *all* decoded frames, planned plus recovery, and
recovery runs for up to 2 rounds (`recovery_rounds`, `tracking_method`
are in the config tag). Regression test:
`test_recovery_frame_bridges_a_moving_face_instead_of_quarantining`.
The partial run's store is refused as stale and was set aside
(`superseded_run2_p5d-c02fd00a89fb7d96`). Run 2 used the fixed config
`p5d-b451b5ca770c8923`.

---

## 2026-10-01 — Five-landmark similarity alignment; one PNG format for every class

- Umeyama least-squares **similarity** (rotation + one uniform scale +
  translation) maps YuNet's 5 landmarks onto the standard 112-px
  five-point template, scaled into 224×224 with `margin_ratio = 0.25`.
  The face outline, hairline and jaw stay in the crop, so blending
  boundaries are not cropped away.
- Uniform scale means aspect ratio is **never** changed: any anisotropic
  distortion in a source would remain measurable (tested with a 3%
  squeeze). No method-specific correction exists.
- Borders: reflect-101 (no black padding). The share of output pixels
  that fall outside the frame is recorded for the audit.
- Downscales beyond 2× are pre-filtered with `INTER_AREA`, then the
  image is warped bilinearly.
- Output: 224×224 RGB PNG, compression 3, no text/time chunks
  (byte-deterministic). Paths are keyed by input SHA-256 and frame
  index only, with no label, method or split.
- PNG was kept after measuring ~62 KB per crop. The trial projected
  ~5 GB for 80,000 crops against ~44 GB of headroom above the floor.

---

## 2026-10-01 — The model-facing crop manifest is a field whitelist; source facts live in an audit sidecar

`crops_<split>.jsonl` rows may contain only `MODEL_ROW_FIELDS`, with
lineage under `metadata`. Source width/height, duration, fps, frame
count, codec, file size and frame indices go to `crop_audit.jsonl` and
the family records instead. `validate_crop_leakage` rejects any
non-whitelisted field. This makes "resolution and duration never enter
the model" a structural property rather than a convention.

---

## 2026-10-01 — Crop store accepts exactly one config and refuses stale outputs

The config tag hashes the detector name, version and **model SHA-256**
(YuNet now hash-pinned in code: `verify_yunet_model`), and every
sampling, linking, recovery, alignment and PNG parameter. A store root
records its config and refuses any other. Family records are checked
against the tag, the member input SHA-256s, and the crop sizes (full
SHA-256 on demand). Mismatch raises `StaleCropError`; nothing is silently
reused. This was demonstrated live: after the linking change, the first
trial store was refused and moved aside (not deleted) to
`D:\ConfiGuard-Data\cache\ffpp_face_crops\superseded_trial1_p5d-2ce67d23e6a64f82`.

---

## 2026-09-30 — FF++ split = the authors' official files, pinned to commit `b952e41cba01`

**Decision:** Use `dataset/splits/{train,val,test}.json` from
`ondyari/FaceForensics` at commit `b952e41cba017eb37593c39e12bd884a934791e1` (the repository head; last
commit 2020-07-15). Membership is used exactly as published: 360/70/70
pairs, i.e. 720/140/140 originals. `configuard.datasets.faceforensics_splits`
pins each file's size, SHA-256, and git blob SHA, and refuses a file that
differs. The whole assignment is refused, and no manifest is written, if
any of these hold:
- a pair or original is in two splits, or in none;
- a split lists a non-official pair;
- a fake's target and source originals are in different splits;
- a parent or paired link crosses splits;
- any Phase 3 leakage group spans splits;
- Phase 3 cross-split validation reports leakage.

**Why:** It keeps results comparable with the FF++ literature and
avoids inventing a split. The files were fetched only from the
authorized source, and all their git blob SHAs matched GitHub's.

---

## 2026-09-30 — Official FF++ split files are NOT committed; only their pins are

**Context:** The repository's LICENSE is MIT for code, but its README
states the *data* is released under the FaceForensics Terms of Use. The
split files enumerate dataset video IDs, so they arguably belong to the
data.

**Decision:** Keep working copies only at
`D:\ConfiGuard-Data\datasets\FaceForensics++\_official_splits\b952e41cba017eb37593c39e12bd884a934791e1\`.
Commit the revision, URLs, sizes, SHA-256, and git blob SHAs, which is
enough to re-fetch and verify them exactly. Tests use synthetic files
in the same format; one integration test uses the real pinned copies
when present and skips otherwise.

**Why:** They are small enough that committing would be convenient, but
the license ambiguity means the conservative choice costs nothing.

---

## 2026-09-30 — Duration, resolution, and source metadata are never model inputs

The Phase 5c audit found systematic, class-correlated metadata:
- FaceSwap and NeuralTextures clips are shorter.
- Face2Face and NeuralTextures round the frame width down to a multiple
  of 16 (282 of 1000 videos each), while originals, Deepfakes, and
  FaceSwap keep the native width.

**Rules for all later phases:**
1. Frame sampling uses a **fixed frame budget** per video (4/8/16, by
   position within the clip; Phase 2 nested sampling). Clip length,
   frame count, and fps must not change what the model sees.
2. Frames reach the model only as **aligned 224×224 face crops**, so
   native resolution is never a direct feature.
3. File metadata (duration, resolution, codec, bitrate, filename, IDs)
   is **never** a model input or a feature for calibration.

---

## 2026-09-30 — FF++ access URLs are kept out of Git

The download-script URL (and the server paths derived from it) are
distributed by the FF++ authors only to approved users. It contains no
token, but it is still access information. So it is recorded only in
`D:\ConfiGuard-Data\datasets\FaceForensics++\_official_script\PROVENANCE.md`,
outside the repository. Committed docs/code keep only hashes, sizes, and
times, which is enough to verify a copy but not to obtain one.

---

## 2026-09-30 — FF++ download runs through a hash-pinned, allow-listed, free-space-guarded wrapper

**Context:** The official `faceforensics_download_v4.py` defaults to
`-c raw` and `-d all` (which includes DeepFakeDetection and FaceShifter),
blocks on an interactive TOS prompt, and has no disk-space awareness.

**Decision:** `scripts/download_faceforensics_c23.py` wraps it:
- It refuses to run unless the script's SHA-256 matches the reviewed
  copy (`5d0b220a…`).
- It only allows `original`, `Deepfakes`, `Face2Face`, `FaceSwap` and
  `NeuralTextures`, with `-c c23 -t videos --server EU2` hard-coded.
- It refuses output paths inside the repo.
- It terminates the download if free space would fall below 40 GB
  (+256 MB headroom).
- It logs JSONL progress to `CONFIGUARD_OUTPUT_DIR/acquisition/faceforensics/`.

The five datasets ran as five independent single-stream instances to
cut wall-clock time about 5×. The TOS prompt is acknowledged with a
newline, because the user has official access and explicitly instructed
the download.

**Added mid-download (2026-09-30, both observed live):**
- **Stall watchdog.** The official script's `urlretrieve()` has no
  timeout. The `original` stream hung for about 8 minutes on a dead
  connection (partial frozen at 3,375,104 bytes). If neither the
  completed-file count nor the in-flight `tmp*` size changes for 5
  minutes, the wrapper kills the script, removes its partial, and
  relaunches it; finished files are skipped, at most 10 times. It later
  recovered a Face2Face stall with no manual intervention.
- **Process-tree kill.** On Windows the venv `python.exe` is a launcher
  that spawns the real interpreter, so `Popen.terminate()` would have
  orphaned the actual downloader, and the low-space stop would not have
  stopped anything. The wrapper now uses `taskkill /T /F`.
  All five streams were moved onto the patched wrapper. This cost only
  the 5 in-flight partial files.

**Why:** It makes the Phase 5b constraints mechanical rather than a
matter of typing the right flags. The official script writes to a
`tmp*` file and renames only on completion, so a watchdog stop can
never leave a truncated `.mp4`, and re-running resumes.

---

## 2026-09-30 — Followed the approved script URL's HTTP→HTTPS redirect

The approved URL (from the approval email; recorded only in
`_official_script/PROVENANCE.md` on D:, never in Git, because FF++
distributes it only to approved users) answers `301` with the same host
and path over HTTPS. The first fetch saved the 353-byte redirect page. It was replaced
by fetching with `curl -L --proto-redir =https`. The script was read in
full before its first execution.

---

## 2026-09-30 — Completeness is measured against the official pair list, not a hard-coded count

The expected file set is derived exactly as the official script derives
it: from `v3/misc/filelist.json` on the same EU2 server (500 pairs;
originals = both IDs, each method = `a_b` and `b_a`). That file (21,002
bytes, SHA-256 `7099a119…`) is stored next to the script and pinned in
the acquisition report. This is metadata the official script itself
downloads, not an additional dataset.

---

## 2026-09-30 — FF++ fakes link to BOTH originals; identities are never invented

**Context:** The generic folder engine grouped a fake only by its leading
filename token, so `000_003.mp4` was tied to `000.mp4` but not to
`003.mp4`, whose face (swap methods) or expressions (reenactment
methods) it contains. A split built on that grouping could leak.

**Decision:** Introduce `FaceForensicsAdapter` (FF++ only). For
`<target>_<source>.mp4` it sets `source_id=<target>`,
`parent_sample_id` to the target original and `paired_sample_id` to the
source original. Phase 3's union-find grouping then puts each official
pair's 2 originals and all 8 derived fakes into one leakage group.
`identity_id` stays `None`: FF++ publishes no identity labels, and the
schema reserves that field for dataset-provided identities.

**Why:** This is a correctness fix required for any future FF++ split.
It is verified by a test that fails against the old adapter (4 groups
instead of 2) and on the real data (500 groups × 10 members).

---

## 2026-09-30 — No FF++ train/val/test split applied

The approved source, the official script and EU2 server, provides no
split files. FF++'s published split JSONs live in the authors' GitHub
repository (`ondyari/FaceForensics`, `dataset/splits/`), which was not
part of the approval, so they were not fetched and **no split was
invented**. Manifests carry `official_split=None`. The leakage grouping
the split must respect is written alongside the manifest. Applying the
official split needs the user's approval to fetch those files.

---

## 2026-09-30 — "Readable by ffprobe" means header + full packet demux

`ffprobe -count_packets` demuxes every packet of the video stream, so a
truncated file with an intact header still fails; a plain header probe
would pass it. A full decode (`ffmpeg -f null`) would take hours for
5,000 files and belongs to face-crop extraction, which is out of scope.

---

## 2026-09-30 — Phase 5 is the reproducible training pipeline; roadmap renumbered

**Context:** `docs/PROJECT_PLAN.md` had planned Phase 5 as GenD
distillation. The user's Phase 5 instruction instead defined a
reproducible training pipeline (no teacher, no new downloads).

**Decision:** Phase 5 = reproducible training pipeline. Every later
planned phase shifts by one (distillation becomes Phase 6, etc.).

**Why:** Distillation, augmentation, the temporal GRU, and calibration
all need a trustworthy training loop, checkpointing, and metrics first.

---

## 2026-09-30 — Exact resume = epoch-boundary checkpoints + per-epoch data seeding + persisted trainer state

**Context:** The first Phase 5 draft saved model/optimizer/scheduler/
scaler/RNG state, but (a) the balanced sampler's generator advanced
across epochs and was not saved, so a resumed run saw different batches;
(b) best-metric and early-stopping counters were not saved, so a resume
could re-save a worse "best" and ignore patience; (c) the checkpoint was
loaded with `map_location="cuda"`, which moved the saved RNG ByteTensors
to the GPU where `torch.set_rng_state` rejects them - GPU resume could
never have worked.

**Decision:** Checkpoints are written only at epoch boundaries;
`Trainer._reseed_loader_for_epoch` reseeds the sampler and DataLoader
generators with `seed + epoch` every epoch; `TrainState` (epoch, step,
best metric + tie-break, best epoch, patience counter, AMP-skipped steps)
is stored in the checkpoint (format version 2); checkpoints are always
loaded on CPU and RNG states restored via `.cpu()`.

**Why:** Makes data order a pure function of (seed, epoch), so resumed
and uninterrupted runs are identical. Measured: max parameter difference
**0.0** after interrupt+resume on CPU and on the RTX 4050 (AMP), both
backbones (`docs/EXPERIMENT_LOG.md`). Mid-epoch resume is deliberately
not supported (it would need DataLoader iterator state); an interruption
loses at most the current epoch.

---

## 2026-09-30 — Refuse any resume whose data, preprocessing, model, or result-affecting config differs

**Decision:** `verify_checkpoint_compatible` compares encoder name, HF
model id, face-preprocessing `version_tag`, SHA-256 of the train *and*
validation manifests, and every key in `RESUME_CRITICAL_CONFIG_KEYS`
(seed, batch size, epochs, lr, weight decay, warm-up, accumulation,
clipping, freezing, AMP + init scale, selection metric, patience,
pretrained, balancing, frames per video), and lists **every** mismatch in
one `CheckpointMismatchError`. Bookkeeping fields (run name, paths,
workers, device, log cadence) are exempt. `TrainingConfig.from_dict` now
**rejects unknown keys** (the draft silently ignored them).

**Why:** Task 11 forbids silent resumes across changed data/
preprocessing/model config. `epochs` is critical because the cosine
schedule length depends on it. A typo'd YAML key (e.g. `learning_rate`)
silently falling back to a default would make a run irreproducible from
its own config file.

---

## 2026-09-30 — Best checkpoint: validation AUROC, then validation loss, then training loss; exact ties broken by validation loss

**Decision:** Selection value (higher is better): (1) validation AUROC
when defined; (2) otherwise, i.e. the validation split lacks one class,
negative validation loss; (3) with no validation set, negative training
loss. Exact ties (AUROC saturating at 1.0) go to the lower validation loss.

**Why:** AUROC is threshold-free and the planned calibration phase
re-derives thresholds anyway. The fallback can't flip between epochs:
whether AUROC is defined depends only on the fixed validation labels.
The tie-break was added after the RTX 4050 smoke run picked epoch 1
(AUROC 1.0, val loss 0.51) over epoch 3 (AUROC 1.0, val loss 0.028).

---

## 2026-09-30 — AMP via `torch.amp`, fp32 loss, `init_scale=1024`, and AMP-skipped steps are counted

**Context:** A 6-step RTX 4050 smoke run finished "successfully" with
chance-level loss. Inspection showed the optimizer state was empty:
`GradScaler`'s default initial scale (65536) overflowed fp16 gradients,
so **all 6 steps were skipped** (scale 65536 → 1024), and nothing
reported it. A 20-step run skipped 9 steps.

**Decision:** Use only `torch.amp.autocast` / `torch.amp.GradScaler`
(not the deprecated `torch.cuda.amp` interfaces); compute BCE on fp32
logits outside autocast; default `amp_init_scale` to 1024 (configurable);
advance the LR schedule only when the scaler actually applied the step;
count skipped steps in `TrainState`, the epoch log, and the smoke summary.

**Why:** With `init_scale=1024` the same 20-step run skips 3 steps
instead of 9 and converges faster. A silent "trained without updating a
weight" run is exactly the failure a pipeline-verification phase must
catch.

---

## 2026-09-30 — CUDA OOM and non-finite loss stop training; nothing adapts automatically

**Decision:** `torch.cuda.OutOfMemoryError` anywhere in forward,
backward, or optimizer step becomes `TrainingOutOfMemoryError`, whose
message carries batch size, accumulation, AMP flag, and allocated/peak/
reserved memory, and says the batch size was **not** changed. A NaN/Inf
loss raises `NonFiniteLossError` before backward, so no optimizer step
and no checkpoint happens for that epoch.

**Why:** Tasks 17/18. Silently shrinking the batch would change the
experiment (and the resume-critical config) behind the user's back.

---

## 2026-09-30 — Training re-checks split leakage; mixed image+video manifests are refused

**Decision:** `configuard.training.splits.assert_no_cross_split_leakage`
runs in `build_trainer` over the train/validation manifests actually
handed to training (reusing Phase 3's source/identity/pair checks, plus
duplicate `sample_id` across splits). `build_manifest_dataset` refuses
manifests mixing IMAGE and VIDEO samples.

**Why:** Phase 3 assigns splits leakage-safely, but the trainer must not
trust that files weren't edited or swapped afterwards (task 5). All
frames of a video share its Sample's `source_id`, so the source check
also keeps frames of one source inside one split. The previous draft
silently dropped image samples from a mixed manifest.

---

## 2026-09-30 — Synthetic pipeline data: blue- vs. red-tinted checkerboards, full-frame mock face detector

**Context:** The first draft's synthetic real/fake pair was a
checkerboard vs. its phase-inverted copy - identical global statistics,
separable by a global-pooled CNN only through padding/border effects,
so not an "obvious" signal (task 20).

**Decision:** `configuard.training.synthetic` generates checkerboards
with a blue (real) vs. red (fake) tint and per-sample phase/brightness
variation. Smoke runs use a mock detector that reports the whole frame
as the face, so Phase 2 alignment/cropping/caching runs for real. Every
synthetic result carries `SYNTHETIC_RESULT_DISCLAIMER`.

**Why:** Pipeline verification needs a signal whose learnability is
beyond doubt; the result must never be mistaken for detection accuracy
(task 21).

---

## 2026-09-30 — Training and evaluation CLIs force `HF_HUB_OFFLINE=1`

**Decision:** `scripts/train.py` and `scripts/evaluate.py` set
`HF_HUB_OFFLINE=1` (via `setdefault`) right after loading `.env`.

**Why:** Phase 5 must not download any model. Offline mode makes
huggingface_hub serve only the already-cached Phase 4 weights and fail
instead of silently fetching something new.

---

## 2026-09-30 — Pinned CUDA build pair in `constraints-cuda.txt` + a runtime regression guard

**Decision:** `constraints-cuda.txt` pins `torch==2.5.1` /
`torchvision==0.20.1` and documents the only correct index
(`https://download.pytorch.org/whl/cu121`).
`configuard.dependency_safety` hard-fails if torch is CPU-only, if
torchvision fails to import **or fails its first compiled op**
(`torchvision.ops.nms` - ABI breaks often import fine), if torchvision
isn't the release paired with the installed torch, or if CUDA was
expected but is gone. Version drift from the pin is a warning. It runs
in `scripts/verify_environment.py` and at the start of both CLIs.

**Why:** Directly guards the Phase 4 regression (a routine `pip install`
silently swapped in a CPU-only torch). No packages were installed or
upgraded in Phase 5.

---

## 2026-09-30 — Structured logs: JSONL is the source of truth, per-record-type CSVs are views

**Decision:** `ExperimentLogger` appends every record to `<run>.jsonl`
(nested) and to `<run>_train.csv` / `<run>_epoch.csv` (recursively
flattened), all under `CONFIGUARD_OUTPUT_DIR`. A resumed run appends to
the same files.

**Why:** The draft used one CSV whose header was frozen from the first
record; since that is a train-step row, every epoch/validation column
would have been silently dropped from the CSV.

---

## 2026-09-30 — Checkpoints use `torch.load(weights_only=False)`, trusted-source only

**Decision:** Kept `weights_only=False` because checkpoints contain
Python/NumPy RNG state tuples that the weights-only unpickler rejects.
Documented in code and `docs/KNOWN_ISSUES.md`: only load checkpoints
this project wrote itself.

---

## 2026-09-29 — `HF_HOME`/`HF_HUB_CACHE`/`TORCH_HOME` loaded via a tiny custom `.env` parser, not python-dotenv

**Context:** Phase 4 needs pretrained weight downloads to land under
`D:\ConfiGuard-Data\cache\...`, never the default `C:\Users\...\.cache`.
`huggingface_hub` and `torch` read these as **module-level constants at
import time** - setting them after `import timm`/`import torch` has
already happened in the process does nothing.

**Decision:** Added `configuard.env_loader.load_dotenv()` (a ~20-line
`KEY=VALUE` parser using `os.environ.setdefault`, no `python-dotenv`
dependency), and call it at the very top of every entrypoint that might
construct a `DeepfakeVisualEncoder` - `scripts/download_baseline_models.py`,
`scripts/benchmark_models.py`, `scripts/export_onnx_models.py`, and
critically the **root** `tests/conftest.py` (not a subpackage conftest),
before any test module can trigger `import timm` via
`DeepfakeVisualEncoder.__init__` - even with `pretrained=False`, that
constructor still imports `timm`, which imports `huggingface_hub`.

**Why:** A full dotenv library is unnecessary for this project's simple
flat `KEY=VALUE` file. The critical, easy-to-miss part is *where* the
loader must run - not "before training starts" but "before the first
`import timm`/`import torch`/`import huggingface_hub` anywhere in the
process," which is why it lives in the root conftest rather than a
per-directory one.

---

## 2026-09-29 — `DeepfakeVisualEncoder` probes `num_features` empirically, not via `backbone.num_features`

**Context:** Constructing the binary head as
`nn.Linear(backbone.num_features, 1)` crashed for
`mobilenetv4_conv_small` specifically: `backbone.num_features` reports
960, but a real forward pass (with `num_classes=0`) returns a 1280-dim
pooled feature vector, because timm's MobileNetV4 has an internal "head
conv" channel-expansion layer (960 -> 1280) that isn't reflected in the
`num_features` attribute the way it is for `tf_efficientnet_b0`.

**Decision:** `DeepfakeVisualEncoder.__init__` runs one dummy forward
pass through the backbone (in `eval()` mode - see the next entry) and
uses the *actual output tensor's* last dimension to size the head,
instead of trusting `backbone.num_features`.

**Why:** More robust across arbitrary timm architectures than relying on
an attribute whose meaning apparently isn't 100% consistent across model
families with `num_classes=0`. Caught by this phase's own smoke testing,
not a documented timm caveat found in advance - see
`tests/models/test_encoder.py::test_forward_features_shape`, which would
have failed loudly (a matmul shape error) if this regressed.

---

## 2026-09-29 — The `num_features` probe forward pass forces `eval()` mode

**Context:** The probe above initially crashed for MobileNetV4 with
`ValueError: Expected more than 1 value per channel when training, got
input size torch.Size([1, 1280, 1, 1])` - BatchNorm computes batch
statistics in `train()` mode (the default state of a freshly constructed
`nn.Module`) and rejects a batch of size 1, but the probe used a single
dummy sample.

**Decision:** The probe temporarily switches `self.backbone` to `eval()`
mode (using running statistics, not batch statistics - safe for batch
size 1), runs the forward pass, then restores whatever training/eval
state the module was actually in before the probe.

**Why:** This is exactly the same reason batch size 1 must always go
through `.eval()` at real inference time too (see
`tests/models/test_encoder.py::test_batch_size_one_works_in_eval_mode`) -
the construction-time probe hit the same underlying BatchNorm constraint
one step earlier than a caller would have.

---

## 2026-09-29 — ONNX export uses the legacy TorchScript-based exporter (`dynamo=False`)

**Context:** `torch.onnx.export(...)` with this project's torch version
defaults to the newer "dynamo" exporter path, which failed immediately
with `ModuleNotFoundError: No module named 'onnxscript'`.

**Decision:** Pass `dynamo=False` to use the older, mature
TorchScript-tracing-based exporter, rather than adding `onnxscript` as a
dependency.

**Why:** The encoder's graph (a timm backbone + one `nn.Linear` head) is
simple and has no control flow the legacy tracer would mishandle -
verified by `docs/EXPERIMENT_LOG.md`'s parity numbers (max abs diff on
the order of 1e-7, far inside the 1e-3 documented tolerance). Avoids a
new dependency for a code path this project doesn't need.

---

## 2026-09-29 — Video inference aggregates *embeddings* (mean-pooled), not per-frame probabilities

**Context:** Requirement 8: "fixed-frame video inference using ordered
frame embeddings and mean aggregation."

**Decision:** `infer_video_fixed_frames` runs every sampled frame through
`forward_features` (preserving order), mean-pools the resulting (N,
num_features) tensor along the frame axis into one (1, num_features)
vector, and only then applies the binary head - producing one logit/
probability per clip, not N per-frame ones later averaged.

**Why:** Averaging in feature space (before the classifier) is a strictly
more expressive aggregation than averaging post-hoc probabilities, and
matches the literal requirement wording ("ordered frame embeddings and
mean aggregation"). No temporal model (GRU) yet, by explicit task scope -
this is a placeholder aggregation, replaced when the GRU phase lands.

---

## 2026-09-29 — Real-model pipeline kept separate from the Phase 1 dummy pipeline

**Context:** Requirement 10: wire real media preprocessing into an
*optional* real-model pipeline, retaining dependency injection.

**Decision:** `configuard.models.real_pipeline.run_real_image_pipeline` /
`run_real_video_pipeline` are new functions, not a modification of
`configuard.pipeline.run_pipeline` (the Phase 1 dummy vertical slice,
still hash-based and deterministic). Both take `detector`, `cache`, and
`encoder` as parameters (dependency injection), so tests use
`MockFaceDetector` + a `pretrained=False` encoder with no network access.

**Why:** `run_pipeline`'s contract (`DetectionResult`, deterministic
dummy score) is still exercised by Phase 1's tests and documented as the
"vertical slice proof" - conflating it with a real, evolving model
pipeline would break that contract or force a confusing dual-purpose
function. Keeping them separate lets each evolve independently until a
later phase deliberately unifies them (see `docs/PROJECT_PLAN.md`).

---

## 2026-09-29 — Added `CONFIGUARD_OUTPUT_DIR` as a fourth storage-checked env var

**Context:** The project already had a gitignored `outputs/` directory
(generated reports/artifacts) but no corresponding configurable external
path or storage-check coverage - `STORAGE_ENV_VARS` only listed
data/cache/checkpoint.

**Decision:** Added `CONFIGUARD_OUTPUT_DIR` to
`configuard.datasets.storage.STORAGE_ENV_VARS` and `.env.example`.

**Why:** Requested explicitly ("add a configurable output-directory
variable if the project supports it") as part of moving all generated
project data off the small `C:` system drive - `outputs/` needed the same
external-path treatment as the other three, for the same reason (avoid
growing an in-repo directory with real generated artifacts).

---

## 2026-09-29 — Real datasets/cache/checkpoints/outputs live on `D:`, not `C:`

**Context:** Storage audit found `C:` (the repo's drive) at 12.9 GB free
and `D:` (a separate local fixed NTFS volume, not OneDrive-synced) at
92.9 GB free.

**Decision:** Configured local `.env` (never committed) to point all four
storage variables at `D:\ConfiGuard-Data\{datasets,cache,checkpoints,outputs}`.

**Why:** `C:` doesn't have enough headroom for real deepfake-detection
datasets (these commonly run tens to hundreds of GB) without risking
filling the system drive; `D:` does. Every path was verified writable via
an actual write test (not just a permission-bit check), outside the git
repository, and outside OneDrive (OneDrive sync could otherwise silently
upload access-controlled dataset content to the cloud, or choke on
folder sizes/counts typical of a dataset).

---

## 2026-09-29 — JSONL manifests, not Parquet

**Context:** Requirement: "Use JSONL as the initial portable manifest
format. Do not add a heavy Parquet dependency unless there is a measured
need."

**Decision:** `configuard.datasets.manifest` reads/writes one JSON object
per line, no `pyarrow`/`pandas` dependency added.

**Why:** At this project's scale (laptop-scale training data, not
web-scale), JSONL is human-diffable in `git diff`/code review, trivially
streamable line-by-line without loading a whole file, and needs zero new
dependencies. Revisit only if a measured manifest size/load-time problem
actually appears.

---

## 2026-09-29 — Leakage groups are connected components (union-find), not pairwise rules

**Context:** Requirement: group all derivatives of one source, all
samples of one identity, and any real/fake pair into the same split.

**Decision:** `compute_leakage_groups` unions samples sharing a
`source_id`, sharing an `identity_id`, linked by `parent_sample_id`, or
linked by `paired_sample_id` into connected components via a union-find
structure, then assigns each *component* (not each sample) to a split.

**Why:** These relationships chain (e.g. a fake derived from another fake
derived from a real - see the derivative-of-derivative case in
`tests/datasets/test_fixture_scenarios.py`), and a person can appear
across sources. Handling each relationship as an independent pairwise
rule would miss transitive leakage (A-B linked, B-C linked, but A-C not
directly checked); connected components close over all transitive links
in one pass.

---

## 2026-09-29 — Split assignment is a deterministic hash, not a seeded shuffle

**Context:** Requirement: "Deterministic splits reproduce exactly with
the same seed," independent of how the caller happens to order samples.

**Decision:** Each leakage group's split is chosen by hashing
`f"{seed}:{group_key}"` into `[0, 1)` and bucketing against cumulative
split fractions, rather than seeding `random.shuffle` on the group list.

**Why:** A seeded shuffle's result depends on the *order* items are fed
into it and on Python's specific PRNG algorithm/version - two callers
with the same groups but a different starting order, or a future
different Python version, aren't guaranteed to get bit-identical output.
Hashing each group's own stable key is order-independent by construction
(verified in `tests/datasets/test_splitting.py::test_split_samples_reordered_input_same_result`)
and has no dependency on `random` module internals.

---

## 2026-09-29 — Two adapter engines cover five named datasets + the generic case

**Context:** Requirement: typed adapters for FaceForensics++, Celeb-DF-v2,
DFDC, DF40, DeeperForensics-1.0, plus "future datasets through a generic
adapter."

**Decision:** Rather than five independent adapter classes (mostly
duplicated scanning logic) plus a sixth generic one, built two reusable,
declaratively-configured engines - `FolderConventionAdapter` (real/fake
media in known subdirectories, identity from filename; backs FF++,
Celeb-DF-v2, DeeperForensics-1.0) and `MetadataSidecarAdapter` (a JSON/
JSONL/CSV sidecar declares path/label/etc per row, no folder assumptions;
backs DFDC, DF40, and is *also* the generic adapter via
`make_generic_metadata_adapter()`). Each named dataset is a thin factory
function in `known_datasets.py` that configures one engine.

**Why:** DRY - one tested scanning/pairing implementation per engine
rather than five. `MetadataSidecarAdapter` was the natural pick for
"generic," since it needs zero folder-layout knowledge (any dataset that
ships a path+label sidecar works with it out of the box), matching the
open-ended "future image-only face-deepfake datasets" requirement better
than a folder-convention engine would.

---

## 2026-09-29 — `FolderConventionAdapter` tolerates partially-missing buckets

**Context:** Initially, `validate_structure` required every declared
bucket directory (e.g. all 3 compression tiers x 6 manipulation methods
for FF++) to exist, which failed a synthetic test fixture containing only
one method/tier and would equally fail a real, legitimately *partial*
local download (many practitioners only fetch the `c23` compression
tier).

**Decision:** A missing individual bucket directory is tolerated (that
bucket just contributes zero samples); `validate_structure`/
`build_manifest` only fail if the dataset root itself is missing, or if
*none* of the declared buckets are present at all.

**Why:** Matches realistic partial local copies, still satisfies "fail
clearly when access-controlled data is missing" (root missing, or nothing
usable found, both still raise `DatasetAccessError`) without being overly
strict about which subset a user happened to download.

---

## 2026-09-29 — Fake→real pairing is inferred one-directionally by matching `source_id`

**Context:** Requirement 2's "paired-real/fake relationship" field.

**Decision:** `FolderConventionAdapter` and `MetadataSidecarAdapter` both
build a real-sample-by-`source_id` lookup after scanning, then set
`parent_sample_id`/`paired_sample_id` on each FAKE sample whose extracted
`source_id` matches a REAL sample's `source_id` in the same dataset. This
is one-directional (only the fake sample carries the link); nothing sets
it on the real sample.

**Why:** This is the general pattern across all five datasets' naming/
metadata conventions (a fake's filename or metadata references its real
source, not the reverse). A reverse index is trivial for a caller to
build from the forward links if needed, so storing it redundantly on both
sides wasn't worth the complexity of updating an already-frozen `Sample`.

---

## 2026-09-29 — Near-duplicate detection uses a hand-rolled average-hash via OpenCV

**Context:** Requirement: perceptual hashes for near-duplicate images,
with a configurable threshold.

**Decision:** `configuard.datasets.duplicates.compute_average_hash`
implements aHash directly (grayscale, resize to `hash_size`, threshold
against the resized image's own mean) using OpenCV, which is already a
project dependency, rather than adding the `imagehash` package.

**Why:** aHash is ~10 lines of OpenCV calls; not worth a new dependency
for one algorithm this project's scale needs. Revisit if a more
sophisticated perceptual hash (pHash/dHash ensembles) becomes necessary.

---

## 2026-09-29 — Storage check refuses (raises) for in-repo paths, only warns for low space

**Context:** Requirement 11: "must warn or refuse before placing real
datasets inside the repository or on an insufficient-volume location."

**Decision:** `check_storage_path` reports both conditions as warnings;
`assert_safe_storage_path` additionally *raises* `UnsafeStoragePathError`
specifically for the in-repo case, not for low free space.

**Why:** An in-repo dataset path risks an accidental `git add -A`/commit
of real (possibly access-controlled, non-redistributable) data - a hard
stop is warranted. Low free space is a softer condition a caller may
still want to proceed past (e.g. to download a small subset, or because
they're about to free space) - a warning that surfaces in
`scripts/check_storage.py` output is proportionate.

---

## 2026-09-29 — Nested 4/8/16 frame sampling is derived, not independently sampled

**Context:** Requirement: the 4-frame selection must be a subset of the
8-frame selection, which must be a subset of the 16-frame selection.

**Decision:** `compute_nested_sampling_plans` computes one uniform
16-frame sample of the video, then derives the 8-frame set as every
second element of that 16-set, and the 4-frame set as every second
element of the 8-set (`indices_16[::2][::2]`).

**Why:** An independently-computed uniform 8-frame sample of a video is
generally *not* a subset of an independently-computed uniform 16-frame
sample - the nesting requirement can only be satisfied by construction.
This also means frames already decoded/analyzed at a smaller frame count
are automatically reusable if a later stage escalates to a larger one
(e.g. an "uncertain" 4-frame result triggering an 8-frame re-analysis),
without any extra bookkeeping.

---

## 2026-09-29 — Video frame decoding is sequential, not seek-based

**Context:** Need to decode a sparse set of target frame indices from a
video reliably.

**Decision:** `decode_sampled_frames` reads frames sequentially from the
start and keeps the ones at wanted indices, rather than seeking via
`cv2.CAP_PROP_POS_FRAMES`.

**Why:** `CAP_PROP_POS_FRAMES` seeking is well-documented as unreliable
across codecs/containers with irregular keyframe intervals (can land on
the wrong frame silently). Sequential decode is slower per frame reached
but correct, and this project's validated video length limits (tens of
seconds - see `configs/*.yaml`) keep the cost acceptable for laptop-scale
preprocessing.

---

## 2026-09-29 — Video metadata prefers ffprobe over OpenCV VideoCapture properties

**Context:** Requirement: reliable video metadata extraction, including
variable-frame-rate video and rotation.

**Decision:** `extract_video_metadata` tries `ffprobe` first (comparing
`r_frame_rate` vs. `avg_frame_rate` to flag VFR, reading `nb_frames` or
estimating from duration, and reading rotation from the legacy `rotate`
tag or Display Matrix side data), falling back to `cv2.VideoCapture`
properties only if ffprobe is unavailable or fails to parse.

**Why:** OpenCV's own `CAP_PROP_FRAME_COUNT`/`CAP_PROP_FPS` are known to
be unreliable for VFR content and don't expose container rotation at all.
ffprobe (already a project dependency since Phase 1) is more accurate for
both. Rotation extraction is best-effort: this ffmpeg build did not
attach rotation metadata to a synthetic `lavfi`-generated test clip via
either convention (verified by direct `ffprobe` inspection), so the
parsing logic is unit-tested directly against synthetic ffprobe JSON
shapes rather than an actual rotated fixture - see `docs/KNOWN_ISSUES.md`.

---

## 2026-09-29 — Face tracking's temporal-gap tolerance is stride-aware

**Context:** `configuard.media.tracking.track_faces` closes a track if a
face isn't matched for more than `max_frame_gap` *video* frames. But
`preprocess_video` only ever runs detection on the sparse sampled frames
(e.g. indices 0, 4, 8, 12, ... for a 30-frame clip), so a face present in
every sampled frame still has a real-frame gap of ~4 between detections.

**Decision:** `preprocess_video` computes the widest stride actually
present in the sampling plan's indices and uses
`max(config.max_track_frame_gap, sample_stride)` as the tracker's gap
tolerance, instead of passing the raw config value straight through.

**Why:** With a fixed small `max_frame_gap` (e.g. the config default of
2), every track would appear to "expire" between consecutive sparse
samples, fragmenting a single continuous face into many spurious
single-frame tracks. `track_faces` itself keeps `max_frame_gap` as a
literal frame-index parameter (unit-tested that way in
`tests/media/test_tracking.py`, matching a caller that tracks over dense/
consecutive frames) - the adaptation lives in the caller that knows its
own sampling density.

---

## 2026-09-29 — Cache key/version derived from a config hash, not a manual bump

**Context:** Requirement: cache keyed by input hash, frame index, track
ID, and preprocessing configuration/version.

**Decision:** `PreprocessingConfig.version_tag` is `f"v1-{sha256(repr of
every alignment/detector field)[:12]}"`, computed automatically, and used
as the cache key's `config_version` component (`FaceCropCache` /
`CacheKey`).

**Why:** A manually-maintained version string is easy to forget to bump
when a margin ratio or detector threshold changes, silently serving stale
crops. Deriving it from the config's own field values makes any change
automatically land in a new cache namespace with no extra step.

---

## 2026-09-29 — YuNet `2026may` (dynamic input shape) over `2023mar`

**Context:** opencv_zoo publishes several YuNet ONNX exports; task
instructions permitted downloading only this official asset.

**Decision:** Downloaded `face_detection_yunet_2026may.onnx` (dynamic
input-shape re-export), not `face_detection_yunet_2023mar.onnx` (fixed
input shape).

**Why:** This project installed `opencv-python-headless` 5.0.0, and
opencv_zoo's own documentation states the `2026may` export is the one
compatible with OpenCV 5.x's ONNX graph engine, while `2023mar` targets
OpenCV 4.x's DNN module. Verified working end-to-end (loads on CPU,
correctly returns no detections on a non-face image) - see
`docs/EXPERIMENT_LOG.md`. Full provenance (source URL, license, size,
SHA-256) recorded in `docs/DATASETS.md`.

---

## 2026-09-29 — `opencv-python-headless` over `opencv-python`

**Context:** Need OpenCV for image/video decoding and `FaceDetectorYN`.

**Decision:** Added `opencv-python-headless` to `requirements.txt`, not
the GUI-enabled `opencv-python`.

**Why:** Requirement 14 is "run on CPU-only servers" - the headless build
skips Qt/GUI dependencies this project never uses (`cv2.imshow`, etc.),
which matters for a lean server deployment and avoids pulling in system
GUI libraries on Linux later.

---

## 2026-09-29 — Fixed a Phase 0 `.gitignore` bug: `dir/` excludes its own `!dir/.gitkeep` exception

**Context:** While adding `models/` (for the YuNet asset) to `.gitignore`,
noticed `data/.gitkeep`, `checkpoints/.gitkeep`, `cache/.gitkeep`, and
`outputs/.gitkeep` were never actually committed in Phase 0 despite the
`!dir/.gitkeep` exception lines - only `configs/.gitkeep` was (because
`configs/` was never itself ignored).

**Decision:** Changed the ignore patterns from `dir/` to `dir/*` for
`data/`, `checkpoints/`, `models/`, `cache/`, and `outputs/`.

**Why:** Git cannot re-include a file whose *parent directory* is itself
excluded - `!dir/.gitkeep` only works if the directory match is `dir/*`
(contents excluded, directory itself not excluded), not `dir/` (the
directory itself excluded). Verified with `git check-ignore -v` on every
affected `.gitkeep` path after the fix.

---

## 2026-09-29 — Media type is decided by content sniffing, not file extension

**Context:** Phase 1 needs secure file-type validation for uploaded
images/video.

**Decision:** `configuard.validation` reads the first ~64 bytes of the file
and matches known magic-byte signatures (JPEG/PNG/WEBP/MP4/MOV/MKV/AVI) to
decide the real media type. The extension is still checked, but only to
pick the right size limit and as a consistency cross-check - a mismatch
between sniffed content and extension is its own rejection reason.

**Why:** Extensions are trivially spoofable (rename `payload.mp4` to
`image.png`); trusting them for type-based branching is an OWASP-flagged
file-upload weakness. Content sniffing is the standard mitigation and adds
no new dependency (stdlib only).

---

## 2026-09-29 — Video duration/corruption check uses `ffprobe` with a timeout, never raises

**Context:** Requirement: "invalid and corrupted files are rejected
safely."

**Decision:** `_ffprobe_duration_seconds()` runs `ffprobe` with a 10s
timeout and `check=False`; any non-zero exit, timeout, or unparsable JSON
output returns `None`, which `validate_media_file` turns into a
`VIDEO_UNREADABLE` `ValidationResult` entry - never an unhandled exception.

**Why:** A corrupted or adversarial video file must not be able to hang or
crash the service. Bounding subprocess time and treating every failure
mode as "reject, don't raise" keeps that guarantee at the validation
boundary, before any decoding is attempted.

---

## 2026-09-29 — Phase 1 preprocessing/model/provenance are explicit deterministic placeholders

**Context:** Phase 1 must prove the full pipeline shape end-to-end without
implementing real models, face detection, or C2PA (per phase scope).

**Decision:** `_dummy_predict()` derives a pseudo-score from
`sha256(filename:size)` rather than random numbers, so every run on the
same file is reproducible and testable. `_preprocess_placeholder()` only
decides frame *count* (via the real adaptive 4/8/16 rule) without decoding
any pixels. `_provenance_placeholder()` always returns `NOT_CHECKED`.

**Why:** Deterministic placeholders let the test suite assert exact
behavior (`test_pipeline_is_deterministic`) instead of tolerating
randomness, while keeping a clear, greppable seam (`_dummy_predict`,
`placeholder`) for where later phases plug in real implementations.

---

## 2026-09-29 — `ProjectConfig` gains `environment` + `validation`, stays backward compatible

**Context:** Requirement: config schemas for development/training/testing/
production.

**Decision:** Extended the existing `ProjectConfig` (rather than
introducing a parallel config type) with `environment: Literal[...]`
(default `"development"`) and `validation: ValidationLimits` (a new frozen
dataclass, default-constructed if the YAML has no `validation:` block).
Added `configs/{development,training,testing,production}.yaml`;
`configs/base.yaml` from Phase 0 is kept as a generic default and still
passes its original tests unchanged.

**Why:** Avoids a second, parallel config type while still giving each
environment (especially `testing`, with deliberately tiny limits) its own
file. Backward compatible: existing Phase 0 config-loading tests pass
without modification.

---

## 2026-09-29 — Hand-built PNG fixture instead of adding Pillow

**Context:** Tests need a real, valid tiny image to exercise the full
validate → preprocess → predict path.

**Decision:** `tests/conftest.py` constructs a minimal valid PNG (IHDR +
zlib-compressed IDAT + IEND chunks) using only `struct`/`zlib`
(stdlib), rather than adding Pillow as a test dependency. Video fixtures
use `ffmpeg`'s `lavfi testsrc` generator (already installed, no download).

**Why:** Keeps dependencies minimal per project rules; a hand-built PNG is
~15 lines of stdlib code and needs no image library at all.

---

## 2026-09-29 — Python 3.11 for the project virtual environment

**Context:** The machine's default `python`/`py` resolves to Python 3.14.6,
but a `py -3.11` interpreter (3.11.9) is also installed.

**Decision:** Use Python 3.11 for `.venv`, pinned via `pyproject.toml`
(`requires-python = ">=3.11,<3.13"`).

**Why:** PyTorch, ONNX Runtime, and common CV libraries (OpenCV, timm) do not
yet reliably support Python 3.14 at the time of this phase. 3.11 is a
mature, broadly-supported version across the full expected dependency set
(PyTorch, ONNX Runtime, timm, OpenCV, face-detection libraries).

---

## 2026-09-29 — PyTorch installed via explicit CUDA index (cu121)

**Context:** The laptop has an NVIDIA RTX 4050 Laptop GPU (6 GB VRAM,
driver 591.66) but no system CUDA Toolkit (`nvcc`) installed.

**Decision:** Install PyTorch from `https://download.pytorch.org/whl/cu121`
rather than the bare PyPI default, and do not require a system CUDA Toolkit
install. PyTorch's CUDA wheels bundle the CUDA runtime they need.

**Why:** Avoids requiring a system-level CUDA Toolkit install (which the
project rules require asking about first), while still enabling GPU
acceleration through the NVIDIA driver already present.

---

## 2026-09-29 — Config loading uses stdlib dataclasses + PyYAML, not pydantic

**Context:** Phase 0 needs a "configuration loading" mechanism and test.

**Decision:** Implement `ProjectConfig` as a plain `dataclasses.dataclass`
loaded from YAML via PyYAML, rather than adding `pydantic` as a dependency.

**Why:** Keeps Phase 0 dependencies minimal, per instructions. Revisit if a
later phase needs richer validation (nested schemas, custom validators).

---

## 2026-09-29 — src-layout package (`src/configuard`)

**Context:** Need an importable, testable Python package from the start.

**Decision:** Use a `src/` layout with package name `configuard`, added to
`pythonpath` in `pyproject.toml`'s `[tool.pytest.ini_options]` rather than
installing the package in editable mode.

**Why:** Avoids requiring an editable install step for Phase 0 while keeping
the package importable by both `scripts/` and `tests/`. Revisit if
packaging/distribution becomes a requirement.
