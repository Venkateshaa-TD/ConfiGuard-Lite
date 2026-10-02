# Datasets

Status: **FaceForensics++ c23 (videos: original + Deepfakes, Face2Face,
FaceSwap, NeuralTextures) acquired and validated on 2026-09-30 (Phase 5b)**,
see "FaceForensics++ c23 acquisition" below. No other dataset has been
downloaded. Text written before Phase 5b follows: Phase 3 added a typed
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

### GenD CLIP-L/14 teacher (Phase 6a — frozen, offline distillation target only)

| Field | Value |
|---|---|
| Purpose | Frozen teacher whose logits on FF++ train/val crops are the distillation signal. Never trained, exported, or served |
| Source | Hugging Face `yermandy/GenD_CLIP_L_14`, model card: "the GenD (CLIP) model from Tab. 2" of Yermakov et al., *Deepfake Detection that Generalizes Across Benchmarks*, WACV 2026 (arXiv 2508.06248) |
| Revision | `891ce014a0308386c4d7d25b3dcf436a22db5504` (pinned as `GEND_REVISION`) |
| Training code reviewed | github.com/yermandy/GenD @ `387a42266dd385fbe8f7626c5d3aa03eea3bfaab` |
| License | MIT (model card `license: mit`; the GitHub repo includes an MIT `LICENSE`) |
| Local path | `D:\ConfiGuard-Data\cache\huggingface\hub\models--yermandy--GenD_CLIP_L_14\snapshots\891ce014…` |
| Training data | FF++ c23 **official train split only** (720 real + 4×720 fake = 3600 videos, ~115k frames). Model selection used DeepSpeak v1.1/v2, CDFv3 and FFIW; FF++ val was not used and FF++ test only for evaluation. See `docs/DECISIONS.md`. |
| Not downloaded | `openai/clip-vit-large-patch14` (the official loader would fetch it; we rebuild the architecture instead) |

| File | Bytes | SHA-256 |
|---|---|---|
| `model.safetensors` | 1,215,928,160 | `d76f0bdfd74a29fe1b1c1b84a80ac92486993e426878e8c7a3944281fbb96833` |
| `modeling_gend.py` | 5,738 | `d2bdc7d57ea208def628f16064c7d22317ec92878b982ac6ac5ab4bcda4991ac` |
| `config.json` | 193 | `7f2761e13678191774a152121f1630d19baa33a74216dd264d54445d96b16229` |
| `model_index.json` | 107 | `c0baa48ac78a64c2806fab0164e810ef931e9a03be184649b04c8b8b5ebe639e` |
| `README.md` | 1,205 | `19e3a142ac3fc32e531ba5eeff132c5de7e8134d268572e62fc69152804a69f9` |
| `requirements.txt` | 71 | `1724d833db5403c1a36341ff2898ce9500767a48a8ce3135e233c9519da39e55` |

Teacher logits (train/val only, never test) are cached under
`D:\ConfiGuard-Data\cache	eacher_logits\gend_clip_l14\<tag>\` — see
`docs/EXPERIMENT_LOG.md` (Phase 6a) for the tag and file hashes.

## Phase 5 — synthetic training fixtures (generated, not downloaded)

Phase 5 downloaded **nothing** (no dataset, no GenD, no DINOv2, no
additional model). Training runs used:

- **Synthetic images**, generated at run time by
  `configuard.training.synthetic`: 64×64 checkerboards, blue-tinted =
  "real", red-tinted = "fake". These are engineering fixtures, not face
  imagery or deepfakes. Smoke runs write them under
  `CONFIGUARD_OUTPUT_DIR/smoke/<run>/synthetic_data/`; tests write them
  under pytest's `tmp_path`.
- **Synthetic video clips** (tests only): ffmpeg `lavfi` `testsrc` /
  `testsrc2` patterns.
- **The two Phase 4-authorized backbones**, loaded from the existing
  `D:\ConfiGuard-Data\cache\huggingface` cache at the revisions
  recorded above (now also stored in `EncoderSpec.revision` and in every
  checkpoint's provenance), with `HF_HUB_OFFLINE=1`.

## FaceForensics++ c23 acquisition (Phase 5b, 2026-09-30)

Acquired under the user's official FaceForensics++ access grant, using
only the official download script URL from the approval email. **Never
committed**: all data lives outside the repository.

| Item | Value |
|---|---|
| Location | `D:\ConfiGuard-Data\datasets\FaceForensics++` (`CONFIGUARD_DATA_DIR`) |
| Official script URL | The URL from the user's approval email (HTTP, 301-redirects to HTTPS on the same host). **Access information, not committed**: recorded in `D:\ConfiGuard-Data\datasets\FaceForensics++\_official_script\PROVENANCE.md` (outside Git) |
| Script downloaded (UTC) | 2026-09-30T16:32:53Z, 10,727 bytes |
| Script SHA-256 | `5d0b220ad0c88bba9d80f45426aef48a89d182e88956ded85c8a9d310f8d04d0` (pinned in `scripts/download_faceforensics_c23.py`) |
| Official pair list | `v3/misc/filelist.json` on the EU2 server (the file the script itself uses; full URL in `PROVENANCE.md`), fetched 2026-09-30T16:39:03Z, 21,002 bytes, SHA-256 `7099a119c0992751a8fd58fda33569ccb01442d8d13aad7fea9e69dc1d2510f4`, 500 pairs / 1000 unique IDs |
| Server | `EU2` |
| Downloaded | `-c c23 -t videos` for `original`, `Deepfakes`, `Face2Face`, `FaceSwap`, `NeuralTextures` only |
| NOT downloaded | raw/c0, c40, masks, Deepfakes models, extracted images, DeepFakeDetection (+ actors), FaceShifter, YouTube source zips, benchmark data |
| Download window (UTC) | 2026-09-30 16:34:50 (trial) / 16:37 (full) → 17:52:52 |
| Terms of use | The official script's TOS prompt (TOS PDF on the chosen server) was acknowledged on the user's behalf; the user holds official access. Restricted: research use, no redistribution |

Files (all h264, validated by `scripts/validate_faceforensics.py`):

| Class | Files | Bytes | Duration range (s) |
|---|---|---|---|
| original (`original_sequences/youtube/c23/videos/NNN.mp4`) | 1000 | 1,936,585,266 | 5.07-72.56 |
| Deepfakes (`manipulated_sequences/Deepfakes/c23/videos/TTT_SSS.mp4`) | 1000 | 1,992,801,584 | 5.60-72.56 |
| Face2Face | 1000 | 1,946,827,903 | 5.07-72.56 |
| FaceSwap | 1000 | 1,639,456,740 | 5.07-41.52 |
| NeuralTextures | 1000 | 1,525,872,246 | 5.07-41.52 |
| **Total** | **5000** | **9,041,543,739 (8.42 GiB)** | |

Verified filename convention: manipulated videos are `<target>_<source>.mp4`
where both are original IDs forming an official pair; each pair yields
both orders (`a_b`, `b_a`) per method.

Derived (on D:, never committed):
- `...\FaceForensics++\_manifests\faceforensics++_c23.jsonl`: canonical
  Phase 3 manifest, 5000 samples, SHA-256
  `966a4b272184a12ca76f4d4920fac029f90b0b8ee1b2e38af5aaa61b6c95a253`.
- `...\_manifests\faceforensics++_c23_leakage_groups.json`: 500 groups
  × 10 samples (one official pair's 2 originals + 8 fakes).
- Acquisition report: `D:\ConfiGuard-Data\outputs\acquisition\faceforensics\acquisition_report_20260930-232450.{json,md}`.

**Split (Phase 5b): none applied.** The approved acquisition source
provided no split. **Superseded in Phase 5c:** the official split was
fetched with the user's approval and applied (next section).

## FaceForensics++ official split (Phase 5c, 2026-09-30)

Fetched with the user's explicit authorization, **only** from the
authors' public repository. **Not committed** (see `docs/DECISIONS.md`):
working copies are at
`D:\ConfiGuard-Data\datasets\FaceForensics++\_official_splits\b952e41cba017eb37593c39e12bd884a934791e1\`.

| File | URL (pinned revision) | Bytes | SHA-256 | Git blob SHA |
|---|---|---|---|---|
| train.json | `https://raw.githubusercontent.com/ondyari/FaceForensics/b952e41cba017eb37593c39e12bd884a934791e1/dataset/splits/train.json` | 10,802 | `e59386911255e6fb0a79a7808ec2210536b8c0474268342694c4bca766d1b347` | `979240f9091412ad17b998ab7b06899e54c3b771` |
| val.json | `.../dataset/splits/val.json` | 2,102 | `b48cc511f66938e05356aaa9c67e150b1e3638c355db3d745225b0022884b36e` | `731b584efea371a81a55c59b3200c2b1320d72f4` |
| test.json | `.../dataset/splits/test.json` | 2,102 | `886f5a0da623c25820692e0d8dc33d197ddb1db527a7f1cfcb9bcbca60fe4f40` | `854b8019a5e279473ac41671d0946b8476920d6b` |

- Repository: `ondyari/FaceForensics`, default branch `master`, head
  commit `b952e41cba017eb37593c39e12bd884a934791e1` (2020-07-15, "Fixed alignment on faceshifter"). Fetched
  2026-09-30T18:05:57Z. All three git blob SHAs matched GitHub's API.
- License: repository code is MIT (`LICENSE`); the README states the data
  is under the FaceForensics Terms of Use.
- Format: JSON list of `["NNN", "NNN"]` original-ID pairs.

Reconciliation (all passed; `scripts/apply_faceforensics_splits.py`):

| Split | Pairs | Originals | Deepfakes | Face2Face | FaceSwap | NeuralTextures | Total videos | Leakage groups |
|---|---|---|---|---|---|---|---|---|
| train | 360 | 720 | 720 | 720 | 720 | 720 | 3600 | 360 × 10 |
| val | 70 | 140 | 140 | 140 | 140 | 140 | 700 | 70 × 10 |
| test | 70 | 140 | 140 | 140 | 140 | 140 | 700 | 70 × 10 |
| **total** | **500** | **1000** | 1000 | 1000 | 1000 | 1000 | **5000** | 500 |

- The 500 split pairs equal the official download pair list exactly;
  every original is in exactly one split.
- Both originals of every fake are in its split. No source, parent,
  pair, or leakage group crosses partitions.

Split manifests (on D:, never committed), in `...\FaceForensics++\_manifests\`:

| File | SHA-256 |
|---|---|
| `faceforensics++_c23_train.jsonl` | `c036d648a7e8e2dfab89c0027bb6b314cf603c471589dfb57d47ba1a880a0809` |
| `faceforensics++_c23_val.jsonl` | `f7d28952d5894e4851d24d76cf36fd7e98fc66d7761d36e067e7e0d254479db9` |
| `faceforensics++_c23_test.jsonl` | `e8d734379518c4427e40abbd96b38f158c6b590d89b51e395d7666e6b104e2ec` |
| `faceforensics++_c23_official_split.jsonl` (all 5000, labelled) | `78225658f9c5247be8105631ffc8db99547f8d463a2e518d9df701930890db55` |

The Phase 5b base manifest (`faceforensics++_c23.jsonl`, `966a4b27…`,
`official_split = null`) is kept unchanged.

## FaceForensics++ c23 matched face crops (Phase 5d, 2026-10-01)

Derived from the Phase 5c official-split manifest (`78225658…`). Nothing
was downloaded in this phase, and nothing here is committed.

**Official filename convention (verified from official sources, not
guessed):** `dataset/README.md` @ `ondyari/FaceForensics` `b952e41c`:
"All filenames are of the form `<target sequence>_<source sequence>`".
Rössler et al. 2019 (arXiv:1901.08971v3, appendix) defines the roles:
reenactment transfers "the expressions of the source video … to the
target video while retaining the identity of the target person"; face
swapping replaces "the face in the target video with the face in the
source video". So for `TTT_SSS.mp4`:
- content parent = original `TTT` (frames);
- donor parent = original `SSS` (face/expressions);
- both are leakage parents.

The same appendix explains clip lengths, which the full probe of all
5000 videos confirmed: DF = target length (1000/1000); F2F = source
length, target rewound (992/1000; the other 8 have equal lengths);
FS/NT = min(target, source) (1000 and 999/1000).

| Item | Value |
|---|---|
| Store root | `D:\ConfiGuard-Data\cache\ffpp_face_crops\store` (`CONFIGUARD_CACHE_DIR`) |
| Config tag | `p5d-b451b5ca770c8923` (YuNet `2026may` SHA-256 `ebafce4e…`, margin 0.25, 224×224 PNG level 3) |
| Families | 1000 (target original + 4 fakes); **991 accepted, 9 quarantined** |
| Accepted videos / crops | 4955 videos × 16 = **79,280 crops** (train 57,040 / val 11,120 / test 11,120) |
| Per class (train/val/test) | 713 / 139 / 139 videos for each of original, DF, F2F, FS, NT |
| Matched pairs | 3,964 fakes paired with their content original; 63,424/63,424 slots exactly matched by frame index |
| Quarantined | 45 videos (9 per class): families 212, 370, 509, 738 (content-wide), 386, 569, 618, 894, 908 (fake-only) |
| Storage | 4.91 GB crops; D: free 77.2 GB at end (floor 40 GB) |

Manifests (`...\store\manifests\p5d-b451b5ca770c8923\full\`, byte-identical on rerun):

| File | Purpose | SHA-256 |
|---|---|---|
| `crops_train.jsonl` | model-facing, whitelisted fields | `38635dd39670371ab3b97e295e6f4ddcdffb41ad4e7a69588c5aea54a7104c2e` |
| `crops_val.jsonl` | model-facing | `5faf4a3dc2175e5f366a2f9a5a608a86432c5d4eb638de951f3c9e38978126fb` |
| `crops_test.jsonl` | model-facing | `e0acefde2628b9d91aa229272688e60742586b299c8c3b7ecdbd8fc5b333a373` |
| `matched_pairs.jsonl` | fake ↔ content original (+ donor), per-slot frame indices | `94cd6492039d90e94a0de4c1a106690aa4258a8e1d867a3988d54b912f890d85` |
| `quarantine.jsonl` | quarantined videos + reasons | `be2cb6472e334f81b04a8b690d4ef6747d3e04d9c22984f603431dbf2cc31fd9` |
| `crop_audit.jsonl` | audit only (resolution, duration, detection stats) — never a model input | `e63b41e6e888be25a7960d0b834a36363f3acf005f7bbf484fda0fd676a18409` |
| `extraction_summary.json` | deterministic summary + leakage re-validation | `761f2aa40cfb92381b10b3cd58593cb4f6c2f05fc204f85ced993ed9b6700e5d` |

Reports, logs, and contact sheets (human review only) are under
`...\store\reports\` and `...\store\contact_sheets\`. Superseded trial
and run stores are kept (refused as stale) in
`...\ffpp_face_crops\superseded_*`.

---

## C2PA SDK and Trust List (Phase 12)

| Item | Source | Version / pin | Integrity | License |
|---|---|---|---|---|
| `c2pa-python` (CAI SDK; native c2pa-rs 0.91.0) | PyPI / github.com/contentauth/c2pa-python | 0.38.0, `c2pa_python-0.38.0-py3-none-win_amd64.whl` (87,904,872 B) | sha256 `5db598a420242229a802c07e84890adb5162aac79afb110e4563c56f33e35256` | MIT OR Apache-2.0 |
| `C2PA-TRUST-LIST.pem` (official signer anchors, 30 certs) | github.com/c2pa-org/conformance-public `trust-list/` | commit `3573be509a793a989f093df4f86744a3632f6155` (2026-10-01) | git blob `a0d20fd7…25b5ef`, sha256 `75cacc98b79ecac33713c7ecfb58d4a0ef383f3c1f886e7409f9e37e8664aea5`, 37,911 B | C2PA Conformance Program public list |
| `C2PA-TSA-TRUST-LIST.pem` (TSA anchors, 22 certs) | same | same commit | git blob `30de202a…5b3f24`, sha256 `c688d3555f4a2f1f8d663472bbd37888ff234abdd234c25934c0f9292e4eb5c9`, 28,863 B | same |

Cached at `D:\ConfiGuard-Data\cache\c2pa_trust\3573be50…\` with `provenance.json`
(URLs, hashes, fetched 2026-10-02T04:48:57Z). Refresh only by bumping the pin in
`configuard/provenance/trust.py` and re-running `scripts/fetch_c2pa_trust_list.py`.
Test credentials are generated at test time and are never stored.

## Landing-page hero head (12d) — MakeHuman CC0 assets

The one 3D asset on the website. It is not training data and never touches the model.

- **Licence:** CC0 1.0. It was independently verified on 2026-10-02 at
  https://static.makehumancommunity.org/about/license.html ("All core
  assets are shared under Creative Commons, CC0"). Each data file also
  carries the CC0 notice. The MakeHuman application code (AGPL) is
  **not** used or installed.
- **Copyright holders at CC0 release:** Data Collection AB, Joel
  Palmius, Jonas Hauquier.
- **Identity:** fictional, generic young adult. It is an equal blend of
  the African/Asian/Caucasian young-male macro targets on the neutral
  hm08 base mesh: not a scan, not a celebrity, not a recognisable person.
- **Never used:** Kimi/GetLayers assets, Mixamo or MetaHuman.

| Input | Source | Bytes | SHA-256 | Retrieved (UTC) |
|---|---|---|---|---|
| `makehuman_system_assets_cc0.zip` (skins, eyes, eyebrows, eyelashes, hair) | https://files.makehumancommunity.org/asset_packs/makehuman_system_assets/makehuman_system_assets_cc0.zip | 280,737,770 | `b542127a8e25547c7c29c19f2d1d2adb9a664c80396ecd694095dbc8028a0107` | 2026-10-02T09:28:06Z |
| `base.obj` (hm08 base mesh) | github.com/makehumancommunity/makehuman @ `a8bc2d54ff0ac92e78ff71431b1023eda42bf482` `makehuman/data/3dobjs/base.obj` | 1,749,303 | `8e761e6624b8f54536409135d1636da63b32486a90d4897f84e121d144f6fb4c` | 2026-10-02 |
| `african-male-young.target` | same commit, `makehuman/data/targets/macrodetails/` | 406,170 | `894abc1fbb3d28543a51fef16f89d5d4bdf9aa2e1534413339811a3d47818b7d` | 2026-10-02 |
| `asian-male-young.target` | same commit, `makehuman/data/targets/macrodetails/` | 421,040 | `ed2e8c191cb6b87b4a2d97c80486acb2604fa549316c7aa2a738a5d5a14334dc` | 2026-10-02 |
| `caucasian-male-young.target` | same commit, `makehuman/data/targets/macrodetails/` | 396,479 | `70e228ba7164737dae664454394536fc5935fa48d333c1a97d77e2dc6eacc5f5` | 2026-10-02 |

- **Components used from the pack:**
  - `skins/young_caucasian_male/young_lightskinned_male_diffuse.png`
  - `eyes/high-poly` with `eyes/materials/brown_eye.png`; the outer
    cornea shell is dropped.
  - `eyebrows/eyebrow001`
  - `eyelashes/eyelashes01`
  - `hair/short02` (diffuse only)
- **Inputs are stored outside the repo** in
  `D:/ConfiGuard-Data/cache/makehuman/` (with `pack.sha256`,
  `github.sha256` and retrieval-time files). They are never committed.
- **Output (committed, self-hosted):**
  - `frontend/public/hero/head.glb`: 1,643,008 bytes, SHA-256
    `85e6942d93753435bbb93c9c2852e5286dc75c79325e0953d4035485bd6b8bea`,
    13,716 triangles.
  - Textures are JPEG/PNG ≤ 1024 px. There is no executable content and
    no Draco/meshopt.
  - The output is CC0 (derived from CC0 inputs). Its provenance record
    is `frontend/public/hero/head.provenance.json`.
- **Posters:** `poster-light.webp` and `poster-dark.webp`, about 44 KB
  each, are still renders of the same head.
- **Rebuild:**
  `python scripts/build_hero_head.py --src <cache> --out frontend/public/hero/head.glb`,
  then `node frontend/scripts/render-hero-posters.mjs <server-url>`.
