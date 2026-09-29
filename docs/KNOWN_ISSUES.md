# Known Issues / Blockers

Format: one entry per issue. Mark resolved issues rather than deleting them.

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
ahead of Phase 3b (wiring in a real dataset).
