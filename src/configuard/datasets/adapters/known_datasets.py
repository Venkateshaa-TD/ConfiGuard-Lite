"""Factory functions for the five named datasets plus the generic adapter,
registered into configuard.datasets.registry.DEFAULT_REGISTRY.

IMPORTANT: none of these datasets were downloaded or accessed to write
this file - every folder/metadata layout below is transcribed from each
dataset's public documentation (paper/official repo) from general
knowledge, not verified against real data (which this project's rules
forbid downloading without asking, and these datasets require their own
access agreements regardless). Confidence varies per dataset - see the
per-dataset docstring and docs/DATASETS.md, which marks anything
uncertain as "verification required" rather than guessing silently.
Adjust the specs below once real access + structure confirmation exists.
"""

from __future__ import annotations

import re
from dataclasses import replace
from pathlib import Path

from configuard.datasets.adapters.folder_convention import (
    FolderBucket,
    FolderConventionAdapter,
    FolderConventionSpec,
)
from configuard.datasets.adapters.metadata_sidecar import MetadataSidecarAdapter, MetadataSidecarSpec
from configuard.datasets.registry import DEFAULT_REGISTRY, DatasetSpec
from configuard.datasets.schema import Sample, SampleLabel, SampleMediaType


# --------------------------------------------------------------- FF++ ---

# Official FF++ manipulated filenames are "<target>_<source>.mp4" (both
# 3-digit original_sequences/youtube video IDs) - verified against the
# real c23 download in Phase 5b (docs/DATASETS.md).
FFPP_MANIPULATED_STEM = re.compile(r"^(?P<target>\d{3})_(?P<source>\d{3})$")


class FaceForensicsAdapter(FolderConventionAdapter):
    """FolderConventionAdapter plus FF++'s two-original lineage.

    The generic engine links a fake only to the original sharing its
    leading filename token (the *target* video). But `000_003.mp4` is
    built from TWO originals: the target `000` (frames/background) and
    the source `003` (the swapped-in face for Deepfakes/FaceSwap, the
    driving expressions for Face2Face/NeuralTextures). A split that
    only kept `000` with it could put `000_003` in train and `003.mp4`
    in test - identity leakage. So each fake gets
    parent_sample_id = target original, paired_sample_id = source
    original; Phase 3's leakage grouping unions on both.

    identity_id is deliberately NOT set: FF++ publishes no identity labels
    (Sample.identity_id is only for dataset-provided identities)."""

    def build_manifest(self, root: Path) -> list[Sample]:
        samples = super().build_manifest(root)
        real_by_key: dict[tuple[str, str | None], str] = {
            (Path(s.media_path).stem, s.compression_level): s.sample_id
            for s in samples if s.label is SampleLabel.REAL
        }
        for i, sample in enumerate(samples):
            if sample.label is not SampleLabel.FAKE:
                continue
            match = FFPP_MANIPULATED_STEM.match(Path(sample.media_path).stem)
            if match is None:
                continue
            target = real_by_key.get((match["target"], sample.compression_level))
            source = real_by_key.get((match["source"], sample.compression_level))
            samples[i] = replace(sample, source_id=match["target"], parent_sample_id=target, paired_sample_id=source)
        return samples


def make_faceforensics_adapter() -> FaceForensicsAdapter:
    """FaceForensics++ (Rossler et al.). Structure per the official repo
    (github.com/ondyari/FaceForensics): original_sequences/youtube/<c>/videos
    and manipulated_sequences/<method>/<c>/videos, c in {raw, c23, c40}.
    Confidence: high for the top-level layout; filename source/target
    convention treated liberally (see folder_convention.py)."""
    methods = ("Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures", "FaceShifter")
    compressions = ("raw", "c23", "c40")

    buckets = [
        FolderBucket(
            relative_dir=f"original_sequences/youtube/{c}/videos",
            label=SampleLabel.REAL,
            compression_level=c,
        )
        for c in compressions
    ] + [
        FolderBucket(
            relative_dir=f"manipulated_sequences/{method}/{c}/videos",
            label=SampleLabel.FAKE,
            manipulation_family="face-swap" if method in ("Deepfakes", "FaceSwap", "FaceShifter") else "reenactment",
            generator_method=method,
            compression_level=c,
        )
        for method in methods
        for c in compressions
    ]

    spec = FolderConventionSpec(
        dataset_name="faceforensics++",
        dataset_version="v1",
        license_status="restricted - requires signing the official FaceForensics++ EULA (see docs/DATASETS.md)",
        buckets=tuple(buckets),
        media_type=SampleMediaType.VIDEO,
    )
    return FaceForensicsAdapter(spec)


# ---------------------------------------------------------- Celeb-DF ---

def make_celebdf_adapter() -> FolderConventionAdapter:
    """Celeb-DF-v2 (Li et al.). Structure per the official repo
    (github.com/yuezunli/celeb-deepfakeforensics): Celeb-real/,
    Celeb-synthesis/, YouTube-real/ (all *.mp4), with
    List_of_testing_videos.txt marking the official test split.
    Confidence: high for folder names; the split-file line format is
    parsed leniently (last whitespace-separated token = path)."""
    buckets = (
        FolderBucket(relative_dir="Celeb-real", label=SampleLabel.REAL),
        FolderBucket(relative_dir="YouTube-real", label=SampleLabel.REAL),
        FolderBucket(
            relative_dir="Celeb-synthesis",
            label=SampleLabel.FAKE,
            manipulation_family="face-swap",
            generator_method="Celeb-DF-synthesis",
        ),
    )
    spec = FolderConventionSpec(
        dataset_name="celeb-df-v2",
        dataset_version="v2",
        license_status="restricted - requires requesting access via the official Celeb-DF Google Form (see docs/DATASETS.md)",
        buckets=buckets,
        media_type=SampleMediaType.VIDEO,
        official_split_file="List_of_testing_videos.txt",
    )
    return FolderConventionAdapter(spec)


# -------------------------------------------------------------- DFDC ---

def make_dfdc_adapter() -> MetadataSidecarAdapter:
    """DFDC (Deepfake Detection Challenge, Dolhansky et al.). Each part
    directory ships its own metadata.json: {"<file>.mp4": {"label":
    "REAL"|"FAKE", "split": "train", "original": "<file>.mp4"|null}}.
    Confidence: high (this format is extensively documented in the public
    Kaggle competition materials). This adapter targets ONE part
    directory at a time (call build_manifest once per
    dfdc_train_part_N/ root and concatenate)."""
    spec = MetadataSidecarSpec(
        dataset_name="dfdc",
        dataset_version="v1",
        license_status="restricted - requires accepting the official DFDC competition rules/EULA (see docs/DATASETS.md)",
        metadata_relative_path="metadata.json",
        media_type=SampleMediaType.VIDEO,
        format="json",
        path_key="path",
        label_key="label",
        real_values=("REAL", "real"),
        fake_values=("FAKE", "fake"),
        split_key="split",
        parent_key="original",
    )
    return MetadataSidecarAdapter(spec)


# -------------------------------------------------------------- DF40 ---

def make_df40_adapter() -> MetadataSidecarAdapter:
    """DF40 (a 2024+ forgery-method-diversity benchmark). VERIFICATION
    REQUIRED: exact on-disk layout not independently confirmed for this
    project - this adapter assumes a `metadata.jsonl` at the dataset root
    with (at least) `path`, `label`, `method` columns, which is a common
    convention for recent benchmark releases but not verified against the
    real DF40 distribution. Adjust metadata_relative_path/keys once real
    access exists - see docs/DATASETS.md."""
    spec = MetadataSidecarSpec(
        dataset_name="df40",
        dataset_version="v1",
        license_status="unknown - verification required (see docs/DATASETS.md)",
        metadata_relative_path="metadata.jsonl",
        media_type=SampleMediaType.IMAGE,
        format="jsonl",
        path_key="path",
        label_key="label",
        method_key="method",
        manipulation_family_key="family",
        split_key="split",
    )
    return MetadataSidecarAdapter(spec)


# ----------------------------------------------------- DeeperForensics ---

def make_deeperforensics_adapter() -> FolderConventionAdapter:
    """DeeperForensics-1.0 (Jiang et al.). VERIFICATION REQUIRED: the
    `source_videos/` bucket name below is a best-effort guess (the paper
    describes real source actor recordings distinct from FaceForensics++)
    and has NOT been confirmed against the official release structure -
    only `manipulated_videos/` (fake, end-to-end perturbed) is stated with
    moderate confidence. Adjust bucket paths once real access exists."""
    buckets = (
        FolderBucket(relative_dir="source_videos", label=SampleLabel.REAL),
        FolderBucket(
            relative_dir="manipulated_videos",
            label=SampleLabel.FAKE,
            manipulation_family="face-swap",
            generator_method="DF-VAE",
        ),
    )
    spec = FolderConventionSpec(
        dataset_name="deeperforensics-1.0",
        dataset_version="v1",
        license_status="restricted - requires signing the official DeeperForensics-1.0 EULA (see docs/DATASETS.md)",
        buckets=buckets,
        media_type=SampleMediaType.VIDEO,
    )
    return FolderConventionAdapter(spec)


# ------------------------------------------------------------ generic ---

def make_generic_metadata_adapter(
    dataset_name: str,
    dataset_version: str,
    metadata_relative_path: str = "metadata.jsonl",
    media_type: SampleMediaType = SampleMediaType.IMAGE,
    format: str = "jsonl",  # noqa: A002 - matches MetadataSidecarSpec.format
    license_status: str = "unknown - verification required",
    **overrides: object,
) -> MetadataSidecarAdapter:
    """Generic adapter for any future image-only (or video) face-deepfake
    dataset that ships a metadata sidecar with path/label columns -
    docs/PROJECT_PLAN.md requirement 1's "generic adapter" for datasets
    not named above. No folder-layout assumptions; fully configurable via
    MetadataSidecarSpec keyword overrides (path_key, label_key,
    identity_id_key, method_key, split_key, parent_key, ...)."""
    spec = MetadataSidecarSpec(
        dataset_name=dataset_name,
        dataset_version=dataset_version,
        license_status=license_status,
        metadata_relative_path=metadata_relative_path,
        media_type=media_type,
        format=format,  # type: ignore[arg-type]
        **overrides,  # type: ignore[arg-type]
    )
    return MetadataSidecarAdapter(spec)


def _register_known_datasets() -> None:
    DEFAULT_REGISTRY.register(
        DatasetSpec(
            name="faceforensics++",
            version="v1",
            description="FaceForensics++: 1000 real YouTube videos, 5 manipulation methods, 3 compression levels.",
            license_status="restricted - EULA required",
            adapter_factory=make_faceforensics_adapter,
        )
    )
    DEFAULT_REGISTRY.register(
        DatasetSpec(
            name="celeb-df-v2",
            version="v2",
            description="Celeb-DF-v2: celebrity real/synthesized face-swap videos with an official test split.",
            license_status="restricted - access request required",
            adapter_factory=make_celebdf_adapter,
        )
    )
    DEFAULT_REGISTRY.register(
        DatasetSpec(
            name="dfdc",
            version="v1",
            description="Deepfake Detection Challenge dataset (Meta/AWS/Kaggle).",
            license_status="restricted - competition EULA required",
            adapter_factory=make_dfdc_adapter,
        )
    )
    DEFAULT_REGISTRY.register(
        DatasetSpec(
            name="df40",
            version="v1",
            description="DF40: multi-forgery-method benchmark. Structure verification required.",
            license_status="unknown - verification required",
            adapter_factory=make_df40_adapter,
        )
    )
    DEFAULT_REGISTRY.register(
        DatasetSpec(
            name="deeperforensics-1.0",
            version="v1",
            description="DeeperForensics-1.0: large-scale end-to-end perturbed face-swap videos.",
            license_status="restricted - EULA required",
            adapter_factory=make_deeperforensics_adapter,
        )
    )


_register_known_datasets()
