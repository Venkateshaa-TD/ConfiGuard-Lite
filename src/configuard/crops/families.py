"""FaceForensics++ content families.

Official convention (verified in Phase 5d against official sources, see
docs/DATASETS.md):
- dataset/README.md @ ondyari/FaceForensics: "All filenames are of the
  form `<target sequence>_<source sequence>`".
- Rossler et al. 2019, appendix: reenactment transfers "the expressions
  of the source video ... to the target video while retaining the
  identity of the target person"; face swapping replaces "the face in
  the target video with the face in the source video".

So for `TTT_SSS.mp4` the TARGET original `TTT` supplies the frames
(content parent) and the SOURCE original `SSS` supplies the swapped face
or the driving expressions (donor parent). Both are leakage parents.

A content family is one target original plus every fake whose frames come
from it (one per method). Every video in the dataset belongs to exactly
one content family, and a family never crosses a split because both
parents of every fake share the fake's official split (Phase 5c).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from configuard.datasets.adapters.known_datasets import FFPP_MANIPULATED_STEM
from configuard.datasets.schema import Sample, SampleLabel
from configuard.datasets.splitting import compute_leakage_groups

FFPP_METHODS = ("Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures")


class FamilyError(Exception):
    """The manifest cannot be partitioned into well-formed content families."""


@dataclass(frozen=True)
class FamilyMember:
    sample_id: str
    media_path: str
    label: str  # "real" | "fake"
    method: str | None  # None for the real member
    split: str
    checksum_sha256: str
    source_id: str
    content_parent_sample_id: str | None  # target original (None for the original itself)
    donor_parent_sample_id: str | None  # source original (None for the original itself)
    leakage_group: str

    @property
    def is_real(self) -> bool:
        return self.label == SampleLabel.REAL.value


@dataclass(frozen=True)
class ContentFamily:
    family_id: str  # the target original's 3-digit ID
    split: str
    members: tuple[FamilyMember, ...]  # real member first, then fakes in FFPP_METHODS order

    @property
    def real(self) -> FamilyMember:
        return self.members[0]

    @property
    def fakes(self) -> tuple[FamilyMember, ...]:
        return self.members[1:]


def _ids_of(sample: Sample) -> tuple[str, str | None]:
    stem = Path(sample.media_path).stem
    if sample.label is SampleLabel.REAL:
        return stem, None
    match = FFPP_MANIPULATED_STEM.match(stem)
    if match is None:
        raise FamilyError(f"{sample.sample_id}: not an official '<target>_<source>' FF++ fake")
    return match["target"], match["source"]


def build_content_families(
    samples: list[Sample], methods: tuple[str, ...] = FFPP_METHODS
) -> list[ContentFamily]:
    """Partition official-split samples into content families, or raise
    FamilyError listing every structural problem. Nothing is guessed: each
    family must have exactly one original and exactly one fake per method,
    and every member, plus both parents of every fake, must be in one split."""
    problems: list[str] = []
    if any(s.official_split is None for s in samples):
        raise FamilyError("every sample must carry its official_split (use the Phase 5c manifest)")
    groups = compute_leakage_groups(samples)
    group_of = {sid: key for key, members in groups.items() for sid in members}
    by_id = {s.sample_id: s for s in samples}
    real_by_id: dict[str, Sample] = {}
    fakes_by_target: dict[str, list[tuple[Sample, str]]] = {}

    for sample in samples:
        try:
            target, source = _ids_of(sample)
        except FamilyError as exc:
            problems.append(str(exc))
            continue
        if source is None:
            if target in real_by_id:
                problems.append(f"duplicate original for ID {target}")
            real_by_id[target] = sample
        else:
            fakes_by_target.setdefault(target, []).append((sample, source))

    families: list[ContentFamily] = []
    for target in sorted(set(real_by_id) | set(fakes_by_target)):
        real = real_by_id.get(target)
        if real is None:
            problems.append(f"family {target}: fakes present but the target original is missing")
            continue
        fakes = fakes_by_target.get(target, [])
        found = sorted(s.generator_method or "" for s, _ in fakes)
        if found != sorted(methods):
            problems.append(f"family {target}: methods {found} != expected {sorted(methods)}")
            continue
        members = [FamilyMember(
            sample_id=real.sample_id, media_path=real.media_path, label=real.label.value, method=None,
            split=str(real.official_split), checksum_sha256=str(real.checksum_sha256), source_id=real.source_id,
            content_parent_sample_id=None, donor_parent_sample_id=None, leakage_group=group_of[real.sample_id],
        )]
        for fake, source in sorted(fakes, key=lambda item: methods.index(item[0].generator_method or "")):
            donor = real_by_id.get(source)
            if donor is None:
                problems.append(f"{fake.sample_id}: donor original {source} missing")
                continue
            if fake.parent_sample_id != real.sample_id or fake.paired_sample_id != donor.sample_id:
                problems.append(f"{fake.sample_id}: manifest lineage disagrees with the filename convention")
            if not (fake.official_split == real.official_split == donor.official_split):
                problems.append(f"{fake.sample_id}: fake/target/donor splits differ")
            if not (group_of[fake.sample_id] == group_of[real.sample_id] == group_of[donor.sample_id]):
                problems.append(f"{fake.sample_id}: not in one leakage group with both parents")
            members.append(FamilyMember(
                sample_id=fake.sample_id, media_path=fake.media_path, label=fake.label.value,
                method=fake.generator_method, split=str(fake.official_split),
                checksum_sha256=str(fake.checksum_sha256), source_id=fake.source_id,
                content_parent_sample_id=real.sample_id, donor_parent_sample_id=donor.sample_id,
                leakage_group=group_of[fake.sample_id],
            ))
        if any(m.checksum_sha256 in ("", "None") for m in members):
            problems.append(f"family {target}: a member has no checksum_sha256")
        families.append(ContentFamily(family_id=target, split=str(real.official_split), members=tuple(members)))

    covered = {m.sample_id for f in families for m in f.members}
    missing = sorted(set(by_id) - covered)
    if missing and not problems:
        problems.append(f"{len(missing)} samples belong to no family (e.g. {missing[:3]})")
    if problems:
        shown = "\n".join(f"  - {p}" for p in problems[:50])
        raise FamilyError(f"{len(problems)} family problem(s):\n{shown}")
    return families
