"""Official FaceForensics++ train/val/test split integration (Phase 5c).

The split files come from the authors' repository, pinned to one commit:
    https://github.com/ondyari/FaceForensics  @ OFFICIAL_SPLIT_REVISION
    dataset/splits/{train,val,test}.json   - JSON lists of [id, id] pairs

They are NOT committed (the repo's README puts the data under the
FaceForensics Terms of Use; only the code is MIT). Working copies live on
the data drive and are refused unless their size and SHA-256 match the
pins below (docs/DATASETS.md, docs/DECISIONS.md).

Membership is never modified: a pair is assigned to exactly the split
its file lists. The whole assignment is REFUSED if any source ID, pair,
fake's parent/paired original, or Phase 3 leakage group would cross
partitions.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from configuard.datasets.adapters.known_datasets import FFPP_MANIPULATED_STEM
from configuard.datasets.manifest import validate_samples
from configuard.datasets.schema import Sample, SampleLabel

OFFICIAL_SPLIT_REPO = "ondyari/FaceForensics"
OFFICIAL_SPLIT_REVISION = "b952e41cba017eb37593c39e12bd884a934791e1"
OFFICIAL_SPLIT_URL_TEMPLATE = (
    "https://raw.githubusercontent.com/" + OFFICIAL_SPLIT_REPO + "/{revision}/dataset/splits/{split}.json"
)
SPLIT_NAMES = ("train", "val", "test")


@dataclass(frozen=True)
class PinnedFile:
    size: int
    sha256: str
    git_blob_sha: str


OFFICIAL_SPLIT_PINS: dict[str, PinnedFile] = {
    "train": PinnedFile(10802, "e59386911255e6fb0a79a7808ec2210536b8c0474268342694c4bca766d1b347",
                        "979240f9091412ad17b998ab7b06899e54c3b771"),
    "val": PinnedFile(2102, "b48cc511f66938e05356aaa9c67e150b1e3638c355db3d745225b0022884b36e",
                      "731b584efea371a81a55c59b3200c2b1320d72f4"),
    "test": PinnedFile(2102, "886f5a0da623c25820692e0d8dc33d197ddb1db527a7f1cfcb9bcbca60fe4f40",
                       "854b8019a5e279473ac41671d0946b8476920d6b"),
}

LEAKAGE_ISSUE_CODES = frozenset(
    {"cross_split_source_leakage", "cross_split_identity_leakage", "real_fake_pair_leakage"}
)

Pair = tuple[str, str]


class OfficialSplitError(Exception):
    """The split files are missing, altered, malformed, or would leak."""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_split_pairs(data: Any, split: str) -> list[Pair]:
    """A split file is a JSON list of 2-element lists of 3-digit ID strings."""
    if not isinstance(data, list):
        raise OfficialSplitError(f"{split}: expected a JSON list, got {type(data).__name__}")
    pairs: list[Pair] = []
    for index, item in enumerate(data):
        if not (isinstance(item, list) and len(item) == 2
                and all(isinstance(i, str) and len(i) == 3 and i.isdigit() for i in item)):
            raise OfficialSplitError(f"{split}[{index}]: expected [\"NNN\", \"NNN\"], got {item!r}")
        if item[0] == item[1]:
            raise OfficialSplitError(f"{split}[{index}]: pair of an ID with itself: {item!r}")
        pairs.append((item[0], item[1]))
    return pairs


def load_official_splits(
    split_dir: str | Path, pins: dict[str, PinnedFile] | None = OFFICIAL_SPLIT_PINS
) -> dict[str, list[Pair]]:
    """Reads {train,val,test}.json from `split_dir`. With `pins` (default),
    each file's size and SHA-256 must match exactly; pass pins=None only
    for synthetic test fixtures."""
    split_dir = Path(split_dir)
    splits: dict[str, list[Pair]] = {}
    for split in SPLIT_NAMES:
        path = split_dir / f"{split}.json"
        if not path.is_file():
            raise OfficialSplitError(f"missing official split file: {path}")
        if pins is not None:
            pin = pins[split]
            size, digest = path.stat().st_size, _sha256(path)
            if (size, digest) != (pin.size, pin.sha256):
                raise OfficialSplitError(
                    f"{path}: size/SHA-256 {size}/{digest} != pinned {pin.size}/{pin.sha256} "
                    f"({OFFICIAL_SPLIT_REPO}@{OFFICIAL_SPLIT_REVISION})"
                )
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise OfficialSplitError(f"{path}: invalid JSON: {exc}") from exc
        splits[split] = parse_split_pairs(data, split)
    return splits


@dataclass
class SplitReconciliation:
    pairs_per_split: dict[str, int]
    originals_per_split: dict[str, int]
    problems: list[str] = field(default_factory=list)
    id_to_split: dict[str, str] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.problems


def reconcile_pairs(splits: dict[str, list[Pair]], official_pairs: list[Pair]) -> SplitReconciliation:
    """Split files vs the official 500-pair list: every official pair in
    exactly one split, nothing extra, no original ID in two splits."""
    problems: list[str] = []
    official = {frozenset(p) for p in official_pairs}
    pair_to_split: dict[frozenset[str], str] = {}
    id_to_split: dict[str, str] = {}

    for split, pairs in splits.items():
        for pair in pairs:
            key = frozenset(pair)
            if key in pair_to_split:
                problems.append(f"pair {sorted(key)} listed in both {pair_to_split[key]!r} and {split!r}")
                continue
            pair_to_split[key] = split
            if key not in official:
                problems.append(f"{split}: pair {sorted(key)} is not an official FF++ pair")
            for original in pair:
                previous = id_to_split.setdefault(original, split)
                if previous != split:
                    problems.append(f"original {original} appears in both {previous!r} and {split!r}")

    for key in sorted(official - set(pair_to_split), key=sorted):
        problems.append(f"official pair {sorted(key)} is in no split")
    all_official_ids = {i for p in official_pairs for i in p}
    for original in sorted(all_official_ids - set(id_to_split)):
        problems.append(f"original {original} is in no split")

    return SplitReconciliation(
        pairs_per_split={s: len(p) for s, p in splits.items()},
        originals_per_split={s: sum(1 for v in id_to_split.values() if v == s) for s in splits},
        problems=problems,
        id_to_split=id_to_split,
    )


def assign_official_splits(
    samples: list[Sample],
    reconciliation: SplitReconciliation,
    leakage_groups: dict[str, list[str]],
) -> list[Sample]:
    """Returns copies of `samples` with official_split set, or raises
    OfficialSplitError listing every problem - nothing is partially
    assigned. Checks: every sample gets a split; a fake's target and source
    originals (and its parent/paired samples) share its split; every
    leakage group is inside one split; Phase 3 cross-split validation is clean."""
    problems = list(reconciliation.problems)
    id_to_split = reconciliation.id_to_split
    by_id = {s.sample_id: s for s in samples}
    assigned: dict[str, str] = {}

    for sample in samples:
        stem = Path(sample.media_path).stem
        if sample.label is SampleLabel.REAL:
            ids = [stem]
        else:
            match = FFPP_MANIPULATED_STEM.match(stem)
            if match is None:
                problems.append(f"{sample.sample_id}: not a '<target>_<source>' FF++ fake")
                continue
            ids = [match["target"], match["source"]]
        splits = {id_to_split.get(i) for i in ids}
        if None in splits:
            problems.append(f"{sample.sample_id}: original(s) {ids} not in any official split")
        elif len(splits) > 1:
            problems.append(f"{sample.sample_id}: its originals {ids} are in different splits {sorted(splits)}")
        else:
            assigned[sample.sample_id] = splits.pop()

    for sample in samples:
        split = assigned.get(sample.sample_id)
        for ref in (sample.parent_sample_id, sample.paired_sample_id):
            if split and ref and ref in by_id and assigned.get(ref) != split:
                problems.append(f"{sample.sample_id} ({split}) references {ref} ({assigned.get(ref)})")

    for key, members in leakage_groups.items():
        member_splits = {assigned.get(m) for m in members}
        if len(member_splits) != 1 or None in member_splits:
            problems.append(f"leakage group {key} spans {sorted(map(str, member_splits))}")

    if not problems:
        report = validate_samples(samples, split_assignments=assigned)
        problems += [f"{i.code}: {i.message}" for i in report.issues if i.code in LEAKAGE_ISSUE_CODES]

    if problems:
        shown = "\n".join(f"  - {p}" for p in problems[:50])
        more = f"\n  ... and {len(problems) - 50} more" if len(problems) > 50 else ""
        raise OfficialSplitError(
            f"Refusing the official split: {len(problems)} problem(s) - no split was assigned.\n{shown}{more}"
        )
    return [replace(s, official_split=assigned[s.sample_id]) for s in samples]
