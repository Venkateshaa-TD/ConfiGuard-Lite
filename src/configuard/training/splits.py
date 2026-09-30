"""Training-time guard that no source/identity/real-fake pair spans two
of the train/validation/calibration/test splits (task 5).

Phase 3's configuard.datasets.splitting already *assigns* splits
leakage-safely; this re-checks the manifests actually handed to the
trainer, since they may have been edited, regenerated, or mixed up after
splitting. Every frame of a video shares its Sample's source_id, so a
source-level check also guarantees frames of one source never cross
split boundaries.
"""

from __future__ import annotations

from configuard.datasets.manifest import validate_samples
from configuard.datasets.schema import Sample

LEAKAGE_CODES = frozenset(
    {"cross_split_source_leakage", "cross_split_identity_leakage", "real_fake_pair_leakage"}
)


class SplitLeakageError(Exception):
    """Raised when the splits handed to training share a source,
    identity, real/fake pair, or sample_id."""


def find_cross_split_leakage(splits: dict[str, list[Sample]]) -> list[str]:
    """Returns human-readable leakage problems (empty list == clean)."""
    problems: list[str] = []
    assignments: dict[str, str] = {}
    all_samples: list[Sample] = []
    for split_name, samples in splits.items():
        for sample in samples:
            previous = assignments.get(sample.sample_id)
            if previous is not None and previous != split_name:
                problems.append(f"sample_id {sample.sample_id!r} appears in both {previous!r} and {split_name!r}")
                continue
            assignments[sample.sample_id] = split_name
            all_samples.append(sample)

    report = validate_samples(all_samples, split_assignments=assignments)
    problems.extend(f"{issue.code}: {issue.message}" for issue in report.issues if issue.code in LEAKAGE_CODES)
    return problems


def assert_no_cross_split_leakage(splits: dict[str, list[Sample]]) -> None:
    problems = find_cross_split_leakage(splits)
    if problems:
        raise SplitLeakageError(
            "Refusing to train: data leaks across splits:\n" + "\n".join(f"  - {p}" for p in problems)
        )
