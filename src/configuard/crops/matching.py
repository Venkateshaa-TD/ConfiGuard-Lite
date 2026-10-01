"""Shared-range nested sampling and deterministic nearby-frame recovery for
one content family. Every rule here is applied identically to every member
of the family, real or fake: nothing depends on the label or method.

Temporal correspondence: a fake's frame i shows the target original's
frame i (verified in Phase 5d by frame registration, docs/DECISIONS.md),
so equivalent temporal positions are equal FRAME INDICES, not timestamps:
16 of the 64 fakes whose fps header differs from their target still align
at offset 0 by index.

Shared range: the official methods differ in length (paper appendix and
Phase 5d probe): Deepfakes = target length, FaceSwap/NeuralTextures =
min(target, source), Face2Face = source length (the target is *rewound*
past its end). The family's shared range is [0, min(frame counts) - 1];
inside it every fake frame has a genuine content counterpart, and the
rewound Face2Face tail is excluded. All members, including the original,
sample the same 16 indices from that range, so the sampled span carries
no duration information that differs between real and fake.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from configuard.media.sampling import MAX_FRAME_COUNT, compute_nested_sampling_plans

SLOTS_16 = tuple(range(MAX_FRAME_COUNT))
# Mirrors configuard.media.sampling: the 8-set is every 2nd element of the
# 16-set and the 4-set every 2nd element of the 8-set.
SLOTS_8 = SLOTS_16[::2]
SLOTS_4 = SLOTS_8[::2]

MODE_PLANNED = "planned"
MODE_JOINT = "joint_recovery"
MODE_INDIVIDUAL = "individual_recovery"
MODE_FAILED = "failed"


class SamplingError(Exception):
    """The shared range cannot provide 16 distinct ordered frames."""


def shared_frame_count(frame_counts: Sequence[int]) -> int:
    if not frame_counts or min(frame_counts) <= 0:
        raise SamplingError(f"invalid frame counts {list(frame_counts)}")
    return min(frame_counts)


def planned_indices(shared_count: int) -> tuple[int, ...]:
    """The existing nested 4/8/16 contract applied to the shared range."""
    indices = compute_nested_sampling_plans(shared_count)[MAX_FRAME_COUNT].indices
    if len(indices) != MAX_FRAME_COUNT:
        raise SamplingError(f"shared range of {shared_count} frames cannot give {MAX_FRAME_COUNT} distinct frames")
    return tuple(indices)


def nested_levels(slot: int) -> tuple[int, ...]:
    """Frame budgets (4/8/16) whose selection contains this slot."""
    return tuple(n for n, slots in ((4, SLOTS_4), (8, SLOTS_8), (16, SLOTS_16)) if slot in slots)


def recovery_offsets(planned: Sequence[int], slot: int, shared_count: int, max_offset: int) -> list[int]:
    """Deterministic candidate offsets for one slot: +1, -1, +2, -2, ...

    Bounded by `max_offset` and by HALF the gap to each neighbouring
    planned index, so two recovered neighbours can never meet or swap:
    the final indices stay strictly increasing whatever is recovered."""
    index = planned[slot]
    left_gap = index - planned[slot - 1] if slot > 0 else index + 1
    right_gap = planned[slot + 1] - index if slot + 1 < len(planned) else shared_count - index
    left = min(max_offset, (left_gap - 1) // 2 if slot > 0 else index)
    right = min(max_offset, (right_gap - 1) // 2 if slot + 1 < len(planned) else shared_count - 1 - index)
    offsets: list[int] = []
    for d in range(1, max_offset + 1):
        if d <= right:
            offsets.append(d)
        if d <= left:
            offsets.append(-d)
    return offsets


@dataclass(frozen=True)
class SlotResolution:
    slot: int
    planned_index: int
    mode: str
    member_indices: tuple[int | None, ...]  # final frame index per member (None = failed)

    @property
    def exact_match(self) -> bool:
        """True when every member uses the same frame index."""
        return self.mode in (MODE_PLANNED, MODE_JOINT)


def resolve_slot(
    slot: int,
    planned_index: int,
    offsets: Sequence[int],
    member_valid: Sequence[Mapping[int, bool]],
) -> SlotResolution:
    """Choose the frame for one slot across all members.

    1. planned index if every member has a valid face there;
    2. else the first offset at which EVERY member is valid (joint
       recovery, keeps the real/fake temporal match exact);
    3. else each member independently: planned if valid, otherwise its
       first valid offset (approximate match, recorded as such);
    4. a member with no valid candidate fails the slot.
    """
    def ok(member: int, index: int) -> bool:
        return bool(member_valid[member].get(index, False))

    members = range(len(member_valid))
    if all(ok(m, planned_index) for m in members):
        return SlotResolution(slot, planned_index, MODE_PLANNED, tuple(planned_index for _ in members))
    for d in offsets:
        if all(ok(m, planned_index + d) for m in members):
            return SlotResolution(slot, planned_index, MODE_JOINT, tuple(planned_index + d for _ in members))
    chosen: list[int | None] = []
    for m in members:
        if ok(m, planned_index):
            chosen.append(planned_index)
            continue
        chosen.append(next((planned_index + d for d in offsets if ok(m, planned_index + d)), None))
    mode = MODE_FAILED if any(c is None for c in chosen) else MODE_INDIVIDUAL
    return SlotResolution(slot, planned_index, mode, tuple(chosen))
