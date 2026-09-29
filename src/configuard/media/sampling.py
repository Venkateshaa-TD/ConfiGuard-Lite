"""Deterministic, nested uniform frame sampling for adaptive 4/8/16-frame
video analysis (docs/PROJECT_PLAN.md requirement 9).

Nesting is achieved by construction rather than by independently sampling
each level: the 16-frame set is a uniform sample of the video, the 8-frame
set is every second element of the 16-frame set, and the 4-frame set is
every second element of the 8-frame set. This guarantees
indices(4) subset-of indices(8) subset-of indices(16) exactly, and as a
side effect means frames already decoded/analyzed at a smaller frame count
can be reused if a later stage escalates to a larger one - see
docs/DECISIONS.md for the full rationale.
"""

from __future__ import annotations

import numpy as np

from configuard.media.types import FrameSamplingPlan

MAX_FRAME_COUNT = 16
MID_FRAME_COUNT = 8
MIN_FRAME_COUNT = 4


def _uniform_indices(frame_count: int, n: int) -> list[int]:
    """n (or fewer, if frame_count < n) unique, ascending, ~evenly spaced
    indices in [0, frame_count - 1]. Deduplicates rather than repeating a
    frame for short videos."""
    if frame_count <= 0 or n <= 0:
        return []
    n = min(n, frame_count)
    if n == 1:
        return [0]
    raw = np.linspace(0, frame_count - 1, num=n)
    # dict.fromkeys preserves first-seen order while deduplicating, then we
    # sort ascending (rounding can occasionally produce a slightly
    # out-of-order duplicate collision at the boundary).
    seen = dict.fromkeys(int(round(x)) for x in raw)
    return sorted(seen)


def compute_nested_sampling_plans(frame_count: int) -> dict[int, FrameSamplingPlan]:
    """Returns {16: plan, 8: plan, 4: plan}, each a FrameSamplingPlan whose
    .indices are a subset of the next larger level's .indices."""
    indices_16 = _uniform_indices(frame_count, MAX_FRAME_COUNT)
    indices_8 = indices_16[::2]
    indices_4 = indices_8[::2]

    return {
        MAX_FRAME_COUNT: FrameSamplingPlan(MAX_FRAME_COUNT, frame_count, tuple(indices_16)),
        MID_FRAME_COUNT: FrameSamplingPlan(MID_FRAME_COUNT, frame_count, tuple(indices_8)),
        MIN_FRAME_COUNT: FrameSamplingPlan(MIN_FRAME_COUNT, frame_count, tuple(indices_4)),
    }


def compute_sampling_plan(frame_count: int, requested_count: int) -> FrameSamplingPlan:
    """Convenience accessor for one level (4, 8, or 16)."""
    if requested_count not in (MIN_FRAME_COUNT, MID_FRAME_COUNT, MAX_FRAME_COUNT):
        raise ValueError(
            f"requested_count must be one of {MIN_FRAME_COUNT}, {MID_FRAME_COUNT}, "
            f"{MAX_FRAME_COUNT}; got {requested_count}"
        )
    return compute_nested_sampling_plans(frame_count)[requested_count]
