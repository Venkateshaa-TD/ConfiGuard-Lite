"""Phase 2: nested uniform frame sampling."""

from __future__ import annotations

import pytest

from configuard.media.sampling import compute_nested_sampling_plans, compute_sampling_plan


@pytest.mark.parametrize("frame_count", [16, 30, 100, 301, 1000])
def test_nesting_holds_for_ample_frame_counts(frame_count):
    plans = compute_nested_sampling_plans(frame_count)
    idx4, idx8, idx16 = set(plans[4].indices), set(plans[8].indices), set(plans[16].indices)

    assert idx4.issubset(idx8)
    assert idx8.issubset(idx16)
    assert len(plans[16].indices) == 16
    assert len(plans[8].indices) == 8
    assert len(plans[4].indices) == 4


@pytest.mark.parametrize("frame_count", [1, 2, 3, 5, 7, 9, 15])
def test_nesting_holds_for_short_videos_with_dedup(frame_count):
    plans = compute_nested_sampling_plans(frame_count)
    idx4, idx8, idx16 = set(plans[4].indices), set(plans[8].indices), set(plans[16].indices)

    assert idx4.issubset(idx8)
    assert idx8.issubset(idx16)

    # No duplicates within any single level (avoid duplicate frames requirement).
    assert len(plans[16].indices) == len(idx16)
    assert len(plans[8].indices) == len(idx8)
    assert len(plans[4].indices) == len(idx4)

    # Never request more frames than actually exist.
    assert len(idx16) <= frame_count
    for idx in idx16:
        assert 0 <= idx < frame_count


def test_indices_are_ascending():
    plans = compute_nested_sampling_plans(100)
    for level in (4, 8, 16):
        indices = plans[level].indices
        assert list(indices) == sorted(indices)


def test_zero_frame_count_returns_empty():
    plans = compute_nested_sampling_plans(0)
    assert plans[16].indices == ()
    assert plans[8].indices == ()
    assert plans[4].indices == ()


def test_single_frame_video():
    plans = compute_nested_sampling_plans(1)
    assert plans[16].indices == (0,)
    assert plans[8].indices == (0,)
    assert plans[4].indices == (0,)


def test_compute_sampling_plan_single_level():
    plan = compute_sampling_plan(100, 8)
    assert plan.requested_count == 8
    assert len(plan.indices) == 8


def test_compute_sampling_plan_rejects_invalid_level():
    with pytest.raises(ValueError):
        compute_sampling_plan(100, 5)


def test_full_16_frame_span_covers_start_and_end():
    plans = compute_nested_sampling_plans(100)
    indices = plans[16].indices
    assert indices[0] == 0
    assert indices[-1] == 99
