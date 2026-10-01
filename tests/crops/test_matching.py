"""Phase 5d: shared-range nested sampling, recovery offsets, slot resolution."""

from __future__ import annotations

import random

import pytest

from configuard.crops.matching import (
    MODE_FAILED,
    MODE_INDIVIDUAL,
    MODE_JOINT,
    MODE_PLANNED,
    SLOTS_4,
    SLOTS_8,
    SamplingError,
    nested_levels,
    planned_indices,
    recovery_offsets,
    resolve_slot,
    shared_frame_count,
)
from configuard.media.sampling import compute_nested_sampling_plans


@pytest.mark.parametrize("n", [16, 17, 287, 375, 1038])
def test_planned_indices_follow_existing_nested_contract(n):
    planned = planned_indices(n)
    plans = compute_nested_sampling_plans(n)
    assert planned == plans[16].indices
    assert tuple(planned[s] for s in SLOTS_8) == plans[8].indices
    assert tuple(planned[s] for s in SLOTS_4) == plans[4].indices
    assert set(plans[4].indices) <= set(plans[8].indices) <= set(planned)
    assert list(planned) == sorted(set(planned)) and len(planned) == 16
    assert planned[0] == 0 and planned[-1] == n - 1


def test_shared_range_is_minimum_and_too_short_is_refused():
    assert shared_frame_count([400, 300, 512]) == 300
    with pytest.raises(SamplingError):
        planned_indices(15)
    with pytest.raises(SamplingError):
        shared_frame_count([400, 0])


def test_nested_levels():
    assert nested_levels(0) == (4, 8, 16)
    assert nested_levels(2) == (8, 16)
    assert nested_levels(1) == (16,)


def test_recovery_offsets_order_and_bounds():
    planned = planned_indices(300)
    middle = recovery_offsets(planned, 5, 300, 6)
    assert middle[:4] == [1, -1, 2, -2] and len(middle) == 12
    assert all(d > 0 for d in recovery_offsets(planned, 0, 300, 6))  # cannot go before frame 0
    assert all(d < 0 for d in recovery_offsets(planned, 15, 300, 6))  # nor past the shared range


def test_recovery_offsets_respect_half_gap():
    planned = planned_indices(40)  # gaps of 2-3 frames
    for slot in range(16):
        for d in recovery_offsets(planned, slot, 40, 6):
            new = planned[slot] + d
            assert 0 <= new < 40
            if slot:
                assert new - planned[slot - 1] > (planned[slot] - planned[slot - 1]) / 2
            if slot < 15:
                assert planned[slot + 1] - new > (planned[slot + 1] - planned[slot]) / 2


def test_any_combination_of_recoveries_keeps_strict_order():
    rng = random.Random(0)
    for n in (40, 100, 287):
        planned = planned_indices(n)
        for _ in range(200):
            final = [p + rng.choice([0] + recovery_offsets(planned, s, n, 6)) for s, p in enumerate(planned)]
            assert final == sorted(set(final))


def _valid(*maps):
    return [dict(m) for m in maps]


def test_resolve_planned_and_joint_recovery():
    full = {i: True for i in range(90, 111)}
    res = resolve_slot(3, 100, [1, -1, 2, -2], _valid(full, full))
    assert res.mode == MODE_PLANNED and res.member_indices == (100, 100) and res.exact_match

    fake_missing = {**full, 100: False, 101: False}
    res = resolve_slot(3, 100, [1, -1, 2, -2], _valid(full, fake_missing))
    assert res.mode == MODE_JOINT and res.member_indices == (99, 99) and res.exact_match


def test_resolve_individual_recovery_then_failure():
    real = {100: False, 101: True, 99: True, 102: False}
    fake = {100: False, 101: False, 99: False, 102: True}
    res = resolve_slot(0, 100, [1, -1, 2], _valid(real, fake))
    assert res.mode == MODE_INDIVIDUAL and res.member_indices == (101, 102) and not res.exact_match

    res = resolve_slot(0, 100, [1, -1, 2], _valid(real, {}))
    assert res.mode == MODE_FAILED and res.member_indices == (101, None)
