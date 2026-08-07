from typing import get_args

import pytest

from chart_worker.stages.timing_feedback import (
    MapTimingFailureSignature,
    RetryTimingSignal,
    TimingFailureFamily,
    record_timing_failure,
)


def test_timing_failure_families_include_observed_resnap_collision():
    assert "RESNAP_COLLISION" in get_args(TimingFailureFamily)


def _signature(
    seed: int,
    *,
    segment: int = 2,
    family: str = "DUPLICATE_NOTE",
    grid_aligned: bool = True,
) -> MapTimingFailureSignature:
    return MapTimingFailureSignature(
        authority_sha256="authority",
        key_mode=4,
        difficulty="EASY",
        seed=seed,
        timing_segment_id=segment,
        failure_family=family,
        time_ms=10_000 + seed,
        grid_aligned=grid_aligned,
    )


def test_one_timing_failure_remains_a_map_retry():
    signatures = []

    record_timing_failure(signatures, _signature(0))

    assert [signature.seed for signature in signatures] == [0]


def test_two_distinct_seeds_in_same_segment_and_family_escalate():
    signatures = []
    record_timing_failure(signatures, _signature(0))

    with pytest.raises(RetryTimingSignal) as captured:
        record_timing_failure(signatures, _signature(12))

    assert [signature.seed for signature in captured.value.signatures] == [0, 12]
    assert captured.value.to_context()["failureFamily"] == "DUPLICATE_NOTE"


@pytest.mark.parametrize(
    "second",
    [
        _signature(0),
        _signature(12, segment=3),
        _signature(12, family="ACTIVE_MIDDLE_GAP"),
        _signature(12, grid_aligned=False),
    ],
)
def test_non_corroborating_failures_do_not_escalate(second):
    signatures = []
    record_timing_failure(signatures, _signature(0))

    record_timing_failure(signatures, second)

    assert len(signatures) == 2


def test_two_active_middle_gaps_follow_the_same_distinct_seed_rule():
    signatures = []
    record_timing_failure(
        signatures,
        _signature(0, family="ACTIVE_MIDDLE_GAP"),
    )

    with pytest.raises(RetryTimingSignal):
        record_timing_failure(
            signatures,
            _signature(12, family="ACTIVE_MIDDLE_GAP"),
        )


def test_two_resnap_collisions_follow_the_same_distinct_seed_rule():
    signatures = []
    record_timing_failure(
        signatures,
        _signature(0, family="RESNAP_COLLISION"),
    )

    with pytest.raises(RetryTimingSignal):
        record_timing_failure(
            signatures,
            _signature(12, family="RESNAP_COLLISION"),
        )
