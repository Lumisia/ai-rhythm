import pytest

from chart_worker.generation.osu_parser import OsuBpmEvent
from chart_worker.validation.timing_authority import (
    TimingAuthorityValidationError,
    validate_timing_events,
    validate_timing_identity,
)


def test_timing_identity_rejects_a_different_event():
    expected = (OsuBpmEvent(0, 120.0),)

    with pytest.raises(TimingAuthorityValidationError, match="identity"):
        validate_timing_identity((OsuBpmEvent(0, 121.0),), expected)


def test_timing_authority_allows_a_late_structurally_valid_first_event():
    validate_timing_events((OsuBpmEvent(2_678, 222.0),), duration_ms=150_000)


@pytest.mark.parametrize("first_event_ms", [-1_000, 0, 250, 500])
def test_timing_authority_allows_a_first_event_within_one_beat(
    first_event_ms: int,
):
    validate_timing_events(
        (OsuBpmEvent(first_event_ms, 120.0), OsuBpmEvent(1_000, 150.0)),
        duration_ms=10_000,
    )
