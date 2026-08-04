import pytest

from chart_worker.generation.osu_parser import OsuBpmEvent
from chart_worker.validation.timing_authority import (
    TimingAuthorityValidationError,
    validate_timing_identity,
)


def test_timing_identity_rejects_a_different_event():
    expected = (OsuBpmEvent(0, 120.0),)

    with pytest.raises(TimingAuthorityValidationError, match="identity"):
        validate_timing_identity((OsuBpmEvent(0, 121.0),), expected)
