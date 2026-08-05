"""Validation for the immutable per-song timing authority."""

import math

from chart_worker.generation.osu_parser import OsuBpmEvent


class TimingAuthorityValidationError(ValueError):
    """A generated timing candidate cannot become the song authority."""


def validate_timing_events(events: tuple[OsuBpmEvent, ...], duration_ms: int) -> None:
    """Validate ordered, finite timing events that fit within the audio."""
    if not events:
        raise TimingAuthorityValidationError("timing authority has no events")
    first_event = events[0]
    if (
        math.isfinite(first_event.bpm)
        and first_event.bpm > 0
        and first_event.time_ms > 60_000.0 / first_event.bpm
    ):
        raise TimingAuthorityValidationError(
            "timing authority first event must be within one beat of audio start"
        )

    previous_time: int | None = None
    for event in events:
        if previous_time is not None and event.time_ms <= previous_time:
            raise TimingAuthorityValidationError(
                "timing authority events must be sorted without duplicate times"
            )
        if not math.isfinite(event.bpm) or event.bpm <= 0:
            raise TimingAuthorityValidationError(
                "timing authority BPM values must be positive and finite"
            )
        if event.time_ms >= duration_ms:
            raise TimingAuthorityValidationError(
                f"timing authority event at {event.time_ms}ms exceeds audio duration {duration_ms}ms"
            )
        previous_time = event.time_ms


def validate_timing_identity(
    actual: tuple[OsuBpmEvent, ...], expected: tuple[OsuBpmEvent, ...]
) -> None:
    """Require reparsed timing to retain every authority event."""
    if len(actual) != len(expected):
        raise TimingAuthorityValidationError("timing authority identity event count differs")

    for index, (actual_event, expected_event) in enumerate(zip(actual, expected)):
        if actual_event.time_ms != expected_event.time_ms or not math.isclose(
            actual_event.bpm,
            expected_event.bpm,
            rel_tol=1e-9,
            abs_tol=1e-6,
        ):
            raise TimingAuthorityValidationError(
                f"timing authority identity differs at event {index}"
            )
