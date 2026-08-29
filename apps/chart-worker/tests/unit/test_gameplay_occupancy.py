from __future__ import annotations

import pytest

from chart_worker.analysis.gameplay_occupancy import (
    gameplay_intervals,
    hold_occupancy_ms,
)
from chart_worker.schema.note import NoteEvent


def tap(time_ms: int, lane: int = 0) -> NoteEvent:
    return NoteEvent(time_ms, lane)


def hold(time_ms: int, duration_ms: int, lane: int = 0) -> NoteEvent:
    return NoteEvent(time_ms, lane, kind="HOLD", duration_ms=duration_ms)


def test_tap_to_tap_leaves_the_start_span_unoccupied() -> None:
    intervals = gameplay_intervals(
        [tap(1_000), tap(4_000)],
        start_ms=0,
        end_ms=5_000,
    )

    middle = next(interval for interval in intervals if interval.row_span_start_ms == 1_000)
    assert middle.unoccupied_start_ms == 1_000
    assert middle.end_ms == 4_000
    assert middle.row_span_duration_ms == 3_000
    assert middle.unoccupied_duration_ms == 3_000
    assert middle.hold_occupied_ms == 0


def test_hold_to_next_row_starts_vacancy_at_hold_end() -> None:
    intervals = gameplay_intervals(
        [hold(1_000, 2_000), tap(5_000)],
        start_ms=0,
        end_ms=6_000,
    )

    middle = next(interval for interval in intervals if interval.row_span_start_ms == 1_000)
    assert middle.unoccupied_start_ms == 3_000
    assert middle.end_ms == 5_000
    assert middle.hold_occupied_ms == 2_000
    assert middle.unoccupied_duration_ms == 2_000


def test_overlapping_holds_use_the_latest_end_without_double_counting() -> None:
    notes = [hold(1_000, 4_000, 0), hold(2_000, 5_000, 1), tap(8_000)]

    assert hold_occupancy_ms(notes, start_ms=1_000, end_ms=8_000) == 6_000
    intervals = gameplay_intervals(notes, start_ms=0, end_ms=8_000)
    crossing = next(interval for interval in intervals if interval.row_span_start_ms == 2_000)
    assert crossing.unoccupied_start_ms == 7_000
    assert crossing.end_ms == 8_000


def test_hold_crossing_intermediate_tap_keeps_occupancy_continuous() -> None:
    intervals = gameplay_intervals(
        [hold(1_000, 7_000), tap(3_000), tap(9_000)],
        start_ms=0,
        end_ms=10_000,
    )

    after_tap = next(interval for interval in intervals if interval.row_span_start_ms == 3_000)
    assert after_tap.unoccupied_start_ms == 8_000
    assert after_tap.end_ms == 9_000
    assert after_tap.hold_occupied_ms == 5_000


def test_hold_end_is_clamped_to_analysis_boundary() -> None:
    intervals = gameplay_intervals(
        [hold(1_000, 20_000)],
        start_ms=0,
        end_ms=5_000,
    )

    trailing = next(interval for interval in intervals if interval.row_span_start_ms == 1_000)
    assert trailing.unoccupied_start_ms == 5_000
    assert trailing.unoccupied_duration_ms == 0
    assert trailing.hold_occupied_ms == 4_000


@pytest.mark.parametrize(
    ("start_ms", "end_ms"),
    [(-1, 5_000), (5_000, 5_000), (6_000, 5_000)],
)
def test_invalid_analysis_boundaries_fail_explicitly(start_ms: int, end_ms: int) -> None:
    with pytest.raises(ValueError):
        gameplay_intervals([tap(1_000)], start_ms=start_ms, end_ms=end_ms)
