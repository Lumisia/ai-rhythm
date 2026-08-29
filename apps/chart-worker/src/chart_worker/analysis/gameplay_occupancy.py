"""Pure HOLD-aware gameplay occupancy evidence.

This module measures player input occupancy only.  It deliberately knows
nothing about audio, BPM, difficulty labels, song identity, or retry policy.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise

from chart_worker.schema.note import Chart


@dataclass(frozen=True, slots=True)
class GameplayInterval:
    """One unique-row span and the actually unoccupied tail inside it."""

    row_span_start_ms: int
    unoccupied_start_ms: int
    end_ms: int
    hold_occupied_ms: int

    def __post_init__(self) -> None:
        if not (
            0 <= self.row_span_start_ms <= self.unoccupied_start_ms <= self.end_ms
        ):
            raise ValueError("gameplay interval boundaries must be ordered")
        if self.hold_occupied_ms != self.unoccupied_start_ms - self.row_span_start_ms:
            raise ValueError("hold_occupied_ms must match the occupied prefix")

    @property
    def row_span_duration_ms(self) -> int:
        return self.end_ms - self.row_span_start_ms

    @property
    def unoccupied_duration_ms(self) -> int:
        return self.end_ms - self.unoccupied_start_ms


def _validate_bounds(*, start_ms: int, end_ms: int) -> None:
    if type(start_ms) is not int or type(end_ms) is not int:
        raise TypeError("analysis boundaries must be exact integers")
    if start_ms < 0:
        raise ValueError("start_ms must be non-negative")
    if end_ms <= start_ms:
        raise ValueError("end_ms must be after start_ms")


def merged_hold_intervals(
    notes: Chart,
    *,
    start_ms: int,
    end_ms: int,
) -> tuple[tuple[int, int], ...]:
    """Return clamped unioned HOLD intervals using half-open semantics."""

    _validate_bounds(start_ms=start_ms, end_ms=end_ms)
    intervals: list[tuple[int, int]] = []
    for note in notes:
        if note.kind != "HOLD":
            continue
        note_end_ms = note.time_ms + (note.duration_ms or 0)
        left = max(start_ms, note.time_ms)
        right = min(end_ms, note_end_ms)
        if right > left:
            intervals.append((left, right))
    if not intervals:
        return ()
    intervals.sort()
    merged: list[tuple[int, int]] = []
    current_start, current_end = intervals[0]
    for left, right in intervals[1:]:
        if left <= current_end:
            current_end = max(current_end, right)
            continue
        merged.append((current_start, current_end))
        current_start, current_end = left, right
    merged.append((current_start, current_end))
    return tuple(merged)


def hold_occupancy_ms(
    notes: Chart,
    *,
    start_ms: int,
    end_ms: int,
) -> int:
    """Return unioned HOLD occupancy inside one analysis interval."""

    return sum(
        right - left
        for left, right in merged_hold_intervals(
            notes,
            start_ms=start_ms,
            end_ms=end_ms,
        )
    )


def gameplay_intervals(
    notes: Chart,
    *,
    start_ms: int,
    end_ms: int,
) -> tuple[GameplayInterval, ...]:
    """Return unique-row spans with a HOLD-aware unoccupied start.

    A span can contain an occupied prefix because no new row starts between
    adjacent boundaries.  A HOLD that began on an earlier row may cross any
    number of later rows, so the latest overlapping HOLD end is used.
    """

    _validate_bounds(start_ms=start_ms, end_ms=end_ms)
    rows = sorted(
        {
            note.time_ms
            for note in notes
            if start_ms <= note.time_ms <= end_ms
        }
    )
    boundaries = sorted({start_ms, end_ms, *rows})
    holds = merged_hold_intervals(notes, start_ms=start_ms, end_ms=end_ms)
    result: list[GameplayInterval] = []
    for left, right in pairwise(boundaries):
        occupied_until = left
        for hold_start, hold_end in holds:
            if hold_start > left:
                break
            if hold_start <= left < hold_end:
                occupied_until = max(occupied_until, hold_end)
        unoccupied_start = min(right, occupied_until)
        result.append(
            GameplayInterval(
                row_span_start_ms=left,
                unoccupied_start_ms=unoccupied_start,
                end_ms=right,
                hold_occupied_ms=unoccupied_start - left,
            )
        )
    return tuple(result)
