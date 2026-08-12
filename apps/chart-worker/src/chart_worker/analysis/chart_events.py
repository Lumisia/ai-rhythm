"""Immutable row, lane, HOLD, and section indices for one chart."""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
from math import ceil

from chart_worker.schema.note import Chart, NoteEvent


@dataclass(frozen=True, slots=True)
class ChartRow:
    time_ms: int
    lanes: tuple[int, ...]
    note_indices: tuple[int, ...]

    @property
    def size(self) -> int:
        return len(self.lanes)


@dataclass(frozen=True, slots=True)
class HoldInterval:
    lane: int
    start_ms: int
    end_ms: int


@dataclass(frozen=True, slots=True)
class ChartEventIndex:
    notes: tuple[NoteEvent, ...]
    key_mode: int
    duration_ms: int
    section_ms: int
    rows: tuple[ChartRow, ...]
    row_times: tuple[int, ...]
    lane_times: tuple[tuple[int, ...], ...]
    holds: tuple[HoldInterval, ...]
    section_rows: tuple[tuple[int, ...], ...]

    @classmethod
    def build(
        cls,
        notes: Chart,
        key_mode: int,
        duration_ms: int,
        *,
        section_ms: int = 400,
    ) -> ChartEventIndex:
        if key_mode <= 0:
            raise ValueError("key_mode must be positive")
        if duration_ms <= 0:
            raise ValueError("duration_ms must be positive")
        if section_ms <= 0:
            raise ValueError("section_ms must be positive")
        ordered = tuple(
            sorted(
                notes,
                key=lambda note: (
                    note.time_ms,
                    note.lane,
                    note.kind,
                    note.duration_ms or 0,
                ),
            )
        )
        seen: set[tuple[int, int]] = set()
        holds: list[HoldInterval] = []
        grouped_indices: dict[int, list[int]] = {}
        lane_times: list[list[int]] = [[] for _ in range(key_mode)]
        for index, note in enumerate(ordered):
            if note.lane >= key_mode:
                raise ValueError(f"note lane {note.lane} is outside {key_mode}K")
            if note.time_ms >= duration_ms:
                raise ValueError("note time exceeds chart duration")
            key = (note.time_ms, note.lane)
            if key in seen:
                raise ValueError("duplicate note at the same time and lane")
            seen.add(key)
            if note.kind == "HOLD":
                end_ms = note.time_ms + (note.duration_ms or 0)
                if end_ms > duration_ms:
                    raise ValueError("HOLD end exceeds chart duration")
                holds.append(HoldInterval(note.lane, note.time_ms, end_ms))
            grouped_indices.setdefault(note.time_ms, []).append(index)
            lane_times[note.lane].append(note.time_ms)

        rows = tuple(
            ChartRow(
                time_ms=time_ms,
                lanes=tuple(ordered[index].lane for index in indices),
                note_indices=tuple(indices),
            )
            for time_ms, indices in grouped_indices.items()
        )
        row_times = tuple(row.time_ms for row in rows)
        section_count = ceil(duration_ms / section_ms)
        section_rows: list[list[int]] = [[] for _ in range(section_count)]
        for row_index, row in enumerate(rows):
            section_rows[row.time_ms // section_ms].append(row_index)
        return cls(
            notes=ordered,
            key_mode=key_mode,
            duration_ms=duration_ms,
            section_ms=section_ms,
            rows=rows,
            row_times=row_times,
            lane_times=tuple(tuple(times) for times in lane_times),
            holds=tuple(holds),
            section_rows=tuple(tuple(indices) for indices in section_rows),
        )

    def has_other_row_between(
        self,
        left_ms: int,
        right_ms: int,
        *,
        excluded_lanes: set[int] | frozenset[int] = frozenset(),
    ) -> bool:
        """Return whether a non-excluded lane occurs in the strict interval."""
        if right_ms <= left_ms:
            return False
        if not excluded_lanes:
            position = bisect_right(self.row_times, left_ms)
            return (
                position < len(self.row_times)
                and self.row_times[position] < right_ms
            )
        for lane, times in enumerate(self.lane_times):
            if lane in excluded_lanes:
                continue
            position = bisect_right(times, left_ms)
            if position < len(times) and times[position] < right_ms:
                return True
        return False
