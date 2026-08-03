"""Structural validation that rejects, but never rewrites, generated charts."""

from collections import defaultdict

from chart_worker.generation.mapperatorinator import GeneratedChart
from chart_worker.schema.note import NoteEvent


class GeneratedChartValidationError(ValueError):
    """Mapperatorinator returned a chart that cannot be played as requested."""


def _end_ms(note: NoteEvent) -> int:
    return note.time_ms + (note.duration_ms or 0)


def validate_generated_chart(
    chart: GeneratedChart,
    *,
    key_mode: int,
    duration_ms: int,
) -> None:
    """Validate key, timing, bounds, duplicates and same-lane hold overlap."""
    if chart.key_mode != key_mode:
        raise GeneratedChartValidationError(
            f"requested {key_mode}K but generated {chart.key_mode}K"
        )
    if not chart.bpm_events:
        raise GeneratedChartValidationError("generated chart has no timing events")

    by_lane: dict[int, list[NoteEvent]] = defaultdict(list)
    for note in chart.notes:
        if not 0 <= note.lane < key_mode:
            raise GeneratedChartValidationError(
                f"lane {note.lane} is outside requested {key_mode}K"
            )
        if note.time_ms > duration_ms or _end_ms(note) > duration_ms:
            raise GeneratedChartValidationError(
                f"note at {note.time_ms}ms exceeds canonical audio duration {duration_ms}ms"
            )
        by_lane[note.lane].append(note)

    for lane, lane_notes in by_lane.items():
        active_hold_end = -1
        previous_time = -1
        for note in sorted(lane_notes, key=lambda item: item.time_ms):
            if note.time_ms == previous_time:
                raise GeneratedChartValidationError(
                    f"duplicate note in lane {lane} at {note.time_ms}ms"
                )
            if note.time_ms < active_hold_end:
                raise GeneratedChartValidationError(
                    f"note overlap in lane {lane} at {note.time_ms}ms "
                    f"before hold end {active_hold_end}ms"
                )
            previous_time = note.time_ms
            if note.kind == "HOLD":
                active_hold_end = _end_ms(note)
