"""Structural validation that rejects, but never rewrites, generated charts."""

import math
from collections import defaultdict

from chart_worker.generation.mapperatorinator import GeneratedChart
from chart_worker.schema.note import NoteEvent


class GeneratedChartValidationError(ValueError):
    """Mapperatorinator returned a chart that cannot be played as requested."""

    def __init__(
        self,
        message: str,
        *,
        reason_code: str,
        context: dict[str, object] | None = None,
    ) -> None:
        super().__init__(message)
        self.reason_code = reason_code
        self.context = context or {}

    def to_report(self) -> dict[str, object]:
        """Serialize stable retry evidence without parsing the message."""
        return {
            "reasonCode": self.reason_code,
            "message": str(self),
            "context": self.context,
        }


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
            f"requested {key_mode}K but generated {chart.key_mode}K",
            reason_code="KEY_MODE_MISMATCH",
            context={"requestedKeyMode": key_mode, "generatedKeyMode": chart.key_mode},
        )
    if not chart.bpm_events:
        raise GeneratedChartValidationError(
            "generated chart has no timing events",
            reason_code="MISSING_TIMING_EVENTS",
        )
    timing_times = [event.time_ms for event in chart.bpm_events]
    if timing_times != sorted(set(timing_times)):
        raise GeneratedChartValidationError(
            "generated chart timing events must be sorted without duplicates",
            reason_code="INVALID_TIMING_ORDER",
            context={"timingEventTimesMs": timing_times},
        )
    if any(not math.isfinite(event.bpm) or event.bpm <= 0 for event in chart.bpm_events):
        raise GeneratedChartValidationError(
            "generated chart bpm values must be positive and finite",
            reason_code="INVALID_BPM",
        )
    if not chart.notes:
        raise GeneratedChartValidationError(
            "generated chart has no notes",
            reason_code="EMPTY_CHART",
        )

    by_lane: dict[int, list[NoteEvent]] = defaultdict(list)
    for note in chart.notes:
        if not 0 <= note.lane < key_mode:
            raise GeneratedChartValidationError(
                f"lane {note.lane} is outside requested {key_mode}K",
                reason_code="NOTE_LANE_OUT_OF_RANGE",
                context={
                    "lane": note.lane,
                    "timeMs": note.time_ms,
                    "noteKind": note.kind,
                    "requestedKeyMode": key_mode,
                },
            )
        if note.time_ms >= duration_ms or _end_ms(note) > duration_ms:
            raise GeneratedChartValidationError(
                f"note at {note.time_ms}ms exceeds canonical audio duration {duration_ms}ms",
                reason_code="NOTE_OUT_OF_RANGE",
                context={
                    "lane": note.lane,
                    "timeMs": note.time_ms,
                    "noteKind": note.kind,
                    "noteEndMs": _end_ms(note),
                    "durationMs": duration_ms,
                },
            )
        by_lane[note.lane].append(note)

    for lane, lane_notes in by_lane.items():
        active_hold_end = -1
        previous_time = -1
        for note in sorted(lane_notes, key=lambda item: item.time_ms):
            if note.time_ms == previous_time:
                raise GeneratedChartValidationError(
                    f"duplicate note in lane {lane} at {note.time_ms}ms",
                    reason_code="DUPLICATE_NOTE",
                    context={
                        "lane": lane,
                        "timeMs": note.time_ms,
                        "noteKind": note.kind,
                    },
                )
            if note.time_ms < active_hold_end:
                raise GeneratedChartValidationError(
                    f"note overlap in lane {lane} at {note.time_ms}ms "
                    f"before hold end {active_hold_end}ms",
                    reason_code="HOLD_OVERLAP",
                    context={
                        "lane": lane,
                        "timeMs": note.time_ms,
                        "noteKind": note.kind,
                        "holdEndMs": active_hold_end,
                    },
                )
            previous_time = note.time_ms
            if note.kind == "HOLD":
                active_hold_end = _end_ms(note)
