"""Plan a narrow Mapperatorinator remap window without cutting existing holds."""

from dataclasses import dataclass

from chart_worker.analysis.timing_diagnostics import TimingCoverageGap
from chart_worker.generation.osu_parser import OsuBpmEvent
from chart_worker.schema.note import Chart

BEATS_PER_REPAIR_MARGIN = 4
MAX_REPAIR_FRACTION = 0.80


@dataclass(frozen=True, slots=True)
class PartialRemapWindow:
    start_ms: int
    end_ms: int


def _bpm_at(time_ms: int, bpm_events: tuple[OsuBpmEvent, ...]) -> float:
    if not bpm_events:
        raise ValueError("partial remap requires BPM events")
    event = bpm_events[0]
    for candidate in bpm_events:
        if candidate.time_ms > time_ms:
            break
        event = candidate
    return event.bpm


def _margin_ms(time_ms: int, bpm_events: tuple[OsuBpmEvent, ...]) -> int:
    return round(BEATS_PER_REPAIR_MARGIN * 60_000.0 / _bpm_at(time_ms, bpm_events))


def _expand_across_holds(
    notes: Chart,
    *,
    start_ms: int,
    end_ms: int,
    duration_ms: int,
) -> tuple[int, int]:
    while True:
        expanded_start = start_ms
        expanded_end = end_ms
        for note in notes:
            if note.kind != "HOLD":
                continue
            hold_end_ms = note.time_ms + (note.duration_ms or 0)
            if note.time_ms < start_ms < hold_end_ms:
                expanded_start = min(expanded_start, note.time_ms)
            if note.time_ms <= end_ms < hold_end_ms:
                expanded_end = max(expanded_end, hold_end_ms)
        expanded_start = max(0, expanded_start)
        expanded_end = min(duration_ms, expanded_end)
        if (expanded_start, expanded_end) == (start_ms, end_ms):
            return start_ms, end_ms
        start_ms, end_ms = expanded_start, expanded_end


def build_partial_remap_window(
    notes: Chart,
    coverage_gaps: tuple[TimingCoverageGap, ...],
    bpm_events: tuple[OsuBpmEvent, ...],
    *,
    duration_ms: int,
) -> PartialRemapWindow | None:
    """Cover all active gaps plus local one-bar context, or decline a near-full remap."""
    if not coverage_gaps:
        return None
    if duration_ms <= 0:
        raise ValueError("duration_ms must be positive")

    gap_start_ms = min(gap.start_ms for gap in coverage_gaps)
    gap_end_ms = max(gap.end_ms for gap in coverage_gaps)
    start_ms = max(0, gap_start_ms - _margin_ms(gap_start_ms, bpm_events))
    end_ms = min(duration_ms, gap_end_ms + _margin_ms(gap_end_ms, bpm_events))
    start_ms, end_ms = _expand_across_holds(
        notes,
        start_ms=start_ms,
        end_ms=end_ms,
        duration_ms=duration_ms,
    )
    if (end_ms - start_ms) / duration_ms > MAX_REPAIR_FRACTION:
        return None
    return PartialRemapWindow(start_ms=start_ms, end_ms=end_ms)
