"""Deterministic last-resort charts on the canonical song timeline."""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass, replace
from math import ceil
from typing import Literal

from chart_worker.analysis.onset import OnsetAnalysis
from chart_worker.generation.mapperatorinator import GeneratedChart
from chart_worker.generation.params import GenerationRequest
from chart_worker.schema.note import Chart, NoteEvent
from chart_worker.stages.types import SongTimingAuthority

ONSET_SUPPORT_MS = 70
SUSTAINED_ACTIVITY_RATIO = 0.70


@dataclass(frozen=True, slots=True)
class RecoveryBudget:
    subdivisions: int
    min_row_gap_ms: int
    chord_every: int | None
    hold_every: int | None


@dataclass(frozen=True, slots=True)
class RecoveryRowPlan:
    rows: tuple[int, ...]
    subdivisions: int
    selection_reason: Literal[
        "DEFAULT",
        "PREFLIGHT_ALTERNATE",
        "NO_VIABLE_ALTERNATE",
    ] = "DEFAULT"
    viable_divisors: tuple[int, ...] = ()

    def to_report(self) -> dict[str, object]:
        return {
            "subdivisions": self.subdivisions,
            "selectionReason": self.selection_reason,
            "viableDivisors": list(self.viable_divisors),
            "rowCount": len(self.rows),
        }


RECOVERY_BUDGETS: dict[str, RecoveryBudget] = {
    "EASY": RecoveryBudget(1, 320, None, None),
    "NORMAL": RecoveryBudget(2, 220, None, 32),
    "HARD": RecoveryBudget(3, 160, 12, 24),
    "EXPERT": RecoveryBudget(4, 110, 8, 16),
}


def _active_onsets(onsets: OnsetAnalysis) -> tuple[int, ...]:
    if onsets.activity is not None:
        return tuple(sorted(set(onsets.activity.active_onset_ms)))
    return tuple(sorted(set(onsets.onset_ms)))


def _has_nearby_onset(time_ms: int, active_onsets: tuple[int, ...]) -> bool:
    index = bisect_left(active_onsets, time_ms)
    return any(
        abs(active_onsets[candidate] - time_ms) <= ONSET_SUPPORT_MS
        for candidate in (index - 1, index)
        if 0 <= candidate < len(active_onsets)
    )


def _tempo_rows(
    request: GenerationRequest,
    authority: SongTimingAuthority,
    budget: RecoveryBudget,
    active_onsets: tuple[int, ...],
) -> tuple[int, ...]:
    rows: list[int] = []
    events = authority.bpm_events
    if not events:
        raise ValueError("recovery requires timing authority BPM events")
    note_start_end_ms = min(
        request.duration_ms,
        (
            request.max_note_start_ms
            if request.max_note_start_ms is not None
            else request.duration_ms
        ),
    )

    for index, event in enumerate(events):
        segment_start = event.time_ms
        segment_end = (
            events[index + 1].time_ms
            if index + 1 < len(events)
            else note_start_end_ms
        )
        if segment_end <= 0 or segment_start >= note_start_end_ms:
            continue
        beat_ms = 60_000.0 / event.bpm
        first_beat = event.time_ms
        if first_beat < 0:
            first_beat += ceil(-first_beat / beat_ms) * beat_ms

        beat_time = first_beat
        while beat_time < min(segment_end, note_start_end_ms):
            for subdivision in range(budget.subdivisions):
                time_ms = round(
                    beat_time + subdivision * beat_ms / budget.subdivisions
                )
                if time_ms < max(0, segment_start) or time_ms >= min(
                    segment_end, note_start_end_ms
                ):
                    continue
                if subdivision and not _has_nearby_onset(time_ms, active_onsets):
                    continue
                if rows and time_ms - rows[-1] < budget.min_row_gap_ms:
                    continue
                if not rows or time_ms != rows[-1]:
                    rows.append(time_ms)
            beat_time += beat_ms
    return tuple(rows)


def plan_recovery_rows(
    request: GenerationRequest,
    authority: SongTimingAuthority,
    onsets: OnsetAnalysis,
    *,
    subdivisions: int | None = None,
) -> RecoveryRowPlan:
    """Plan deterministic recovery rows without building note patterns."""
    budget = RECOVERY_BUDGETS[request.difficulty]
    if subdivisions is not None:
        if subdivisions <= 0:
            raise ValueError("recovery subdivisions must be positive")
        budget = replace(budget, subdivisions=subdivisions)
    rows = _tempo_rows(request, authority, budget, _active_onsets(onsets))
    return RecoveryRowPlan(rows=rows, subdivisions=budget.subdivisions)


def select_recovery_plan(
    request: GenerationRequest,
    authority: SongTimingAuthority,
    onsets: OnsetAnalysis,
) -> RecoveryRowPlan:
    """Select one bounded recovery grid from timing-stage preflight evidence."""
    default_plan = plan_recovery_rows(request, authority, onsets)
    preflight = authority.recovery_preflight
    if preflight is None:
        return default_plan
    difficulty = preflight.for_difficulty(request.difficulty)
    viable_divisors = difficulty.viable_divisors
    if difficulty.action.value == "REVIEW" and viable_divisors:
        selected_divisor = min(
            viable_divisors,
            key=lambda divisor: (
                abs(divisor - default_plan.subdivisions),
                divisor,
            ),
        )
        return replace(
            plan_recovery_rows(
                request,
                authority,
                onsets,
                subdivisions=selected_divisor,
            ),
            selection_reason="PREFLIGHT_ALTERNATE",
            viable_divisors=viable_divisors,
        )
    if difficulty.action.value == "DAMAGED":
        return replace(
            default_plan,
            selection_reason="NO_VIABLE_ALTERNATE",
            viable_divisors=viable_divisors,
        )
    return replace(default_plan, viable_divisors=viable_divisors)


def _is_sustained(
    onsets: OnsetAnalysis,
    active_onsets: tuple[int, ...],
    start_ms: int,
    end_ms: int,
) -> bool:
    if onsets.activity is None or end_ms <= start_ms:
        return False
    if onsets.activity.active_frame_ratio(start_ms, end_ms) < SUSTAINED_ACTIVITY_RATIO:
        return False
    left = bisect_left(active_onsets, start_ms + 1)
    return left >= len(active_onsets) or active_onsets[left] >= end_ms


def _lane_for(row_index: int, key_mode: int, difficulty: str) -> int:
    stride = {"EASY": 1, "NORMAL": 1, "HARD": 2, "EXPERT": 3}[difficulty]
    if key_mode % stride == 0:
        stride = 1
    return (row_index * stride + row_index // key_mode) % key_mode


def _build_notes(
    request: GenerationRequest,
    rows: tuple[int, ...],
    onsets: OnsetAnalysis,
    active_onsets: tuple[int, ...],
    budget: RecoveryBudget,
) -> Chart:
    notes: list[NoteEvent] = []
    for row_index, time_ms in enumerate(rows):
        lane = _lane_for(row_index, request.key_mode, request.difficulty)
        next_time = rows[row_index + 1] if row_index + 1 < len(rows) else None
        make_hold = (
            budget.hold_every is not None
            and row_index > 0
            and row_index % budget.hold_every == 0
            and next_time is not None
            and _is_sustained(onsets, active_onsets, time_ms, next_time)
        )
        if make_hold:
            notes.append(
                NoteEvent(
                    time_ms=time_ms,
                    lane=lane,
                    kind="HOLD",
                    duration_ms=next_time - time_ms,
                )
            )
        else:
            notes.append(NoteEvent(time_ms=time_ms, lane=lane))

        if (
            budget.chord_every is not None
            and row_index > 0
            and row_index % budget.chord_every == 0
        ):
            opposite = (lane + request.key_mode // 2) % request.key_mode
            if opposite != lane:
                notes.append(NoteEvent(time_ms=time_ms, lane=opposite))
    return sorted(notes, key=lambda note: (note.time_ms, note.lane))


def build_recovery_chart(
    request: GenerationRequest,
    authority: SongTimingAuthority,
    onsets: OnsetAnalysis,
    *,
    plan: RecoveryRowPlan | None = None,
) -> GeneratedChart:
    """Build a bounded fallback only after model candidates are exhausted."""
    budget = RECOVERY_BUDGETS[request.difficulty]
    active_onsets = _active_onsets(onsets)
    row_plan = plan or plan_recovery_rows(request, authority, onsets)
    rows = row_plan.rows
    if not rows:
        raise ValueError("recovery timing grid produced no playable rows")
    notes = _build_notes(request, rows, onsets, active_onsets, budget)
    return GeneratedChart(
        notes=notes,
        key_mode=request.key_mode,
        osu_text="",
        generator_name="adaptive-recovery-v1",
        seed=request.seed,
        bpm_events=authority.bpm_events,
    )
