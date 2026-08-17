"""Deterministic TAP-only repair for proven active coverage gaps."""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass

from chart_worker.analysis.coverage_opportunity import CoverageKind
from chart_worker.analysis.onset import OnsetAnalysis
from chart_worker.generation.mapperatorinator import GeneratedChart
from chart_worker.generation.params import GenerationRequest
from chart_worker.generation.recovery import (
    ONSET_SUPPORT_MS,
    RECOVERY_BUDGETS,
    select_recovery_plan,
)
from chart_worker.schema.note import NoteEvent
from chart_worker.stages.types import SongTimingAuthority
from chart_worker.validation.quality_gate import ChartAcceptance, GateAction, GateAxis


@dataclass(frozen=True, slots=True)
class CoverageRepairPlan:
    repaired_intervals: tuple[tuple[int, int], ...]
    inserted_notes: tuple[NoteEvent, ...]

    @property
    def repaired_gap_count(self) -> int:
        return len(self.repaired_intervals)

    def to_report(self) -> dict[str, object]:
        return {
            "repairedGapCount": self.repaired_gap_count,
            "insertedNoteCount": len(self.inserted_notes),
            "repairedIntervals": [
                {"startMs": start_ms, "endMs": end_ms}
                for start_ms, end_ms in self.repaired_intervals
            ],
        }


def _active_onsets(onsets: OnsetAnalysis) -> tuple[int, ...]:
    if onsets.activity is not None:
        return tuple(sorted(set(onsets.activity.active_onset_ms)))
    return tuple(sorted(set(onsets.onset_ms)))


def _nearest_grid_row(
    time_ms: int,
    rows: tuple[int, ...],
    *,
    start_ms: int,
    end_ms: int,
) -> int | None:
    index = bisect_left(rows, time_ms)
    candidates = (
        rows[candidate]
        for candidate in (index - 1, index)
        if 0 <= candidate < len(rows)
        and start_ms < rows[candidate] < end_ms
        and abs(rows[candidate] - time_ms) <= ONSET_SUPPORT_MS
    )
    return min(candidates, key=lambda row: (abs(row - time_ms), row), default=None)


def _far_enough(time_ms: int, rows: list[int], minimum_gap_ms: int) -> bool:
    index = bisect_left(rows, time_ms)
    return all(
        abs(rows[candidate] - time_ms) >= minimum_gap_ms
        for candidate in (index - 1, index)
        if 0 <= candidate < len(rows)
    )


def _lane_for(row_index: int, key_mode: int, difficulty: str) -> int:
    stride = {"EASY": 1, "NORMAL": 1, "HARD": 2, "EXPERT": 3}[difficulty]
    if key_mode % stride == 0:
        stride = 1
    return (row_index * stride + row_index // key_mode) % key_mode


def _available_lane(
    row_index: int,
    key_mode: int,
    difficulty: str,
    *,
    time_ms: int,
    holds: tuple[NoteEvent, ...],
) -> int | None:
    preferred = _lane_for(row_index, key_mode, difficulty)
    blocked = {
        note.lane
        for note in holds
        if note.time_ms <= time_ms < note.time_ms + (note.duration_ms or 0)
    }
    return next(
        (
            (preferred + offset) % key_mode
            for offset in range(key_mode)
            if (preferred + offset) % key_mode not in blocked
        ),
        None,
    )


def build_coverage_repair_chart(
    request: GenerationRequest,
    source: GeneratedChart,
    acceptance: ChartAcceptance,
    authority: SongTimingAuthority,
    onsets: OnsetAnalysis,
) -> tuple[GeneratedChart, CoverageRepairPlan]:
    """Add deterministic TAP rows only inside proven attack-required gaps."""

    if source.key_mode != request.key_mode:
        raise ValueError("coverage repair source key mode differs from request")
    coverage = acceptance.decision(GateAxis.COVERAGE)
    attack_gaps = tuple(
        gap
        for gap in acceptance.timing.coverage_gaps
        if gap.opportunity is None
        or gap.opportunity.kind is CoverageKind.ATTACK_REQUIRED
    )
    if coverage.action is not GateAction.RETRY_MAP or not attack_gaps:
        raise ValueError("coverage repair requires an attack-required coverage gap")

    active_onsets = _active_onsets(onsets)
    canonical_rows = select_recovery_plan(request, authority, onsets).rows
    minimum_gap_ms = RECOVERY_BUDGETS[request.difficulty].min_row_gap_ms
    occupied_rows = sorted({note.time_ms for note in source.notes})
    holds = tuple(note for note in source.notes if note.kind == "HOLD")
    inserted: list[NoteEvent] = []
    repaired_intervals: list[tuple[int, int]] = []

    for gap in attack_gaps:
        gap_inserted = False
        left = bisect_left(active_onsets, gap.start_ms + 1)
        right = bisect_left(active_onsets, gap.end_ms)
        for onset_ms in active_onsets[left:right]:
            time_ms = _nearest_grid_row(
                onset_ms,
                canonical_rows,
                start_ms=gap.start_ms,
                end_ms=gap.end_ms,
            )
            if time_ms is None:
                time_ms = onset_ms
            if not _far_enough(time_ms, occupied_rows, minimum_gap_ms):
                continue
            lane = _available_lane(
                len(occupied_rows),
                request.key_mode,
                request.difficulty,
                time_ms=time_ms,
                holds=holds,
            )
            if lane is None:
                continue
            note = NoteEvent(time_ms=time_ms, lane=lane)
            inserted.append(note)
            occupied_rows.insert(bisect_left(occupied_rows, time_ms), time_ms)
            gap_inserted = True
        if gap_inserted:
            repaired_intervals.append((gap.start_ms, gap.end_ms))

    if not inserted:
        raise ValueError("coverage repair found no supported TAP rows")

    notes = sorted(
        [*source.notes, *inserted],
        key=lambda note: (note.time_ms, note.lane),
    )
    return (
        GeneratedChart(
            notes=notes,
            key_mode=source.key_mode,
            osu_text="",
            generator_name=f"coverage-repair-v1:{source.generator_name}",
            seed=source.seed,
            bpm_events=source.bpm_events,
            resnap_diagnostics=source.resnap_diagnostics,
        ),
        CoverageRepairPlan(
            repaired_intervals=tuple(repaired_intervals),
            inserted_notes=tuple(inserted),
        ),
    )
