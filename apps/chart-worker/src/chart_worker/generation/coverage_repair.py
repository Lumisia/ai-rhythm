"""Deterministic TAP-only repair for proven active coverage gaps."""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass
from math import ceil
from typing import Literal

from chart_worker.analysis.coverage_opportunity import MIN_PHRASE_DURATION_MS
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

MAX_INSERTED_NOTE_RATIO = 0.15
MIN_INSERTED_NOTE_BUDGET = 8
MAX_INSERTED_NOTE_BUDGET = 64
MAX_REPAIRED_DURATION_RATIO = 0.20
MIN_REPAIRED_DURATION_BUDGET_MS = 8_000
CoverageRepairMode = Literal["VACANT_INTERVAL", "UNDER_HOLD_POLYPHONY"]


@dataclass(frozen=True, slots=True)
class _RepairAuthority:
    start_ms: int
    end_ms: int
    mode: CoverageRepairMode


@dataclass(frozen=True, slots=True)
class CoverageRepairPlan:
    repaired_intervals: tuple[tuple[int, int], ...]
    repair_modes: tuple[CoverageRepairMode, ...]
    inserted_notes: tuple[NoteEvent, ...]

    def __post_init__(self) -> None:
        if len(self.repaired_intervals) != len(self.repair_modes):
            raise ValueError("repair modes must align with repaired intervals")

    @property
    def repaired_gap_count(self) -> int:
        return len(self.repaired_intervals)

    def to_report(self) -> dict[str, object]:
        return {
            "repairedGapCount": self.repaired_gap_count,
            "insertedNoteCount": len(self.inserted_notes),
            "repairedIntervals": [
                {"startMs": start_ms, "endMs": end_ms, "mode": mode}
                for (start_ms, end_ms), mode in zip(
                    self.repaired_intervals,
                    self.repair_modes,
                    strict=True,
                )
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


def _repair_authority(
    request: GenerationRequest,
    gap: object,
) -> _RepairAuthority | None:
    opportunity = getattr(gap, "opportunity", None)
    if opportunity is None or not opportunity.actionable:
        return None
    row_span_start_ms = getattr(gap, "row_span_start_ms", None)
    if row_span_start_ms is None:
        row_span_start_ms = gap.start_ms  # type: ignore[attr-defined]
    unoccupied_start_ms = getattr(gap, "unoccupied_start_ms", None)
    if unoccupied_start_ms is None:
        unoccupied_start_ms = gap.start_ms  # type: ignore[attr-defined]
    end_ms = gap.end_ms  # type: ignore[attr-defined]
    if end_ms - unoccupied_start_ms >= MIN_PHRASE_DURATION_MS:
        return _RepairAuthority(
            start_ms=unoccupied_start_ms,
            end_ms=end_ms,
            mode="VACANT_INTERVAL",
        )
    if (
        request.difficulty in {"HARD", "EXPERT"}
        and opportunity.attack_evidence_scope == "GLOBAL"
        and opportunity.hold_occupancy_ratio > 0.0
        and end_ms - row_span_start_ms >= MIN_PHRASE_DURATION_MS
    ):
        return _RepairAuthority(
            start_ms=row_span_start_ms,
            end_ms=end_ms,
            mode="UNDER_HOLD_POLYPHONY",
        )
    return None


def build_coverage_repair_chart(
    request: GenerationRequest,
    source: GeneratedChart,
    acceptance: ChartAcceptance,
    authority: SongTimingAuthority,
    onsets: OnsetAnalysis,
    *,
    approved_intervals: tuple[tuple[int, int], ...] = (),
) -> tuple[GeneratedChart, CoverageRepairPlan]:
    """Add deterministic TAP rows only inside gate- or jury-proven gaps."""

    if source.key_mode != request.key_mode:
        raise ValueError("coverage repair source key mode differs from request")
    coverage = acceptance.decision(GateAxis.COVERAGE)
    if approved_intervals:
        if any(
            type(start_ms) is not int
            or type(end_ms) is not int
            or start_ms < 0
            or end_ms <= start_ms
            for start_ms, end_ms in approved_intervals
        ):
            raise ValueError("approved intervals must be positive exact integer bounds")
        if len(set(approved_intervals)) != len(approved_intervals):
            raise ValueError("approved intervals must be unique")
        eligible_by_interval = {
            (gap.start_ms, gap.end_ms): gap
            for gap in acceptance.timing.coverage_gaps
            if _repair_authority(request, gap) is not None
        }
        if any(interval not in eligible_by_interval for interval in approved_intervals):
            raise ValueError(
                "approved interval is absent from an attack-required coverage gap "
                "in gate evidence"
            )
        if coverage.action is not GateAction.RETRY_MAP:
            raise ValueError(
                "approved attack-required repair requires a RETRY_MAP coverage decision"
            )
        attack_gaps = tuple(eligible_by_interval[interval] for interval in approved_intervals)
    else:
        attack_gaps = tuple(
            gap
            for gap in acceptance.timing.coverage_gaps
            if _repair_authority(request, gap) is not None
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
    repair_modes: list[CoverageRepairMode] = []

    for gap in attack_gaps:
        repair_authority = _repair_authority(request, gap)
        if repair_authority is None:
            continue
        repair_start_ms = repair_authority.start_ms
        repair_end_ms = repair_authority.end_ms
        gap_inserted = False
        left = bisect_left(active_onsets, repair_start_ms + 1)
        right = bisect_left(active_onsets, repair_end_ms)
        for onset_ms in active_onsets[left:right]:
            time_ms = _nearest_grid_row(
                onset_ms,
                canonical_rows,
                start_ms=repair_start_ms,
                end_ms=repair_end_ms,
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
            repaired_intervals.append((repair_start_ms, repair_end_ms))
            repair_modes.append(repair_authority.mode)

    if not inserted:
        raise ValueError("coverage repair found no supported TAP rows")

    inserted_note_budget = max(
        MIN_INSERTED_NOTE_BUDGET,
        min(
            MAX_INSERTED_NOTE_BUDGET,
            ceil(len(source.notes) * MAX_INSERTED_NOTE_RATIO),
        ),
    )
    repaired_duration_ms = sum(end_ms - start_ms for start_ms, end_ms in repaired_intervals)
    if len(inserted) > inserted_note_budget:
        raise ValueError(
            "coverage repair exceeds inserted-note budget: "
            f"{len(inserted)} > {inserted_note_budget}"
        )
    repaired_duration_budget_ms = max(
        MIN_REPAIRED_DURATION_BUDGET_MS,
        request.duration_ms * MAX_REPAIRED_DURATION_RATIO,
    )
    if repaired_duration_ms > repaired_duration_budget_ms:
        raise ValueError(
            "coverage repair exceeds repaired-duration budget: "
            f"{repaired_duration_ms}ms > {repaired_duration_budget_ms:g}ms"
        )

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
            repair_modes=tuple(repair_modes),
            inserted_notes=tuple(inserted),
        ),
    )
