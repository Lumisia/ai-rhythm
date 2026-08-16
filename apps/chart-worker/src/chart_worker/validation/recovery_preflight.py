"""Read-only recovery grid coverage review for a shared timing authority."""

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from chart_worker.analysis.activity import BoundaryPolicyMode, build_song_boundary_contract
from chart_worker.analysis.onset import OnsetAnalysis
from chart_worker.analysis.timing_diagnostics import TimingCoverageGap, diagnose_chart_timing
from chart_worker.generation.params import GenerationRequest
from chart_worker.generation.recovery import RECOVERY_BUDGETS, plan_recovery_rows
from chart_worker.schema.note import NoteEvent
from chart_worker.schema.types import DIFFICULTIES
from chart_worker.stages.types import SongTimingAuthority

PREFLIGHT_VERSION = "recovery-preflight-v1"
ALLOWED_DIVISORS = (1, 2, 3, 4, 6, 8)


class RecoveryPreflightAction(StrEnum):
    PASS = "PASS"
    REVIEW = "REVIEW"
    DAMAGED = "DAMAGED"


@dataclass(frozen=True, slots=True)
class DifficultyRecoveryPreflight:
    difficulty: str
    action: RecoveryPreflightAction
    selected_divisor: int
    selected_row_count: int
    active_gaps: tuple[TimingCoverageGap, ...]
    quiet_gaps: tuple[TimingCoverageGap, ...]
    viable_divisors: tuple[int, ...]

    def to_report(self) -> dict[str, object]:
        return {
            "difficulty": self.difficulty,
            "action": self.action.value,
            "selectedDivisor": self.selected_divisor,
            "selectedRowCount": self.selected_row_count,
            "activeGaps": [gap.to_report() for gap in self.active_gaps],
            "quietGaps": [gap.to_report() for gap in self.quiet_gaps],
            "viableDivisors": list(self.viable_divisors),
        }


@dataclass(frozen=True, slots=True)
class RecoveryPreflight:
    action: RecoveryPreflightAction
    difficulties: tuple[DifficultyRecoveryPreflight, ...]

    def for_difficulty(self, difficulty: str) -> DifficultyRecoveryPreflight:
        return next(
            item for item in self.difficulties if item.difficulty == difficulty
        )

    def to_report(self) -> dict[str, object]:
        return {
            "version": PREFLIGHT_VERSION,
            "action": self.action.value,
            "difficulties": [item.to_report() for item in self.difficulties],
        }


def _request(difficulty: str, authority: SongTimingAuthority, duration_ms: int) -> GenerationRequest:
    return GenerationRequest(
        audio_path=Path("preflight.flac"),
        timing_reference_path=authority.reference_path,
        key_mode=4,
        difficulty=difficulty,
        duration_ms=duration_ms,
        seed=0,
    )


def _diagnose_rows(
    rows: tuple[int, ...],
    onsets: OnsetAnalysis,
    authority: SongTimingAuthority,
    duration_ms: int,
    boundary_policy_mode: BoundaryPolicyMode,
    difficulty: str,
):
    notes = [NoteEvent(time_ms=time_ms, lane=0) for time_ms in rows]
    coverage_end_ms = (
        build_song_boundary_contract(
            onsets.activity,
            duration_ms,
            enforcement_mode=boundary_policy_mode,
        )
        .required_coverage_end_ms
        if onsets.activity is not None
        else duration_ms
    )
    return diagnose_chart_timing(
        notes,
        onsets.onset_ms,
        duration_ms=duration_ms,
        coverage_end_ms=coverage_end_ms,
        bpm_events=authority.bpm_events,
        activity=onsets.activity,
        onset_analysis=onsets,
        difficulty=difficulty,
    )


def review_recovery_preflight(
    authority: SongTimingAuthority,
    onsets: OnsetAnalysis,
    *,
    duration_ms: int,
    boundary_policy_mode: BoundaryPolicyMode = "SHADOW",
) -> RecoveryPreflight:
    """Compare current recovery divisors with bounded alternatives without mutation."""
    difficulty_reports = []
    for difficulty in DIFFICULTIES:
        request = _request(difficulty, authority, duration_ms)
        selected = plan_recovery_rows(request, authority, onsets)
        selected_diagnostics = _diagnose_rows(
            selected.rows,
            onsets,
            authority,
            duration_ms,
            boundary_policy_mode,
            difficulty,
        )

        viable_divisors = []
        for divisor in ALLOWED_DIVISORS:
            candidate = plan_recovery_rows(
                request,
                authority,
                onsets,
                subdivisions=divisor,
            )
            diagnostics = _diagnose_rows(
                candidate.rows,
                onsets,
                authority,
                duration_ms,
                boundary_policy_mode,
                difficulty,
            )
            if candidate.rows and not diagnostics.coverage_gaps:
                viable_divisors.append(divisor)

        if not selected_diagnostics.coverage_gaps:
            action = RecoveryPreflightAction.PASS
        elif viable_divisors:
            action = RecoveryPreflightAction.REVIEW
        else:
            action = RecoveryPreflightAction.DAMAGED
        difficulty_reports.append(
            DifficultyRecoveryPreflight(
                difficulty=difficulty,
                action=action,
                selected_divisor=RECOVERY_BUDGETS[difficulty].subdivisions,
                selected_row_count=len(selected.rows),
                active_gaps=selected_diagnostics.coverage_gaps,
                quiet_gaps=selected_diagnostics.quiet_coverage_gaps,
                viable_divisors=tuple(viable_divisors),
            )
        )

    priority = {
        RecoveryPreflightAction.PASS: 0,
        RecoveryPreflightAction.REVIEW: 1,
        RecoveryPreflightAction.DAMAGED: 2,
    }
    action = max(
        (report.action for report in difficulty_reports),
        key=priority.__getitem__,
    )
    return RecoveryPreflight(action=action, difficulties=tuple(difficulty_reports))
