"""Pure acceptance decisions for generated chart candidates."""

from dataclasses import dataclass
from enum import StrEnum

from chart_worker.analysis.activity import BoundaryPolicyMode, build_song_boundary_contract
from chart_worker.analysis.chart_profile import (
    ChartQualityProfile,
    build_chart_quality_profile,
)
from chart_worker.analysis.grid_alignment import NoteGridMetrics, measure_note_grid_alignment
from chart_worker.analysis.onset import OnsetAnalysis
from chart_worker.analysis.timing_diagnostics import (
    ACTIVE_GAP_MIN_FRAME_RATIO,
    ACTIVE_GAP_MIN_ONSETS,
    SECTION_PHASE_DRIFT_MAX_MS,
    TimingDiagnostics,
    diagnose_chart_timing,
)
from chart_worker.generation.mapperatorinator import GeneratedChart
from chart_worker.generation.osu_parser import OsuBpmEvent
from chart_worker.schema.types import DIFFICULTIES
from chart_worker.stages.types import SongTimingAuthority
from chart_worker.validation.generated_chart import (
    GeneratedChartValidationError,
    validate_generated_chart,
)
from chart_worker.validation.timing_authority import (
    TimingAuthorityValidationError,
    validate_timing_identity,
)

QUALITY_GATE_VERSION = "quality-gate-v3-outro-review"
QUIET_TRAILING_REVIEW_PROXIMITY = 0.5


class GateAxis(StrEnum):
    STRUCTURE = "STRUCTURE"
    SONG_BOUNDS = "SONG_BOUNDS"
    TIMING_IDENTITY = "TIMING_IDENTITY"
    TIMING_ALIGNMENT = "TIMING_ALIGNMENT"
    COVERAGE = "COVERAGE"
    PATTERN = "PATTERN"


class GateAction(StrEnum):
    PASS = "PASS"
    RETRY_MAP = "RETRY_MAP"
    REVIEW = "REVIEW"


@dataclass(frozen=True, slots=True)
class GateDecision:
    axis: GateAxis
    action: GateAction
    reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ChartAcceptance:
    action: GateAction
    decisions: tuple[GateDecision, ...]
    timing: TimingDiagnostics
    note_grid: NoteGridMetrics
    profile: ChartQualityProfile | None = None
    structure_error: dict[str, object] | None = None

    def decision(self, axis: GateAxis) -> GateDecision:
        """Return the independently recorded decision for one acceptance axis."""
        for decision in self.decisions:
            if decision.axis is axis:
                return decision
        raise ValueError(f"missing decision for axis: {axis}")

    def to_report(self) -> dict[str, object]:
        """Serialize only stable machine-readable acceptance evidence."""
        return {
            "action": self.action.value,
            "decisions": {
                decision.axis.value: {
                    "action": decision.action.value,
                    "reasons": list(decision.reasons),
                }
                for decision in self.decisions
            },
            "timing": self.timing.to_report(),
            "noteGrid": {
                "uniqueRowCount": self.note_grid.unique_row_count,
                "cleanRowCount": self.note_grid.clean_row_count,
                "cleanRate": self.note_grid.clean_rate,
                "absoluteP95Beats": self.note_grid.absolute_p95_beats,
            },
            "qualityProfile": self.profile.to_report() if self.profile is not None else None,
            "structureError": self.structure_error,
        }


def _structure_decision(
    chart: GeneratedChart, *, key_mode: int, duration_ms: int
) -> tuple[GateDecision, dict[str, object] | None]:
    try:
        validate_generated_chart(chart, key_mode=key_mode, duration_ms=duration_ms)
    except GeneratedChartValidationError as error:
        return (
            GateDecision(
                GateAxis.STRUCTURE, GateAction.RETRY_MAP, ("STRUCTURE_INVALID",)
            ),
            {"reasonCode": error.reason_code, "context": error.context},
        )
    return GateDecision(GateAxis.STRUCTURE, GateAction.PASS, ()), None


def _timing_identity_decision(
    chart: GeneratedChart, authority: SongTimingAuthority
) -> GateDecision:
    try:
        validate_timing_identity(chart.bpm_events, authority.bpm_events)
    except TimingAuthorityValidationError:
        return GateDecision(
            GateAxis.TIMING_IDENTITY,
            GateAction.RETRY_MAP,
            ("TIMING_REFERENCE_MISMATCH",),
        )
    return GateDecision(GateAxis.TIMING_IDENTITY, GateAction.PASS, ())


def _song_bounds_decision(
    chart: GeneratedChart,
    *,
    key_mode: int,
    duration_ms: int,
    max_note_start_ms: int,
    max_hold_end_ms: int,
) -> GateDecision:
    try:
        validate_generated_chart(
            chart,
            key_mode=key_mode,
            duration_ms=duration_ms,
            max_note_start_ms=max_note_start_ms,
            max_hold_end_ms=max_hold_end_ms,
        )
    except GeneratedChartValidationError as error:
        return GateDecision(
            GateAxis.SONG_BOUNDS,
            GateAction.RETRY_MAP,
            (error.reason_code,),
        )
    return GateDecision(GateAxis.SONG_BOUNDS, GateAction.PASS, ())


def _coverage_decision(timing: TimingDiagnostics) -> GateDecision:
    active_reasons = tuple(f"ACTIVE_{gap.position}_GAP" for gap in timing.coverage_gaps)
    if active_reasons:
        return GateDecision(GateAxis.COVERAGE, GateAction.RETRY_MAP, active_reasons)
    quiet_reasons = tuple(
        f"QUIET_{gap.position}_GAP" for gap in timing.quiet_coverage_gaps
    )
    if quiet_reasons:
        near_active_trailing = any(
            gap.position == "TRAILING"
            and min(
                gap.active_onset_count / ACTIVE_GAP_MIN_ONSETS,
                gap.active_frame_ratio / ACTIVE_GAP_MIN_FRAME_RATIO,
            )
            >= QUIET_TRAILING_REVIEW_PROXIMITY
            for gap in timing.quiet_coverage_gaps
        )
        if near_active_trailing:
            return GateDecision(
                GateAxis.COVERAGE,
                GateAction.REVIEW,
                (*quiet_reasons, "QUIET_TRAILING_GAP_NEAR_ACTIVE_THRESHOLD"),
            )
        return GateDecision(GateAxis.COVERAGE, GateAction.PASS, quiet_reasons)
    return GateDecision(GateAxis.COVERAGE, GateAction.PASS, ())


def _metrical_phase_distance_ms(
    section_start_ms: int,
    phase_delta_ms: float | None,
    bpm_events: tuple[OsuBpmEvent, ...],
) -> float | None:
    """Measure phase modulo the local beat so one-beat ambiguity is equivalent."""
    if phase_delta_ms is None or not bpm_events:
        return None
    event = bpm_events[0]
    for candidate in bpm_events:
        if candidate.time_ms > section_start_ms:
            break
        event = candidate
    beat_ms = 60_000.0 / event.bpm
    remainder = abs(phase_delta_ms) % beat_ms
    return min(remainder, beat_ms - remainder)


def _timing_alignment_decision(
    timing: TimingDiagnostics,
    note_grid: NoteGridMetrics,
    *,
    activity_present: bool,
    bpm_events: tuple[OsuBpmEvent, ...],
) -> GateDecision:
    overall = timing.overall
    if (
        overall.precision_50 is not None
        and overall.absolute_p95_ms is not None
        and overall.signed_median_ms is not None
        and overall.precision_50 < 0.70
        and overall.absolute_p95_ms >= 60
        and abs(overall.signed_median_ms) > SECTION_PHASE_DRIFT_MAX_MS
    ):
        return GateDecision(
            GateAxis.TIMING_ALIGNMENT,
            GateAction.RETRY_MAP,
            ("OVERALL_TIMING_MISALIGNED",),
        )

    grid_weak_clean_rate = note_grid.clean_rate < 0.80
    grid_weak_p95 = note_grid.absolute_p95_beats > 0.04
    if grid_weak_clean_rate and grid_weak_p95:
        return GateDecision(
            GateAxis.TIMING_ALIGNMENT,
            GateAction.RETRY_MAP,
            ("NOTE_GRID_MISALIGNED",),
        )

    advisory_reasons: list[str] = []
    review_reasons: list[str] = []
    if overall.precision_50 is None:
        review_reasons.append("LOW_OVERALL_ONSET_SUPPORT")
    elif overall.precision_50 < 0.70 or (
        overall.absolute_p95_ms is not None and overall.absolute_p95_ms >= 60
    ):
        advisory_reasons.append("OVERALL_TIMING_WEAK_SUPPORT")
    if activity_present and timing.active_onset_count < ACTIVE_GAP_MIN_ONSETS:
        review_reasons.append("LOW_ACTIVE_ONSET_SUPPORT")
    if any(section.status == "INSUFFICIENT" for section in timing.sections):
        advisory_reasons.append("SECTION_TIMING_INSUFFICIENT")
    if any(
        section.status != "INSUFFICIENT"
        and section.metrics.precision_50 is not None
        and section.metrics.absolute_p95_ms is not None
        and (
            section.metrics.precision_50 < 0.60
            or section.metrics.absolute_p95_ms >= 60
        )
        for section in timing.sections
    ):
        advisory_reasons.append("SECTION_TIMING_WEAK_SUPPORT")
    if any(
        section.status != "INSUFFICIENT"
        and (
            _metrical_phase_distance_ms(
                section.start_ms, section.phase_delta_ms, bpm_events
            )
            or 0.0
        )
        > SECTION_PHASE_DRIFT_MAX_MS
        for section in timing.sections
    ):
        advisory_reasons.append("SECTION_PHASE_DELTA")
    if grid_weak_clean_rate or grid_weak_p95:
        review_reasons.append("NOTE_GRID_WEAK_SUPPORT")
    reasons = tuple(advisory_reasons + review_reasons)
    if review_reasons:
        return GateDecision(GateAxis.TIMING_ALIGNMENT, GateAction.REVIEW, reasons)
    return GateDecision(GateAxis.TIMING_ALIGNMENT, GateAction.PASS, reasons)


def _overall_action(decisions: tuple[GateDecision, ...]) -> GateAction:
    if any(decision.action is GateAction.RETRY_MAP for decision in decisions):
        return GateAction.RETRY_MAP
    if any(decision.action is GateAction.REVIEW for decision in decisions):
        return GateAction.REVIEW
    return GateAction.PASS


def evaluate_chart_candidate(
    chart: GeneratedChart,
    authority: SongTimingAuthority,
    onset_analysis: OnsetAnalysis,
    *,
    requested_key_mode: int,
    requested_difficulty: str,
    duration_ms: int,
    boundary_policy_mode: BoundaryPolicyMode = "SHADOW",
) -> ChartAcceptance:
    """Evaluate a candidate without changing its notes, holds, or timing events."""
    if requested_difficulty not in DIFFICULTIES:
        raise ValueError(f"unsupported difficulty: {requested_difficulty}")

    structure, structure_error = _structure_decision(
        chart, key_mode=requested_key_mode, duration_ms=duration_ms
    )
    boundary = (
        build_song_boundary_contract(
            onset_analysis.activity,
            duration_ms,
            enforcement_mode=boundary_policy_mode,
        )
        if onset_analysis.activity is not None
        else None
    )
    coverage_end_ms = (
        boundary.required_coverage_end_ms if boundary is not None else duration_ms
    )
    timing = diagnose_chart_timing(
        chart.notes,
        onset_analysis.onset_ms,
        duration_ms=duration_ms,
        coverage_end_ms=coverage_end_ms,
        bpm_events=authority.bpm_events,
        activity=onset_analysis.activity,
    )
    unique_rows = tuple(sorted({note.time_ms for note in chart.notes}))
    note_grid = measure_note_grid_alignment(unique_rows, authority.bpm_events)
    profile: ChartQualityProfile | None = None
    if structure.action is GateAction.PASS:
        profile = build_chart_quality_profile(
            chart.notes,
            key_mode=requested_key_mode,
            duration_ms=duration_ms,
            beat_ms=60_000.0 / authority.bpm_events[0].bpm,
            bpm_events=authority.bpm_events,
            activity=onset_analysis.activity,
        )
        from chart_worker.validation.profile_review import review_profile

        profile_decisions = review_profile(
            profile,
            key_mode=requested_key_mode,
            difficulty=requested_difficulty,
        )
    else:
        profile_decisions = (
            GateDecision(
                GateAxis.PATTERN,
                GateAction.PASS,
                ("PROFILE_UNAVAILABLE_STRUCTURE_INVALID",),
            ),
        )
    decisions = (
        structure,
        (
            _song_bounds_decision(
                chart,
                key_mode=requested_key_mode,
                duration_ms=duration_ms,
                max_note_start_ms=(
                    boundary.max_note_start_ms
                    if boundary is not None
                    else duration_ms
                ),
                max_hold_end_ms=(
                    min(
                        duration_ms,
                        boundary.release_end_ms
                        + boundary.quantization_tolerance_ms,
                    )
                    if boundary is not None
                    else duration_ms
                ),
            )
            if structure.action is GateAction.PASS
            else GateDecision(
                GateAxis.SONG_BOUNDS,
                GateAction.PASS,
                ("NOT_EVALUATED_STRUCTURE_INVALID",),
            )
        ),
        _timing_identity_decision(chart, authority),
        _timing_alignment_decision(
            timing,
            note_grid,
            activity_present=onset_analysis.activity is not None,
            bpm_events=authority.bpm_events,
        ),
        _coverage_decision(timing),
        *profile_decisions,
    )
    return ChartAcceptance(
        action=_overall_action(decisions),
        decisions=decisions,
        timing=timing,
        note_grid=note_grid,
        profile=profile,
        structure_error=structure_error,
    )
