"""Pure acceptance decisions for generated chart candidates."""

from dataclasses import dataclass
from enum import StrEnum

from chart_worker.analysis.chart_profile import (
    ChartQualityProfile,
    build_chart_quality_profile,
)
from chart_worker.analysis.grid_alignment import NoteGridMetrics, measure_note_grid_alignment
from chart_worker.analysis.onset import OnsetAnalysis
from chart_worker.analysis.timing_diagnostics import (
    ACTIVE_GAP_MIN_ONSETS,
    SECTION_PHASE_DRIFT_MAX_MS,
    TimingDiagnostics,
    diagnose_chart_timing,
)
from chart_worker.generation.mapperatorinator import GeneratedChart
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


class GateAxis(StrEnum):
    STRUCTURE = "STRUCTURE"
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
        }


def _structure_decision(
    chart: GeneratedChart, *, key_mode: int, duration_ms: int
) -> GateDecision:
    try:
        validate_generated_chart(chart, key_mode=key_mode, duration_ms=duration_ms)
    except GeneratedChartValidationError:
        return GateDecision(
            GateAxis.STRUCTURE, GateAction.RETRY_MAP, ("STRUCTURE_INVALID",)
        )
    return GateDecision(GateAxis.STRUCTURE, GateAction.PASS, ())


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


def _coverage_decision(timing: TimingDiagnostics) -> GateDecision:
    active_reasons = tuple(f"ACTIVE_{gap.position}_GAP" for gap in timing.coverage_gaps)
    if active_reasons:
        return GateDecision(GateAxis.COVERAGE, GateAction.RETRY_MAP, active_reasons)
    quiet_reasons = tuple(
        f"QUIET_{gap.position}_GAP" for gap in timing.quiet_coverage_gaps
    )
    if quiet_reasons:
        return GateDecision(GateAxis.COVERAGE, GateAction.PASS, quiet_reasons)
    return GateDecision(GateAxis.COVERAGE, GateAction.PASS, ())


def _timing_alignment_decision(
    timing: TimingDiagnostics, note_grid: NoteGridMetrics, *, activity_present: bool
) -> GateDecision:
    overall = timing.overall
    if (
        overall.precision_50 is not None
        and overall.absolute_p95_ms is not None
        and overall.precision_50 < 0.70
        and overall.absolute_p95_ms >= 60
    ):
        return GateDecision(
            GateAxis.TIMING_ALIGNMENT,
            GateAction.RETRY_MAP,
            ("OVERALL_TIMING_MISALIGNED",),
        )

    if any(
        section.status == "REVIEW"
        and section.metrics.precision_50 is not None
        and section.metrics.absolute_p95_ms is not None
        and section.metrics.precision_50 < 0.60
        and section.metrics.absolute_p95_ms >= 60
        for section in timing.sections
    ):
        return GateDecision(
            GateAxis.TIMING_ALIGNMENT,
            GateAction.RETRY_MAP,
            ("SECTION_TIMING_MISALIGNED",),
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
        and section.phase_delta_ms is not None
        and abs(section.phase_delta_ms) > SECTION_PHASE_DRIFT_MAX_MS
        for section in timing.sections
    ):
        review_reasons.append("SECTION_PHASE_DELTA")
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
) -> ChartAcceptance:
    """Evaluate a candidate without changing its notes, holds, or timing events."""
    if requested_difficulty not in DIFFICULTIES:
        raise ValueError(f"unsupported difficulty: {requested_difficulty}")

    structure = _structure_decision(
        chart, key_mode=requested_key_mode, duration_ms=duration_ms
    )
    timing = diagnose_chart_timing(
        chart.notes,
        onset_analysis.onset_ms,
        duration_ms=duration_ms,
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
        _timing_identity_decision(chart, authority),
        _timing_alignment_decision(
            timing,
            note_grid,
            activity_present=onset_analysis.activity is not None,
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
    )
