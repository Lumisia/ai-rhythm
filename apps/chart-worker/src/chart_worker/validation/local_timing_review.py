"""Conjunctive decision policy for segment-local timing evidence."""

from dataclasses import dataclass

from chart_worker.analysis.local_timing import (
    ACTIVE_FRAME_RATIO_MIN,
    ACTIVE_ONSET_MIN,
    LocalTimingMetrics,
    LocalTimingSegmentMetrics,
)
from chart_worker.validation.recovery_preflight import RecoveryPreflight
from chart_worker.validation.timing_review import TimingAuthorityAction

LOCAL_TIMING_REVIEW_VERSION = "local-timing-review-v2-duration-weighted"


@dataclass(frozen=True, slots=True)
class LocalTimingSegmentReview:
    metrics: LocalTimingSegmentMetrics
    grid_damage: bool
    action: TimingAuthorityAction
    reasons: tuple[str, ...]

    @property
    def active_confident(self) -> bool:
        return self.metrics.active_confident

    @property
    def current_grid_support(self) -> float:
        return self.metrics.current_grid_support

    @property
    def isolated_metrical_outlier(self) -> bool:
        return self.metrics.isolated_metrical_outlier

    @property
    def pulse_conflict(self) -> bool:
        return self.metrics.pulse_conflict

    @property
    def evidence_status(self) -> str:
        return self.metrics.evidence_status

    @property
    def contradiction_count(self) -> int:
        return sum(
            (
                self.metrics.isolated_metrical_outlier,
                self.metrics.pulse_conflict,
            )
        )

    def to_report(self) -> dict[str, object]:
        metrics = self.metrics
        return {
            "index": metrics.index,
            "startMs": metrics.start_ms,
            "endMs": metrics.end_ms,
            "durationMs": metrics.end_ms - metrics.start_ms,
            "bpm": metrics.bpm,
            "action": self.action.value,
            "reasons": list(self.reasons),
            "onsetCount": metrics.onset_count,
            "activeOnsetCount": metrics.active_onset_count,
            "activeFrameRatio": metrics.active_frame_ratio,
            "activeConfident": metrics.active_confident,
            "currentGridSupport": metrics.current_grid_support,
            "neighborGridSupport": metrics.neighbor_grid_support,
            "currentResidualP95Ms": metrics.current_residual_p95_ms,
            "neighborResidualP95Ms": metrics.neighbor_residual_p95_ms,
            "isolatedMetricalOutlier": metrics.isolated_metrical_outlier,
            "pulseConflict": metrics.pulse_conflict,
            "phaseConflict": metrics.phase_conflict,
            "gridDamage": self.grid_damage,
            "contradictionCount": self.contradiction_count,
            "evidenceStatus": metrics.evidence_status,
            "boundaryOnsetDistanceMs": metrics.boundary_onset_distance_ms,
            "boundarySupported": metrics.boundary_supported,
        }


@dataclass(frozen=True, slots=True)
class DurationWeightedTimingEvidence:
    active_evidence_ms: int
    supported_active_ms: int
    contradicted_active_ms: int
    insufficient_active_ms: int
    quiet_ms: int
    supported_ratio: float
    contradicted_ratio: float
    insufficient_ratio: float
    segment_count: int

    def to_report(self) -> dict[str, object]:
        return {
            "activeEvidenceMs": self.active_evidence_ms,
            "supportedActiveMs": self.supported_active_ms,
            "contradictedActiveMs": self.contradicted_active_ms,
            "insufficientActiveMs": self.insufficient_active_ms,
            "quietMs": self.quiet_ms,
            "supportedRatio": self.supported_ratio,
            "contradictedRatio": self.contradicted_ratio,
            "insufficientRatio": self.insufficient_ratio,
            "segmentCount": self.segment_count,
        }


@dataclass(frozen=True, slots=True)
class LocalTimingAuthorityReview:
    action: TimingAuthorityAction
    reasons: tuple[str, ...]
    segments: tuple[LocalTimingSegmentReview, ...]

    @property
    def duration_evidence(self) -> DurationWeightedTimingEvidence:
        active_evidence_ms = 0
        supported_active_ms = 0
        contradicted_active_ms = 0
        insufficient_active_ms = 0
        quiet_ms = 0
        for segment in self.segments:
            metrics = segment.metrics
            duration_ms = metrics.end_ms - metrics.start_ms
            if "QUIET_LOCAL_TIMING_SEGMENT" in segment.reasons:
                quiet_ms += duration_ms
                continue
            active_evidence_ms += duration_ms
            contradicted = metrics.active_confident and (
                segment.contradiction_count >= 2
                or (segment.grid_damage and segment.contradiction_count >= 1)
            )
            if contradicted:
                contradicted_active_ms += duration_ms
            elif not metrics.active_confident:
                insufficient_active_ms += duration_ms
            elif (
                metrics.current_grid_support >= 0.55
                and segment.contradiction_count == 0
                and not segment.grid_damage
            ):
                supported_active_ms += duration_ms

        def ratio(value: int) -> float:
            return round(value / active_evidence_ms, 6) if active_evidence_ms else 0.0

        return DurationWeightedTimingEvidence(
            active_evidence_ms=active_evidence_ms,
            supported_active_ms=supported_active_ms,
            contradicted_active_ms=contradicted_active_ms,
            insufficient_active_ms=insufficient_active_ms,
            quiet_ms=quiet_ms,
            supported_ratio=ratio(supported_active_ms),
            contradicted_ratio=ratio(contradicted_active_ms),
            insufficient_ratio=ratio(insufficient_active_ms),
            segment_count=len(self.segments),
        )

    def to_report(self) -> dict[str, object]:
        return {
            "version": LOCAL_TIMING_REVIEW_VERSION,
            "action": self.action.value,
            "reasons": list(self.reasons),
            "durationEvidence": self.duration_evidence.to_report(),
            "segments": [segment.to_report() for segment in self.segments],
        }


def _overlaps_active_gap(
    metrics: LocalTimingSegmentMetrics,
    preflight: RecoveryPreflight,
) -> bool:
    return any(
        gap.start_ms < metrics.end_ms and gap.end_ms > metrics.start_ms
        for difficulty in preflight.difficulties
        for gap in difficulty.active_gaps
    )


def _review_segment(
    metrics: LocalTimingSegmentMetrics,
    preflight: RecoveryPreflight,
) -> LocalTimingSegmentReview:
    grid_damage = _overlaps_active_gap(metrics, preflight)
    contradictions = sum(
        (metrics.isolated_metrical_outlier, metrics.pulse_conflict)
    )
    quiet_confirmed = (
        metrics.onset_count >= ACTIVE_ONSET_MIN
        and metrics.active_onset_count < ACTIVE_ONSET_MIN
        and metrics.active_frame_ratio < ACTIVE_FRAME_RATIO_MIN
    )
    if metrics.active_confident and grid_damage and contradictions >= 2:
        action = TimingAuthorityAction.RETRY_TIMING
        reasons = ("ACTIVE_LOCAL_GRID_DAMAGE_WITH_CORROBORATION",)
    elif quiet_confirmed:
        action = TimingAuthorityAction.PASS
        reasons = ("QUIET_LOCAL_TIMING_SEGMENT",)
    elif not metrics.active_confident:
        action = TimingAuthorityAction.REVIEW
        reasons = ("INSUFFICIENT_LOCAL_TIMING_EVIDENCE",)
    elif grid_damage or contradictions >= 2:
        action = TimingAuthorityAction.REVIEW
        reasons = ("LOCAL_TIMING_DAMAGE_NEEDS_REVIEW",)
    else:
        action = TimingAuthorityAction.PASS
        reasons = ()
    return LocalTimingSegmentReview(
        metrics=metrics,
        grid_damage=grid_damage,
        action=action,
        reasons=reasons,
    )


def review_local_timing_authority(
    metrics: LocalTimingMetrics,
    preflight: RecoveryPreflight,
) -> LocalTimingAuthorityReview:
    """Hard-retry only active grid damage corroborated by two evidence axes."""
    segments = tuple(
        _review_segment(segment, preflight) for segment in metrics.segments
    )
    if any(segment.action is TimingAuthorityAction.RETRY_TIMING for segment in segments):
        action = TimingAuthorityAction.RETRY_TIMING
    elif any(segment.action is TimingAuthorityAction.REVIEW for segment in segments):
        action = TimingAuthorityAction.REVIEW
    else:
        action = TimingAuthorityAction.PASS
    reasons = tuple(
        dict.fromkeys(
            reason
            for segment in segments
            if segment.action is not TimingAuthorityAction.PASS
            for reason in segment.reasons
        )
    )
    return LocalTimingAuthorityReview(
        action=action,
        reasons=reasons,
        segments=segments,
    )
