"""Conservative integrity classification for variable-BPM timing candidates."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from math import log2

from chart_worker.analysis.local_timing import (
    ACTIVE_DURATION_MIN_MS,
    METRICAL_DISTANCE_MAX_OCTAVES,
    METRICAL_RATIOS,
)
from chart_worker.validation.local_timing_review import (
    LocalTimingAuthorityReview,
    LocalTimingSegmentReview,
)
from chart_worker.validation.recovery_preflight import (
    RecoveryPreflight,
    RecoveryPreflightAction,
)
from chart_worker.validation.timing_review import TimingAuthorityAction

TIMING_INTEGRITY_VERSION = "timing-integrity-v1"
MAX_RETURN_ISLAND_SEGMENTS = 4
RETURN_ISLAND_CURRENT_SUPPORT_MAX = 0.5
RETURN_ISLAND_NEIGHBOR_SUPPORT_MIN = 0.55
RETURN_ISLAND_SUPPORT_GAP_MIN = 0.25


class TimingIntegrityStatus(StrEnum):
    HEALTHY = "HEALTHY"
    NEEDS_CORROBORATION = "NEEDS_CORROBORATION"
    DAMAGED = "DAMAGED"


@dataclass(frozen=True, slots=True)
class TimingIslandEvidence:
    segment_indices: tuple[int, ...]
    start_ms: int
    end_ms: int
    bpms: tuple[float, ...]
    active_duration_ms: int
    current_grid_support: float
    neighbor_grid_support: float
    pulse_conflict_count: int
    grid_damage_count: int

    def to_report(self) -> dict[str, object]:
        return {
            "segmentIndices": list(self.segment_indices),
            "startMs": self.start_ms,
            "endMs": self.end_ms,
            "bpms": list(self.bpms),
            "activeDurationMs": self.active_duration_ms,
            "currentGridSupport": self.current_grid_support,
            "neighborGridSupport": self.neighbor_grid_support,
            "pulseConflictCount": self.pulse_conflict_count,
            "gridDamageCount": self.grid_damage_count,
        }


@dataclass(frozen=True, slots=True)
class TimingIntegrityAssessment:
    status: TimingIntegrityStatus
    reasons: tuple[str, ...]
    islands: tuple[TimingIslandEvidence, ...]

    def to_report(self) -> dict[str, object]:
        return {
            "version": TIMING_INTEGRITY_VERSION,
            "status": self.status.value,
            "reasons": list(self.reasons),
            "islands": [island.to_report() for island in self.islands],
        }


def _metrical_distance(left_bpm: float, right_bpm: float) -> float:
    return min(
        abs(log2(left_bpm / (ratio * right_bpm)))
        for ratio in METRICAL_RATIOS
    )


def _metrically_matches(left_bpm: float, right_bpm: float) -> bool:
    return _metrical_distance(left_bpm, right_bpm) <= METRICAL_DISTANCE_MAX_OCTAVES


def _weighted_support(
    segments: tuple[LocalTimingSegmentReview, ...],
    *,
    neighbor: bool,
) -> float:
    active = tuple(segment for segment in segments if _has_active_evidence(segment))
    duration_ms = sum(segment.metrics.end_ms - segment.metrics.start_ms for segment in active)
    if not duration_ms:
        return 0.0
    total = sum(
        (segment.metrics.end_ms - segment.metrics.start_ms)
        * (
            segment.metrics.neighbor_grid_support
            if neighbor
            else segment.metrics.current_grid_support
        )
        for segment in active
    )
    return round(total / duration_ms, 6)


def _has_active_evidence(segment: LocalTimingSegmentReview) -> bool:
    metrics = segment.metrics
    return metrics.active_onset_count > 0 and metrics.active_frame_ratio >= 0.35


def _return_islands(
    segments: tuple[LocalTimingSegmentReview, ...],
) -> tuple[TimingIslandEvidence, ...]:
    islands: list[TimingIslandEvidence] = []
    for start in range(1, len(segments) - 1):
        max_end = min(
            len(segments) - 2,
            start + MAX_RETURN_ISLAND_SEGMENTS - 1,
        )
        for end in range(start, max_end + 1):
            left = segments[start - 1].metrics
            right = segments[end + 1].metrics
            if not _metrically_matches(left.bpm, right.bpm):
                continue
            body = segments[start : end + 1]
            if not all(
                not _metrically_matches(segment.metrics.bpm, left.bpm)
                and not _metrically_matches(segment.metrics.bpm, right.bpm)
                for segment in body
            ):
                continue
            total_duration_ms = sum(
                segment.metrics.end_ms - segment.metrics.start_ms for segment in body
            )
            active_duration_ms = sum(
                segment.metrics.end_ms - segment.metrics.start_ms
                for segment in body
                if _has_active_evidence(segment)
            )
            active_onset_count = sum(
                segment.metrics.active_onset_count for segment in body
            )
            current_support = _weighted_support(body, neighbor=False)
            neighbor_support = _weighted_support(body, neighbor=True)
            corroborated = any(
                segment.metrics.pulse_conflict or segment.grid_damage
                for segment in body
            )
            if not (
                active_duration_ms >= ACTIVE_DURATION_MIN_MS
                and active_onset_count >= 8
                and active_duration_ms >= 0.8 * total_duration_ms
                and current_support < RETURN_ISLAND_CURRENT_SUPPORT_MAX
                and neighbor_support >= RETURN_ISLAND_NEIGHBOR_SUPPORT_MIN
                and neighbor_support - current_support >= RETURN_ISLAND_SUPPORT_GAP_MIN
                and corroborated
            ):
                continue
            islands.append(
                TimingIslandEvidence(
                    segment_indices=tuple(
                        segment.metrics.index for segment in body
                    ),
                    start_ms=body[0].metrics.start_ms,
                    end_ms=body[-1].metrics.end_ms,
                    bpms=tuple(segment.metrics.bpm for segment in body),
                    active_duration_ms=active_duration_ms,
                    current_grid_support=current_support,
                    neighbor_grid_support=neighbor_support,
                    pulse_conflict_count=sum(
                        segment.metrics.pulse_conflict for segment in body
                    ),
                    grid_damage_count=sum(segment.grid_damage for segment in body),
                )
            )
            break
    return tuple(islands)


def assess_timing_integrity(
    local_review: LocalTimingAuthorityReview,
    recovery_preflight: RecoveryPreflight,
) -> TimingIntegrityAssessment:
    """Classify without editing timing events or imposing an absolute BPM range."""
    islands = _return_islands(local_review.segments)
    reasons: list[str] = []
    if islands:
        reasons.append("ACTIVE_RETURN_TIMING_ISLAND")
    if recovery_preflight.action is RecoveryPreflightAction.DAMAGED:
        reasons.append("RECOVERY_PREFLIGHT_DAMAGED")
    if local_review.action is TimingAuthorityAction.RETRY_TIMING:
        reasons.append("LOCAL_TIMING_HARD_REJECT")

    if local_review.action is TimingAuthorityAction.RETRY_TIMING or (
        islands and recovery_preflight.action is RecoveryPreflightAction.DAMAGED
    ):
        status = TimingIntegrityStatus.DAMAGED
    elif islands or recovery_preflight.action is RecoveryPreflightAction.DAMAGED:
        status = TimingIntegrityStatus.NEEDS_CORROBORATION
    else:
        status = TimingIntegrityStatus.HEALTHY
    return TimingIntegrityAssessment(
        status=status,
        reasons=tuple(reasons),
        islands=islands,
    )
