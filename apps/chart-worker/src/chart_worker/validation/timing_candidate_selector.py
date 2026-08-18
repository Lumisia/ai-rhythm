"""Length-normalized evidence and deterministic timing-candidate selection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from chart_worker.analysis.grid_alignment import TempoCandidateMetrics
from chart_worker.validation.local_timing_review import LocalTimingAuthorityReview
from chart_worker.validation.timing_integrity import (
    TimingIntegrityAssessment,
    TimingIntegrityStatus,
)

TimingMode = Literal["STANDARD", "SUPER_TIMING", "BEAT_THIS_FALLBACK"]
MetricalLevel = Literal["HALF", "BASE", "DOUBLE"]


@dataclass(frozen=True, slots=True)
class TimingCandidateEvidence:
    epoch: int
    mode: TimingMode
    structurally_valid: bool
    active_evidence_ms: int
    supported_active_ms: int
    contradicted_active_ms: int
    insufficient_active_ms: int
    quiet_ms: int
    supported_ratio: float
    contradicted_ratio: float
    insufficient_ratio: float
    beat_consensus_by_level: dict[str, float]
    best_metrical_level: MetricalLevel
    unsupported_transition_rate_per_minute: float | None
    fragmentation_active_ratio: float
    boundary_risk_count: int
    confidence: Literal["HIGH", "MEDIUM", "LOW"]
    integrity: TimingIntegrityAssessment
    external_beat_f1_by_level: dict[str, float] | None = None
    best_external_beat_f1: float | None = None

    @property
    def best_beat_consensus(self) -> float:
        return self.beat_consensus_by_level[self.best_metrical_level]

    def to_report(self) -> dict[str, object]:
        return {
            "epoch": self.epoch,
            "mode": self.mode,
            "structurallyValid": self.structurally_valid,
            "activeEvidenceMs": self.active_evidence_ms,
            "supportedActiveMs": self.supported_active_ms,
            "contradictedActiveMs": self.contradicted_active_ms,
            "insufficientActiveMs": self.insufficient_active_ms,
            "quietMs": self.quiet_ms,
            "supportedRatio": self.supported_ratio,
            "contradictedRatio": self.contradicted_ratio,
            "insufficientRatio": self.insufficient_ratio,
            "beatConsensusByLevel": self.beat_consensus_by_level,
            "bestMetricalLevel": self.best_metrical_level,
            "unsupportedTransitionRatePerMinute": (
                self.unsupported_transition_rate_per_minute
            ),
            "fragmentationActiveRatio": self.fragmentation_active_ratio,
            "boundaryRiskCount": self.boundary_risk_count,
            "confidence": self.confidence,
            "timingIntegrity": self.integrity.to_report(),
            "externalBeatF1ByLevel": self.external_beat_f1_by_level,
            "bestExternalBeatF1": self.best_external_beat_f1,
        }


@dataclass(frozen=True, slots=True)
class TimingCandidateSelection:
    selected_index: int
    reason: str
    candidates: tuple[TimingCandidateEvidence, ...]
    external_beat_status: Literal[
        "NOT_REQUESTED", "AVAILABLE", "UNAVAILABLE", "FAILED"
    ] = "NOT_REQUESTED"
    external_beat_error: str | None = None
    external_beat_elapsed_ms: int | None = None

    def to_report(self) -> dict[str, object]:
        selected = self.candidates[self.selected_index]
        return {
            "selectedIndex": self.selected_index,
            "selectedEpoch": selected.epoch,
            "selectedMode": selected.mode,
            "reason": self.reason,
            "externalBeatStatus": self.external_beat_status,
            "externalBeatError": self.external_beat_error,
            "externalBeatElapsedMs": self.external_beat_elapsed_ms,
            "candidates": [candidate.to_report() for candidate in self.candidates],
        }


def _consensus(metrics: TempoCandidateMetrics) -> dict[str, float]:
    return {
        "HALF": round(
            min(metrics.half_pulse_support, metrics.half_periodicity_support), 6
        ),
        "BASE": round(
            min(metrics.base_pulse_support, metrics.base_periodicity_support), 6
        ),
        "DOUBLE": round(
            min(metrics.double_pulse_support, metrics.double_periodicity_support), 6
        ),
    }


def build_timing_candidate_evidence(
    *,
    epoch: int,
    mode: TimingMode,
    structurally_valid: bool,
    local_review: LocalTimingAuthorityReview,
    tempo_metrics: TempoCandidateMetrics,
    boundary_risk_count: int = 0,
    integrity: TimingIntegrityAssessment | None = None,
) -> TimingCandidateEvidence:
    duration = local_review.duration_evidence
    consensus = _consensus(tempo_metrics)
    best_level = max(
        ("HALF", "BASE", "DOUBLE"),
        key=lambda level: (consensus[level], level == "BASE"),
    )

    transition_segments = local_review.segments[1:]
    supported_boundaries = [
        segment.metrics.boundary_supported
        for segment in transition_segments
        if segment.metrics.boundary_supported is not None
    ]
    total_duration_ms = sum(
        segment.metrics.end_ms - segment.metrics.start_ms
        for segment in local_review.segments
    )
    unsupported_transition_count = sum(value is False for value in supported_boundaries)
    unsupported_rate = (
        round(unsupported_transition_count * 60_000 / total_duration_ms, 6)
        if total_duration_ms > 0 and supported_boundaries
        else None
    )
    fragmented_active_ms = sum(
        segment.metrics.end_ms - segment.metrics.start_ms
        for segment in transition_segments
        if (
            segment.metrics.active_confident
            and segment.metrics.boundary_supported is False
            and segment.metrics.end_ms - segment.metrics.start_ms
            < 60_000.0 / segment.metrics.bpm
        )
    )
    fragmentation_ratio = (
        round(fragmented_active_ms / duration.active_evidence_ms, 6)
        if duration.active_evidence_ms
        else 0.0
    )
    if (
        duration.active_evidence_ms == 0
        or duration.insufficient_active_ms == duration.active_evidence_ms
        or tempo_metrics.evidence_status == "INSUFFICIENT"
    ):
        confidence = "LOW"
    elif duration.insufficient_active_ms:
        confidence = "MEDIUM"
    else:
        confidence = "HIGH"
    return TimingCandidateEvidence(
        epoch=epoch,
        mode=mode,
        structurally_valid=structurally_valid,
        active_evidence_ms=duration.active_evidence_ms,
        supported_active_ms=duration.supported_active_ms,
        contradicted_active_ms=duration.contradicted_active_ms,
        insufficient_active_ms=duration.insufficient_active_ms,
        quiet_ms=duration.quiet_ms,
        supported_ratio=duration.supported_ratio,
        contradicted_ratio=duration.contradicted_ratio,
        insufficient_ratio=duration.insufficient_ratio,
        beat_consensus_by_level=consensus,
        best_metrical_level=best_level,
        unsupported_transition_rate_per_minute=unsupported_rate,
        fragmentation_active_ratio=fragmentation_ratio,
        boundary_risk_count=boundary_risk_count,
        confidence=confidence,
        integrity=(
            integrity
            if integrity is not None
            else TimingIntegrityAssessment(
                status=TimingIntegrityStatus.HEALTHY,
                reasons=(),
                islands=(),
            )
        ),
    )


_REASONS = (
    "STRUCTURALLY_VALID",
    "BETTER_TIMING_INTEGRITY",
    "LOWER_CONTRADICTED_ACTIVE_RATIO",
    "HIGHER_EXTERNAL_BEAT_F1",
    "HIGHER_MULTI_METRIC_BEAT_CONSENSUS",
    "HIGHER_SUPPORTED_ACTIVE_RATIO",
    "LOWER_UNSUPPORTED_TRANSITION_RATE",
    "LOWER_FRAGMENTATION_ACTIVE_RATIO",
    "LOWER_RESNAP_BOUNDARY_RISK",
    "LOWER_CANDIDATE_COST",
)


def _rank(candidate: TimingCandidateEvidence) -> tuple[object, ...]:
    unsupported_rate = candidate.unsupported_transition_rate_per_minute
    candidate_cost = {
        "STANDARD": 0,
        "SUPER_TIMING": 1,
        "BEAT_THIS_FALLBACK": 2,
    }[candidate.mode]
    integrity_rank = {
        TimingIntegrityStatus.HEALTHY: 0,
        TimingIntegrityStatus.NEEDS_CORROBORATION: 1,
        TimingIntegrityStatus.DAMAGED: 2,
    }[candidate.integrity.status]
    return (
        0 if candidate.structurally_valid else 1,
        integrity_rank,
        candidate.contradicted_ratio,
        -(candidate.best_external_beat_f1 or 0.0),
        -candidate.best_beat_consensus,
        -candidate.supported_ratio,
        unsupported_rate if unsupported_rate is not None else float("inf"),
        candidate.fragmentation_active_ratio,
        candidate.boundary_risk_count,
        candidate_cost,
    )


def timing_candidates_need_external_corroboration(
    candidates: tuple[TimingCandidateEvidence, ...],
) -> bool:
    """Use the optional model only for a measured close internal decision."""
    if len(candidates) < 2:
        return False
    if any(candidate.mode == "BEAT_THIS_FALLBACK" for candidate in candidates):
        return False
    ordered = sorted(candidates, key=_rank)
    first, second = ordered[:2]
    if not first.structurally_valid or not second.structurally_valid:
        return False
    if first.integrity.status is not second.integrity.status:
        return False
    if abs(first.contradicted_ratio - second.contradicted_ratio) > 1e-6:
        return False
    # In the latest 33-song corpus only two of four viable Standard/Super
    # pairs fell within this 1 percentage-point band.  It bounds optional
    # inference to actual ties instead of running a second model on every song.
    return abs(first.best_beat_consensus - second.best_beat_consensus) <= 0.01


def select_timing_candidate(
    candidates: tuple[TimingCandidateEvidence, ...],
) -> TimingCandidateSelection:
    if not candidates:
        raise ValueError("at least one timing candidate is required")
    selected_index = min(range(len(candidates)), key=lambda index: _rank(candidates[index]))
    if len(candidates) == 1:
        reason = "ONLY_STRUCTURALLY_VALID_CANDIDATE"
    else:
        selected_rank = _rank(candidates[selected_index])
        alternative_ranks = [
            _rank(candidate)
            for index, candidate in enumerate(candidates)
            if index != selected_index
        ]
        reason = "DETERMINISTIC_TIE_BREAK"
        for position, label in enumerate(_REASONS):
            if any(selected_rank[position] != rank[position] for rank in alternative_ranks):
                reason = label
                break
    return TimingCandidateSelection(
        selected_index=selected_index,
        reason=reason,
        candidates=candidates,
    )
