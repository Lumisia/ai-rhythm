"""Conservative decision policy for diagnostic timing evidence."""

from dataclasses import dataclass
from enum import StrEnum

from chart_worker.analysis.grid_alignment import TempoCandidateMetrics


class TimingAuthorityAction(StrEnum):
    PASS = "PASS"
    RETRY_TIMING = "RETRY_TIMING"
    REVIEW = "REVIEW"


@dataclass(frozen=True, slots=True)
class TimingAuthorityReview:
    action: TimingAuthorityAction
    reasons: tuple[str, ...]


def review_timing_authority(metrics: TempoCandidateMetrics) -> TimingAuthorityReview:
    """Only retry when both independent axes strongly corroborate one alternative."""
    if metrics.evidence_status == "INSUFFICIENT":
        return TimingAuthorityReview(
            TimingAuthorityAction.REVIEW, ("INSUFFICIENT_TEMPO_EVIDENCE",)
        )

    if (
        metrics.pulse_best_alternative is not None
        and metrics.periodicity_best_alternative is not None
        and metrics.pulse_best_alternative != metrics.periodicity_best_alternative
    ):
        return TimingAuthorityReview(
            TimingAuthorityAction.REVIEW, ("TEMPO_EVIDENCE_DISAGREES",)
        )

    pulse_strong = metrics.pulse_alternative_margin >= 0.15
    periodicity_strong = metrics.periodicity_margin >= 0.10
    if pulse_strong or periodicity_strong:
        if (
            pulse_strong
            and periodicity_strong
            and metrics.pulse_best_alternative is not None
            and metrics.pulse_best_alternative == metrics.periodicity_best_alternative
            and metrics.base_pulse_support < 0.55
        ):
            return TimingAuthorityReview(
                TimingAuthorityAction.RETRY_TIMING,
                (f"STRONG_{metrics.pulse_best_alternative}_TEMPO_ALTERNATIVE",),
            )
        return TimingAuthorityReview(
            TimingAuthorityAction.REVIEW, ("TEMPO_EVIDENCE_DISAGREES",)
        )

    if metrics.base_pulse_support < 0.55:
        return TimingAuthorityReview(
            TimingAuthorityAction.REVIEW, ("WEAK_BASE_TEMPO_SUPPORT",)
        )

    if metrics.pulse_alternative_margin < 0.05 and metrics.base_pulse_support >= 0.55:
        return TimingAuthorityReview(
            TimingAuthorityAction.PASS, ("TEMPO_CANDIDATE_AMBIGUOUS",)
        )
    return TimingAuthorityReview(TimingAuthorityAction.PASS, ())
