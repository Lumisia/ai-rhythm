"""Monotonic quality contract for post-selection candidate replacement.

Recovery stages may optimize different family-level objectives, but none may
silently weaken a chart-level gate that an earlier selection already preserved.
This module deliberately knows nothing about stage-private candidate objects.
"""

from dataclasses import dataclass

from chart_worker.validation.quality_gate import GateAction

_ACTION_RANK = {
    GateAction.PASS: 0,
    GateAction.REVIEW: 1,
    GateAction.RETRY_MAP: 2,
}


@dataclass(frozen=True, slots=True)
class CandidateQualitySnapshot:
    provenance: str
    overall_action: GateAction
    retry_axes: tuple[str, ...]
    review_axes: tuple[str, ...]
    structure_pass: bool
    timing_identity_pass: bool
    song_bounds_action: GateAction

    def to_report(self) -> dict[str, object]:
        return {
            "provenance": self.provenance,
            "overallAction": self.overall_action.value,
            "retryAxes": list(self.retry_axes),
            "reviewAxes": list(self.review_axes),
            "structurePass": self.structure_pass,
            "timingIdentityPass": self.timing_identity_pass,
            "songBoundsAction": self.song_bounds_action.value,
        }


@dataclass(frozen=True, slots=True)
class CandidateReplacementDecision:
    accepted: bool
    stage: str
    reasons: tuple[str, ...]
    current: CandidateQualitySnapshot
    challenger: CandidateQualitySnapshot

    def to_report(self) -> dict[str, object]:
        return {
            "accepted": self.accepted,
            "stage": self.stage,
            "reasons": list(self.reasons),
            "current": self.current.to_report(),
            "challenger": self.challenger.to_report(),
        }


def decide_candidate_replacement(
    current: CandidateQualitySnapshot,
    challenger: CandidateQualitySnapshot,
    *,
    stage: str,
    objective_improved: bool,
) -> CandidateReplacementDecision:
    """Accept only a stage improvement that preserves prior gate quality."""
    reasons: list[str] = []
    if challenger.provenance == "RAW_UNVERIFIED":
        reasons.append("RAW_UNVERIFIED_CHALLENGER")
    if not challenger.structure_pass:
        reasons.append("STRUCTURE_NOT_PASS")
    if not challenger.timing_identity_pass:
        reasons.append("TIMING_IDENTITY_NOT_PASS")

    new_retry_axes = sorted(set(challenger.retry_axes) - set(current.retry_axes))
    reasons.extend(f"NEW_RETRY_AXIS:{axis}" for axis in new_retry_axes)
    new_review_axes = sorted(set(challenger.review_axes) - set(current.review_axes))
    reasons.extend(f"NEW_REVIEW_AXIS:{axis}" for axis in new_review_axes)

    if _ACTION_RANK[challenger.song_bounds_action] > _ACTION_RANK[
        current.song_bounds_action
    ]:
        reasons.append(
            "SONG_BOUNDS_DOWNGRADE:"
            f"{current.song_bounds_action.value}->{challenger.song_bounds_action.value}"
        )
    if _ACTION_RANK[challenger.overall_action] > _ACTION_RANK[
        current.overall_action
    ]:
        reasons.append(
            "OVERALL_ACTION_DOWNGRADE:"
            f"{current.overall_action.value}->{challenger.overall_action.value}"
        )
    if not objective_improved:
        reasons.append("STAGE_OBJECTIVE_NOT_IMPROVED")

    accepted = not reasons
    return CandidateReplacementDecision(
        accepted=accepted,
        stage=stage,
        reasons=(
            ("OBJECTIVE_IMPROVED_WITHOUT_QUALITY_DOWNGRADE",)
            if accepted
            else tuple(reasons)
        ),
        current=current,
        challenger=challenger,
    )
