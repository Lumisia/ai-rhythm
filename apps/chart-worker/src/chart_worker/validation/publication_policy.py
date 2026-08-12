"""Pure publication decisions derived from an orthogonal run outcome."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from chart_worker.validation.outcome_status import CompletenessStatus, OutcomeStatus

PUBLICATION_POLICY_VERSION = "PUBLICATION_POLICY_V2"

PublicationDecisionName = Literal[
    "ALLOW_PRODUCTION",
    "PLAYTEST_ONLY",
    "REJECTED",
]
PublicationReasonCode = Literal[
    "BOUNDARY_POLICY_UNCALIBRATED",
    "EXECUTION_FAILED",
    "INCOMPLETE_CHART_SET",
    "QUALITY_REVIEW_REQUIRED",
    "QUALITY_REJECTED",
    "QUALITY_UNKNOWN",
    "STRICT_OUTCOME_FALSE",
]
PublicationStrictBlocker = Literal["BOUNDARY_POLICY_UNCALIBRATED"]
BoundaryPolicyState = Literal["PROVISIONAL", "CALIBRATED"]
BoundaryPolicyConfidence = Literal["UNKNOWN", "VALIDATED"]


@dataclass(frozen=True, slots=True)
class BoundaryPublicationAssessment:
    evidence_status: Literal["AVAILABLE", "UNAVAILABLE"]
    policy_state: BoundaryPolicyState | None
    confidence: BoundaryPolicyConfidence | None
    strict_blockers: tuple[PublicationStrictBlocker, ...]
    version: Literal["boundary-publication-assessment-v1"] = (
        "boundary-publication-assessment-v1"
    )

    def to_report(self) -> dict[str, object]:
        return {
            "version": self.version,
            "evidenceStatus": self.evidence_status,
            "policyState": self.policy_state,
            "confidence": self.confidence,
            "strictBlockers": list(self.strict_blockers),
        }


def assess_boundary_publication(
    *,
    policy_state: BoundaryPolicyState | None,
    confidence: BoundaryPolicyConfidence | None,
) -> BoundaryPublicationAssessment:
    """Translate boundary calibration evidence into publication blockers.

    Detector enforcement and production publication are separate decisions.
    Missing or provisional evidence remains explicit; only a validated,
    calibrated policy may remove the strict blocker.
    """

    if policy_state is None and confidence is None:
        return BoundaryPublicationAssessment(
            evidence_status="UNAVAILABLE",
            policy_state=None,
            confidence=None,
            strict_blockers=("BOUNDARY_POLICY_UNCALIBRATED",),
        )
    if (policy_state, confidence) == ("PROVISIONAL", "UNKNOWN"):
        return BoundaryPublicationAssessment(
            evidence_status="AVAILABLE",
            policy_state="PROVISIONAL",
            confidence="UNKNOWN",
            strict_blockers=("BOUNDARY_POLICY_UNCALIBRATED",),
        )
    if (policy_state, confidence) == ("CALIBRATED", "VALIDATED"):
        return BoundaryPublicationAssessment(
            evidence_status="AVAILABLE",
            policy_state="CALIBRATED",
            confidence="VALIDATED",
            strict_blockers=(),
        )
    raise ValueError("boundary policy state and confidence are inconsistent")


@dataclass(frozen=True, slots=True)
class PublicationDecision:
    decision: PublicationDecisionName
    reason_codes: tuple[PublicationReasonCode, ...]
    policy_version: str = PUBLICATION_POLICY_VERSION

    def to_report(self) -> dict[str, object]:
        return {
            "policyVersion": self.policy_version,
            "decision": self.decision,
            "reasonCodes": list(self.reason_codes),
        }


def _completeness_for_slots(*, published_slots: int, expected_slots: int) -> CompletenessStatus:
    if expected_slots <= 0:
        raise ValueError("expected_slots must be positive")
    if not 0 <= published_slots <= expected_slots:
        raise ValueError("published_slots must be within expected_slots")
    if published_slots == 0:
        return "EMPTY"
    if published_slots == expected_slots:
        return "COMPLETE"
    return "PARTIAL"


def decide_publication(
    *,
    outcome: OutcomeStatus,
    published_slots: int,
    expected_slots: int,
    strict_blockers: tuple[PublicationStrictBlocker, ...] = (),
) -> PublicationDecision:
    """Return the only publication decision valid for the supplied outcome."""

    actual_completeness = _completeness_for_slots(
        published_slots=published_slots,
        expected_slots=expected_slots,
    )
    if outcome.completeness != actual_completeness:
        raise ValueError("outcome completeness disagrees with published slot count")
    if (outcome.execution == "SUCCEEDED") != (outcome.failure_category == "NONE"):
        raise ValueError("outcome failure_category disagrees with execution")

    expected_strict = (
        outcome.execution == "SUCCEEDED"
        and outcome.completeness == "COMPLETE"
        and outcome.quality == "PASS"
    )
    if outcome.publishable_strict != expected_strict:
        raise ValueError("outcome publishable_strict is internally inconsistent")
    if strict_blockers != tuple(sorted(set(strict_blockers))):
        raise ValueError("publication strict blockers must be sorted and unique")

    reasons: list[PublicationReasonCode] = list(strict_blockers)
    if outcome.execution == "FAILED":
        reasons.append("EXECUTION_FAILED")
    if outcome.completeness != "COMPLETE":
        reasons.append("INCOMPLETE_CHART_SET")
    if outcome.quality == "REVIEW":
        reasons.append("QUALITY_REVIEW_REQUIRED")
    elif outcome.quality == "REJECTED":
        reasons.append("QUALITY_REJECTED")
    elif outcome.quality == "UNKNOWN":
        reasons.append("QUALITY_UNKNOWN")
    if not outcome.publishable_strict:
        reasons.append("STRICT_OUTCOME_FALSE")

    reasons = sorted(set(reasons))
    if not reasons:
        decision: PublicationDecisionName = "ALLOW_PRODUCTION"
    elif outcome.execution == "FAILED" or outcome.quality in {"REJECTED", "UNKNOWN"}:
        decision = "REJECTED"
    else:
        decision = "PLAYTEST_ONLY"
    return PublicationDecision(decision=decision, reason_codes=tuple(reasons))
