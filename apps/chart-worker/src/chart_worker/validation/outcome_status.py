"""Orthogonal execution, completeness, quality, and failure reporting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

ExecutionStatus = Literal["SUCCEEDED", "FAILED"]
CompletenessStatus = Literal["COMPLETE", "PARTIAL", "EMPTY"]
QualityStatus = Literal["PASS", "REVIEW", "REJECTED", "UNKNOWN"]
FailureCategory = Literal["NONE", "INFRA", "GENERATION", "VALIDATION", "POLICY"]


@dataclass(frozen=True, slots=True)
class OutcomeStatus:
    execution: ExecutionStatus
    completeness: CompletenessStatus
    quality: QualityStatus
    failure_category: FailureCategory
    publishable_strict: bool

    def to_report(self) -> dict[str, object]:
        return {
            "execution": self.execution,
            "completeness": self.completeness,
            "quality": self.quality,
            "failureCategory": self.failure_category,
            "publishableStrict": self.publishable_strict,
        }


def success_outcome_status(
    *,
    expected_slots: int,
    generated_slots: int,
    requires_review: bool,
) -> OutcomeStatus:
    if expected_slots <= 0:
        raise ValueError("expected_slots must be positive")
    if not 0 <= generated_slots <= expected_slots:
        raise ValueError("generated_slots must be within expected_slots")
    if generated_slots == 0:
        completeness: CompletenessStatus = "EMPTY"
    elif generated_slots == expected_slots:
        completeness = "COMPLETE"
    else:
        completeness = "PARTIAL"
    quality: QualityStatus = "REVIEW" if requires_review else "PASS"
    return OutcomeStatus(
        execution="SUCCEEDED",
        completeness=completeness,
        quality=quality,
        failure_category="NONE",
        publishable_strict=(completeness == "COMPLETE" and quality == "PASS"),
    )


def failure_outcome_status(*, category: FailureCategory) -> OutcomeStatus:
    if category == "NONE":
        raise ValueError("failed outcome requires a failure category")
    return OutcomeStatus(
        execution="FAILED",
        completeness="EMPTY",
        quality="UNKNOWN",
        failure_category=category,
        publishable_strict=False,
    )
