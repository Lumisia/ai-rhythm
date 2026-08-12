import pytest

from chart_worker.validation.outcome_status import (
    OutcomeStatus,
    failure_outcome_status,
    success_outcome_status,
)
from chart_worker.validation.publication_policy import (
    assess_boundary_publication,
    decide_publication,
)


def test_provisional_boundary_assessment_is_an_explicit_strict_blocker():
    assessment = assess_boundary_publication(
        policy_state="PROVISIONAL",
        confidence="UNKNOWN",
    )

    assert assessment.strict_blockers == ("BOUNDARY_POLICY_UNCALIBRATED",)
    assert assessment.to_report() == {
        "version": "boundary-publication-assessment-v1",
        "evidenceStatus": "AVAILABLE",
        "policyState": "PROVISIONAL",
        "confidence": "UNKNOWN",
        "strictBlockers": ["BOUNDARY_POLICY_UNCALIBRATED"],
    }


def test_missing_boundary_evidence_is_not_silently_treated_as_calibrated():
    assessment = assess_boundary_publication(
        policy_state=None,
        confidence=None,
    )

    assert assessment.to_report() == {
        "version": "boundary-publication-assessment-v1",
        "evidenceStatus": "UNAVAILABLE",
        "policyState": None,
        "confidence": None,
        "strictBlockers": ["BOUNDARY_POLICY_UNCALIBRATED"],
    }


def test_only_validated_calibration_removes_boundary_strict_blocker():
    assessment = assess_boundary_publication(
        policy_state="CALIBRATED",
        confidence="VALIDATED",
    )

    assert assessment.strict_blockers == ()


@pytest.mark.parametrize(
    ("policy_state", "confidence"),
    [
        ("CALIBRATED", "UNKNOWN"),
        ("PROVISIONAL", "VALIDATED"),
        (None, "UNKNOWN"),
        ("PROVISIONAL", None),
    ],
)
def test_boundary_assessment_rejects_inconsistent_state_pairs(
    policy_state: str | None,
    confidence: str | None,
):
    with pytest.raises(ValueError, match="boundary policy state"):
        assess_boundary_publication(
            policy_state=policy_state,
            confidence=confidence,
        )


def test_complete_pass_is_allowed_for_production():
    outcome = success_outcome_status(
        expected_slots=12,
        generated_slots=12,
        requires_review=False,
    )

    decision = decide_publication(
        outcome=outcome,
        published_slots=12,
        expected_slots=12,
    )

    assert decision.to_report() == {
        "policyVersion": "PUBLICATION_POLICY_V2",
        "decision": "ALLOW_PRODUCTION",
        "reasonCodes": [],
    }


def test_uncalibrated_boundary_blocks_production_without_rejecting_playtest():
    outcome = success_outcome_status(
        expected_slots=12,
        generated_slots=12,
        requires_review=False,
    )

    decision = decide_publication(
        outcome=outcome,
        published_slots=12,
        expected_slots=12,
        strict_blockers=("BOUNDARY_POLICY_UNCALIBRATED",),
    )

    assert decision.to_report() == {
        "policyVersion": "PUBLICATION_POLICY_V2",
        "decision": "PLAYTEST_ONLY",
        "reasonCodes": ["BOUNDARY_POLICY_UNCALIBRATED"],
    }


def test_complete_review_is_playtest_only_with_literal_reasons():
    outcome = success_outcome_status(
        expected_slots=12,
        generated_slots=12,
        requires_review=True,
    )

    decision = decide_publication(
        outcome=outcome,
        published_slots=12,
        expected_slots=12,
    )

    assert decision.decision == "PLAYTEST_ONLY"
    assert decision.reason_codes == (
        "QUALITY_REVIEW_REQUIRED",
        "STRICT_OUTCOME_FALSE",
    )


def test_partial_pass_is_playtest_only_with_literal_reasons():
    outcome = success_outcome_status(
        expected_slots=12,
        generated_slots=11,
        requires_review=False,
    )

    decision = decide_publication(
        outcome=outcome,
        published_slots=11,
        expected_slots=12,
    )

    assert decision.decision == "PLAYTEST_ONLY"
    assert decision.reason_codes == (
        "INCOMPLETE_CHART_SET",
        "STRICT_OUTCOME_FALSE",
    )


def test_failed_execution_is_rejected_instead_of_becoming_playtest_only():
    outcome = failure_outcome_status(category="VALIDATION")

    decision = decide_publication(
        outcome=outcome,
        published_slots=0,
        expected_slots=12,
    )

    assert decision.to_report() == {
        "policyVersion": "PUBLICATION_POLICY_V2",
        "decision": "REJECTED",
        "reasonCodes": [
            "EXECUTION_FAILED",
            "INCOMPLETE_CHART_SET",
            "QUALITY_UNKNOWN",
            "STRICT_OUTCOME_FALSE",
        ],
    }


@pytest.mark.parametrize(
    ("published_slots", "expected_slots"),
    [(-1, 12), (13, 12), (0, 0)],
)
def test_publication_policy_rejects_invalid_slot_counts(
    published_slots: int,
    expected_slots: int,
):
    outcome = success_outcome_status(
        expected_slots=12,
        generated_slots=12,
        requires_review=False,
    )

    with pytest.raises(ValueError, match="slots"):
        decide_publication(
            outcome=outcome,
            published_slots=published_slots,
            expected_slots=expected_slots,
        )


def test_publication_policy_rejects_outcome_that_disagrees_with_slot_count():
    outcome = OutcomeStatus(
        execution="SUCCEEDED",
        completeness="COMPLETE",
        quality="PASS",
        failure_category="NONE",
        publishable_strict=True,
    )

    with pytest.raises(ValueError, match="completeness"):
        decide_publication(
            outcome=outcome,
            published_slots=11,
            expected_slots=12,
        )


def test_publication_policy_rejects_internally_inconsistent_strict_flag():
    outcome = OutcomeStatus(
        execution="SUCCEEDED",
        completeness="COMPLETE",
        quality="PASS",
        failure_category="NONE",
        publishable_strict=False,
    )

    with pytest.raises(ValueError, match="publishable_strict"):
        decide_publication(
            outcome=outcome,
            published_slots=12,
            expected_slots=12,
        )


@pytest.mark.parametrize(
    "outcome",
    [
        OutcomeStatus(
            execution="SUCCEEDED",
            completeness="COMPLETE",
            quality="PASS",
            failure_category="VALIDATION",
            publishable_strict=True,
        ),
        OutcomeStatus(
            execution="FAILED",
            completeness="EMPTY",
            quality="UNKNOWN",
            failure_category="NONE",
            publishable_strict=False,
        ),
    ],
)
def test_publication_policy_rejects_failure_category_inconsistent_with_execution(
    outcome: OutcomeStatus,
):
    with pytest.raises(ValueError, match="failure_category"):
        decide_publication(
            outcome=outcome,
            published_slots=12 if outcome.execution == "SUCCEEDED" else 0,
            expected_slots=12,
        )
