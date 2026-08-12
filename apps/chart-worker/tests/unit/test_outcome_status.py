from chart_worker.validation.outcome_status import (
    failure_outcome_status,
    success_outcome_status,
)


def test_complete_review_is_not_strictly_publishable():
    status = success_outcome_status(
        expected_slots=12,
        generated_slots=12,
        requires_review=True,
    )

    assert status.to_report() == {
        "execution": "SUCCEEDED",
        "completeness": "COMPLETE",
        "quality": "REVIEW",
        "failureCategory": "NONE",
        "publishableStrict": False,
    }


def test_partial_pass_is_not_strictly_publishable():
    status = success_outcome_status(
        expected_slots=12,
        generated_slots=11,
        requires_review=False,
    )

    assert status.completeness == "PARTIAL"
    assert status.quality == "PASS"
    assert status.publishable_strict is False


def test_failure_keeps_execution_and_failure_category_orthogonal():
    status = failure_outcome_status(category="VALIDATION")

    assert status.execution == "FAILED"
    assert status.completeness == "EMPTY"
    assert status.quality == "UNKNOWN"
    assert status.failure_category == "VALIDATION"
    assert status.publishable_strict is False
