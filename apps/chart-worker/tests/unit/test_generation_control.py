import pytest

from chart_worker.generation.generation_control import (
    AdditionalInferenceBudget,
    AttemptBudgetState,
    RecoveryKind,
    RecoveryRouterState,
)


def test_attempt_budget_reserves_monotonic_attempts_and_separates_failure_classes():
    budget = AttemptBudgetState(max_quality_attempts=3, max_crash_attempts=2, max_total_attempts=4)

    assert budget.reserve_attempt(seed=10) == 1
    budget.record_quality_attempt()
    assert budget.reserve_attempt(seed=22) == 2
    budget.record_crash_attempt()

    assert budget.next_attempt == 3
    assert budget.attempted_seeds == [10, 22]
    assert budget.quality_attempts == 1
    assert budget.crash_attempts == 1
    assert budget.can_attempt is True


def test_attempt_budget_rejects_reservation_after_any_hard_limit():
    budget = AttemptBudgetState(max_quality_attempts=1, max_crash_attempts=2, max_total_attempts=4)
    budget.record_quality_attempt()

    with pytest.raises(RuntimeError, match="attempt budget exhausted"):
        budget.reserve_attempt(seed=10)


def test_additional_recovery_reservation_is_independent_of_primary_quality_limit():
    budget = AttemptBudgetState(
        max_quality_attempts=1,
        max_crash_attempts=1,
        max_total_attempts=1,
        next_attempt=2,
        quality_attempts=1,
    )

    assert budget.reserve_additional_attempt(seed=22) == 2
    assert budget.next_attempt == 3
    assert budget.attempted_seeds == [22]


def test_recovery_router_tracks_each_recovery_kind_independently():
    router = RecoveryRouterState()

    assert router.claim(RecoveryKind.PARTIAL_REMAP) is True
    assert router.claim(RecoveryKind.PARTIAL_REMAP) is False
    assert router.claim(RecoveryKind.INTRO) is True
    assert router.was_attempted(RecoveryKind.TIMING_FAMILY) is False


def test_additional_inference_budget_reports_remaining_capacity():
    budget = AdditionalInferenceBudget(limit=1)

    assert budget.remaining == 1
    assert budget.consume() is True
    assert budget.remaining == 0
    assert budget.consume() is False
