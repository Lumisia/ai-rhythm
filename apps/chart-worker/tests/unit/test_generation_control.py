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


def test_additional_inference_budget_limits_song_equivalent_work_not_just_calls():
    budget = AdditionalInferenceBudget(limit=14, work_limit_ms=180_000)

    assert budget.consume(20_000) is True
    assert budget.consume(30_000) is True
    assert budget.used == 2
    assert budget.used_work_ms == 50_000
    assert budget.remaining_work_ms == 130_000
    assert budget.consume(140_000) is False
    assert budget.used == 2
    assert budget.to_report() == {
        "policyVersion": "SONG_EQUIVALENT_WORK_V1",
        "callLimit": 14,
        "callsUsed": 2,
        "workLimitMs": 180_000,
        "workUsedMs": 50_000,
        "workRemainingMs": 130_000,
    }


@pytest.mark.parametrize("song_ms", (30_000, 180_000, 600_000))
def test_song_equivalent_budget_scales_with_song_duration(song_ms: int):
    budget = AdditionalInferenceBudget(limit=14, work_limit_ms=song_ms)

    assert budget.consume(song_ms) is True
    assert budget.consume(1) is False


@pytest.mark.parametrize("work_limit_ms", (True, 1.5, -1))
def test_song_equivalent_budget_rejects_invalid_work_limits(work_limit_ms):
    with pytest.raises(ValueError, match="work_limit_ms"):
        AdditionalInferenceBudget(limit=14, work_limit_ms=work_limit_ms)
