from chart_worker.generation.candidate_state import VariantState
from chart_worker.generation.generation_control import (
    MAX_CRASH_ATTEMPTS,
    MAX_TOTAL_ATTEMPTS,
    MAX_VARIANT_ATTEMPTS,
)


def test_variant_state_owns_isolated_candidate_and_budget_state() -> None:
    first = VariantState(key_mode=4, difficulty="HARD", flat_index=0)
    second = VariantState(key_mode=4, difficulty="EXPERT", flat_index=1)

    assert first.budget.max_quality_attempts == MAX_VARIANT_ATTEMPTS
    assert first.budget.max_crash_attempts == MAX_CRASH_ATTEMPTS
    assert first.budget.max_total_attempts == MAX_TOTAL_ATTEMPTS
    assert first.budget_left is True

    first.budget.reserve_attempt(seed=17)
    first.candidates.reject(object())

    assert first.budget.attempted_seeds == [17]
    assert second.budget.attempted_seeds == []
    assert len(first.candidates.raw_rejected) == 1
    assert second.candidates.raw_rejected == ()
