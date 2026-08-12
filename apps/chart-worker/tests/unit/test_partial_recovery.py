from types import SimpleNamespace

from chart_worker.generation.generation_control import RecoveryRouterState
from chart_worker.generation.partial_recovery import plan_partial_repair


def test_partial_repair_planner_does_not_claim_or_infer_without_a_source():
    state = SimpleNamespace(
        recovery=RecoveryRouterState(),
        candidates=SimpleNamespace(partial_sources=[]),
    )
    authority = SimpleNamespace(bpm_events=())

    assert (
        plan_partial_repair(
            state,
            authority=authority,
            duration_ms=60_000,
        )
        is None
    )
    assert state.recovery.attempted == set()
