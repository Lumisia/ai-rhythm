import pytest

from chart_worker.generation.generation_control import RecoveryKind
from chart_worker.generation.recovery_router import (
    RecoveryPriority,
    RecoveryRequest,
    intro_phrase_recovery_request,
    partial_remap_recovery_request,
    plan_recoveries,
    timing_family_recovery_request,
)


def _request(
    request_id: str,
    *,
    kind: RecoveryKind,
    priority: RecoveryPriority,
    estimated_generation_ms: int,
    key_mode: int = 4,
    difficulty: str = "EXPERT",
) -> RecoveryRequest:
    return RecoveryRequest(
        request_id=request_id,
        kind=kind,
        key_mode=key_mode,
        difficulty=difficulty,
        priority=priority,
        estimated_generation_ms=estimated_generation_ms,
        reason=request_id,
    )


def test_router_chooses_completeness_before_code_or_input_order():
    timing = _request(
        "timing-first-in-code",
        kind=RecoveryKind.TIMING_FAMILY,
        priority=RecoveryPriority.QUALITY_BLOCKING,
        estimated_generation_ms=180_000,
    )
    missing = _request(
        "missing-chart",
        kind=RecoveryKind.PARTIAL_REMAP,
        priority=RecoveryPriority.COMPLETENESS_BLOCKING,
        estimated_generation_ms=20_000,
        difficulty="EASY",
    )

    plan = plan_recoveries((timing, missing), available_slots=1)

    assert plan.selected == (missing,)
    assert plan.deferred == (timing,)
    assert plan.to_report()["selectedRequestIds"] == ["missing-chart"]


def test_router_uses_smaller_generation_scope_for_equal_priority():
    full_map = _request(
        "intro-full-map",
        kind=RecoveryKind.INTRO,
        priority=RecoveryPriority.COMPLETENESS_BLOCKING,
        estimated_generation_ms=180_000,
    )
    partial = _request(
        "partial-window",
        kind=RecoveryKind.PARTIAL_REMAP,
        priority=RecoveryPriority.COMPLETENESS_BLOCKING,
        estimated_generation_ms=12_000,
        difficulty="NORMAL",
    )

    plan = plan_recoveries((full_map, partial), available_slots=1)

    assert plan.selected == (partial,)
    assert plan.deferred == (full_map,)


def test_router_is_deterministic_when_priority_and_cost_tie():
    seven_key = _request(
        "7k",
        kind=RecoveryKind.INTRO,
        priority=RecoveryPriority.COMPLETENESS_BLOCKING,
        estimated_generation_ms=10_000,
        key_mode=7,
    )
    four_key = _request(
        "4k",
        kind=RecoveryKind.INTRO,
        priority=RecoveryPriority.COMPLETENESS_BLOCKING,
        estimated_generation_ms=10_000,
        key_mode=4,
    )

    forward = plan_recoveries((seven_key, four_key), available_slots=1)
    reverse = plan_recoveries((four_key, seven_key), available_slots=1)

    assert forward.selected == reverse.selected == (four_key,)


def test_router_rejects_duplicate_request_ids_and_negative_capacity():
    request = _request(
        "duplicate",
        kind=RecoveryKind.INTRO,
        priority=RecoveryPriority.COMPLETENESS_BLOCKING,
        estimated_generation_ms=10_000,
    )

    with pytest.raises(ValueError, match="duplicate recovery request id"):
        plan_recoveries((request, request), available_slots=1)
    with pytest.raises(ValueError, match="available_slots"):
        plan_recoveries((request,), available_slots=-1)


def test_domain_request_builders_make_policy_explicit():
    partial = partial_remap_recovery_request(
        key_mode=6,
        difficulty="NORMAL",
        window_ms=12_000,
    )
    intro = intro_phrase_recovery_request(
        key_mode=4,
        song_duration_ms=180_000,
    )
    timing = timing_family_recovery_request(
        key_mode=7,
        difficulty="HARD",
        song_duration_ms=180_000,
    )

    assert partial.priority is RecoveryPriority.COMPLETENESS_BLOCKING
    assert intro.priority is RecoveryPriority.COMPLETENESS_BLOCKING
    assert timing.priority is RecoveryPriority.QUALITY_BLOCKING
    assert partial.estimated_generation_ms == 12_000
    assert intro.estimated_generation_ms == timing.estimated_generation_ms == 180_000
