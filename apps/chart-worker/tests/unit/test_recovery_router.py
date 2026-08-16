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

    plan = plan_recoveries((timing, missing), available_generation_ms=180_000)

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

    plan = plan_recoveries((full_map, partial), available_generation_ms=180_000)

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

    forward = plan_recoveries((seven_key, four_key), available_generation_ms=10_000)
    reverse = plan_recoveries((four_key, seven_key), available_generation_ms=10_000)

    assert forward.selected == reverse.selected == (four_key,)


def test_router_rejects_duplicate_request_ids_and_negative_capacity():
    request = _request(
        "duplicate",
        kind=RecoveryKind.INTRO,
        priority=RecoveryPriority.COMPLETENESS_BLOCKING,
        estimated_generation_ms=10_000,
    )

    with pytest.raises(ValueError, match="duplicate recovery request id"):
        plan_recoveries((request, request), available_generation_ms=10_000)
    with pytest.raises(ValueError, match="available_generation_ms"):
        plan_recoveries((request,), available_generation_ms=-1)


def test_router_can_admit_multiple_small_partial_repairs_within_song_work_budget():
    requests = (
        _request(
            "partial-20s",
            kind=RecoveryKind.PARTIAL_REMAP,
            priority=RecoveryPriority.COMPLETENESS_BLOCKING,
            estimated_generation_ms=20_000,
        ),
        _request(
            "partial-30s",
            kind=RecoveryKind.PARTIAL_REMAP,
            priority=RecoveryPriority.COMPLETENESS_BLOCKING,
            estimated_generation_ms=30_000,
            key_mode=6,
        ),
        _request(
            "whole-song",
            kind=RecoveryKind.INTRO,
            priority=RecoveryPriority.COMPLETENESS_BLOCKING,
            estimated_generation_ms=180_000,
            key_mode=7,
        ),
    )

    plan = plan_recoveries(requests, available_generation_ms=180_000)

    assert [request.request_id for request in plan.selected] == [
        "partial-20s",
        "partial-30s",
    ]
    assert [request.request_id for request in plan.deferred] == ["whole-song"]
    assert plan.selected_generation_ms == 50_000
    assert plan.remaining_generation_ms == 130_000
    report = plan.to_report()
    assert report["policyVersion"] == "RECOVERY_WORK_BUDGET_V2"
    assert report["availableGenerationMs"] == 180_000
    assert report["remainingGenerationMs"] == 130_000
    assert report["requests"][-1]["decision"] == "DEFERRED_WORK_BUDGET"


def test_router_does_not_spend_on_quality_while_completeness_is_deferred():
    completeness = _request(
        "missing-100s",
        kind=RecoveryKind.PARTIAL_REMAP,
        priority=RecoveryPriority.COMPLETENESS_BLOCKING,
        estimated_generation_ms=100_000,
    )
    second_completeness = _request(
        "missing-another-100s",
        kind=RecoveryKind.PARTIAL_REMAP,
        priority=RecoveryPriority.COMPLETENESS_BLOCKING,
        estimated_generation_ms=100_000,
        key_mode=6,
    )
    quality = _request(
        "quality-40s",
        kind=RecoveryKind.TIMING_FAMILY,
        priority=RecoveryPriority.QUALITY_BLOCKING,
        estimated_generation_ms=40_000,
        key_mode=7,
    )

    plan = plan_recoveries(
        (quality, second_completeness, completeness),
        available_generation_ms=150_000,
    )

    assert [request.request_id for request in plan.selected] == ["missing-100s"]
    assert {request.request_id for request in plan.deferred} == {
        "missing-another-100s",
        "quality-40s",
    }


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
