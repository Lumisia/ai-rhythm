import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from chart_worker.errors import ErrorCode, WorkerError, disposition_of
from chart_worker.generation import partial_recovery
from chart_worker.generation.generation_control import (
    RecoveryKind,
    RecoveryRouterState,
)
from chart_worker.generation.params import GenerationRequest
from chart_worker.generation.partial_recovery import (
    PartialRepairPlan,
    execute_partial_repair,
    plan_partial_repair,
)
from chart_worker.generation.partial_remap import PartialRemapWindow


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


def test_tail_full_length_block_does_not_disable_local_partial_repair(monkeypatch):
    source = SimpleNamespace(
        generated=SimpleNamespace(notes=[]),
        acceptance=SimpleNamespace(timing=SimpleNamespace(coverage_gaps=[])),
        attempt=1,
        seed=0,
    )
    state = SimpleNamespace(
        key_mode=4,
        difficulty="EASY",
        full_length_retry_blocked_by={"signature": "tail:500:11"},
        recovery=RecoveryRouterState(),
        candidates=SimpleNamespace(partial_sources=[source]),
    )
    monkeypatch.setattr(
        partial_recovery,
        "build_partial_remap_window",
        lambda *args, **kwargs: PartialRemapWindow(8_000, 12_000),
    )

    plan = plan_partial_repair(
        state,
        authority=SimpleNamespace(bpm_events=()),
        duration_ms=60_000,
    )

    assert plan is not None
    assert plan.source is source
    assert plan.window == PartialRemapWindow(8_000, 12_000)
    assert state.recovery.attempted == set()


PARTIAL_REJOIN_CONTEXT = {
    "scope": "REFERENCE",
    "partialStartMs": 8_000,
    "partialEndMs": 12_000,
    "validationError": "reference Mania stream is invalid: unclosed HOLD lanes: [0]",
    "earliestGeneratedSourceWindowId": None,
}


def _executing_plan(tmp_path: Path):
    request = GenerationRequest(
        audio_path=tmp_path / "game.flac",
        key_mode=4,
        difficulty="EASY",
        duration_ms=60_000,
    )
    source = SimpleNamespace(
        osu_text="osu file format v14\n",
        request=request,
        attempt=1,
        seed=0,
    )
    state = SimpleNamespace(
        key_mode=4,
        difficulty="EASY",
        flat_index=0,
        recovery=RecoveryRouterState(),
        attempt_errors=[],
        attempt_evidence=[],
        full_length_retry_blocked_by=None,
    )
    plan = PartialRepairPlan(
        state=state,
        source=source,
        window=PartialRemapWindow(8_000, 12_000),
        request=request,
    )
    return plan, state, {
        "prepared": SimpleNamespace(normalized=SimpleNamespace(duration_ms=60_000)),
        "authority": SimpleNamespace(bpm_events=()),
        "onset_analysis": SimpleNamespace(),
        "run_dir": tmp_path,
        "generator": SimpleNamespace(),
        "base_seed": 0,
        "authority_epoch": 1,
        "evaluate_candidate": lambda *a, **k: pytest.fail("must not evaluate"),
        "serialize_candidate": lambda *a, **k: pytest.fail("must not serialize"),
        "intro_anchor_covered": lambda *a, **k: True,
        "should_retry": lambda error: disposition_of(error.code).name == "RETRYABLE",
    }


def test_typed_partial_rejoin_failure_declines_the_repair_without_killing_the_run(
    monkeypatch, tmp_path
):
    """PARTIAL_REMAP 은 선택적 복구다. 실패해도 기존 후보까지 버리면 안 된다."""
    plan, state, kwargs = _executing_plan(tmp_path)
    failure = WorkerError(
        ErrorCode.MANIA_PARTIAL_REJOIN_INVALID,
        "Mapperatorinator could not rejoin the partial Mania interval",
        context=PARTIAL_REJOIN_CONTEXT,
    )

    def raise_typed(*_args, **_kwargs):
        raise failure

    monkeypatch.setattr(partial_recovery, "run_inference_with_journal", raise_typed)

    assert execute_partial_repair(plan, **kwargs) is None
    assert state.recovery.attempted == {RecoveryKind.PARTIAL_REMAP}
    recorded = [json.loads(entry) for entry in state.attempt_errors]
    assert [entry["code"] for entry in recorded] == [
        ErrorCode.MANIA_PARTIAL_REJOIN_INVALID.value
    ]
    assert recorded[0]["context"] == PARTIAL_REJOIN_CONTEXT


def test_typed_partial_rejoin_failure_does_not_block_the_full_length_retry(
    monkeypatch, tmp_path
):
    """참조가 깨진 것과 새 seed 가 같은 실패를 낸다는 것은 다른 주장이다."""
    plan, state, kwargs = _executing_plan(tmp_path)

    def raise_typed(*_args, **_kwargs):
        raise WorkerError(
            ErrorCode.MANIA_PARTIAL_REJOIN_INVALID,
            "Mapperatorinator could not rejoin the partial Mania interval",
            context=PARTIAL_REJOIN_CONTEXT,
        )

    monkeypatch.setattr(partial_recovery, "run_inference_with_journal", raise_typed)

    execute_partial_repair(plan, **kwargs)

    assert state.full_length_retry_blocked_by is None


def test_other_final_worker_errors_still_propagate(monkeypatch, tmp_path):
    """복구 경로가 모든 FINAL 오류를 삼키면 진짜 실패가 조용히 사라진다."""
    plan, _state, kwargs = _executing_plan(tmp_path)

    def raise_final(*_args, **_kwargs):
        raise WorkerError(ErrorCode.CHART_OSU_PARSE_FAILED, "broken output")

    monkeypatch.setattr(partial_recovery, "run_inference_with_journal", raise_final)

    with pytest.raises(WorkerError) as caught:
        execute_partial_repair(plan, **kwargs)

    assert caught.value.code is ErrorCode.CHART_OSU_PARSE_FAILED
