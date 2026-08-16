from pathlib import Path
from types import SimpleNamespace

from chart_worker.generation import timing_family_recovery
from chart_worker.generation.candidate_state import VariantState
from chart_worker.generation.generation_control import (
    AdditionalInferenceBudget,
    RecoveryKind,
)
from chart_worker.generation.params import GenerationRequest
from chart_worker.generation.timing_family_recovery import timing_family_reviews


def test_timing_family_review_keeps_missing_siblings_insufficient():
    reviews = timing_family_reviews([])

    assert len(reviews) == 4
    assert {review.status for review in reviews} == {"INSUFFICIENT"}
    assert all(review.target_key_mode is None for review in reviews)


def test_timing_family_retry_does_not_spend_budget_after_tail_exhaustion(
    monkeypatch, tmp_path: Path
):
    blocked_by = {"signature": "END_BOUNDARY_CROSSES_HOLD:500:11"}
    state = VariantState(
        key_mode=4,
        difficulty="EXPERT",
        flat_index=3,
        full_length_retry_blocked_by=blocked_by,
    )
    source = SimpleNamespace(
        request=GenerationRequest(
            audio_path=Path("game.flac"),
            timing_reference_path=Path("timing.osu"),
            key_mode=4,
            difficulty="EXPERT",
            seed=3,
            duration_ms=2_000,
        )
    )
    budget = AdditionalInferenceBudget(limit=1)

    def forbidden_inference(*args, **kwargs):
        del args, kwargs
        raise AssertionError("blocked timing-family retry reached inference")

    monkeypatch.setattr(
        timing_family_recovery,
        "run_inference_with_journal",
        forbidden_inference,
    )

    result = timing_family_recovery._try_timing_family_retry(
        state,
        source,
        prepared=None,
        authority=SimpleNamespace(reference_path=Path("timing.osu")),
        onset_analysis=None,
        run_dir=tmp_path,
        generator=object(),
        base_seed=0,
        authority_epoch=1,
        inference_budget=budget,
        evaluate_candidate=lambda *args, **kwargs: None,
        serialize_candidate=lambda *args, **kwargs: "",
        intro_anchor_covered=lambda *args, **kwargs: None,
    )

    assert result is None
    assert budget.used == 0
    assert state.recovery.was_attempted(RecoveryKind.TIMING_FAMILY) is False
    assert state.attempt_evidence == [
        {
            "reason": "TIMING_FAMILY_RETRY_SUPPRESSED_BY_TAIL_EXHAUSTION",
            "blockedBy": blocked_by,
        }
    ]
