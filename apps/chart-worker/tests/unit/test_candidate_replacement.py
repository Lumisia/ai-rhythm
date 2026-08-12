from chart_worker.validation.candidate_replacement import (
    CandidateQualitySnapshot,
    decide_candidate_replacement,
)
from chart_worker.validation.quality_gate import GateAction


def _snapshot(
    *,
    provenance: str = "PRIMARY",
    overall_action: GateAction = GateAction.PASS,
    retry_axes: tuple[str, ...] = (),
    review_axes: tuple[str, ...] = (),
    structure_pass: bool = True,
    timing_identity_pass: bool = True,
    song_bounds_action: GateAction = GateAction.PASS,
) -> CandidateQualitySnapshot:
    return CandidateQualitySnapshot(
        provenance=provenance,
        overall_action=overall_action,
        retry_axes=retry_axes,
        review_axes=review_axes,
        structure_pass=structure_pass,
        timing_identity_pass=timing_identity_pass,
        song_bounds_action=song_bounds_action,
    )


def test_rejects_verified_pass_to_raw_unverified():
    decision = decide_candidate_replacement(
        _snapshot(),
        _snapshot(provenance="RAW_UNVERIFIED"),
        stage="INTRO_EXACT_RESELECT",
        objective_improved=True,
    )

    assert decision.accepted is False
    assert "RAW_UNVERIFIED_CHALLENGER" in decision.reasons


def test_rejects_pass_to_new_review_axis():
    decision = decide_candidate_replacement(
        _snapshot(),
        _snapshot(
            overall_action=GateAction.REVIEW,
            review_axes=("PATTERN",),
        ),
        stage="INTRO_EXACT_RESELECT",
        objective_improved=True,
    )

    assert decision.accepted is False
    assert "NEW_REVIEW_AXIS:PATTERN" in decision.reasons
    assert "OVERALL_ACTION_DOWNGRADE:PASS->REVIEW" in decision.reasons


def test_rejects_new_retry_axis_even_when_overall_action_is_inconsistent():
    decision = decide_candidate_replacement(
        _snapshot(review_axes=("PATTERN",), overall_action=GateAction.REVIEW),
        _snapshot(
            retry_axes=("COVERAGE",),
            overall_action=GateAction.REVIEW,
        ),
        stage="TIMING_FAMILY_RESELECT",
        objective_improved=True,
    )

    assert decision.accepted is False
    assert "NEW_RETRY_AXIS:COVERAGE" in decision.reasons


def test_accepts_hard_equivalent_candidate_when_stage_objective_improves():
    decision = decide_candidate_replacement(
        _snapshot(review_axes=("PATTERN",), overall_action=GateAction.REVIEW),
        _snapshot(review_axes=("PATTERN",), overall_action=GateAction.REVIEW),
        stage="INTRO_PHRASE_RESELECT",
        objective_improved=True,
    )

    assert decision.accepted is True
    assert decision.reasons == ("OBJECTIVE_IMPROVED_WITHOUT_QUALITY_DOWNGRADE",)
    assert decision.to_report()["accepted"] is True


def test_rejects_same_quality_candidate_without_objective_improvement():
    decision = decide_candidate_replacement(
        _snapshot(),
        _snapshot(),
        stage="TIMING_FAMILY_RESELECT",
        objective_improved=False,
    )

    assert decision.accepted is False
    assert decision.reasons == ("STAGE_OBJECTIVE_NOT_IMPROVED",)
