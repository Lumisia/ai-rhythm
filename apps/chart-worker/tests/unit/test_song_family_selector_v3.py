from dataclasses import replace

import pytest

from chart_worker.analysis.intro_anchor import IntroAnchorEvidence
from chart_worker.validation.difficulty_hazard_v1 import CandidateDifficultyEvidenceV1
from chart_worker.validation.family_evidence_v3 import (
    CandidateFamilyEvidenceV3,
    CandidateSafetyEvidenceV3,
    GapIntervalEvidence,
    IntroCandidateVoteV3,
    SongSelectionEvidenceV3,
    build_intro_selection_evidence,
)
from chart_worker.validation.mania_star_evidence import ManiaStarEvidenceV1
from chart_worker.validation.song_family_selector_v3 import (
    CalibrationPredictionV3,
    evaluate_shadow_v3_hazard_proposal,
    evaluate_shadow_v3_proposal,
    propose_shadow_v3_assignment,
    propose_shadow_v3_hazard_assignment,
)


def _intro():
    return build_intro_selection_evidence(
        IntroAnchorEvidence(
            status="CONFIRMED",
            anchor_ms=1_020,
            anchor_grid_ms=1_000,
            grid_distance_ms=20,
            aggregate_percentile_rank=0.95,
            prominent_band_count=2,
            pulse_continuation_matches=3,
            pulse_continuation_opportunities=4,
        ),
        active_onset_ms=(1_020,),
        votes=(IntroCandidateVoteV3("4K:EASY", "vote", 1_000),),
    )


def _gap(start_ms: int, end_ms: int) -> GapIntervalEvidence:
    return GapIntervalEvidence(
        start_ms=start_ms,
        end_ms=end_ms,
        position="MIDDLE",
        active_onset_count=8,
        active_frame_ratio=0.7,
        opportunity_kind="ATTACK_REQUIRED",
        local_audio_evidence_digest="a" * 64,
    )


def _candidate(
    candidate_id: str,
    difficulty: str,
    *,
    gap: GapIntervalEvidence | None = None,
    first_row_ms: int = 1_000,
    audio_supported: bool = True,
    publication_tier: str = "PRODUCTION_CANDIDATE",
    matched_f1_50: float = 0.8,
    review_rank: int = 0,
    eligible_target_difficulties: tuple[str, ...] | None = None,
) -> CandidateFamilyEvidenceV3:
    safety = CandidateSafetyEvidenceV3(
        candidate_id=candidate_id,
        structure_safe=True,
        timing_identity_safe=True,
        song_bounds_safe=True,
        serialization_safe=True,
        publication_tier=publication_tier,
        model_backed=True,
        recovery_trust_rank=0,
        active_gaps=() if gap is None else (gap,),
    )
    return CandidateFamilyEvidenceV3(
        candidate_id=candidate_id,
        key_mode=4,
        difficulty=difficulty,
        provenance="PRIMARY",
        candidate_payload_ref=f"candidate-payloads/{candidate_id}.osu",
        candidate_payload_sha256=(candidate_id[0] * 64),
        safety=safety,
        first_row_ms=first_row_ms,
        first_row_audio_supported=audio_supported,
        first_row_grid_distance_ms=0,
        intro_reference_state="CONFIRMED_AUDIO",
        matched_f1_50=matched_f1_50,
        review_rank=review_rank,
        eligible_target_difficulties=(
            (difficulty,)
            if eligible_target_difficulties is None
            else eligible_target_difficulties
        ),
    )


def _evidence(*candidates: CandidateFamilyEvidenceV3) -> SongSelectionEvidenceV3:
    current = tuple(
        sorted(
            (f"4K:{candidate.difficulty}", candidate.candidate_id)
            for candidate in candidates
            if candidate.candidate_id.endswith("0")
        )
    )
    return SongSelectionEvidenceV3(
        context_id="context-v3",
        intro_selection=_intro(),
        candidates=tuple(sorted(candidates, key=lambda item: item.candidate_id)),
        current_assignment=current,
    )


def _base_family():
    return (
        _candidate("a0", "EASY"),
        _candidate("b0", "NORMAL"),
        _candidate("c0", "HARD"),
        _candidate("d0", "EXPERT"),
    )


def _predictions(**scores: float) -> tuple[CalibrationPredictionV3, ...]:
    return tuple(
        CalibrationPredictionV3(
            candidate_id=candidate_id,
            state="IN_DOMAIN",
            score=float(score),
            calibration_sha256="f" * 64,
        )
        for candidate_id, score in sorted(scores.items())
    )


def _metric(candidate_id: str, official: float, project: float):
    payload = candidate_id[0] * 64
    return CandidateDifficultyEvidenceV1(
        candidate_id=candidate_id,
        candidate_payload_sha256=payload,
        official_star=ManiaStarEvidenceV1(
            input_osu_sha256=payload,
            tool_binary_sha256="1" * 64,
            osu_tools_source_commit="2" * 40,
            osu_source_commit="3" * 40,
            calculator_version=20241007,
            star_rating=float(official),
            attributes_sha256="4" * 64,
            mods=(),
            verification_state="VERIFIED_PINNED_TOOL_EXECUTION",
        ),
        project_rating=float(project),
        project_rating_evidence_sha256="5" * 64,
    )


def test_shadow_v3_without_calibration_never_changes_assignment():
    easy, normal, hard, expert = _base_family()
    challenger = _candidate("e1", "EXPERT")
    evidence = _evidence(easy, normal, hard, expert, challenger)
    proposal = tuple(
        sorted(
            {
                **dict(evidence.current_assignment),
                "4K:EXPERT": challenger.candidate_id,
            }.items()
        )
    )

    result = evaluate_shadow_v3_proposal(evidence, proposal=proposal)

    assert result.selected_assignment == evidence.current_assignment
    assert result.shadow_assignment == evidence.current_assignment
    assert result.proposal_eligible is False
    assert result.blockers == ("CALIBRATION_UNAVAILABLE",)
    assert result.mutates_selection is False


def test_calibrated_safe_proposal_resolves_existing_inversion_in_shadow_only():
    easy, normal, hard, expert = _base_family()
    challenger = _candidate("e1", "EXPERT")
    evidence = _evidence(easy, normal, hard, expert, challenger)
    proposal = tuple(sorted({**dict(evidence.current_assignment), "4K:EXPERT": "e1"}.items()))
    predictions = _predictions(a0=1, b0=2, c0=4, d0=3, e1=5)

    result = evaluate_shadow_v3_proposal(
        evidence,
        proposal=proposal,
        predictions=predictions,
    )

    assert result.proposal_eligible is True
    assert dict(result.shadow_assignment)["4K:EXPERT"] == "e1"
    assert result.selected_assignment == evidence.current_assignment
    assert result.resolved_inversions == (("4K:HARD", "4K:EXPERT"),)
    assert result.created_inversions == ()


def test_longer_active_gap_blocks_better_calibrated_candidate():
    easy, normal, hard, expert = _base_family()
    current_with_gap = replace(
        expert,
        safety=replace(expert.safety, active_gaps=(_gap(10_000, 18_000),)),
    )
    challenger = _candidate("e1", "EXPERT", gap=_gap(9_000, 29_000))
    evidence = _evidence(easy, normal, hard, current_with_gap, challenger)
    proposal = tuple(sorted({**dict(evidence.current_assignment), "4K:EXPERT": "e1"}.items()))

    result = evaluate_shadow_v3_proposal(
        evidence,
        proposal=proposal,
        predictions=_predictions(a0=1, b0=2, c0=4, d0=3, e1=5),
    )

    assert result.proposal_eligible is False
    assert "ACTIVE_GAP_REGRESSION:4K:EXPERT" in result.blockers


def test_confirmed_intro_support_cannot_regress_for_difficulty_gain():
    easy, normal, hard, expert = _base_family()
    challenger = _candidate(
        "e1",
        "EXPERT",
        first_row_ms=5_000,
        audio_supported=False,
    )
    evidence = _evidence(easy, normal, hard, expert, challenger)
    proposal = tuple(sorted({**dict(evidence.current_assignment), "4K:EXPERT": "e1"}.items()))

    result = evaluate_shadow_v3_proposal(
        evidence,
        proposal=proposal,
        predictions=_predictions(a0=1, b0=2, c0=4, d0=3, e1=5),
    )

    assert result.proposal_eligible is False
    assert "INTRO_EVIDENCE_REGRESSION:4K:EXPERT" in result.blockers


def test_new_adjacent_inversion_blocks_otherwise_safe_replacement():
    easy, normal, hard, expert = _base_family()
    challenger = _candidate("c1", "HARD")
    evidence = _evidence(easy, normal, hard, expert, challenger)
    proposal = tuple(sorted({**dict(evidence.current_assignment), "4K:HARD": "c1"}.items()))

    result = evaluate_shadow_v3_proposal(
        evidence,
        proposal=proposal,
        predictions=_predictions(a0=1, b0=2, c0=3, c1=5, d0=4),
    )

    assert result.proposal_eligible is False
    assert result.created_inversions == (("4K:HARD", "4K:EXPERT"),)
    assert "NEW_DIFFICULTY_INVERSION" in result.blockers


def test_unknown_prediction_fails_closed():
    easy, normal, hard, expert = _base_family()
    challenger = _candidate("e1", "EXPERT")
    evidence = _evidence(easy, normal, hard, expert, challenger)
    proposal = tuple(sorted({**dict(evidence.current_assignment), "4K:EXPERT": "e1"}.items()))
    predictions = (
        *_predictions(a0=1, b0=2, c0=4, d0=3),
        CalibrationPredictionV3(
            candidate_id="e1",
            state="UNKNOWN",
            score=None,
            calibration_sha256="f" * 64,
        ),
    )

    result = evaluate_shadow_v3_proposal(
        evidence,
        proposal=proposal,
        predictions=predictions,
    )

    assert result.proposal_eligible is False
    assert result.blockers == ("CALIBRATION_UNKNOWN:e1",)


def test_proposal_cannot_assign_candidate_from_another_slot():
    easy, normal, hard, expert = _base_family()
    evidence = _evidence(easy, normal, hard, expert)
    proposal = tuple(sorted({**dict(evidence.current_assignment), "4K:EXPERT": "a0"}.items()))

    with pytest.raises(ValueError, match="does not belong to slot"):
        evaluate_shadow_v3_proposal(evidence, proposal=proposal)


def test_relabelled_current_assignment_is_a_valid_v3_baseline():
    easy = _candidate(
        "a0",
        "EASY",
        eligible_target_difficulties=("EASY", "NORMAL"),
    )
    normal = _candidate(
        "b0",
        "NORMAL",
        eligible_target_difficulties=("EASY", "NORMAL"),
    )
    hard = _candidate("c0", "HARD")
    expert = _candidate("d0", "EXPERT")
    evidence = replace(
        _evidence(easy, normal, hard, expert),
        current_assignment=(
            ("4K:EASY", "b0"),
            ("4K:EXPERT", "d0"),
            ("4K:HARD", "c0"),
            ("4K:NORMAL", "a0"),
        ),
    )

    result = propose_shadow_v3_assignment(
        evidence,
        predictions=_predictions(a0=2, b0=1, c0=3, d0=4),
    )

    assert result.selected_assignment == evidence.current_assignment
    assert result.shadow_assignment == evidence.current_assignment
    assert result.blockers == ("NO_CALIBRATED_DIFFICULTY_IMPROVEMENT",)


def test_calibrated_search_uses_explicit_cross_slot_eligibility():
    easy, normal, hard, expert = _base_family()
    easy_replacement = _candidate(
        "e1",
        "NORMAL",
        eligible_target_difficulties=("EASY", "NORMAL"),
    )
    evidence = _evidence(easy, normal, hard, expert, easy_replacement)

    result = propose_shadow_v3_assignment(
        evidence,
        predictions=_predictions(a0=3, b0=2, c0=4, d0=5, e1=1),
    )

    assert result.proposal_eligible is True
    assert dict(result.shadow_assignment)["4K:EASY"] == "e1"
    assert result.resolved_inversions == (("4K:EASY", "4K:NORMAL"),)


def test_calibrated_proposal_never_reuses_one_candidate_in_two_slots():
    easy, normal, hard, expert = _base_family()
    cross_slot_hard = replace(
        hard,
        eligible_target_difficulties=("EASY", "HARD"),
    )
    evidence = _evidence(easy, normal, cross_slot_hard, expert)
    proposal = tuple(
        sorted(
            {
                **dict(evidence.current_assignment),
                "4K:EASY": cross_slot_hard.candidate_id,
            }.items()
        )
    )

    result = evaluate_shadow_v3_proposal(
        evidence,
        proposal=proposal,
        predictions=_predictions(a0=3, b0=2, c0=1, d0=4),
    )

    assert result.proposal_eligible is False
    assert result.shadow_assignment == evidence.current_assignment
    assert "DUPLICATE_CANDIDATE_ASSIGNMENT" in result.blockers


def test_two_axis_proposal_never_reuses_one_candidate_in_two_slots():
    easy, normal, hard, expert = _base_family()
    cross_slot_hard = replace(
        hard,
        eligible_target_difficulties=("EASY", "HARD"),
    )
    evidence = _evidence(easy, normal, cross_slot_hard, expert)
    proposal = tuple(
        sorted(
            {
                **dict(evidence.current_assignment),
                "4K:EASY": cross_slot_hard.candidate_id,
            }.items()
        )
    )

    result = evaluate_shadow_v3_hazard_proposal(
        evidence,
        proposal=proposal,
        difficulty_evidence=(
            _metric("a0", 3, 3),
            _metric("b0", 2, 2),
            _metric("c0", 1, 1),
            _metric("d0", 4, 4),
        ),
    )

    assert result.proposal_eligible is False
    assert result.shadow_assignment == evidence.current_assignment
    assert "DUPLICATE_CANDIDATE_ASSIGNMENT" in result.blockers


def test_calibrated_search_finds_safe_family_replacement_without_model_calls():
    easy, normal, hard, expert = _base_family()
    challenger = _candidate("e1", "EXPERT")
    evidence = _evidence(easy, normal, hard, expert, challenger)

    result = propose_shadow_v3_assignment(
        evidence,
        predictions=_predictions(a0=1, b0=2, c0=4, d0=3, e1=5),
    )

    assert result.proposal_eligible is True
    assert dict(result.shadow_assignment)["4K:EXPERT"] == "e1"
    assert result.selected_assignment == evidence.current_assignment
    assert result.resolved_inversions == (("4K:HARD", "4K:EXPERT"),)


def test_calibrated_search_does_not_replace_with_gap_regression():
    easy, normal, hard, expert = _base_family()
    current = replace(
        expert,
        safety=replace(expert.safety, active_gaps=(_gap(10_000, 18_000),)),
    )
    challenger = _candidate("e1", "EXPERT", gap=_gap(9_000, 29_000))
    evidence = _evidence(easy, normal, hard, current, challenger)

    result = propose_shadow_v3_assignment(
        evidence,
        predictions=_predictions(a0=1, b0=2, c0=4, d0=3, e1=5),
    )

    assert result.shadow_assignment == evidence.current_assignment
    assert result.proposal_eligible is False
    assert "NO_SAFE_CALIBRATED_PROPOSAL" in result.blockers


def test_calibrated_search_with_monotonic_current_family_is_noop():
    evidence = _evidence(*_base_family())

    result = propose_shadow_v3_assignment(
        evidence,
        predictions=_predictions(a0=1, b0=2, c0=3, d0=4),
    )

    assert result.shadow_assignment == evidence.current_assignment
    assert result.proposal_eligible is False
    assert result.blockers == ("NO_CALIBRATED_DIFFICULTY_IMPROVEMENT",)


def test_two_axis_shadow_proposal_resolves_project_only_risk_without_mutation():
    easy, normal, hard, expert = _base_family()
    challenger = _candidate("e1", "EXPERT")
    evidence = _evidence(easy, normal, hard, expert, challenger)
    proposal = tuple(sorted({**dict(evidence.current_assignment), "4K:EXPERT": "e1"}.items()))
    metrics = (
        _metric("a0", 1.0, 1.0),
        _metric("b0", 2.0, 2.0),
        _metric("c0", 4.0, 4.0),
        _metric("d0", 4.1, 3.5),
        _metric("e1", 5.0, 5.0),
    )

    result = evaluate_shadow_v3_hazard_proposal(
        evidence,
        proposal=proposal,
        difficulty_evidence=metrics,
    )

    assert result.proposal_eligible is True
    assert result.selected_assignment == evidence.current_assignment
    assert dict(result.shadow_assignment)["4K:EXPERT"] == "e1"
    assert result.current_at_risk == (("4K:HARD", "4K:EXPERT"),)
    assert result.proposed_at_risk == ()
    assert result.resolved_at_risk == (("4K:HARD", "4K:EXPERT"),)
    assert result.created_at_risk == ()
    assert result.mutates_selection is False


def test_two_axis_shadow_fails_closed_when_current_metric_is_missing():
    evidence = _evidence(*_base_family())
    metrics = (
        _metric("a0", 1.0, 1.0),
        _metric("b0", 2.0, 2.0),
        _metric("c0", 3.0, 3.0),
    )

    result = propose_shadow_v3_hazard_assignment(
        evidence,
        difficulty_evidence=metrics,
    )

    assert result.proposal_eligible is False
    assert result.shadow_assignment == evidence.current_assignment
    assert result.current_unknown == (("4K:HARD", "4K:EXPERT"),)
    assert any(blocker.startswith("DIFFICULTY_EVIDENCE_UNKNOWN:") for blocker in result.blockers)


def test_two_axis_shadow_rejects_metric_bound_to_different_payload():
    easy, normal, hard, expert = _base_family()
    mismatched_easy = replace(easy, candidate_payload_sha256="9" * 64)
    evidence = _evidence(mismatched_easy, normal, hard, expert)

    with pytest.raises(ValueError, match="payload differs"):
        evaluate_shadow_v3_hazard_proposal(
            evidence,
            proposal=evidence.current_assignment,
            difficulty_evidence=(
                _metric("a0", 1.0, 1.0),
                _metric("b0", 2.0, 2.0),
                _metric("c0", 3.0, 3.0),
                _metric("d0", 4.0, 4.0),
            ),
        )


def test_two_axis_search_keeps_quality_safety_veto_and_finds_no_alternative():
    easy, normal, hard, expert = _base_family()
    current = replace(
        expert,
        safety=replace(expert.safety, active_gaps=(_gap(10_000, 18_000),)),
    )
    challenger = _candidate("e1", "EXPERT", gap=_gap(9_000, 29_000))
    evidence = _evidence(easy, normal, hard, current, challenger)
    metrics = (
        _metric("a0", 1.0, 1.0),
        _metric("b0", 2.0, 2.0),
        _metric("c0", 4.0, 4.0),
        _metric("d0", 3.0, 3.0),
        _metric("e1", 5.0, 5.0),
    )

    result = propose_shadow_v3_hazard_assignment(
        evidence,
        difficulty_evidence=metrics,
    )

    assert result.proposal_eligible is False
    assert result.shadow_assignment == evidence.current_assignment
    assert result.blockers == ("NO_SAFE_TWO_AXIS_PROPOSAL",)
