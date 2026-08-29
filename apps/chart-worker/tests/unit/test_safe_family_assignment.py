import hashlib

import pytest

from chart_worker.validation.safe_family_assignment import (
    SafeFamilyCandidate,
    select_safe_family_assignment,
)


def _candidate(
    candidate_id: str,
    source_difficulty: str,
    score: float | None,
    *,
    payload: str | None = None,
    hard_safe: bool = True,
    intro_state: str = "CONFIRMED_SAFE",
    boundary_state: str = "CONFIRMED_SAFE",
    provenance: str = "PRIMARY",
    attack_gap_count: int = 0,
    publication_rank: int = 0,
    intro_distance_ms: int | None = 0,
    tail_coverage_deficit_ms: int = 0,
    tail_active_onset_count: int = 0,
    terminal_overflow_ms: int = 0,
    terminal_overflow_confidence: str = "CONFIRMED",
    ordering_score: float | None = None,
) -> SafeFamilyCandidate:
    return SafeFamilyCandidate(
        candidate_id=candidate_id,
        candidate_payload_sha256=(
            payload * 64
            if payload is not None
            else hashlib.sha256(candidate_id.encode("utf-8")).hexdigest()
        ),
        key_mode=4,
        source_difficulty=source_difficulty,
        provenance=provenance,
        hard_safe=hard_safe,
        intro_state=intro_state,
        boundary_state=boundary_state,
        attack_gap_count=attack_gap_count,
        attack_gap_total_ms=attack_gap_count * 5_000,
        active_gap_count=attack_gap_count,
        max_active_gap_ms=attack_gap_count * 5_000,
        difficulty_scores=(
            ()
            if score is None
            else (
                (
                    "ORDERING_SCORE",
                    float(score if ordering_score is None else ordering_score),
                ),
                ("PROJECT_RATING", float(score)),
            )
        ),
        review_rank=0,
        publication_rank=publication_rank,
        recovery_trust_rank=0,
        matched_f1_50=0.8,
        attempt=1,
        intro_distance_ms=intro_distance_ms,
        tail_coverage_deficit_ms=tail_coverage_deficit_ms,
        tail_active_onset_count=tail_active_onset_count,
        terminal_overflow_ms=terminal_overflow_ms,
        terminal_overflow_confidence=terminal_overflow_confidence,
    )


def _current(*candidate_ids: str) -> tuple[tuple[str, str], ...]:
    return tuple(
        zip(("EASY", "NORMAL", "HARD", "EXPERT"), candidate_ids, strict=True)
    )


def test_confirmed_intro_failure_cannot_authorize_a_duplicate_payload() -> None:
    candidates = (
        _candidate("easy", "EASY", 1.0),
        _candidate("normal", "NORMAL", 2.0),
        _candidate("hard", "HARD", 3.0),
        _candidate("late", "EXPERT", 4.0, intro_state="VIOLATION"),
    )

    decision = select_safe_family_assignment(
        candidates,
        current_assignment=_current("easy", "normal", "hard", "late"),
    )

    selected = dict(decision.selected_assignment)
    assert selected["EXPERT"] == "late"
    assert decision.unique_payload_status == "SATISFIED"
    assert decision.emergency_duplicate_slots == ()
    assert decision.selected_score.intro_violations == 1
    assert decision.selected_score.duplicate_payloads == 0


def test_hard_unsafe_candidate_cannot_be_hidden_by_reusing_a_payload() -> None:
    candidates = (
        _candidate("easy", "EASY", 1.0),
        _candidate("normal", "NORMAL", 2.0),
        _candidate("hard", "HARD", 3.0),
        _candidate("broken", "EXPERT", 7.0, hard_safe=False),
    )

    decision = select_safe_family_assignment(
        candidates,
        current_assignment=_current("easy", "normal", "hard", "broken"),
    )

    assert dict(decision.selected_assignment)["EXPERT"] == "broken"
    assert decision.unique_payload_status == "SATISFIED"
    assert decision.selected_score.hard_violations == 1
    assert decision.selected_score.duplicate_payloads == 0


def test_safe_fallback_is_not_rejected_only_because_of_provenance() -> None:
    candidates = (
        _candidate("fallback", "EASY", 1.0, provenance="SAFE_FALLBACK"),
        _candidate("raw", "EASY", 1.2, hard_safe=False, provenance="RAW_UNVERIFIED"),
        _candidate("normal", "NORMAL", 2.0),
        _candidate("hard", "HARD", 3.0),
        _candidate("expert", "EXPERT", 4.0),
    )

    decision = select_safe_family_assignment(
        candidates,
        current_assignment=_current("raw", "normal", "hard", "expert"),
    )

    assert dict(decision.selected_assignment)["EASY"] == "fallback"
    assert decision.selected_score.hard_violations == 0


def test_current_safe_fallback_can_be_replaced_by_a_safer_ranked_model_candidate() -> None:
    candidates = (
        _candidate("easy", "EASY", 1.0),
        _candidate("normal", "NORMAL", 2.0),
        _candidate("hard", "HARD", 3.0),
        _candidate(
            "fallback",
            "EXPERT",
            4.0,
            provenance="SAFE_FALLBACK",
            publication_rank=1,
        ),
        _candidate("model-alt", "HARD", 4.0),
    )

    decision = select_safe_family_assignment(
        candidates,
        current_assignment=_current("easy", "normal", "hard", "fallback"),
    )

    assert dict(decision.selected_assignment)["EXPERT"] == "model-alt"
    assert decision.changed is True


def test_four_unique_safe_candidates_are_relabelled_by_relative_difficulty() -> None:
    candidates = (
        _candidate("ten", "EXPERT", 10.0),
        _candidate("twenty", "HARD", 20.0),
        _candidate("thirty", "NORMAL", 30.0),
        _candidate("forty", "EASY", 40.0),
    )

    decision = select_safe_family_assignment(
        candidates,
        current_assignment=_current("forty", "thirty", "twenty", "ten"),
    )

    assert decision.selected_assignment == _current("ten", "twenty", "thirty", "forty")
    assert decision.selected_score.difficulty_violations == 0
    assert decision.reassigned_slots == ("EASY", "NORMAL", "HARD", "EXPERT")


def test_relative_difficulty_cannot_introduce_a_new_active_audio_tail_gap() -> None:
    candidates = (
        _candidate("easy", "EASY", 40.0, provenance="SAFE_FALLBACK"),
        _candidate("normal", "NORMAL", 30.0, provenance="SAFE_FALLBACK"),
        _candidate("hard", "HARD", 20.0, provenance="SAFE_FALLBACK"),
        _candidate(
            "expert",
            "EXPERT",
            10.0,
            provenance="SAFE_FALLBACK",
            tail_coverage_deficit_ms=10_567,
            tail_active_onset_count=5,
        ),
    )
    current = _current("easy", "normal", "hard", "expert")

    decision = select_safe_family_assignment(candidates, current_assignment=current)

    selected = dict(decision.selected_assignment)
    assert selected["EASY"] != "expert"
    assert selected["EXPERT"] == "expert"
    assert decision.selected_score.tail_active_onset_count == 5
    assert decision.selected_score.tail_coverage_deficit_ms == 10_567


def test_post_resolution_ordering_relabels_the_same_four_payloads_despite_slot_veto() -> None:
    candidates = (
        _candidate(
            "easy-with-tail",
            "EASY",
            2.65,
            provenance="COVERAGE_REPAIR",
            tail_coverage_deficit_ms=10_567,
            tail_active_onset_count=5,
        ),
        _candidate("normal-easier", "NORMAL", 1.94, provenance="SAFE_FALLBACK"),
        _candidate("hard", "HARD", 3.68),
        _candidate("expert", "EXPERT", 4.96),
    )
    current = _current("easy-with-tail", "normal-easier", "hard", "expert")

    protected = select_safe_family_assignment(candidates, current_assignment=current)
    ordered = select_safe_family_assignment(
        candidates,
        current_assignment=current,
        post_resolution_ordering=True,
    )

    assert protected.selected_assignment == current
    assert ordered.selected_assignment == _current(
        "normal-easier", "easy-with-tail", "hard", "expert"
    )
    assert ordered.post_resolution_ordering_status == "ORDERED"
    assert ordered.reason == "POST_RESOLUTION_DIFFICULTY_ORDERED"
    assert ordered.assignments_evaluated == 1
    assert ordered.selected_score.difficulty_violations == 0
    assert ordered.selected_score.tail_active_onset_count == 5
    assert ordered.selected_score.tail_coverage_deficit_ms == 10_567
    assert {
        candidate.candidate_payload_sha256 for candidate in candidates
    } == {
        next(
            candidate.candidate_payload_sha256
            for candidate in candidates
            if candidate.candidate_id == candidate_id
        )
        for _difficulty, candidate_id in ordered.selected_assignment
    }


def test_post_resolution_ordering_fails_closed_without_required_metrics() -> None:
    candidates = (
        _candidate("easy", "EASY", 2.0),
        _candidate("normal", "NORMAL", None),
        _candidate("hard", "HARD", 3.0),
        _candidate("expert", "EXPERT", 4.0),
    )
    current = _current("easy", "normal", "hard", "expert")

    decision = select_safe_family_assignment(
        candidates,
        current_assignment=current,
        post_resolution_ordering=True,
    )

    assert decision.selected_assignment == current
    assert decision.post_resolution_ordering_status == "UNAVAILABLE"
    assert decision.reason == "POST_RESOLUTION_DIFFICULTY_ORDERING_UNAVAILABLE"
    assert "DIFFICULTY_EVIDENCE" in decision.family_feasibility_reasons


def test_post_resolution_ordering_does_not_hide_duplicate_payloads() -> None:
    candidates = (
        _candidate("easy", "EASY", 2.0, payload="a"),
        _candidate("normal-alias", "NORMAL", 1.0, payload="a"),
        _candidate("hard", "HARD", 3.0),
        _candidate("expert", "EXPERT", 4.0),
    )
    current = _current("easy", "normal-alias", "hard", "expert")

    decision = select_safe_family_assignment(
        candidates,
        current_assignment=current,
        post_resolution_ordering=True,
    )

    assert decision.selected_assignment == current
    assert decision.unique_payload_status == "UNAVAILABLE"
    assert decision.post_resolution_ordering_status == "UNAVAILABLE"
    assert "PAYLOAD_UNIQUENESS" in decision.family_feasibility_reasons


def test_post_resolution_ordering_uses_project_rating_but_reports_metric_conflict() -> None:
    candidates = (
        _candidate("easy-harder", "EASY", 2.0, ordering_score=1.0),
        _candidate("normal-easier", "NORMAL", 1.0, ordering_score=2.0),
        _candidate("hard", "HARD", 3.0, ordering_score=3.0),
        _candidate("expert", "EXPERT", 4.0, ordering_score=4.0),
    )

    decision = select_safe_family_assignment(
        candidates,
        current_assignment=_current(
            "easy-harder", "normal-easier", "hard", "expert"
        ),
        post_resolution_ordering=True,
    )

    assert decision.selected_assignment == _current(
        "normal-easier", "easy-harder", "hard", "expert"
    )
    assert decision.post_resolution_ordering_status == "ORDERED"
    assert "DIFFICULTY_ORDER" in decision.family_feasibility_reasons


def test_post_resolution_ordering_keeps_equal_rating_family_unavailable() -> None:
    candidates = (
        _candidate("easy", "EASY", 1.0, ordering_score=1.0),
        _candidate("normal", "NORMAL", 1.0, ordering_score=2.0),
        _candidate("hard", "HARD", 3.0, ordering_score=3.0),
        _candidate("expert", "EXPERT", 4.0, ordering_score=4.0),
    )

    decision = select_safe_family_assignment(
        candidates,
        current_assignment=_current("easy", "normal", "hard", "expert"),
        post_resolution_ordering=True,
    )

    assert decision.post_resolution_ordering_status == "UNAVAILABLE"
    assert decision.family_feasibility_status == "UNAVAILABLE"
    assert "DIFFICULTY_ORDER" in decision.family_feasibility_reasons


def test_confirmed_attack_gap_cannot_use_a_duplicate_wrong_label() -> None:
    candidates = (
        _candidate("easy", "EASY", 10.0),
        _candidate("normal", "NORMAL", 20.0),
        _candidate("hard", "HARD", 30.0),
        _candidate("gapped", "EXPERT", 40.0, attack_gap_count=1),
    )

    decision = select_safe_family_assignment(
        candidates,
        current_assignment=_current("easy", "normal", "hard", "gapped"),
    )

    assert dict(decision.selected_assignment)["EXPERT"] == "gapped"
    assert decision.selected_score.attack_gap_count == 1
    assert decision.selected_score.duplicate_payloads == 0
    assert decision.emergency_duplicate_slots == ()


def test_one_confirmed_defect_is_not_replaced_by_a_different_confirmed_defect() -> None:
    candidates = (
        _candidate("easy", "EASY", 1.0, attack_gap_count=1),
        _candidate("normal", "NORMAL", 2.0, boundary_state="VIOLATION"),
        _candidate("hard", "HARD", 3.0, intro_state="VIOLATION"),
        _candidate("expert", "EXPERT", 4.0, hard_safe=False),
    )
    current = _current("easy", "normal", "hard", "expert")

    decision = select_safe_family_assignment(candidates, current_assignment=current)

    assert decision.selected_assignment == current
    assert decision.selected_score == decision.current_score


def test_relative_fit_cannot_beat_payload_uniqueness() -> None:
    candidates = (
        _candidate("easy", "EASY", 1.0),
        _candidate("easy-alt", "EASY", 1.4),
        _candidate("normal", "NORMAL", 2.0),
        _candidate("hard", "HARD", 3.0),
        _candidate("broken", "EXPERT", 4.0, hard_safe=False),
    )

    decision = select_safe_family_assignment(
        candidates,
        current_assignment=_current("easy", "normal", "hard", "broken"),
    )

    assert decision.selected_assignment == _current(
        "easy", "easy-alt", "normal", "hard"
    )
    assert decision.unique_payload_status == "SATISFIED"
    assert decision.family_feasibility_status == "SATISFIED"
    assert decision.emergency_duplicate_slots == ()


def test_safe_monotonic_current_family_is_left_unchanged() -> None:
    candidates = (
        _candidate("easy", "EASY", 1.0),
        _candidate("normal", "NORMAL", 2.0),
        _candidate("hard", "HARD", 3.0),
        _candidate("expert", "EXPERT", 4.0),
    )
    current = _current("easy", "normal", "hard", "expert")

    decision = select_safe_family_assignment(candidates, current_assignment=current)

    assert decision.selected_assignment == current
    assert decision.changed is False
    assert decision.emergency_duplicate_slots == ()


def test_monotonic_family_with_material_project_gap_is_feasible() -> None:
    candidates = (
        _candidate("easy", "EASY", 1.0),
        _candidate("normal", "NORMAL", 1.4),
        _candidate("hard", "HARD", 2.1),
        _candidate("expert", "EXPERT", 3.0),
    )

    decision = select_safe_family_assignment(
        candidates,
        current_assignment=_current("easy", "normal", "hard", "expert"),
    )

    assert decision.family_feasibility_status == "SATISFIED"
    assert decision.family_feasibility_reasons == ()


def test_inverted_best_effort_family_is_preserved_but_not_feasible() -> None:
    candidates = (
        _candidate("easy", "EASY", 2.0),
        _candidate("normal", "NORMAL", 2.0),
        _candidate("hard", "HARD", 2.0),
        _candidate("expert", "EXPERT", 2.0),
    )
    current = _current("easy", "normal", "hard", "expert")

    decision = select_safe_family_assignment(candidates, current_assignment=current)

    assert decision.selected_assignment == current
    assert decision.unique_payload_status == "SATISFIED"
    assert decision.family_feasibility_status == "UNAVAILABLE"
    assert "DIFFICULTY_ORDER" in decision.family_feasibility_reasons


def test_narrow_project_gap_is_not_production_feasible() -> None:
    candidates = (
        _candidate("easy", "EASY", 1.0),
        _candidate("normal", "NORMAL", 1.1),
        _candidate("hard", "HARD", 2.0),
        _candidate("expert", "EXPERT", 3.0),
    )

    decision = select_safe_family_assignment(
        candidates,
        current_assignment=_current("easy", "normal", "hard", "expert"),
    )

    assert decision.family_feasibility_status == "UNAVAILABLE"
    assert "PROJECT_RATING_SEPARATION" in decision.family_feasibility_reasons
    assert decision.selected_score.minimum_project_rating_gap == pytest.approx(0.1)


def test_narrow_current_family_can_use_a_safe_candidate_with_material_gap() -> None:
    candidates = (
        _candidate("easy", "EASY", 1.0),
        _candidate("normal", "NORMAL", 1.1),
        _candidate("hard", "HARD", 2.0),
        _candidate("expert", "EXPERT", 3.0),
        _candidate("normal-wide", "NORMAL", 1.4),
    )

    decision = select_safe_family_assignment(
        candidates,
        current_assignment=_current("easy", "normal", "hard", "expert"),
    )

    assert decision.family_feasibility_status == "SATISFIED"
    assert dict(decision.selected_assignment)["NORMAL"] == "normal-wide"
    assert decision.selected_score.minimum_project_rating_gap == pytest.approx(0.4)


def test_payload_hash_not_candidate_id_defines_emergency_duplication() -> None:
    candidates = (
        _candidate("easy", "EASY", 1.0),
        _candidate("normal", "NORMAL", 2.0),
        _candidate("hard", "HARD", 3.0, payload="a"),
        _candidate("alias", "EXPERT", 4.0, payload="a"),
    )
    candidates = (
        candidates[0],
        candidates[1],
        _candidate("hard", "HARD", 3.0, payload="a"),
        candidates[3],
    )

    decision = select_safe_family_assignment(
        candidates,
        current_assignment=_current("easy", "normal", "hard", "alias"),
    )

    assert decision.current_score.duplicate_payloads == 1
    assert decision.selected_score.duplicate_payloads == 1
    assert decision.unique_payload_status == "UNAVAILABLE"
    assert decision.reason == "UNIQUE_PAYLOAD_UNAVAILABLE"


def test_source_matching_later_slot_remains_primary_when_earlier_slot_reuses_it() -> None:
    candidates = (
        _candidate("easy", "EASY", 1.0),
        _candidate("normal", "NORMAL", 2.0),
        _candidate("expert", "EXPERT", 4.0),
    )

    decision = select_safe_family_assignment(
        candidates,
        current_assignment=_current("easy", "normal", "expert", "expert"),
    )

    assert decision.emergency_duplicate_slots == ("HARD",)
    assert decision.unique_payload_status == "UNAVAILABLE"


def test_unique_safe_alternative_is_selected_instead_of_duplicate_reuse() -> None:
    candidates = (
        _candidate("easy", "EASY", 1.0),
        _candidate("normal", "NORMAL", 2.0),
        _candidate("hard", "HARD", 3.0),
        _candidate("broken", "EXPERT", 4.0, hard_safe=False),
        _candidate("expert-alt", "HARD", 4.2),
    )

    decision = select_safe_family_assignment(
        candidates,
        current_assignment=_current("easy", "normal", "hard", "broken"),
    )

    assert dict(decision.selected_assignment)["EXPERT"] == "expert-alt"
    assert decision.unique_payload_status == "SATISFIED"
    assert decision.selected_score.duplicate_payloads == 0
    assert decision.emergency_duplicate_slots == ()


def test_family_level_duplicate_authorizes_a_non_regressing_compiled_candidate() -> None:
    candidates = (
        _candidate("easy", "EASY", 1.0),
        _candidate("normal", "NORMAL", 2.0),
        _candidate("hard", "HARD", 3.0, payload="a"),
        _candidate("expert-alias", "EXPERT", 4.0, payload="a"),
        _candidate("compiled-expert", "EXPERT", 4.2, provenance="SAFE_FALLBACK"),
    )

    decision = select_safe_family_assignment(
        candidates,
        current_assignment=_current("easy", "normal", "hard", "expert-alias"),
    )

    assert dict(decision.selected_assignment)["EXPERT"] == "compiled-expert"
    assert decision.unique_payload_status == "SATISFIED"
    assert decision.selected_score.duplicate_payloads == 0


def test_candidates_from_different_key_modes_are_rejected() -> None:
    candidate = _candidate("easy", "EASY", 1.0)
    mixed = SafeFamilyCandidate(
        candidate_id="six",
        candidate_payload_sha256="6" * 64,
        key_mode=6,
        source_difficulty="NORMAL",
        provenance="PRIMARY",
        hard_safe=True,
        intro_state="UNKNOWN",
        boundary_state="UNKNOWN",
        attack_gap_count=0,
        attack_gap_total_ms=0,
        active_gap_count=0,
        max_active_gap_ms=0,
        difficulty_scores=(("PROJECT_RATING", 2.0),),
        review_rank=0,
        publication_rank=0,
        recovery_trust_rank=0,
        matched_f1_50=0.8,
        attempt=1,
    )

    with pytest.raises(ValueError, match="one key mode"):
        select_safe_family_assignment(
            (candidate, mixed),
            current_assignment=_current("easy", "easy", "easy", "easy"),
        )


def test_large_candidate_repository_cannot_break_an_existing_safe_family() -> None:
    current_candidates = (
        _candidate("easy", "EASY", 1.0),
        _candidate("normal", "NORMAL", 2.0),
        _candidate("hard", "HARD", 3.0),
        _candidate("expert", "EXPERT", 4.0),
    )
    historical = tuple(
        _candidate(f"historical-{index:02d}", "EASY", float(index + 10))
        for index in range(30)
    )
    current = _current("easy", "normal", "hard", "expert")

    decision = select_safe_family_assignment(
        (*current_candidates, *historical),
        current_assignment=current,
    )

    assert decision.selected_assignment == current
    assert decision.candidates_evaluated == 34
    assert decision.assignments_evaluated == 256
