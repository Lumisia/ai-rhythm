from chart_worker.validation.difficulty_selector import (
    DifficultyCandidateView,
    compare_family_candidates,
)


def candidate(
    difficulty: str,
    candidate_id: str,
    seed: int,
    current_rating: float,
    v2_score: float | None,
) -> DifficultyCandidateView:
    return DifficultyCandidateView(
        candidate_id=candidate_id,
        difficulty=difficulty,
        seed=seed,
        attempt=1,
        provenance="PRIMARY",
        intro_anchor_covered=True,
        current_rating=current_rating,
        v2_ordering_score=v2_score,
        vector_v2={"orderingScore": v2_score} if v2_score is not None else None,
    )


def test_shadow_comparison_never_changes_the_current_assignment():
    easy = candidate("EASY", "easy", 1, 1.0, 10.0)
    normal_current = candidate("NORMAL", "normal-current", 2, 2.0, 30.0)
    normal_shadow = candidate("NORMAL", "normal-shadow", 3, 2.1, 20.0)
    hard = candidate("HARD", "hard", 4, 3.0, 25.0)
    expert = candidate("EXPERT", "expert", 5, 4.0, 40.0)
    pools = {
        "EASY": (easy,),
        "NORMAL": (normal_current, normal_shadow),
        "HARD": (hard,),
        "EXPERT": (expert,),
    }
    current = {
        "EASY": "easy",
        "NORMAL": "normal-current",
        "HARD": "hard",
        "EXPERT": "expert",
    }

    selected, comparison = compare_family_candidates(
        pools,
        current,
        mode="SHADOW_V2",
    )

    assert selected == current
    assert comparison.shadow_assignment["NORMAL"] == "normal-shadow"
    assert comparison.current_inversions == (("NORMAL", "HARD"),)
    assert comparison.shadow_inversions == ()
    assert comparison.reason == "FEWER_V2_INVERSIONS"


def test_current_mode_does_not_compute_a_shadow_assignment():
    current_candidate = candidate("EASY", "easy", 1, 1.0, 10.0)
    current = {"EASY": "easy", "NORMAL": None, "HARD": None, "EXPERT": None}

    selected, comparison = compare_family_candidates(
        {"EASY": (current_candidate,), "NORMAL": (), "HARD": (), "EXPERT": ()},
        current,
        mode="CURRENT",
    )

    assert selected == current
    assert comparison is None


def test_missing_current_vector_is_reported_without_crashing():
    easy = candidate("EASY", "easy", 1, 1.0, None)
    current = {"EASY": "easy", "NORMAL": None, "HARD": None, "EXPERT": None}

    selected, comparison = compare_family_candidates(
        {"EASY": (easy,), "NORMAL": (), "HARD": (), "EXPERT": ()},
        current,
        mode="SHADOW_V2",
    )

    assert selected == current
    assert comparison is not None
    assert comparison.reason == "CURRENT_CANDIDATE_MISSING_VECTOR"


def test_v2_keeps_current_candidates_when_ordering_quality_is_tied():
    easy_old = candidate("EASY", "easy-old", 1, 1.0, 10.0)
    easy_current = candidate("EASY", "easy-current", 2, 1.0, 10.0)
    normal = candidate("NORMAL", "normal", 3, 2.0, 20.0)
    hard = candidate("HARD", "hard", 4, 3.0, 30.0)
    expert = candidate("EXPERT", "expert", 5, 4.0, 40.0)
    pools = {
        "EASY": (easy_old, easy_current),
        "NORMAL": (normal,),
        "HARD": (hard,),
        "EXPERT": (expert,),
    }
    current = {
        "EASY": "easy-current",
        "NORMAL": "normal",
        "HARD": "hard",
        "EXPERT": "expert",
    }

    selected, comparison = compare_family_candidates(pools, current, mode="V2")

    assert selected == current
    assert comparison is not None
    assert comparison.reason == "SAME_SELECTION"


def test_v2_does_not_reintroduce_a_current_rating_inversion():
    easy = candidate("EASY", "easy", 1, 1.0, 10.0)
    normal_current = candidate("NORMAL", "normal-current", 2, 2.0, 30.1)
    normal_v2_better = candidate("NORMAL", "normal-v2-better", 3, 3.5, 20.0)
    hard = candidate("HARD", "hard", 4, 3.0, 30.0)
    expert = candidate("EXPERT", "expert", 5, 4.0, 40.0)
    pools = {
        "EASY": (easy,),
        "NORMAL": (normal_current, normal_v2_better),
        "HARD": (hard,),
        "EXPERT": (expert,),
    }
    current = {
        "EASY": "easy",
        "NORMAL": "normal-current",
        "HARD": "hard",
        "EXPERT": "expert",
    }

    selected, comparison = compare_family_candidates(pools, current, mode="V2")

    assert selected == current
    assert comparison is not None
    assert comparison.reason == "SAME_SELECTION"
