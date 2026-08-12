import pytest

from chart_worker.analysis.chart_profile import DifficultyProfile
from chart_worker.validation.difficulty_order import review_difficulty_order


def _profile(rating: float) -> DifficultyProfile:
    return DifficultyProfile(
        project_rating=rating,
        avg_nps=rating,
        p95_nps=rating,
        peak_nps=rating,
        chord_ratio=0.0,
        max_jack=1,
        section_peak_nps=(rating,),
    )


def test_reports_only_the_inverted_adjacent_pair():
    review = review_difficulty_order(
        {
            "EASY": _profile(1.2),
            "NORMAL": _profile(2.4),
            "HARD": _profile(4.1),
            "EXPERT": _profile(3.0),
        }
    )

    assert review.status == "RETRY"
    assert review.inverted_pairs == (("HARD", "EXPERT"),)
    assert review.ambiguous_pairs == ()
    assert review.retry_difficulties == frozenset({"HARD", "EXPERT"})


def test_equal_profiles_pass_with_ambiguity_metadata():
    review = review_difficulty_order(
        {difficulty: _profile(2.0) for difficulty in ("EASY", "NORMAL", "HARD", "EXPERT")}
    )

    assert review.status == "PASS"
    assert review.inverted_pairs == ()
    assert review.ambiguous_pairs == (
        ("EASY", "NORMAL"),
        ("NORMAL", "HARD"),
        ("HARD", "EXPERT"),
    )
    assert review.retry_difficulties == frozenset()


def test_separate_inversion_is_retried_while_ambiguity_is_preserved():
    review = review_difficulty_order(
        {
            "EASY": _profile(2.0),
            "NORMAL": _profile(2.0),
            "HARD": _profile(4.0),
            "EXPERT": _profile(3.0),
        }
    )

    assert review.status == "RETRY"
    assert review.ambiguous_pairs == (("EASY", "NORMAL"),)
    assert review.inverted_pairs == (("HARD", "EXPERT"),)
    assert review.retry_difficulties == frozenset({"HARD", "EXPERT"})


def test_monotonic_ratings_pass_and_serialize_in_label_order():
    review = review_difficulty_order(
        {
            "EXPERT": _profile(4.0),
            "EASY": _profile(1.0),
            "HARD": _profile(3.0),
            "NORMAL": _profile(2.0),
        }
    )

    assert review.status == "PASS"
    assert review.ordered_ratings == (
        ("EASY", 1.0),
        ("NORMAL", 2.0),
        ("HARD", 3.0),
        ("EXPERT", 4.0),
    )
    assert review.to_report() == {
        "status": "PASS",
        "orderedRatings": {
            "EASY": 1.0,
            "NORMAL": 2.0,
            "HARD": 3.0,
            "EXPERT": 4.0,
        },
        "invertedPairs": [],
        "ambiguousPairs": [],
        "narrowPairs": [],
        "retryDifficulties": [],
    }


def test_rejects_unknown_labels():
    with pytest.raises(ValueError, match="unknown difficulties"):
        review_difficulty_order({"EASY": _profile(1.0), "LUNATIC": _profile(9.0)})


def test_requires_at_least_one_label():
    with pytest.raises(ValueError, match="at least one"):
        review_difficulty_order({})


def test_reviews_a_subset_when_a_label_has_no_publishable_candidate():
    """발행 가능한 후보가 없는 난이도는 단조성 검사에서 빠진다.

    망가진 조합을 기준점으로 삼아 멀쩡한 조합을 끌어내리면 안 된다.
    """
    review = review_difficulty_order(
        {"EASY": _profile(1.0), "NORMAL": _profile(2.0), "HARD": _profile(3.0)}
    )

    assert review.status == "PASS"
    assert review.ordered_ratings == (("EASY", 1.0), ("NORMAL", 2.0), ("HARD", 3.0))
    assert review.retry_difficulties == frozenset()


def test_adjacent_gap_below_minimum_is_reported_as_narrow():
    review = review_difficulty_order(
        {
            "EASY": _profile(1.0),
            "NORMAL": _profile(1.1),
            "HARD": _profile(3.0),
            "EXPERT": _profile(4.0),
        }
    )

    # 역전은 아니므로 차단하지 않는다. 라벨이 겹친다는 근거만 남긴다.
    assert review.status == "PASS"
    assert review.narrow_pairs == (("EASY", "NORMAL"),)
    assert review.to_report()["narrowPairs"] == [["EASY", "NORMAL"]]
