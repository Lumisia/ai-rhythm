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


def test_equal_profiles_require_review_not_arbitrary_relabel():
    review = review_difficulty_order(
        {difficulty: _profile(2.0) for difficulty in ("EASY", "NORMAL", "HARD", "EXPERT")}
    )

    assert review.status == "REVIEW"
    assert review.inverted_pairs == ()
    assert review.ambiguous_pairs == (
        ("EASY", "NORMAL"),
        ("NORMAL", "HARD"),
        ("HARD", "EXPERT"),
    )
    assert review.retry_difficulties == frozenset()


def test_ambiguity_takes_precedence_over_a_separate_inversion():
    review = review_difficulty_order(
        {
            "EASY": _profile(2.0),
            "NORMAL": _profile(2.0),
            "HARD": _profile(4.0),
            "EXPERT": _profile(3.0),
        }
    )

    assert review.status == "REVIEW"
    assert review.ambiguous_pairs == (("EASY", "NORMAL"),)
    assert review.inverted_pairs == (("HARD", "EXPERT"),)
    assert review.retry_difficulties == frozenset()


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
        "retryDifficulties": [],
    }


def test_requires_exactly_the_four_supported_labels():
    with pytest.raises(ValueError, match="exactly"):
        review_difficulty_order({"EASY": _profile(1.0)})
