from chart_worker.generation.timing_family_recovery import timing_family_reviews


def test_timing_family_review_keeps_missing_siblings_insufficient():
    reviews = timing_family_reviews([])

    assert len(reviews) == 4
    assert {review.status for review in reviews} == {"INSUFFICIENT"}
    assert all(review.target_key_mode is None for review in reviews)
