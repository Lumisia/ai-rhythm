from chart_worker.validation.outro_family_review import (
    OutroChartView,
    review_outro_family,
)


def _view(
    key_mode: int,
    difficulty: str,
    last_start_ms: int,
    last_end_ms: int | None = None,
) -> OutroChartView:
    return OutroChartView(
        key_mode=key_mode,
        difficulty=difficulty,
        last_note_start_ms=last_start_ms,
        last_note_end_ms=last_end_ms if last_end_ms is not None else last_start_ms,
    )


def test_flags_an_ivory_shaped_early_key_against_two_agreeing_siblings():
    review = review_outro_family(
        (
            _view(4, "HARD", 99_200, 99_587),
            _view(6, "HARD", 99_400, 99_887),
            _view(7, "HARD", 89_200, 89_587),
        )
    )

    assert review.mode == "SHADOW"
    assert review.status == "REVIEW"
    assert len(review.findings) == 1
    finding = review.findings[0]
    assert finding.reason == "OUTRO_FAMILY_EARLY_START"
    assert finding.key_mode == 7
    assert finding.difficulty == "HARD"
    assert finding.sibling_key_modes == (4, 6)
    assert finding.reference_start_ms == 99_300
    assert finding.early_by_ms == 10_100
    report = review.to_report()
    assert report["version"] == "outro-family-review-v3-tiered-start-shadow"
    assert report["mutatesCharts"] is False
    assert report["additionalInferenceCalls"] == 0
    assert report["findings"][0]["supportLevel"] == "TWO_SIBLING_CONSENSUS"


def test_flags_a_final_step_shaped_early_key_but_does_not_modify_any_chart():
    charts = (
        _view(4, "NORMAL", 89_500, 90_000),
        _view(6, "NORMAL", 96_000, 96_400),
        _view(7, "NORMAL", 96_200, 96_600),
    )

    review = review_outro_family(charts)

    assert review.status == "REVIEW"
    assert review.findings[0].key_mode == 4
    assert review.findings[0].early_by_ms == 6_600
    assert review.charts == charts


def test_flags_a_large_single_sibling_gap_as_provisional_when_third_key_is_missing():
    review = review_outro_family(
        (
            _view(4, "NORMAL", 163_475),
            _view(7, "NORMAL", 170_075),
        )
    )

    assert review.status == "REVIEW"
    assert len(review.findings) == 1
    finding = review.findings[0]
    assert finding.key_mode == 4
    assert finding.reason == "OUTRO_FAMILY_EARLY_START_SINGLE_SIBLING"
    assert finding.support_level == "SINGLE_SIBLING_PROVISIONAL"
    assert finding.early_by_ms == 6_600


def test_naturally_divergent_siblings_do_not_create_a_false_consensus():
    review = review_outro_family(
        (
            _view(4, "EXPERT", 89_500, 90_000),
            _view(6, "EXPERT", 93_500, 94_000),
            _view(7, "EXPERT", 99_500, 100_000),
        )
    )

    assert review.status == "PASS"
    assert review.findings == ()


def test_hold_release_consensus_does_not_masquerade_as_late_attack_consensus():
    review = review_outro_family(
        (
            _view(4, "HARD", 174_182, 174_182),
            _view(6, "HARD", 169_679, 173_831),
            _view(7, "HARD", 169_679, 169_679),
        )
    )

    # The 4K late tap and 6K earlier HOLD happen to end together, but the two
    # siblings do not agree that a late *attack* exists.  This is therefore not
    # evidence that 7K omitted a final playable onset.
    assert review.status == "PASS"
    assert review.findings == ()


def test_only_compares_sibling_keys_within_the_same_difficulty():
    review = review_outro_family(
        (
            _view(4, "HARD", 80_000),
            _view(6, "EXPERT", 100_000),
            _view(7, "EXPERT", 100_200),
        )
    )

    assert review.status == "PASS"
    assert review.findings == ()
