import pytest

from chart_worker.validation.intro_phrase_family import (
    IntroPhraseChartView,
    review_intro_phrase_pair,
)


def view(
    difficulty: str,
    first: int | None,
    second: int | None,
    beats: float | None,
    *,
    audio_supported: bool | None = True,
) -> IntroPhraseChartView:
    return IntroPhraseChartView(
        key_mode=4,
        difficulty=difficulty,
        first_row_ms=first,
        second_row_ms=second,
        post_first_gap_beats=beats,
        first_row_audio_supported=audio_supported,
        candidate_id=f"4K-{difficulty}",
        seed=7,
        attempt=1,
    )


@pytest.mark.parametrize(
    ("hard", "expert", "start_delta_beats"),
    [
        (view("HARD", 156, 355, 0.50), view("EXPERT", 0, 12_745, 32.07), 0.39),
        (view("HARD", 129, 174, 0.12), view("EXPERT", 0, 17_667, 48.58), 0.35),
    ],
)
def test_flags_measured_fog_and_koe_shapes_as_high_confidence_defects(
    hard,
    expert,
    start_delta_beats,
):
    result = review_intro_phrase_pair(
        hard,
        expert,
        start_delta_beats=start_delta_beats,
    )

    assert result.status == "DEFECT"
    assert result.reason == "ISOLATED_EXPERT_FIRST_ROW"
    assert result.should_recover is True
    assert result.should_block_publication is True


def test_salt_trace_shape_is_an_early_start_review_not_the_same_defect():
    result = review_intro_phrase_pair(
        view("HARD", 15_790, 15_915, 0.25, audio_supported=False),
        view("EXPERT", 0, 11_848, 23.70, audio_supported=False),
        start_delta_beats=31.58,
    )

    assert result.status == "REVIEW"
    assert result.reason == "EXPERT_EARLY_GHOST"
    assert result.should_recover is False
    assert result.should_block_publication is False


def test_shared_long_silence_is_kept_as_a_possible_musical_break():
    result = review_intro_phrase_pair(
        view("HARD", 100, 12_100, 24.0),
        view("EXPERT", 150, 12_350, 24.4),
        start_delta_beats=0.1,
    )

    assert result.status == "REVIEW"
    assert result.reason == "SHARED_LONG_SILENCE_POSSIBLE_NORMAL"
    assert result.should_recover is False


def test_late_expert_start_is_separate_from_post_first_gap_defect():
    result = review_intro_phrase_pair(
        view("HARD", 0, 250, 0.5),
        view("EXPERT", 12_000, 12_250, 0.5),
        start_delta_beats=24.0,
    )

    assert result.status == "REVIEW"
    assert result.reason == "EXPERT_LATE_START"


def test_missing_second_row_is_insufficient_not_a_defect():
    result = review_intro_phrase_pair(
        view("HARD", 0, 250, 0.5),
        view("EXPERT", 0, None, None),
        start_delta_beats=0.0,
    )

    assert result.status == "INSUFFICIENT"
    assert result.reason == "INSUFFICIENT_ROWS"
    assert result.should_block_publication is False


def test_report_exposes_measurements_thresholds_and_provisional_policy():
    result = review_intro_phrase_pair(
        view("HARD", 100, 300, 0.4),
        view("EXPERT", 0, 12_000, 24.0),
        start_delta_beats=0.2,
    )

    report = result.to_report()
    assert report["version"] == "intro-phrase-family-v1"
    assert report["policyState"] == "PROVISIONAL"
    assert report["status"] == "DEFECT"
    assert report["reason"] == "ISOLATED_EXPERT_FIRST_ROW"
    assert report["startDeltaMs"] == 100
    assert report["gapDeltaMs"] == 11_800
    assert report["gapRatio"] == 60.0
    assert report["thresholds"] == {
        "nearStartMaxMs": 1000,
        "nearStartMaxBeats": 1.0,
        "hardImmediateMaxMs": 1500,
        "hardImmediateMaxBeats": 2.0,
        "expertLongGapMinMs": 10000,
        "expertLongGapMinBeats": 16.0,
        "gapRatioMin": 8.0,
    }


@pytest.mark.parametrize(
    ("hard", "expert", "start_delta_beats"),
    [
        (view("HARD", 1_001, 1_201, 0.4), view("EXPERT", 0, 12_000, 24.0), 0.2),
        (view("HARD", 1_000, 1_200, 0.4), view("EXPERT", 0, 12_000, 24.0), 1.001),
        (view("HARD", 100, 1_601, 0.4), view("EXPERT", 0, 12_000, 24.0), 0.2),
        (view("HARD", 100, 1_600, 2.001), view("EXPERT", 0, 12_000, 24.0), 0.2),
        (view("HARD", 100, 300, 0.4), view("EXPERT", 0, 9_999, 24.0), 0.2),
        (view("HARD", 100, 300, 0.4), view("EXPERT", 0, 10_000, 15.999), 0.2),
        (view("HARD", 100, 1_600, 2.0), view("EXPERT", 0, 10_000, 16.0), 0.2),
    ],
)
def test_does_not_flag_when_any_high_confidence_boundary_is_missed(
    hard,
    expert,
    start_delta_beats,
):
    result = review_intro_phrase_pair(
        hard,
        expert,
        start_delta_beats=start_delta_beats,
    )

    assert result.status != "DEFECT"
    assert result.should_block_publication is False


def test_inclusive_high_confidence_boundaries_are_a_defect():
    result = review_intro_phrase_pair(
        view("HARD", 1_000, 2_250, 2.0),
        view("EXPERT", 0, 10_000, 16.0),
        start_delta_beats=1.0,
    )

    assert result.status == "DEFECT"
    assert result.gap_ratio == 8.0


def test_rejects_mismatched_key_modes_and_reversed_rows():
    with pytest.raises(ValueError, match="same key mode"):
        review_intro_phrase_pair(
            IntroPhraseChartView(4, "HARD", 0, 100, 0.2),
            IntroPhraseChartView(6, "EXPERT", 0, 12_000, 24.0),
            start_delta_beats=0.0,
        )

    with pytest.raises(ValueError, match="second_row_ms"):
        IntroPhraseChartView(4, "EXPERT", 100, 0, 0.2)
