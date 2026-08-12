from chart_worker.analysis.chart_profile import (
    ChartQualityProfile,
    DifficultyProfile,
    HoldProfile,
    PatternProfile,
)
from chart_worker.analysis.difficulty_vector import DifficultyVectorV2
from chart_worker.validation.profile_review import review_profile
from chart_worker.validation.quality_gate import GateAction, GateAxis


def _profile(
    *,
    occupancy: tuple[float, ...] = (0.1, 0.1, 0.1, 0.1),
    releases: tuple[int, ...] = (1, 1, 1, 1),
    lane_imbalance: tuple[float, ...] = (0.1, 0.1, 0.1, 0.1),
    row_repeats: tuple[int, ...] = (1, 1, 1, 1),
    note_counts: tuple[int, ...] | None = None,
    active: tuple[bool, ...] | None = None,
) -> ChartQualityProfile:
    section_count = len(occupancy)
    assert all(
        len(values) == section_count
        for values in (releases, lane_imbalance, row_repeats)
    )
    return ChartQualityProfile(
        hold=HoldProfile(
            note_ratio=0.1,
            time_occupancy_ratio=0.1,
            mean_duration_ms=500.0,
            p95_duration_ms=500.0,
            max_duration_ms=500,
            max_concurrent=1,
            max_held_lane_ratio=0.1,
            max_release_count_250ms=max(releases, default=0),
            section_hold_counts=(1,) * section_count,
            section_occupancy_ratios=occupancy,
            section_release_counts_250ms=releases,
        ),
        pattern=PatternProfile(
            histogram={},
            sections=tuple({} for _ in range(section_count)),
            transition_counts={},
            longest_row_ngram_repeat=max(row_repeats, default=0),
            lane_usage_ratios=(0.25,) * 4,
            section_note_counts=note_counts or (400,) * section_count,
            section_lane_imbalances=lane_imbalance,
            section_longest_row_ngram_repeats=row_repeats,
        ),
        difficulty=DifficultyProfile(
            project_rating=2.0,
            avg_nps=2.0,
            p95_nps=3.0,
            peak_nps=4.0,
            chord_ratio=0.0,
            max_jack=1,
            section_peak_nps=(2.0,) * section_count,
        ),
        difficulty_vector_v2=DifficultyVectorV2.empty(section_count),
        active_section_mask=active or (True,) * section_count,
    )


def _pattern_decision(profile: ChartQualityProfile):
    decisions = review_profile(profile, key_mode=4, difficulty="HARD")
    assert len(decisions) == 1
    assert decisions[0].axis is GateAxis.PATTERN
    return decisions[0]


def test_extreme_single_section_hold_concentration_is_advisory():
    decision = _pattern_decision(
        _profile(occupancy=(0.05, 0.06, 0.82, 0.04))
    )

    assert decision.action is GateAction.PASS
    assert decision.reasons == ("HOLD_SECTION_OUTLIER",)


def test_zero_mad_does_not_invent_an_outlier():
    decision = _pattern_decision(_profile())

    assert decision.action is GateAction.PASS
    assert decision.reasons == ("INSUFFICIENT_PROFILE_VARIATION",)


def test_fewer_than_four_active_sections_is_insufficient():
    decision = _pattern_decision(
        _profile(
            occupancy=(0.05, 0.06, 0.82, 0.04),
            active=(True, True, True, False),
        )
    )

    assert decision.action is GateAction.PASS
    assert decision.reasons == ("INSUFFICIENT_PROFILE_VARIATION",)


def test_inactive_outlier_is_not_used_as_evidence():
    decision = _pattern_decision(
        _profile(
            occupancy=(0.05, 0.06, 0.07, 0.82, 0.04),
            releases=(1, 1, 1, 1, 1),
            lane_imbalance=(0.1, 0.1, 0.1, 0.1, 0.1),
            row_repeats=(1, 1, 1, 1, 1),
            active=(True, True, True, False, True),
        )
    )

    assert decision.action is GateAction.PASS
    assert decision.reasons == ()


def test_low_side_deviation_is_not_treated_as_concentration():
    decision = _pattern_decision(
        _profile(lane_imbalance=(0.023, 0.022, 0.022, 0.0))
    )

    assert decision.action is GateAction.PASS
    assert "LANE_IMBALANCE_SECTION_OUTLIER" not in decision.reasons


def test_lane_outlier_requires_enough_notes_for_statistical_support():
    decision = _pattern_decision(
        _profile(
            lane_imbalance=(0.02, 0.02, 1.0, 0.02),
            note_counts=(45, 45, 9, 45),
        )
    )

    assert decision.action is GateAction.PASS


def test_independent_profile_metrics_have_stable_advisory_reasons():
    cases = (
        (_profile(releases=(1, 1, 12, 2)), "HOLD_RELEASE_SECTION_OUTLIER"),
        (
            _profile(lane_imbalance=(0.10, 0.12, 0.90, 0.11)),
            "LANE_IMBALANCE_SECTION_OUTLIER",
        ),
        (_profile(row_repeats=(1, 1, 20, 2)), "ROW_LOOP_SECTION_OUTLIER"),
    )

    for profile, reason in cases:
        decision = _pattern_decision(profile)
        assert decision.action is GateAction.PASS
        assert decision.reasons == (reason,)


def test_multiple_outliers_keep_canonical_reason_order():
    decision = _pattern_decision(
        _profile(
            occupancy=(0.05, 0.06, 0.82, 0.04),
            releases=(1, 1, 12, 2),
        )
    )

    assert decision.reasons == (
        "HOLD_SECTION_OUTLIER",
        "HOLD_RELEASE_SECTION_OUTLIER",
    )


def test_key_mode_and_difficulty_are_validated_without_threshold_branches():
    profile = _profile()

    for key_mode in (4, 6, 7):
        assert review_profile(profile, key_mode=key_mode, difficulty="EXPERT")

    try:
        review_profile(profile, key_mode=5, difficulty="HARD")
    except ValueError as error:
        assert "key_mode" in str(error)
    else:
        raise AssertionError("unsupported key mode must fail")
