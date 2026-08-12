import numpy as np
import pytest

from chart_worker.analysis.activity import AudioActivity
from chart_worker.analysis.chart_profile import build_chart_quality_profile
from chart_worker.schema.note import NoteEvent


def _tap(time_ms: int, lane: int) -> NoteEvent:
    return NoteEvent(time_ms=time_ms, lane=lane)


def _hold(time_ms: int, lane: int, duration_ms: int) -> NoteEvent:
    return NoteEvent(
        time_ms=time_ms,
        lane=lane,
        kind="HOLD",
        duration_ms=duration_ms,
    )


def test_hold_profile_measures_time_occupancy_and_release_density():
    notes = [_hold(0, 0, 2_000), _hold(1_000, 1, 3_000), _tap(4_000, 2)]

    profile = build_chart_quality_profile(
        notes,
        key_mode=4,
        duration_ms=10_000,
        beat_ms=500.0,
        activity=None,
    )

    assert profile.hold.note_ratio == pytest.approx(2 / 3)
    assert profile.hold.time_occupancy_ratio == pytest.approx(5_000 / 40_000)
    assert profile.hold.mean_duration_ms == pytest.approx(2_500)
    assert profile.hold.p95_duration_ms == pytest.approx(2_950)
    assert profile.hold.max_concurrent == 2
    assert profile.hold.max_duration_ms == 3_000
    assert profile.hold.max_held_lane_ratio == pytest.approx(0.3)
    assert profile.hold.max_release_count_250ms == 1
    assert profile.hold.section_hold_counts == (2,)
    assert profile.hold.section_occupancy_ratios == pytest.approx((0.125,))
    assert profile.active_section_mask == (True,)


def test_release_burst_and_same_lane_overlap_do_not_double_count_occupancy():
    notes = [
        _hold(0, 0, 1_000),
        _hold(100, 0, 1_000),
        _hold(250, 1, 1_000),
    ]

    profile = build_chart_quality_profile(
        notes,
        key_mode=4,
        duration_ms=2_000,
        beat_ms=500.0,
        activity=None,
    )

    assert profile.hold.time_occupancy_ratio == pytest.approx(2_100 / 8_000)
    assert profile.hold.max_concurrent == 2
    assert profile.hold.max_release_count_250ms == 3
    assert profile.hold.section_release_counts_250ms == (3,)


def test_release_at_audio_end_belongs_to_the_last_section():
    profile = build_chart_quality_profile(
        [_hold(0, 0, 1_000)],
        key_mode=4,
        duration_ms=1_000,
        beat_ms=500.0,
        activity=None,
    )

    assert profile.hold.max_release_count_250ms == 1
    assert profile.hold.section_release_counts_250ms == (1,)


def test_pattern_profile_keeps_sections_and_longest_row_loop():
    notes = [_tap(time_ms, time_ms // 500 % 2) for time_ms in range(0, 20_000, 500)]

    profile = build_chart_quality_profile(
        notes,
        key_mode=4,
        duration_ms=20_000,
        beat_ms=500.0,
        activity=None,
    )

    assert len(profile.pattern.sections) == 2
    assert profile.pattern.longest_row_ngram_repeat >= 3
    assert profile.pattern.section_longest_row_ngram_repeats[0] >= 3
    assert sum(profile.pattern.histogram.values()) > 0
    assert sum(profile.pattern.lane_usage_ratios) == pytest.approx(1.0)
    assert profile.pattern.section_note_counts == (30, 10)
    assert profile.active_section_mask == (True, True)


def test_audio_activity_controls_active_section_mask():
    notes = [_tap(1_000, 0), _tap(16_000, 1), _tap(31_000, 2)]
    activity = AudioActivity(
        frame_ms=1_000.0,
        rms_db=np.array([-10.0] * 15 + [-30.0] * 15 + [-10.0] * 5),
        floor_db=-20.0,
        active_onset_ms=(),
    )

    profile = build_chart_quality_profile(
        notes,
        key_mode=4,
        duration_ms=35_000,
        beat_ms=500.0,
        activity=activity,
    )

    assert profile.active_section_mask == (True, False, True)
    assert len(profile.hold.section_occupancy_ratios) == 3
    assert len(profile.difficulty.section_peak_nps) == 3


def test_hold_carried_from_previous_section_keeps_section_active_without_audio_activity():
    profile = build_chart_quality_profile(
        [_hold(14_000, 0, 16_000)],
        key_mode=4,
        duration_ms=30_000,
        beat_ms=500.0,
        activity=None,
    )

    assert profile.active_section_mask == (True, True)
    assert profile.hold.section_occupancy_ratios[1] > 0


def test_empty_chart_has_zero_profiles_and_inactive_sections():
    profile = build_chart_quality_profile(
        [],
        key_mode=4,
        duration_ms=20_000,
        beat_ms=500.0,
        activity=None,
    )

    assert profile.active_section_mask == (False, False)
    assert profile.hold.note_ratio == 0.0
    assert profile.hold.section_hold_counts == (0, 0)
    assert profile.pattern.histogram == {}
    assert profile.pattern.lane_usage_ratios == (0.0, 0.0, 0.0, 0.0)
    assert profile.difficulty.project_rating == 0.0


@pytest.mark.parametrize(
    ("key_mode", "duration_ms", "beat_ms"),
    [(0, 1_000, 500.0), (4, 0, 500.0), (4, 1_000, 0.0)],
)
def test_profile_rejects_invalid_dimensions(key_mode, duration_ms, beat_ms):
    with pytest.raises(ValueError):
        build_chart_quality_profile(
            [],
            key_mode=key_mode,
            duration_ms=duration_ms,
            beat_ms=beat_ms,
            activity=None,
        )


def test_profile_report_uses_stable_camel_case_fields():
    profile = build_chart_quality_profile(
        [_tap(time_ms, time_ms // 500 % 2) for time_ms in range(0, 4_000, 500)],
        key_mode=4,
        duration_ms=4_000,
        beat_ms=500.0,
        activity=None,
    )
    report = profile.to_report()

    assert set(report) == {
        "activeSectionMask",
        "difficultyProfile",
        "difficultyVectorV2",
        "holdProfile",
        "patternProfile",
    }
    assert report["difficultyProfile"]["projectRating"] > 0
    assert report["difficultyVectorV2"]["version"] == "difficulty-vector-v2"
    assert report["difficultyVectorV2"]["orderingScore"] > 0
    with pytest.raises(TypeError):
        profile.pattern.histogram["JACK"] = 99
    with pytest.raises(TypeError):
        profile.pattern.sections[0]["JACK"] = 99
