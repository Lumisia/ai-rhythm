import math

import pytest

from chart_worker.analysis.chart_events import ChartEventIndex
from chart_worker.analysis.difficulty_vector import (
    DifficultyCalibration,
    measure_difficulty_vector,
)
from chart_worker.analysis.song_context import LocalTempoMap
from chart_worker.generation.osu_parser import OsuBpmEvent
from chart_worker.schema.note import NoteEvent


def vector(
    notes,
    *,
    key_mode=4,
    duration_ms=4_000,
    bpm_events=None,
):
    if bpm_events is None:
        bpm_events = (OsuBpmEvent(0, 120.0),)
    return measure_difficulty_vector(
        ChartEventIndex.build(notes, key_mode, duration_ms),
        LocalTempoMap(bpm_events),
    )


def test_chord_row_input_order_does_not_change_vector():
    notes = [NoteEvent(100, 0), NoteEvent(100, 2), NoteEvent(500, 1)]

    assert vector(notes) == vector(list(reversed(notes)))


def test_lane_permutation_within_same_hand_preserves_coordination():
    left_a = [NoteEvent(100, 0), NoteEvent(250, 1), NoteEvent(400, 0)]
    left_b = [NoteEvent(100, 1), NoteEvent(250, 0), NoteEvent(400, 1)]

    assert vector(left_a).coordination == vector(left_b).coordination


def test_dense_cross_hand_changes_have_more_coordination_strain_than_sparse_ones():
    sparse = [
        NoteEvent(time_ms, 0 if index % 2 == 0 else 3)
        for index, time_ms in enumerate(range(0, 60_000, 400))
    ]
    dense = [
        NoteEvent(time_ms, 0 if index % 2 == 0 else 3)
        for index, time_ms in enumerate(range(0, 60_000, 100))
    ]

    dense_coordination = vector(dense, duration_ms=60_000).coordination
    sparse_coordination = vector(sparse, duration_ms=60_000).coordination

    assert dense_coordination > sparse_coordination * 1.2


def test_silence_extension_preserves_peak_skill():
    notes = [NoteEvent(100, 0), NoteEvent(200, 1), NoteEvent(300, 2)]

    assert vector(notes, duration_ms=1_000).peak_skill == vector(
        notes,
        duration_ms=10_000,
    ).peak_skill


def test_empty_chart_is_finite_and_has_zero_stamina():
    result = vector([])

    assert all(math.isfinite(value) for value in result.numeric_axes())
    assert result.bounded_stamina == 0.0
    assert result.peak_skill == 0.0


def test_local_bpm_integrates_hold_duration_in_beats():
    result = vector(
        [NoteEvent(500, 0, kind="HOLD", duration_ms=1_000)],
        duration_ms=2_000,
        bpm_events=(
            OsuBpmEvent(0, 120.0),
            OsuBpmEvent(1_000, 240.0),
        ),
    )

    assert result.mean_hold_beats == pytest.approx(3.0)
    assert result.p95_hold_beats == pytest.approx(3.0)


def test_more_dense_physical_input_has_more_peak_skill():
    sparse = [NoteEvent(time_ms, time_ms // 500 % 4) for time_ms in range(0, 4_000, 500)]
    dense = [NoteEvent(time_ms, time_ms // 100 % 4) for time_ms in range(0, 4_000, 100)]

    assert vector(dense).peak_skill > vector(sparse).peak_skill


def test_stamina_is_bounded_by_the_geometric_decay_limit():
    notes = [
        NoteEvent(section * 400 + offset, offset // 100 % 4)
        for section in range(100)
        for offset in (10, 110, 210, 310)
    ]

    result = vector(notes, duration_ms=40_100)

    assert 0 < result.bounded_stamina <= 10.0


def test_uncalibrated_ordering_score_is_the_official_style_peak_skill():
    result = vector([NoteEvent(100, 0), NoteEvent(200, 1)])

    assert result.ordering_score == result.peak_skill


def test_calibration_requires_non_negative_matching_axes():
    with pytest.raises(ValueError, match="same length"):
        DifficultyCalibration(
            key_mode=4,
            axis_names=("density_strain",),
            medians=(0.0, 1.0),
            iqrs=(1.0,),
            weights=(1.0,),
            complete_song_count=11,
        )
    with pytest.raises(ValueError, match="non-negative"):
        DifficultyCalibration(
            key_mode=4,
            axis_names=("density_strain",),
            medians=(0.0,),
            iqrs=(1.0,),
            weights=(-1.0,),
            complete_song_count=11,
        )
