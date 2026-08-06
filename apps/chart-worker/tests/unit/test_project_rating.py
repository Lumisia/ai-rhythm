import numpy as np
import pytest

from chart_worker.rating.project_rating import measure_rating, tier_of
from chart_worker.schema.note import NoteEvent


@pytest.mark.parametrize(
    ("rating", "tier"),
    [(0, "EASY"), (1.99, "EASY"), (2.0, "NORMAL"), (3.2, "HARD"), (4.5, "EXPERT")],
)
def test_tier_boundaries(rating, tier):
    assert tier_of(rating) == tier


def test_empty_chart_is_zero():
    metrics = measure_rating([], duration_ms=10_000)
    assert (metrics.rating, metrics.tier) == (0.0, "EASY")


def test_nonempty_chart_rejects_nonpositive_duration():
    with pytest.raises(ValueError, match="duration_ms"):
        measure_rating([NoteEvent(time_ms=0, lane=0)], duration_ms=0)


def test_counts_ratios_jacks_and_formula():
    notes = [NoteEvent(time_ms=time, lane=time % 4) for time in range(0, 4000, 250)]
    metrics = measure_rating(notes, duration_ms=4000)
    expected = (
        0.38 * metrics.p95_nps
        + 1.10 * metrics.chord_ratio
        + 0.15 * max(0, metrics.max_jack - 2)
        + 0.60 * metrics.hold_ratio
        + 0.10 * metrics.avg_nps
    )
    assert metrics.rating == pytest.approx(round(expected, 2))


def test_one_second_windows_do_not_double_count_boundary_note():
    notes = [NoteEvent(time_ms=0, lane=0), NoteEvent(time_ms=1000, lane=1)]
    metrics = measure_rating(notes, duration_ms=2000)
    assert metrics.peak_nps == 1.0


def test_reports_hand_checked_note_metrics():
    notes = [
        NoteEvent(time_ms=0, lane=0, kind="HOLD", duration_ms=250),
        NoteEvent(time_ms=0, lane=1),
        NoteEvent(time_ms=250, lane=0),
        NoteEvent(time_ms=500, lane=0),
    ]
    metrics = measure_rating(notes, duration_ms=2000)
    assert metrics.note_count == 4
    assert metrics.hold_count == 1
    assert metrics.avg_nps == 2.0
    assert metrics.peak_nps == 4.0
    assert metrics.chord_ratio == 0.5
    assert metrics.max_jack == 3
    assert metrics.hold_ratio == 0.25


def test_jack_run_stops_after_250ms_gap():
    notes = [
        NoteEvent(time_ms=0, lane=0),
        NoteEvent(time_ms=250, lane=0),
        NoteEvent(time_ms=501, lane=0),
    ]

    assert measure_rating(notes, duration_ms=1000).max_jack == 2


@pytest.mark.parametrize(
    "note",
    [
        NoteEvent(time_ms=1000, lane=0),
        NoteEvent(time_ms=900, lane=0, kind="HOLD", duration_ms=101),
    ],
)
def test_rejects_notes_outside_chart_duration(note):
    with pytest.raises(ValueError, match="duration_ms"):
        measure_rating([note], duration_ms=1000)


@pytest.mark.parametrize("duration_ms", [1000.5, True, np.float64(1000.0)])
def test_rejects_non_integer_chart_duration(duration_ms):
    with pytest.raises(ValueError, match="duration_ms"):
        measure_rating([NoteEvent(time_ms=0, lane=0)], duration_ms=duration_ms)


def test_accepts_numpy_integer_chart_duration():
    """오디오 분석이 넘기는 numpy 길이를 그대로 받는다."""
    metrics = measure_rating([NoteEvent(time_ms=0, lane=0)], duration_ms=np.int64(1000))
    assert metrics.note_count == 1
    assert metrics.avg_nps == 1.0
