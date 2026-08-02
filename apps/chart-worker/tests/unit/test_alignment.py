import pytest

from chart_worker.report.alignment import align_notes
from chart_worker.schema.note import NoteEvent


def test_alignment_uses_each_note_time_and_drum_onset_once():
    notes = [NoteEvent(100, 0), NoteEvent(120, 1), NoteEvent(400, 0)]
    result = align_notes(notes, (105, 410, 900), snap_window_ms=50)

    assert result.matched_pairs == ((100, 105), (400, 410))
    assert result.auto_play_onsets == (900,)
    assert result.drum_coverage == pytest.approx(2 / 3, abs=1e-4)
    assert result.drum_precision == pytest.approx(2 / 3, abs=1e-4)
    assert result.mean_abs_err_ms == 7.5


def test_alignment_counts_a_chord_as_one_music_event():
    notes = [NoteEvent(100, 0), NoteEvent(100, 1)]
    result = align_notes(notes, (105,), snap_window_ms=50)

    assert result.matched_pairs == ((100, 105),)
    assert result.drum_coverage == 1.0
    assert result.drum_precision == 1.0


def test_alignment_includes_the_snap_window_boundary():
    result = align_notes([NoteEvent(100, 0)], (150,), snap_window_ms=50)
    assert result.matched_pairs == ((100, 150),)


def test_alignment_sorts_and_deduplicates_drum_onsets():
    result = align_notes([NoteEvent(100, 0)], (900, 105, 105), snap_window_ms=50)
    assert result.matched_pairs == ((100, 105),)
    assert result.auto_play_onsets == (900,)


@pytest.mark.parametrize(
    ("notes", "onsets", "expected_auto"),
    [([], (100,), (100,)), ([NoteEvent(100, 0)], (), ())],
)
def test_alignment_handles_an_empty_side(notes, onsets, expected_auto):
    result = align_notes(notes, onsets)
    assert result.auto_play_onsets == expected_auto
    assert result.drum_coverage == 0.0
    assert result.drum_precision == 0.0
    assert result.mean_abs_err_ms == 0.0


def test_alignment_rejects_a_negative_window():
    with pytest.raises(ValueError, match="non-negative"):
        align_notes([], (), snap_window_ms=-1)
