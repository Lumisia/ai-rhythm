import pytest

from chart_worker.errors import ErrorCode, WorkerError
from chart_worker.schema.note import NoteEvent
from chart_worker.validation.playability import (
    PlayabilityViolation,
    ViolationCode,
    find_violations,
    validate_and_recover,
)


def test_recovery_reduces_a_forbidden_easy_hand_to_an_allowed_jump():
    notes = [
        NoteEvent(500, 0, onset_strength=0.9),
        NoteEvent(500, 1, onset_strength=0.1),
        NoteEvent(500, 2, onset_strength=0.2),
    ]
    result = validate_and_recover(
        notes,
        key_mode=4,
        difficulty="EASY",
        duration_ms=2_000,
        beat_ms=500.0,
    )
    assert [(note.time_ms, note.lane) for note in result.notes] == [(500, 0), (500, 2)]
    assert result.deleted_count == 1
    assert result.violations == ()


def test_recovery_keeps_only_the_stronger_duplicate_note():
    notes = [
        NoteEvent(500, 0, onset_strength=0.9),
        NoteEvent(500, 0, onset_strength=0.1),
    ]
    result = validate_and_recover(
        notes,
        key_mode=4,
        difficulty="NORMAL",
        duration_ms=2_000,
        beat_ms=500.0,
    )
    assert result.notes == [notes[0]]
    assert result.deleted_count == 1


def test_recovery_shrinks_a_hold_at_the_song_end():
    result = validate_and_recover(
        [NoteEvent(900, 0, kind="HOLD", duration_ms=300)],
        key_mode=4,
        difficulty="NORMAL",
        duration_ms=1_000,
        beat_ms=500.0,
    )
    [note] = result.notes
    assert (note.kind, note.duration_ms) == ("HOLD", 100)


def test_recovery_turns_a_too_short_end_hold_into_a_tap():
    result = validate_and_recover(
        [NoteEvent(980, 0, kind="HOLD", duration_ms=100)],
        key_mode=4,
        difficulty="NORMAL",
        duration_ms=1_000,
        beat_ms=500.0,
    )
    [note] = result.notes
    assert (note.kind, note.duration_ms) == ("TAP", None)


def test_recovery_moves_an_out_of_bounds_lane_without_changing_time():
    result = validate_and_recover(
        [NoteEvent(500, 8)],
        key_mode=4,
        difficulty="NORMAL",
        duration_ms=1_000,
        beat_ms=500.0,
    )
    [note] = result.notes
    assert (note.time_ms, note.lane, note.origin_lane) == (500, 3, 8)


def test_find_violations_reports_lane_rules_without_mutating_notes():
    notes = [NoteEvent(100, 0), NoteEvent(200, 0)]
    before = list(notes)
    violations = find_violations(
        notes,
        key_mode=4,
        difficulty="EASY",
        duration_ms=1_000,
        beat_ms=500.0,
    )
    assert ViolationCode.LANE_RULE in {violation.code for violation in violations}
    assert notes == before


def test_far_apart_same_lane_notes_are_not_a_jack_run():
    violations = find_violations(
        [NoteEvent(0, 0), NoteEvent(1_000, 0)],
        key_mode=4,
        difficulty="EASY",
        duration_ms=2_000,
        beat_ms=500.0,
    )
    assert ViolationCode.JACK_RUN not in {violation.code for violation in violations}


def test_validate_raises_worker_error_when_recovery_is_disabled():
    with pytest.raises(WorkerError) as caught:
        validate_and_recover(
            [NoteEvent(500, 0), NoteEvent(500, 1), NoteEvent(500, 2)],
            key_mode=4,
            difficulty="EASY",
            duration_ms=1_000,
            beat_ms=500.0,
            max_passes=0,
        )
    assert caught.value.code is ErrorCode.CHART_VALIDATION_FAILED


def test_recovery_handles_more_independent_violations_than_passes():
    notes = [
        NoteEvent(time_ms, lane, onset_strength=0.9 - lane * 0.1)
        for time_ms in range(1_000, 201_000, 20_000)
        for lane in range(3)
    ]
    result = validate_and_recover(
        notes,
        key_mode=4,
        difficulty="EASY",
        duration_ms=220_000,
        beat_ms=500.0,
        max_passes=3,
    )
    assert result.violations == ()
    assert result.deleted_count == 10
    assert all(len(row) == 2 for row in _notes_at_each_time(result.notes).values())


def _notes_at_each_time(notes):
    grouped = {}
    for note in notes:
        grouped.setdefault(note.time_ms, []).append(note)
    return grouped


def test_default_recovery_allows_a_four_step_progressing_repair(monkeypatch):
    def one_at_a_time(notes, **kwargs):
        del kwargs
        if len(notes) <= 1:
            return ()
        note = notes[0]
        return (
            PlayabilityViolation(
                ViolationCode.CHORD_LIMIT,
                note.time_ms,
                note.time_ms,
                (note.lane,),
                "forced cascading repair",
            ),
        )

    monkeypatch.setattr("chart_worker.validation.playability.find_violations", one_at_a_time)
    result = validate_and_recover(
        [NoteEvent(index * 1_000, index % 4) for index in range(5)],
        key_mode=4,
        difficulty="EASY",
        duration_ms=10_000,
        beat_ms=500.0,
    )
    assert result.deleted_count == 4
    assert result.passes == 5


def test_recovery_reduces_every_handstream_row_in_one_bounded_run():
    notes = [
        NoteEvent(index * 250, lane, onset_strength=0.9 - position * 0.1)
        for index in range(12)
        for position, lane in enumerate((0, 1, 2) if index % 2 == 0 else (3, 4, 5))
    ]
    result = validate_and_recover(
        notes,
        key_mode=6,
        difficulty="HARD",
        duration_ms=10_000,
        beat_ms=500.0,
    )
    assert result.violations == ()
    assert all(len(row) <= 2 for row in _notes_at_each_time(result.notes).values())
