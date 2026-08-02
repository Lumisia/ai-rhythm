import pytest

from chart_worker.errors import ErrorCode, WorkerError
from chart_worker.schema.note import NoteEvent
from chart_worker.validation.playability import (
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
