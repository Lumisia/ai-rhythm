import pytest

from chart_worker.postprocess.patterns import (
    PatternInstance,
    PatternKind,
    detect_patterns,
    pattern_entropy,
    pattern_histogram,
    rows_of,
)
from chart_worker.schema.note import NoteEvent

BEAT_MS = 500.0  # 120 BPM


def _taps(pairs):
    return [NoteEvent(time_ms=time_ms, lane=lane) for time_ms, lane in pairs]


def _kinds(notes, *, key_mode=4, beat_ms=BEAT_MS):
    return {
        instance.kind for instance in detect_patterns(notes, key_mode=key_mode, beat_ms=beat_ms)
    }


def _instances(notes, kind, *, key_mode=4, beat_ms=BEAT_MS):
    return [
        instance
        for instance in detect_patterns(notes, key_mode=key_mode, beat_ms=beat_ms)
        if instance.kind is kind
    ]


# --- 기본 ------------------------------------------------------------------


def test_rows_group_simultaneous_notes():
    rows = rows_of(_taps([(0, 2), (0, 0), (100, 1)]))
    assert [(row.time_ms, row.lanes) for row in rows] == [(0, (0, 2)), (100, (1,))]


def test_beat_length_must_be_positive():
    with pytest.raises(ValueError, match="beat_ms"):
        detect_patterns(_taps([(0, 0)]), key_mode=4, beat_ms=0)


def test_empty_chart_has_no_patterns():
    assert detect_patterns([], key_mode=4, beat_ms=BEAT_MS) == []


# --- 화음 ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("lanes", "kind"),
    [((0, 1), PatternKind.JUMP), ((0, 1, 2), PatternKind.HAND), ((0, 1, 2, 3), PatternKind.QUAD)],
)
def test_chord_size_names_the_pattern(lanes, kind):
    notes = _taps([(0, lane) for lane in lanes])
    assert kind in _kinds(notes)


def test_five_note_chord_stays_a_quad_with_the_real_size():
    """해금표에 5노트 행이 없다. 새 종류를 만들면 집행에 구멍이 생긴다."""
    notes = _taps([(0, lane) for lane in range(5)])
    [quad] = _instances(notes, PatternKind.QUAD, key_mode=7)
    assert quad.size == 5


def test_grace_note_needs_different_lanes_within_a_sixth():
    fast = _taps([(0, 0), (int(BEAT_MS / 6) - 10, 1)])
    slow = _taps([(0, 0), (int(BEAT_MS / 6) + 50, 1)])
    assert PatternKind.GRACE in _kinds(fast)
    assert PatternKind.GRACE not in _kinds(slow)


def test_a_jack_is_not_a_grace_note():
    notes = _taps([(0, 0), (20, 0)])
    assert PatternKind.GRACE not in _kinds(notes)


# --- 잭 --------------------------------------------------------------------


def test_two_in_a_lane_is_a_minijack():
    notes = _taps([(0, 0), (250, 0), (500, 1)])
    assert PatternKind.MINIJACK in _kinds(notes)
    assert PatternKind.JACK not in _kinds(notes)


def test_three_in_a_lane_is_a_jack_but_not_a_longjack():
    notes = _taps([(0, 0), (250, 0), (500, 0), (750, 1)])
    kinds = _kinds(notes)
    assert PatternKind.JACK in kinds
    assert PatternKind.LONGJACK not in kinds


def test_four_in_a_lane_is_both_a_jack_and_a_longjack():
    """해금표가 Jack(3+)과 Longjack 을 따로 다룬다."""
    notes = _taps([(0, 0), (250, 0), (500, 0), (750, 0)])
    kinds = _kinds(notes)
    assert {PatternKind.JACK, PatternKind.LONGJACK} <= kinds


def test_another_lane_between_breaks_the_jack():
    notes = _taps([(0, 0), (125, 1), (250, 0)])
    assert PatternKind.MINIJACK not in _kinds(notes)


def test_anchor_is_a_half_beat_lane_with_others_interleaved():
    notes = _taps(
        [(0, 0), (125, 1), (250, 0), (375, 2), (500, 0), (625, 1), (750, 0), (875, 3)]
    )
    assert PatternKind.ANCHOR in _kinds(notes)


def test_chordjack_shares_two_lanes_between_chords():
    notes = _taps([(0, 0), (0, 1), (250, 0), (250, 1)])
    assert PatternKind.CHORDJACK in _kinds(notes)


def test_chords_sharing_one_lane_are_not_a_chordjack():
    notes = _taps([(0, 0), (0, 1), (250, 1), (250, 2)])
    assert PatternKind.CHORDJACK not in _kinds(notes)


# --- 스트림 ----------------------------------------------------------------


def test_even_single_notes_are_a_stream():
    notes = _taps([(index * 125, index % 4) for index in range(8)])
    assert PatternKind.STREAM in _kinds(notes)


def test_three_notes_are_too_few_for_a_stream():
    notes = _taps([(0, 0), (125, 1), (250, 2)])
    assert PatternKind.STREAM not in _kinds(notes)


def test_a_stream_with_jumps_is_a_jumpstream():
    notes = _taps([(0, 0), (125, 1), (125, 2), (250, 3), (375, 0), (500, 1)])
    assert PatternKind.JUMPSTREAM in _kinds(notes)


def test_a_stream_with_hands_is_a_handstream():
    notes = _taps([(0, 0), (125, 1), (125, 2), (125, 3), (250, 0), (375, 1), (500, 2)])
    kinds = _kinds(notes)
    assert PatternKind.HANDSTREAM in kinds
    assert PatternKind.CHORDSTREAM in kinds


def test_burst_is_a_short_fast_stream():
    fast = _taps([(index * 100, index % 4) for index in range(6)])
    slow = _taps([(index * 250, index % 4) for index in range(6)])
    assert PatternKind.BURST in _kinds(fast)
    assert PatternKind.BURST not in _kinds(slow)


def test_roll_crosses_every_lane_in_order():
    notes = _taps([(index * 125, index % 4) for index in range(8)])
    assert PatternKind.ROLL in _kinds(notes)


def test_a_stream_that_misses_a_lane_is_not_a_roll():
    notes = _taps([(index * 125, index % 3) for index in range(8)])
    assert PatternKind.ROLL not in _kinds(notes)


def test_stairs_move_one_lane_at_a_time():
    notes = _taps([(0, 0), (125, 1), (250, 2), (375, 3)])
    [stairs] = _instances(notes, PatternKind.STAIRS)
    assert stairs.lanes == (0, 1, 2, 3)


def test_a_lane_jump_of_two_breaks_stairs():
    notes = _taps([(0, 0), (125, 2), (250, 3)])
    assert PatternKind.STAIRS not in _kinds(notes)


# --- 트릴 ------------------------------------------------------------------


def test_two_handed_trill_crosses_the_hands():
    notes = _taps([(0, 0), (125, 3), (250, 0), (375, 3)])
    assert PatternKind.TRILL_TWO_HANDED in _kinds(notes)


def test_one_handed_trill_stays_on_one_side():
    notes = _taps([(0, 0), (125, 1), (250, 0), (375, 1)])
    assert PatternKind.TRILL_ONE_HANDED in _kinds(notes)


def test_a_trill_needs_three_notes():
    notes = _taps([(0, 0), (125, 1)])
    kinds = _kinds(notes)
    assert PatternKind.TRILL_ONE_HANDED not in kinds
    assert PatternKind.TRILL_TWO_HANDED not in kinds


def test_jumptrill_alternates_disjoint_pairs():
    notes = _taps([(0, 0), (0, 1), (125, 2), (125, 3), (250, 0), (250, 1)])
    assert PatternKind.JUMPTRILL in _kinds(notes)


def test_denim_alternates_lane_parity_across_more_than_two_lanes():
    notes = _taps([(0, 0), (125, 1), (250, 2), (375, 3), (500, 0), (625, 3), (750, 2)])
    assert PatternKind.DENIM in _kinds(notes)


def test_a_two_lane_trill_is_not_denim():
    notes = _taps([(index * 125, index % 2) for index in range(8)])
    assert PatternKind.DENIM not in _kinds(notes)


# --- 롱노트 ----------------------------------------------------------------


def test_a_hold_is_detected():
    notes = [NoteEvent(time_ms=0, lane=0, kind="HOLD", duration_ms=400)]
    assert PatternKind.HOLD in _kinds(notes)


def test_shield_is_a_tap_just_before_a_hold_in_the_same_lane():
    notes = [
        NoteEvent(time_ms=0, lane=0),
        NoteEvent(time_ms=200, lane=0, kind="HOLD", duration_ms=400),
    ]
    assert PatternKind.SHIELD in _kinds(notes)


def test_reverse_shield_is_a_tap_just_after_a_hold():
    notes = [
        NoteEvent(time_ms=0, lane=0, kind="HOLD", duration_ms=400),
        NoteEvent(time_ms=500, lane=0),
    ]
    assert PatternKind.REVERSE_SHIELD in _kinds(notes)


def test_a_far_tap_is_not_a_shield():
    notes = [
        NoteEvent(time_ms=0, lane=0),
        NoteEvent(time_ms=900, lane=0, kind="HOLD", duration_ms=400),
    ]
    assert PatternKind.SHIELD not in _kinds(notes)


def test_a_tap_in_another_lane_is_not_a_shield():
    notes = [
        NoteEvent(time_ms=0, lane=1),
        NoteEvent(time_ms=200, lane=0, kind="HOLD", duration_ms=400),
    ]
    assert PatternKind.SHIELD not in _kinds(notes)


def test_inverse_is_a_wall_of_touching_holds():
    notes = [
        NoteEvent(time_ms=index * 200, lane=index % 4, kind="HOLD", duration_ms=200)
        for index in range(5)
    ]
    assert PatternKind.INVERSE in _kinds(notes)


def test_holds_with_gaps_are_not_inverse():
    notes = [
        NoteEvent(time_ms=index * 500, lane=index % 4, kind="HOLD", duration_ms=200)
        for index in range(5)
    ]
    assert PatternKind.INVERSE not in _kinds(notes)


# --- 히스토그램과 엔트로피 --------------------------------------------------


def test_histogram_counts_only_what_was_found():
    notes = _taps([(0, 0), (0, 1), (250, 2)])
    histogram = pattern_histogram(detect_patterns(notes, key_mode=4, beat_ms=BEAT_MS))
    assert histogram == {PatternKind.JUMP: 1}


def test_histogram_keeps_the_enum_order():
    instances = [
        PatternInstance(PatternKind.HOLD, 0, 1, (0,), 1),
        PatternInstance(PatternKind.JUMP, 0, 0, (0, 1), 2),
    ]
    assert list(pattern_histogram(instances)) == [PatternKind.JUMP, PatternKind.HOLD]


def test_entropy_is_zero_for_a_single_pattern_kind():
    assert pattern_entropy({PatternKind.JUMP: 12}) == 0.0


def test_entropy_of_an_even_split_is_the_log_of_the_kind_count():
    even = dict.fromkeys(list(PatternKind)[:4], 5)
    assert pattern_entropy(even) == pytest.approx(2.0)


def test_entropy_of_nothing_is_zero():
    assert pattern_entropy({}) == 0.0


def test_a_varied_chart_scores_higher_than_a_repetitive_one():
    """엔트로피가 낮으면 같은 패턴만 반복한다는 뜻이다."""
    repetitive = _taps([(index * 250, 0) for index in range(16)])
    varied = _taps(
        [(0, 0), (0, 1), (125, 2), (250, 3), (375, 2), (500, 1), (625, 0), (750, 1), (875, 2)]
    )
    low = pattern_entropy(pattern_histogram(detect_patterns(repetitive, key_mode=4, beat_ms=BEAT_MS)))
    high = pattern_entropy(pattern_histogram(detect_patterns(varied, key_mode=4, beat_ms=BEAT_MS)))
    assert high > low
