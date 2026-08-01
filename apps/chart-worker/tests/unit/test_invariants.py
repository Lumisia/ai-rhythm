import dataclasses

from chart_worker.schema.note import NoteEvent
from chart_worker.validation.invariants import check_hold_only_shrinks, check_timing_invariant


def _before():
    return [
        NoteEvent(time_ms=100, lane=0),
        NoteEvent(time_ms=200, lane=1, kind="HOLD", duration_ms=500),
        NoteEvent(time_ms=300, lane=2),
    ]


def test_lane_move_and_deletion_keep_timing():
    before = _before()
    assert check_timing_invariant(before, [dataclasses.replace(before[0], lane=3)])


def test_new_or_shifted_time_breaks_timing():
    before = _before()
    assert not check_timing_invariant(before, before + [NoteEvent(time_ms=999, lane=0)])
    assert not check_timing_invariant(before, [dataclasses.replace(before[0], time_ms=105)])


def test_duplicate_at_existing_time_breaks_timing():
    before = _before()
    assert not check_timing_invariant(before, before + [NoteEvent(time_ms=100, lane=3)])


def test_hold_can_shrink_or_become_tap_but_not_grow():
    before = _before()
    shorter = [dataclasses.replace(n, duration_ms=200) if n.kind == "HOLD" else n for n in before]
    longer = [dataclasses.replace(n, duration_ms=900) if n.kind == "HOLD" else n for n in before]
    tap = [
        dataclasses.replace(n, kind="TAP", duration_ms=None) if n.kind == "HOLD" else n
        for n in before
    ]
    assert check_hold_only_shrinks(before, shorter)
    assert check_hold_only_shrinks(before, tap)
    assert not check_hold_only_shrinks(before, longer)


def test_each_duplicate_hold_must_match_a_distinct_original_duration():
    before = [
        NoteEvent(time_ms=100, lane=0, kind="HOLD", duration_ms=100),
        NoteEvent(time_ms=100, lane=0, kind="HOLD", duration_ms=500),
    ]
    after = [
        NoteEvent(time_ms=100, lane=0, kind="HOLD", duration_ms=500),
        NoteEvent(time_ms=100, lane=0, kind="HOLD", duration_ms=500),
    ]
    assert not check_hold_only_shrinks(before, after)
