import pytest

from chart_worker.postprocess.lane_conversion import (
    MAX_PASSES,
    MOVE_BUDGET,
    convert_lanes,
)
from chart_worker.postprocess.lane_rules import FingerClass as Finger
from chart_worker.postprocess.lane_rules import Rule, check_lane_rules, finger_of
from chart_worker.schema.note import NoteEvent
from chart_worker.schema.types import lane_semantics

SIDE_L, MAIN_1, MAIN_2, CENTER, MAIN_3, MAIN_4, SIDE_R = range(7)


def _tap(time_ms, lane, **kwargs):
    return NoteEvent(time_ms=time_ms, lane=lane, **kwargs)


def _convert(notes, difficulty="HARD", key_mode=7, **kwargs):
    return convert_lanes(notes, key_mode=key_mode, difficulty=difficulty, **kwargs)


def _side_jack(count=4, gap=125, lane=SIDE_L):
    return [_tap(index * gap, lane) for index in range(count)]


def _chart_with_side_jack(*, filler=60, jack=4, gap=125):
    """합법적인 본문에 사이드 연타를 하나 심는다.

    이동 예산이 노트 수의 15% 라 짧은 채보로는 이동 자체를 검증할 수 없다.
    실제 채보(1000노트 이상)의 비율에 맞춘다.
    """
    body = [_tap(2000 + index * 400, MAIN_1 + index % 4) for index in range(filler)]
    return sorted(_side_jack(jack, gap) + body, key=lambda n: (n.time_ms, n.lane))


# --- 타이밍 불변 ------------------------------------------------------------


def test_conversion_never_changes_a_note_time():
    """Mapperatorinator 를 채택한 이유가 여기 걸려 있다."""
    notes = _chart_with_side_jack()
    result = _convert(notes)
    assert sorted(n.time_ms for n in result.notes) == sorted(n.time_ms for n in notes)


def test_result_times_are_a_subset_of_the_input():
    notes = _chart_with_side_jack(jack=8)
    result = _convert(notes, difficulty="EASY")
    assert {n.time_ms for n in result.notes} <= {n.time_ms for n in notes}


def test_holds_keep_their_length():
    notes = [
        NoteEvent(time_ms=0, lane=SIDE_L, kind="HOLD", duration_ms=800),
        _tap(100, MAIN_1),
    ]
    result = _convert(notes, difficulty="EXPERT")
    [hold] = [n for n in result.notes if n.kind == "HOLD"]
    assert hold.duration_ms == 800


def test_an_empty_chart_stays_empty():
    result = _convert([])
    assert result.notes == []
    assert result.moved_note_ratio == 0.0


# --- 이동 -------------------------------------------------------------------


def test_a_side_jack_is_moved_onto_main_lanes():
    """125ms 사이드 연타는 HARD 의 새끼손가락 한계(250ms)를 넘는다."""
    notes = _chart_with_side_jack()
    result = _convert(notes)
    assert result.moved_count > 0
    semantics = lane_semantics(7)
    moved = {
        finger_of(semantics[note.lane])
        for note in result.notes
        if note.origin_lane == SIDE_L and note.lane != SIDE_L
    }
    assert moved <= {Finger.MAIN}


def test_notes_are_moved_not_deleted():
    """삭제하면 음악 이벤트에 구멍이 생기고 키음이 안 울린다."""
    notes = _chart_with_side_jack()
    result = _convert(notes)
    assert len(result.notes) == len(notes)
    assert result.deleted_count == 0


def test_the_side_jack_becomes_a_trill_without_a_trill_rule():
    """w1 이 같은 레인 재사용을 눌러 교대 배분을 유도한다."""
    result = _convert(_chart_with_side_jack())
    lanes = [
        note.lane
        for note in sorted(result.notes, key=lambda n: n.time_ms)
        if note.origin_lane == SIDE_L
    ]
    assert len(set(lanes)) >= 2


def test_origin_lane_records_where_the_model_put_it():
    result = _convert(_chart_with_side_jack(filler=0, jack=40))
    assert all(note.origin_lane == SIDE_L for note in result.notes)


def test_a_legal_chart_is_left_alone():
    notes = [_tap(index * 600, lane) for index, lane in enumerate([SIDE_L, MAIN_1, MAIN_4, SIDE_R])]
    result = _convert(notes, difficulty="EASY")
    assert result.moved_count == 0
    assert [n.lane for n in result.notes] == [n.lane for n in notes]


def test_a_main_lane_jack_within_the_limit_is_untouched():
    notes = [_tap(index * 200, MAIN_1) for index in range(4)]
    result = _convert(notes, difficulty="HARD")
    assert result.moved_count == 0


# --- 규칙 해소 --------------------------------------------------------------


def test_conversion_clears_the_finger_limit_violations():
    notes = _chart_with_side_jack()
    before = check_lane_rules(notes, key_mode=7, difficulty="HARD")
    result = _convert(notes)
    assert before
    assert not [v for v in result.remaining_violations if v.rule is Rule.S1_JACK_INTERVAL]


def test_both_sides_at_once_is_split_at_easy():
    notes = _chart_with_side_jack(jack=0) + [_tap(1000, SIDE_L), _tap(1000, SIDE_R)]
    result = _convert(notes, difficulty="EASY")
    semantics = lane_semantics(7)
    sides = [n for n in result.notes if finger_of(semantics[n.lane]) is Finger.SIDE]
    assert len(sides) < 2


def test_center_with_both_sides_is_broken_up_below_expert():
    notes = _chart_with_side_jack(jack=0) + [
        _tap(1000, CENTER), _tap(1000, SIDE_L), _tap(1000, SIDE_R)
    ]
    result = _convert(notes, difficulty="HARD")
    assert not [
        v for v in result.remaining_violations if v.rule is Rule.C3_CENTER_WITH_BOTH_SIDES
    ]


def test_side_hold_density_is_relieved_by_moving_to_the_other_hand():
    notes = _chart_with_side_jack(jack=0) + [
        NoteEvent(time_ms=0, lane=SIDE_L, kind="HOLD", duration_ms=1000),
        *(_tap(index * 200, MAIN_1) for index in range(1, 5)),
    ]
    before = check_lane_rules(notes, key_mode=7, difficulty="NORMAL")
    result = _convert(notes, difficulty="NORMAL")
    assert [v.rule for v in before] == [Rule.S3_SIDE_HOLD_SAME_HAND]
    assert len(result.remaining_violations) < len(before)


def test_four_key_charts_need_no_conversion():
    notes = [_tap(index * 150, index % 4) for index in range(16)]
    result = _convert(notes, difficulty="EXPERT", key_mode=4)
    assert result.moved_count == 0
    assert result.remaining_violations == ()


# --- 예산 -------------------------------------------------------------------


def test_the_move_budget_is_a_fraction_of_the_chart():
    assert 0 < MOVE_BUDGET < 1


def test_moves_stop_at_the_budget():
    notes = _side_jack(count=40, gap=100)
    result = _convert(notes, difficulty="EASY", budget=0.1)
    assert result.moved_count <= int(len(notes) * 0.1)


def test_ratio_matches_the_move_count():
    notes = _chart_with_side_jack(jack=20)
    result = _convert(notes)
    assert result.moved_note_ratio == pytest.approx(result.moved_count / len(notes), abs=1e-4)


def test_unfixable_violations_are_deleted_weakest_first():
    """예산을 다 쓰고도 남은 위반은 지운다."""
    notes = [
        _tap(0, SIDE_L, onset_strength=0.9),
        _tap(60, SIDE_L, onset_strength=0.1),
    ]
    result = _convert(notes, difficulty="EASY", budget=0.0)
    assert result.deleted_count > 0
    assert result.budget_exhausted


def test_a_zero_budget_still_resolves_lane_collisions():
    """같은 시각 같은 레인 겹침은 예산과 무관하게 풀어야 한다."""
    notes = [_tap(1000, CENTER), _tap(1000, SIDE_L), _tap(1000, SIDE_R)]
    result = _convert(notes, difficulty="HARD", budget=0.0)
    times_lanes = [(n.time_ms, n.lane) for n in result.notes]
    assert len(times_lanes) == len(set(times_lanes))


# --- 재검사 -----------------------------------------------------------------


def test_conversion_reruns_until_clean():
    result = _convert(_chart_with_side_jack(jack=6))
    assert 1 <= result.passes <= MAX_PASSES


def test_a_clean_chart_needs_only_one_pass():
    notes = [_tap(index * 600, MAIN_1 + index % 4) for index in range(4)]
    assert _convert(notes, difficulty="EASY").passes == 1


def test_no_two_notes_share_a_lane_at_the_same_time():
    notes = _side_jack(count=12, gap=60)
    result = _convert(notes, difficulty="EASY")
    keys = [(n.time_ms, n.lane) for n in result.notes]
    assert len(keys) == len(set(keys))


# --- 검수 회귀 --------------------------------------------------------------


def test_a_moved_note_never_lands_inside_an_active_hold():
    """롱노트가 물고 있는 레인은 그 끝까지 비어 있지 않다."""
    notes = [NoteEvent(time_ms=0, lane=MAIN_1, kind="HOLD", duration_ms=1000)]
    notes += _side_jack(4, 125)
    notes += [_tap(2000 + i * 400, MAIN_2 + i % 3) for i in range(60)]
    result = _convert(notes)
    hold = next(n for n in result.notes if n.kind == "HOLD")
    end = hold.time_ms + hold.duration_ms
    inside = [
        n
        for n in result.notes
        if n.kind == "TAP" and n.lane == hold.lane and hold.time_ms < n.time_ms < end
    ]
    assert inside == []


def test_deletion_removes_one_note_per_violation_not_both_ends():
    """양쪽을 다 지우면 한 번만 지워도 될 자리에서 강한 타격까지 잃는다."""
    notes = [
        _tap(0, SIDE_L, onset_strength=0.9),
        _tap(100, SIDE_L, onset_strength=0.1),
        _tap(300, SIDE_L, onset_strength=0.8),
    ]
    result = _convert(notes, difficulty="HARD", budget=0.0)
    assert result.deleted_count == 1
    assert [n.time_ms for n in result.notes] == [0, 300]


@pytest.mark.parametrize("gap", [60, 80, 100])
@pytest.mark.parametrize("count", [30, 60, 120])
def test_move_count_is_distinct_notes_not_operations(gap, count):
    """연산을 세면 여러 패스에서 같은 노트가 중복 집계된다."""
    result = _convert(_side_jack(count, gap), difficulty="EASY")
    assert result.moved_count == sum(1 for n in result.notes if n.lane != n.origin_lane)


def test_an_expert_side_chord_is_left_alone():
    """EXPERT 의 양쪽 사이드 동시는 합법이다. 옮기면 예산만 깎는다."""
    notes = _chart_with_side_jack(jack=0) + [_tap(1000, SIDE_L), _tap(1000, SIDE_R)]
    result = _convert(notes, difficulty="EXPERT")
    lanes = {n.lane for n in result.notes if n.time_ms == 1000}
    assert lanes == {SIDE_L, SIDE_R}
    assert result.moved_count == 0


def test_an_active_hold_is_not_a_chord_with_the_current_row():
    """진행 중인 롱노트를 같은 행 화음으로 세면 lane_rules 와 답이 갈린다."""
    notes = [
        NoteEvent(time_ms=0, lane=SIDE_L, kind="HOLD", duration_ms=2000),
        _tap(1000, SIDE_R),
    ]
    notes += [_tap(3000 + i * 400, MAIN_1 + i % 4) for i in range(60)]
    for difficulty in ("NORMAL", "HARD", "EXPERT"):
        assert check_lane_rules(notes, key_mode=7, difficulty=difficulty) == []
        result = _convert(notes, difficulty=difficulty)
        tap = next(n for n in result.notes if n.time_ms == 1000)
        assert (tap.lane, result.moved_count) == (SIDE_R, 0), difficulty


@pytest.mark.parametrize(
    "strengths", [(0.1, 0.9, 0.2), (0.9, 0.1, 0.8), (0.5, 0.5, 0.5)]
)
def test_a_shared_endpoint_is_deleted_once(strengths):
    """연타가 셋이면 가운데 노트가 앞뒤 두 위반에 동시에 얽힌다."""
    notes = [
        _tap(time_ms, SIDE_L, onset_strength=strength)
        for time_ms, strength in zip((0, 100, 300), strengths)
    ]
    result = _convert(notes, difficulty="HARD", budget=0.0)
    assert result.deleted_count == 1
    assert [n.time_ms for n in result.notes] == [0, 300]
