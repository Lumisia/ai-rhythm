import math

import pytest

from chart_worker.postprocess.cost import (
    INFEASIBLE,
    CostWeights,
    PlacementContext,
    best_lane,
    candidate_lanes,
    placement_cost,
)
from chart_worker.schema.note import NoteEvent

# 7키: SIDE_LEFT MAIN_1 MAIN_2 CENTER MAIN_3 MAIN_4 SIDE_RIGHT
SIDE_L, MAIN_1, MAIN_2, CENTER, MAIN_3, MAIN_4, SIDE_R = range(7)
MAIN_LANES = (MAIN_1, MAIN_2, MAIN_3, MAIN_4)


def _context(**overrides):
    base = {
        "key_mode": 7,
        "difficulty": "HARD",
        "last_time_by_lane": {},
    }
    return PlacementContext(**(base | overrides))


def _note(time_ms=1000, lane=SIDE_L, **kwargs):
    return NoteEvent(time_ms=time_ms, lane=lane, **kwargs)


def _total(note, lane, context, **weight_overrides):
    weights = CostWeights(**weight_overrides) if weight_overrides else CostWeights()
    return placement_cost(note, lane, context, weights).total


def _term(note, lane, context, name):
    return placement_cost(note, lane, context).terms[name]


# --- 기본 ------------------------------------------------------------------


def test_a_lane_outside_the_key_mode_is_rejected():
    with pytest.raises(ValueError, match="outside 7K"):
        placement_cost(_note(), 9, _context())


def test_an_occupied_lane_is_infeasible():
    context = _context(occupied_lanes=frozenset({MAIN_1}))
    assert math.isinf(placement_cost(_note(), MAIN_1, context).total)


def test_breakdown_sums_to_the_total():
    context = _context(previous_lane=MAIN_1, hand_counts=(3, 1))
    breakdown = placement_cost(_note(), MAIN_3, context)
    assert breakdown.total == pytest.approx(sum(breakdown.terms.values()))


def test_leaving_a_note_where_it_already_is_costs_nothing():
    assert _total(_note(lane=MAIN_1), MAIN_1, _context()) == 0.0


def test_an_empty_hand_window_does_not_favour_the_thumb():
    """정보가 없을 때 0 을 주지 않으면 손이 없는 CENTER 만 공짜가 된다."""
    context = _context(hand_counts=(0, 0))
    for lane in (*MAIN_LANES, SIDE_L, CENTER):
        assert _term(_note(), lane, context, "w2_hand_balance") == 0.0


# --- w1 같은 레인 반복 ------------------------------------------------------


def test_a_tighter_repeat_costs_more():
    tight = _context(last_time_by_lane={MAIN_1: 950})
    loose = _context(last_time_by_lane={MAIN_1: 880})
    assert _term(_note(), MAIN_1, tight, "w1_same_lane_repeat") > _term(
        _note(), MAIN_1, loose, "w1_same_lane_repeat"
    )


def test_a_repeat_beyond_the_finger_limit_is_free():
    context = _context(last_time_by_lane={MAIN_1: 1000 - 120})
    assert _term(_note(), MAIN_1, context, "w1_same_lane_repeat") == 0.0


def test_the_same_gap_costs_more_on_a_side_lane():
    """손가락이 다르면 한계가 다르다."""
    context = _context(last_time_by_lane={SIDE_L: 850, MAIN_1: 850})
    assert _term(_note(), SIDE_L, context, "w1_same_lane_repeat") > _term(
        _note(), MAIN_1, context, "w1_same_lane_repeat"
    )


def test_an_easy_chart_penalises_the_same_gap_more_than_an_expert_one():
    easy = _context(difficulty="EASY", last_time_by_lane={MAIN_1: 900})
    expert = _context(difficulty="EXPERT", last_time_by_lane={MAIN_1: 900})
    assert _term(_note(), MAIN_1, easy, "w1_same_lane_repeat") > _term(
        _note(), MAIN_1, expert, "w1_same_lane_repeat"
    )


# --- w9 원배치이탈 ----------------------------------------------------------


def test_staying_put_costs_nothing_and_moving_far_costs_most():
    """w9 가 C안의 핵심이다. 모델 배치를 최대한 존중한다."""
    note = _note(lane=MAIN_1)
    context = _context()
    near = _term(note, MAIN_2, context, "w9_origin_drift")
    far = _term(note, SIDE_R, context, "w9_origin_drift")
    assert _term(note, MAIN_1, context, "w9_origin_drift") == 0.0
    assert 0 < near < far


def test_origin_lane_survives_an_earlier_move():
    """이미 옮겨진 노트도 최초 배치를 기준으로 잰다."""
    moved = NoteEvent(time_ms=1000, lane=MAIN_2, origin_lane=SIDE_L)
    context = _context()
    assert _term(moved, SIDE_L, context, "w9_origin_drift") == 0.0
    assert _term(moved, MAIN_2, context, "w9_origin_drift") > 0.0


def test_without_origin_drift_the_conversion_is_just_a_reshuffle():
    note = _note(lane=MAIN_1)
    context = _context()
    with_drift = _total(note, SIDE_R, context)
    without = _total(note, SIDE_R, context, origin_drift=0.0)
    assert with_drift > without


# --- w2~w7 ------------------------------------------------------------------


def test_the_lighter_hand_is_preferred():
    context = _context(hand_counts=(6, 1))
    assert _term(_note(), MAIN_4, context, "w2_hand_balance") < _term(
        _note(), MAIN_1, context, "w2_hand_balance"
    )


def test_center_belongs_to_neither_hand_so_it_never_shifts_the_balance():
    even = _context(hand_counts=(3, 3))
    assert _term(_note(), CENTER, even, "w2_hand_balance") == 0.0


def test_a_longer_reach_costs_more():
    context = _context(previous_lane=MAIN_1)
    assert _term(_note(), SIDE_R, context, "w3_travel") > _term(
        _note(), MAIN_2, context, "w3_travel"
    )


def test_landing_on_the_previous_lane_is_a_jack():
    context = _context(previous_lane=MAIN_2)
    assert _term(_note(), MAIN_2, context, "w4_jack") > 0
    assert _term(_note(), MAIN_3, context, "w4_jack") == 0.0


def test_a_wider_chord_costs_more():
    context = _context(occupied_lanes=frozenset({MAIN_1}))
    assert _term(_note(), SIDE_R, context, "w5_chord_spread") > _term(
        _note(), MAIN_2, context, "w5_chord_spread"
    )


def test_repeating_a_recent_shape_costs_more():
    context = _context(
        occupied_lanes=frozenset({MAIN_1}), recent_shapes=((MAIN_1, MAIN_2), (MAIN_1, MAIN_2))
    )
    assert _term(_note(), MAIN_2, context, "w6_pattern_repeat") > _term(
        _note(), MAIN_3, context, "w6_pattern_repeat"
    )


@pytest.mark.parametrize(
    ("band", "cheap", "dear"), [("LOW", MAIN_1, MAIN_4), ("HIGH", MAIN_4, MAIN_1)]
)
def test_low_leans_left_and_high_leans_right(band, cheap, dear):
    note = _note(band=band)
    context = _context()
    assert _term(note, cheap, context, "w7_band_mismatch") == 0.0
    assert _term(note, dear, context, "w7_band_mismatch") > 0.0


def test_mid_and_unlabelled_notes_have_no_band_preference():
    context = _context()
    for note in (_note(band="MID"), _note()):
        for lane in MAIN_LANES:
            assert _term(note, lane, context, "w7_band_mismatch") == 0.0


# --- w8 규칙 위반 -----------------------------------------------------------


def test_a_placement_that_breaks_the_finger_limit_is_expensive_but_possible():
    """모든 후보가 위반이면 가장 덜 나쁜 곳에 둬야 한다. INF 가 아니다."""
    context = _context(last_time_by_lane={MAIN_1: 990})
    total = _total(_note(), MAIN_1, context)
    assert total > CostWeights().rule_violation
    assert math.isfinite(total)


def test_pressing_the_second_side_is_a_violation():
    context = _context(occupied_lanes=frozenset({SIDE_L}))
    assert _term(_note(), SIDE_R, context, "w8_rule_violation") > 0
    assert _term(_note(), MAIN_1, context, "w8_rule_violation") == 0.0


def test_center_on_top_of_both_sides_is_a_violation():
    context = _context(occupied_lanes=frozenset({SIDE_L, SIDE_R}))
    assert _term(_note(), CENTER, context, "w8_rule_violation") > 0


# --- 후보와 선택 -------------------------------------------------------------


def test_candidates_are_main_lanes_only_by_default():
    """사이드로 옮기면 새끼손가락 문제를 다른 새끼손가락으로 옮기는 것뿐이다."""
    assert candidate_lanes(_note(), _context()) == sorted(
        MAIN_LANES, key=lambda lane: abs(lane - SIDE_L)
    )


def test_candidates_can_include_every_lane_when_asked():
    lanes = candidate_lanes(_note(lane=MAIN_1), _context(), preference="ANY")
    assert set(lanes) == set(range(7))


def test_occupied_lanes_are_not_candidates():
    context = _context(occupied_lanes=frozenset({MAIN_1, MAIN_2}))
    assert set(candidate_lanes(_note(), context)) == {MAIN_3, MAIN_4}


def test_candidates_start_from_the_original_lane():
    note = _note(lane=SIDE_R)
    assert candidate_lanes(note, _context())[0] == MAIN_4


def test_best_lane_prefers_the_nearest_main_lane():
    lane, breakdown = best_lane(_note(lane=SIDE_L), _context())
    assert lane == MAIN_1
    assert math.isfinite(breakdown.total)


def test_best_lane_avoids_a_lane_that_just_fired():
    context = _context(last_time_by_lane={MAIN_1: 995}, previous_lane=MAIN_1)
    lane, _ = best_lane(_note(lane=SIDE_L), context)
    assert lane != MAIN_1


def test_best_lane_is_none_when_every_main_lane_is_taken():
    context = _context(occupied_lanes=frozenset(MAIN_LANES))
    assert best_lane(_note(), context) is None


def test_infeasible_is_infinite():
    assert math.isinf(INFEASIBLE)
