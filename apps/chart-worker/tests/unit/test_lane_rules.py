import dataclasses

import pytest

from chart_worker.postprocess.ergonomics import Hand
from chart_worker.postprocess.lane_rules import (
    BOTH_SIDES,
    JACK_MIN_INTERVAL_MS,
    SAME_HAND_NPS_DURING_SIDE_HOLD,
    BothSidesPolicy,
    ErgonomicRole,
    FingerClass,
    Rule,
    check_both_sides,
    check_center_combinations,
    check_jack_intervals,
    check_lane_rules,
    check_side_hold_density,
    ergonomic_roles,
    finger_of,
    hand_of,
    jack_interval_for_lane,
    jack_interval_ms,
)
from chart_worker.schema.note import NoteEvent
from chart_worker.schema.types import DIFFICULTIES, LaneSemantic

# 7키: SIDE_LEFT MAIN_1 MAIN_2 CENTER MAIN_3 MAIN_4 SIDE_RIGHT
SIDE_L, MAIN_1, MAIN_2, CENTER, MAIN_3, MAIN_4, SIDE_R = range(7)


def _tap(time_ms, lane, **kwargs):
    return NoteEvent(time_ms=time_ms, lane=lane, **kwargs)


def _rules(notes, difficulty, key_mode=7):
    return check_lane_rules(notes, key_mode=key_mode, difficulty=difficulty)


# --- 표 ---------------------------------------------------------------------


def test_every_difficulty_has_a_full_finger_table():
    for difficulty in DIFFICULTIES:
        assert set(JACK_MIN_INTERVAL_MS[difficulty]) == set(FingerClass)
        assert difficulty in SAME_HAND_NPS_DURING_SIDE_HOLD
        assert difficulty in BOTH_SIDES


def test_six_key_uses_six_main_fingers():
    assert ergonomic_roles(6) == (ErgonomicRole.MAIN,) * 6


def test_seven_key_is_six_main_fingers_plus_center_thumb():
    assert ergonomic_roles(7) == (
        ErgonomicRole.MAIN,
        ErgonomicRole.MAIN,
        ErgonomicRole.MAIN,
        ErgonomicRole.CENTER,
        ErgonomicRole.MAIN,
        ErgonomicRole.MAIN,
        ErgonomicRole.MAIN,
    )


def test_outer_lane_uses_main_jack_limit_in_six_and_seven_key():
    for key_mode in (6, 7):
        assert jack_interval_for_lane(key_mode, 0, "HARD") == JACK_MIN_INTERVAL_MS["HARD"][
            FingerClass.MAIN
        ]


def test_thresholds_relax_as_difficulty_rises():
    for finger in FingerClass:
        values = [JACK_MIN_INTERVAL_MS[d][finger] for d in DIFFICULTIES]
        assert values == sorted(values, reverse=True), finger


def test_pinky_is_always_slower_than_thumb_which_is_slower_than_index():
    """물리적 순서다. 어떤 난이도에서도 뒤집히면 안 된다."""
    for difficulty in DIFFICULTIES:
        table = JACK_MIN_INTERVAL_MS[difficulty]
        assert table[FingerClass.SIDE] > table[FingerClass.CENTER] > table[FingerClass.MAIN]


@pytest.mark.parametrize(
    ("semantic", "finger"),
    [
        (LaneSemantic.SIDE_LEFT, FingerClass.SIDE),
        (LaneSemantic.SIDE_RIGHT, FingerClass.SIDE),
        (LaneSemantic.CENTER, FingerClass.CENTER),
        (LaneSemantic.MAIN_1, FingerClass.MAIN),
        (LaneSemantic.MAIN_4, FingerClass.MAIN),
    ],
)
def test_finger_classes(semantic, finger):
    assert finger_of(semantic) is finger


def test_center_belongs_to_no_hand():
    """엄지라 어느 손에도 고정되지 않는다."""
    assert hand_of(LaneSemantic.CENTER) is None
    assert hand_of(LaneSemantic.SIDE_LEFT) is Hand.LEFT
    assert hand_of(LaneSemantic.MAIN_3) is Hand.RIGHT


def test_easy_side_cannot_even_do_a_quarter_note_jack_at_150_bpm():
    """BPM 150 의 4분음표는 400ms 다."""
    assert jack_interval_ms(LaneSemantic.SIDE_LEFT, "EASY") > 400


def test_expert_main_allows_sixteenths_at_150_bpm():
    """BPM 150 의 16분은 100ms 다."""
    assert jack_interval_ms(LaneSemantic.MAIN_1, "EXPERT") < 100


def test_unknown_difficulty_is_rejected():
    with pytest.raises(ValueError, match="unsupported difficulty"):
        check_jack_intervals([], key_mode=7, difficulty="NOMAL")


# --- S1 ---------------------------------------------------------------------


def test_main_jack_that_is_too_fast_is_flagged():
    notes = [_tap(0, SIDE_L), _tap(100, SIDE_L)]
    [violation] = check_jack_intervals(notes, key_mode=7, difficulty="HARD")
    assert violation.rule is Rule.S1_JACK_INTERVAL
    assert violation.lanes == (SIDE_L,)
    assert "120ms" in violation.detail


def test_the_same_gap_passes_on_a_main_lane():
    """손가락이 다르면 한계가 다르다."""
    notes = [_tap(0, MAIN_1), _tap(200, MAIN_1)]
    assert check_jack_intervals(notes, key_mode=7, difficulty="HARD") == []


def test_outer_lane_jack_uses_main_limit_in_six_and_seven_key():
    for key_mode in (6, 7):
        notes = [_tap(0, 0), _tap(200, 0)]
        assert check_jack_intervals(notes, key_mode=key_mode, difficulty="HARD") == []


def test_center_sits_between_side_and_main():
    notes = [_tap(0, CENTER), _tap(220, CENTER)]
    assert check_jack_intervals(notes, key_mode=7, difficulty="HARD") == []
    assert check_jack_intervals(notes, key_mode=7, difficulty="NORMAL")


def test_other_lanes_in_between_do_not_matter():
    """밀도 제한이 아니다. 잭에만 적용된다."""
    notes = [_tap(0, MAIN_1), _tap(60, MAIN_2), _tap(120, MAIN_3), _tap(180, MAIN_4)]
    assert check_jack_intervals(notes, key_mode=7, difficulty="EASY") == []


def test_a_lane_repeat_is_flagged_even_with_other_lanes_between():
    notes = [_tap(0, SIDE_L), _tap(25, MAIN_1), _tap(50, SIDE_L)]
    violations = check_jack_intervals(notes, key_mode=7, difficulty="EXPERT")
    assert [v.lanes for v in violations] == [(SIDE_L,)]


def test_a_gap_exactly_at_the_limit_passes():
    notes = [_tap(0, SIDE_L), _tap(120, SIDE_L)]
    assert check_jack_intervals(notes, key_mode=7, difficulty="HARD") == []


def test_four_key_layout_has_no_side_or_center():
    notes = [_tap(0, lane) for lane in range(4)] + [_tap(100, lane) for lane in range(4)]
    violations = check_jack_intervals(notes, key_mode=4, difficulty="EXPERT")
    assert violations == []


# --- S3 ---------------------------------------------------------------------


def _side_hold(duration_ms=1000, lane=SIDE_L):
    return NoteEvent(time_ms=0, lane=lane, kind="HOLD", duration_ms=duration_ms)


def test_shift_side_hold_density_is_inactive_for_home_position_modes():
    for key_mode, right_lane in ((6, 5), (7, 6)):
        notes = [
            _side_hold(),
            *(_tap(index * 100, right_lane) for index in range(1, 10)),
        ]
        for difficulty in DIFFICULTIES:
            assert check_side_hold_density(notes, key_mode=key_mode, difficulty=difficulty) == []


# --- S4 ---------------------------------------------------------------------


def test_legacy_both_sides_policy_values_remain_schema_compatible():
    assert BOTH_SIDES["EXPERT"] is BothSidesPolicy.FREE


def test_shift_specific_side_rules_do_not_reject_home_position_chords():
    for key_mode, right_lane in ((6, 5), (7, 6)):
        notes = [_tap(1000, 0), _tap(1000, right_lane)]
        for difficulty in DIFFICULTIES:
            assert check_both_sides(notes, key_mode=key_mode, difficulty=difficulty) == []
            if key_mode == 7:
                assert check_center_combinations(
                    [*notes, _tap(1000, 3)], key_mode=key_mode, difficulty=difficulty
                ) == []


# --- C3 ---------------------------------------------------------------------


def _center_and_both_sides(extra_mains=()):
    return [
        _tap(1000, CENTER),
        _tap(1000, SIDE_L),
        _tap(1000, SIDE_R),
        *(_tap(1000, lane) for lane in extra_mains),
    ]


def test_center_with_outer_home_keys_is_allowed_at_every_difficulty():
    notes = _center_and_both_sides()
    for difficulty in DIFFICULTIES:
        assert check_center_combinations(notes, key_mode=7, difficulty=difficulty) == []


def test_six_key_has_no_center_lane():
    notes = [_tap(1000, 0), _tap(1000, 5)]
    assert check_center_combinations(notes, key_mode=6, difficulty="EASY") == []


# --- 통합 -------------------------------------------------------------------


def test_all_rules_run_together_and_come_back_in_time_order():
    notes = [
        _tap(0, SIDE_L),
        _tap(100, SIDE_L),
        _tap(2000, SIDE_L),
        _tap(2000, SIDE_R),
    ]
    violations = _rules(notes, "EASY")
    assert [v.time_ms for v in violations] == sorted(v.time_ms for v in violations)
    assert {v.rule for v in violations} == {Rule.S1_JACK_INTERVAL}


def test_a_clean_easy_chart_has_no_violations():
    notes = [_tap(index * 600, lane) for index, lane in enumerate([SIDE_L, MAIN_1, MAIN_4, SIDE_R])]
    assert _rules(notes, "EASY") == []


def test_violations_are_frozen():
    [violation] = check_jack_intervals(
        [_tap(0, SIDE_L), _tap(10, SIDE_L)], key_mode=7, difficulty="EXPERT"
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        violation.time_ms = 5  # type: ignore[misc]
