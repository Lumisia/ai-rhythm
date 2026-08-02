import dataclasses

import pytest

from chart_worker.postprocess.lane_rules import (
    ACCENT_STRENGTH,
    BOTH_SIDES,
    JACK_MIN_INTERVAL_MS,
    SAME_HAND_NPS_DURING_SIDE_HOLD,
    BothSidesPolicy,
    FingerClass,
    Hand,
    Rule,
    check_both_sides,
    check_center_combinations,
    check_jack_intervals,
    check_lane_rules,
    check_side_hold_density,
    finger_of,
    hand_of,
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


def test_side_jack_that_is_too_fast_is_flagged():
    notes = [_tap(0, SIDE_L), _tap(200, SIDE_L)]
    [violation] = check_jack_intervals(notes, key_mode=7, difficulty="HARD")
    assert violation.rule is Rule.S1_JACK_INTERVAL
    assert violation.lanes == (SIDE_L,)
    assert "250ms" in violation.detail


def test_the_same_gap_passes_on_a_main_lane():
    """손가락이 다르면 한계가 다르다."""
    notes = [_tap(0, MAIN_1), _tap(200, MAIN_1)]
    assert check_jack_intervals(notes, key_mode=7, difficulty="HARD") == []


def test_center_sits_between_side_and_main():
    notes = [_tap(0, CENTER), _tap(220, CENTER)]
    assert check_jack_intervals(notes, key_mode=7, difficulty="HARD") == []
    assert check_jack_intervals(notes, key_mode=7, difficulty="NORMAL")


def test_other_lanes_in_between_do_not_matter():
    """밀도 제한이 아니다. 잭에만 적용된다."""
    notes = [_tap(0, MAIN_1), _tap(60, MAIN_2), _tap(120, MAIN_3), _tap(180, MAIN_4)]
    assert check_jack_intervals(notes, key_mode=7, difficulty="EASY") == []


def test_a_lane_repeat_is_flagged_even_with_other_lanes_between():
    notes = [_tap(0, SIDE_L), _tap(50, MAIN_1), _tap(100, SIDE_L)]
    violations = check_jack_intervals(notes, key_mode=7, difficulty="EXPERT")
    assert [v.lanes for v in violations] == [(SIDE_L,)]


def test_a_gap_exactly_at_the_limit_passes():
    notes = [_tap(0, SIDE_L), _tap(250, SIDE_L)]
    assert check_jack_intervals(notes, key_mode=7, difficulty="HARD") == []


def test_four_key_layout_has_no_side_or_center():
    notes = [_tap(0, lane) for lane in range(4)] + [_tap(100, lane) for lane in range(4)]
    violations = check_jack_intervals(notes, key_mode=4, difficulty="EXPERT")
    assert violations == []


# --- S3 ---------------------------------------------------------------------


def _side_hold(duration_ms=1000, lane=SIDE_L):
    return NoteEvent(time_ms=0, lane=lane, kind="HOLD", duration_ms=duration_ms)


def test_easy_forbids_any_same_hand_note_during_a_side_hold():
    notes = [_side_hold(), _tap(500, MAIN_1)]
    [violation] = check_side_hold_density(notes, key_mode=7, difficulty="EASY")
    assert violation.rule is Rule.S3_SIDE_HOLD_SAME_HAND


def test_the_other_hand_is_free_during_a_side_hold():
    notes = [_side_hold(), *(_tap(index * 100, MAIN_4) for index in range(1, 10))]
    assert check_side_hold_density(notes, key_mode=7, difficulty="EASY") == []


def test_normal_allows_a_couple_of_same_hand_notes():
    sparse = [_side_hold(), _tap(400, MAIN_1), _tap(800, MAIN_2)]
    dense = [_side_hold(), *(_tap(index * 100, MAIN_1) for index in range(1, 10))]
    assert check_side_hold_density(sparse, key_mode=7, difficulty="NORMAL") == []
    assert check_side_hold_density(dense, key_mode=7, difficulty="NORMAL")


def test_expert_tolerates_a_burst_under_the_hold():
    notes = [_side_hold(), *(_tap(index * 200, MAIN_1) for index in range(1, 6))]
    assert check_side_hold_density(notes, key_mode=7, difficulty="EXPERT") == []


def test_center_notes_do_not_count_against_either_hand():
    notes = [_side_hold(), *(_tap(index * 100, CENTER) for index in range(1, 10))]
    assert check_side_hold_density(notes, key_mode=7, difficulty="EASY") == []


def test_a_main_hold_is_not_a_side_hold():
    notes = [
        NoteEvent(time_ms=0, lane=MAIN_1, kind="HOLD", duration_ms=1000),
        _tap(500, MAIN_2),
    ]
    assert check_side_hold_density(notes, key_mode=7, difficulty="EASY") == []


# --- S4 ---------------------------------------------------------------------


def _both_sides(**kwargs):
    return [_tap(1000, SIDE_L, **kwargs), _tap(1000, SIDE_R, **kwargs)]


def test_easy_forbids_both_sides_outright():
    assert check_both_sides(_both_sides(), key_mode=7, difficulty="EASY")


def test_expert_never_complains():
    assert BOTH_SIDES["EXPERT"] is BothSidesPolicy.FREE
    assert check_both_sides(_both_sides(), key_mode=7, difficulty="EXPERT") == []


def test_normal_allows_both_sides_only_on_a_downbeat():
    off = _both_sides()
    on = _both_sides(is_downbeat=True)
    assert check_both_sides(off, key_mode=7, difficulty="NORMAL")
    assert check_both_sides(on, key_mode=7, difficulty="NORMAL") == []


def test_hard_also_accepts_a_strong_onset():
    quiet = _both_sides(onset_strength=0.1)
    loud = _both_sides(onset_strength=ACCENT_STRENGTH)
    assert check_both_sides(quiet, key_mode=7, difficulty="HARD")
    assert check_both_sides(loud, key_mode=7, difficulty="HARD") == []


def test_normal_ignores_onset_strength():
    """다운비트에서만 허용한다. 악센트는 HARD 부터다."""
    loud = _both_sides(onset_strength=1.0)
    assert check_both_sides(loud, key_mode=7, difficulty="NORMAL")


def test_one_side_alone_is_never_a_violation():
    notes = [_tap(1000, SIDE_L), _tap(1000, MAIN_1)]
    assert check_both_sides(notes, key_mode=7, difficulty="EASY") == []


def test_four_key_has_no_sides_to_press_together():
    notes = [_tap(1000, 0), _tap(1000, 3)]
    assert check_both_sides(notes, key_mode=4, difficulty="EASY") == []


# --- C3 ---------------------------------------------------------------------


def _center_and_both_sides(extra_mains=()):
    return [
        _tap(1000, CENTER),
        _tap(1000, SIDE_L),
        _tap(1000, SIDE_R),
        *(_tap(1000, lane) for lane in extra_mains),
    ]


def test_center_with_both_sides_is_expert_only():
    notes = _center_and_both_sides()
    for difficulty in ("EASY", "NORMAL", "HARD"):
        assert check_center_combinations(notes, key_mode=7, difficulty=difficulty)
    assert check_center_combinations(notes, key_mode=7, difficulty="EXPERT") == []


def test_one_extra_main_is_still_allowed_at_expert():
    notes = _center_and_both_sides(extra_mains=(MAIN_1,))
    assert check_center_combinations(notes, key_mode=7, difficulty="EXPERT") == []


def test_two_extra_mains_are_unplayable_at_every_difficulty():
    """엄지와 양쪽 새끼가 이미 묶였는데 일반키가 둘 더 붙는다."""
    notes = _center_and_both_sides(extra_mains=(MAIN_1, MAIN_4))
    for difficulty in DIFFICULTIES:
        [violation] = check_center_combinations(notes, key_mode=7, difficulty=difficulty)
        assert violation.rule is Rule.C3_CENTER_WITH_BOTH_SIDES
        assert "not playable" in violation.detail


def test_center_with_one_side_is_fine():
    notes = [_tap(1000, CENTER), _tap(1000, SIDE_L)]
    assert check_center_combinations(notes, key_mode=7, difficulty="EASY") == []


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
    assert {v.rule for v in violations} == {Rule.S1_JACK_INTERVAL, Rule.S4_BOTH_SIDES}


def test_a_clean_easy_chart_has_no_violations():
    notes = [_tap(index * 600, lane) for index, lane in enumerate([SIDE_L, MAIN_1, MAIN_4, SIDE_R])]
    assert _rules(notes, "EASY") == []


def test_violations_are_frozen():
    [violation] = check_jack_intervals(
        [_tap(0, SIDE_L), _tap(10, SIDE_L)], key_mode=7, difficulty="EXPERT"
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        violation.time_ms = 5  # type: ignore[misc]
