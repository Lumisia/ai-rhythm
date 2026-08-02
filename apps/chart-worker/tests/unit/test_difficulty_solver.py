import pytest

from chart_worker.postprocess.difficulty_solver import (
    DOWNBEAT_WEIGHT,
    RATING_TOLERANCE,
    REMOVAL_BUDGET,
    Operation,
    _jack_trim_candidates,
    musical_cost,
    solve_difficulty,
)
from chart_worker.rating.project_rating import TARGET_RATING, measure_rating
from chart_worker.schema.note import NoteEvent
from chart_worker.schema.types import TARGET_HOLD_RATIO

BEAT_MS = 500.0
DURATION_MS = 60_000


def _dense_chart(count=600, key_mode=4, step=100, hold_every=5):
    notes = []
    for index in range(count):
        time_ms = index * step
        lane = index % key_mode
        if index % hold_every == 0:
            notes.append(
                NoteEvent(
                    time_ms=time_ms,
                    lane=lane,
                    kind="HOLD",
                    duration_ms=step - 10,
                    onset_strength=0.3 + (index % 7) / 10,
                    beat_fraction=(time_ms % BEAT_MS) / BEAT_MS,
                    is_downbeat=time_ms % (BEAT_MS * 4) == 0,
                )
            )
        else:
            notes.append(
                NoteEvent(
                    time_ms=time_ms,
                    lane=lane,
                    onset_strength=0.2 + (index % 9) / 10,
                    beat_fraction=(time_ms % BEAT_MS) / BEAT_MS,
                    is_downbeat=time_ms % (BEAT_MS * 4) == 0,
                )
            )
        if index % 3 == 0:
            notes.append(
                NoteEvent(
                    time_ms=time_ms,
                    lane=(lane + 1) % key_mode,
                    onset_strength=0.1,
                    beat_fraction=(time_ms % BEAT_MS) / BEAT_MS,
                )
            )
    return sorted(notes, key=lambda n: (n.time_ms, n.lane))


def _solve(notes, difficulty="NORMAL", **kwargs):
    return solve_difficulty(
        notes,
        duration_ms=DURATION_MS,
        difficulty=difficulty,
        beat_ms=BEAT_MS,
        **kwargs,
    )


# --- 음악 비용 --------------------------------------------------------------


def test_a_stronger_hit_costs_more_to_lose():
    weak = NoteEvent(time_ms=0, lane=0, onset_strength=0.1)
    strong = NoteEvent(time_ms=0, lane=1, onset_strength=0.9)
    assert musical_cost(strong, isolation=0.0) > musical_cost(weak, isolation=0.0)


def test_a_downbeat_costs_more():
    plain = NoteEvent(time_ms=0, lane=0, onset_strength=0.5)
    downbeat = NoteEvent(time_ms=0, lane=1, onset_strength=0.5, is_downbeat=True)
    difference = musical_cost(downbeat, isolation=0.0) - musical_cost(plain, isolation=0.0)
    assert difference == pytest.approx(DOWNBEAT_WEIGHT)


def test_an_isolated_note_costs_more():
    """주변에 노트가 없으면 빼는 순간 구멍이 된다."""
    note = NoteEvent(time_ms=0, lane=0, onset_strength=0.5)
    assert musical_cost(note, isolation=1.0) > musical_cost(note, isolation=0.0)


def test_the_chorus_is_protected_more_than_the_verse():
    chorus = NoteEvent(time_ms=0, lane=0, onset_strength=0.5, section="chorus")
    verse = NoteEvent(time_ms=0, lane=1, onset_strength=0.5, section="verse")
    assert musical_cost(chorus, isolation=0.0) > musical_cost(verse, isolation=0.0)


def test_an_unanalysed_note_gets_a_middle_cost():
    plain = NoteEvent(time_ms=0, lane=0)
    weak = NoteEvent(time_ms=0, lane=1, onset_strength=0.0)
    strong = NoteEvent(time_ms=0, lane=2, onset_strength=1.0)
    assert (
        musical_cost(weak, isolation=0.0)
        < musical_cost(plain, isolation=0.0)
        < musical_cost(strong, isolation=0.0)
    )


# --- 타이밍 불변 ------------------------------------------------------------


def test_the_solver_never_invents_a_time():
    notes = _dense_chart()
    result = _solve(notes)
    assert {n.time_ms for n in result.notes} <= {n.time_ms for n in notes}


def test_hold_to_tap_keeps_the_start_time():
    """시작 시각을 유지하므로 불변 원칙을 통과한다."""
    notes = [
        NoteEvent(time_ms=index * 100, lane=index % 4, kind="HOLD", duration_ms=90)
        for index in range(80)
    ]
    result = _solve(notes, difficulty="EASY")
    assert {n.time_ms for n in result.notes} <= {n.time_ms for n in notes}
    assert result.converted_count > 0


def test_the_solver_only_goes_down():
    notes = _dense_chart()
    result = _solve(notes)
    assert len(result.notes) <= len(notes)
    assert result.metrics.rating <= measure_rating(notes, DURATION_MS).rating


# --- 목표 도달 --------------------------------------------------------------


@pytest.mark.parametrize("difficulty", ["EASY", "NORMAL", "HARD"])
def test_a_dense_chart_is_brought_down_toward_the_target(difficulty):
    notes = _dense_chart()
    before = measure_rating(notes, DURATION_MS).rating
    result = _solve(notes, difficulty=difficulty)
    assert before > TARGET_RATING[difficulty]
    assert result.metrics.rating < before
    assert result.target_rating == TARGET_RATING[difficulty]


def test_a_chart_already_at_target_is_left_alone():
    notes = [NoteEvent(time_ms=index * 1000, lane=index % 4) for index in range(20)]
    result = _solve(notes, difficulty="EXPERT")
    assert result.removed_count == 0
    assert result.converted_count == 0
    assert result.notes == notes


def test_reaching_the_target_is_reported():
    notes = [NoteEvent(time_ms=index * 1000, lane=index % 4) for index in range(20)]
    result = _solve(notes, difficulty="EXPERT")
    assert result.reached_target
    assert not result.budget_exhausted


def test_the_removal_budget_caps_the_damage():
    """다 써도 목표에 못 닿으면 채보를 살리고 실제 별점을 기록한다."""
    notes = _dense_chart(count=400, step=60)
    result = _solve(notes, difficulty="EASY", budget=0.05)
    assert result.removed_ratio <= 0.05 + 1e-9
    assert result.notes


def test_custom_removal_budget_reports_exhaustion():
    notes = _dense_chart(count=400, step=60)
    result = _solve(notes, difficulty="EASY", budget=0.05)
    assert not result.reached_target
    assert result.removed_count == int(len(notes) * 0.05)
    assert result.budget_exhausted


def test_the_result_metrics_match_the_returned_chart():
    notes = _dense_chart()
    result = _solve(notes)
    assert result.metrics.note_count == len(result.notes)
    assert result.metrics == measure_rating(result.notes, DURATION_MS)


# --- 연산 우선순위 ----------------------------------------------------------


def test_chords_are_broken_before_anything_else():
    """chord_ratio 가중치가 1.10 으로 가장 크고 음악을 가장 덜 해친다."""
    notes = _dense_chart()
    result = _solve(notes, difficulty="HARD")
    assert result.operations[Operation.CHORD_BREAK.value] > 0


def test_only_the_weakest_note_of_a_chord_is_taken():
    notes = [
        NoteEvent(time_ms=0, lane=0, onset_strength=0.9),
        NoteEvent(time_ms=0, lane=1, onset_strength=0.1),
    ]
    notes += [
        NoteEvent(time_ms=100 + index * 100, lane=index % 4, onset_strength=0.5)
        for index in range(200)
    ]
    result = _solve(notes, difficulty="EASY")
    survivors = [n for n in result.notes if n.time_ms == 0]
    assert [n.lane for n in survivors] in ([0], [])


def test_downbeats_survive_longer_than_offbeats():
    notes = []
    for index in range(300):
        time_ms = index * 100
        notes.append(
            NoteEvent(
                time_ms=time_ms,
                lane=index % 4,
                onset_strength=0.5,
                is_downbeat=time_ms % 2000 == 0,
                beat_fraction=(time_ms % BEAT_MS) / BEAT_MS,
            )
        )
    result = _solve(notes, difficulty="EASY")
    kept = result.notes
    downbeats_before = sum(1 for n in notes if n.is_downbeat)
    downbeats_after = sum(1 for n in kept if n.is_downbeat)
    assert downbeats_after / downbeats_before > len(kept) / len(notes)


def test_operation_counts_add_up():
    notes = _dense_chart()
    result = _solve(notes, difficulty="EASY")
    assert sum(result.operations.values()) == result.removed_count + result.converted_count


# --- 입력 검증 --------------------------------------------------------------


def test_an_unknown_difficulty_is_rejected():
    with pytest.raises(ValueError, match="unsupported difficulty"):
        _solve([], difficulty="NOMAL")


def test_a_non_positive_beat_is_rejected():
    with pytest.raises(ValueError, match="beat_ms"):
        solve_difficulty([], duration_ms=1000, difficulty="EASY", beat_ms=0)


def test_an_empty_chart_is_returned_untouched():
    result = _solve([])
    assert result.notes == []
    assert result.removed_ratio == 0.0


def test_the_budget_default_is_a_fraction():
    assert 0 < REMOVAL_BUDGET < 1
    assert RATING_TOLERANCE > 0


def test_no_operation_class_is_starved_by_an_earlier_one():
    """연산을 하나씩 소진하면 앞선 연산이 예산을 다 먹어 뒤가 굶는다.

    잭 축약은 max_jack 항을 줄이는 유일한 수단인데 그게 한 번도 안 돌 수 있다.
    """
    notes = []
    for index in range(400):
        time_ms = index * 100
        notes.append(
            NoteEvent(
                time_ms=time_ms,
                lane=index // 6 % 4,
                kind="HOLD" if index % 4 == 0 else "TAP",
                duration_ms=90 if index % 4 == 0 else None,
                onset_strength=0.2 + (index % 9) / 12,
                beat_fraction=(time_ms % BEAT_MS) / BEAT_MS,
            )
        )
    result = _solve(notes, difficulty="EASY")
    fired = {name for name, count in result.operations.items() if count}
    assert Operation.JACK_TRIM.value in fired
    assert Operation.HOLD_TO_TAP.value in fired


def test_jack_trim_ignores_non_consecutive_lane_repeats():
    notes = [
        NoteEvent(time_ms=time_ms, lane=lane)
        for time_ms, lane in ((0, 0), (100, 1), (200, 0), (300, 1), (400, 0))
    ]
    assert list(_jack_trim_candidates(notes)) == []


def test_jack_trim_keeps_only_the_first_two_consecutive_rows():
    notes = [NoteEvent(time_ms=index * 100, lane=0) for index in range(4)]
    assert [note.time_ms for note in _jack_trim_candidates(notes)] == [200, 300]


def test_hold_conversion_survives_an_exhausted_removal_budget():
    """변환은 노트를 없애지 않으므로 삭제 예산에 막히면 안 된다."""
    notes = [
        NoteEvent(time_ms=index * 100, lane=index % 4, kind="HOLD", duration_ms=90)
        for index in range(80)
    ]
    result = _solve(notes, difficulty="EASY", budget=0.05)
    assert result.removed_ratio <= 0.05 + 1e-9
    assert result.converted_count > 0


# --- 롱노트 바닥 --------------------------------------------------------------


def _hold_ratio_of(notes):
    return sum(1 for note in notes if note.kind == "HOLD") / len(notes)


@pytest.mark.parametrize("difficulty", ["EASY", "NORMAL", "HARD", "EXPERT"])
def test_hold_conversion_never_wipes_out_every_hold(difficulty):
    """삭제 예산에서 뺐다고 상한까지 없애면 롱노트가 전멸한다.

    목표에 못 닿는 채보에서 라운드가 계속 돌면 HOLD_TO_TAP 이 끝까지
    변환한다. 등급 기여는 0.60*hold_ratio 뿐이라 다 없애도 얼마 못 내리면서
    노트 종류 하나를 통째로 지운다.
    """
    notes = _dense_chart(count=600, step=60, hold_every=2)
    result = _solve(notes, difficulty=difficulty)
    assert not result.reached_target, "이 채보는 목표에 못 닿아야 회귀가 드러난다"
    assert sum(1 for note in result.notes if note.kind == "HOLD") > 0


def test_conversion_stops_at_the_difficulty_hold_floor():
    notes = _dense_chart(count=600, step=60, hold_every=2)
    before = _hold_ratio_of(notes)
    result = _solve(notes, difficulty="HARD")
    assert before > TARGET_HOLD_RATIO["HARD"]
    assert _hold_ratio_of(result.notes) >= TARGET_HOLD_RATIO["HARD"] - 1e-9


def test_a_chart_already_under_the_floor_is_not_converted_further():
    notes = [
        NoteEvent(
            time_ms=index * 100,
            lane=index % 4,
            kind="HOLD" if index % 20 == 0 else "TAP",
            duration_ms=90 if index % 20 == 0 else None,
            onset_strength=0.5,
        )
        for index in range(600)
    ]
    assert _hold_ratio_of(notes) < TARGET_HOLD_RATIO["EASY"]
    result = _solve(notes, difficulty="EASY")
    assert result.operations[Operation.HOLD_TO_TAP.value] == 0


def test_the_floor_tracks_holds_lost_to_deletion():
    """삭제 연산도 롱노트를 가져간다. 변환 횟수만 세면 실제와 어긋난다."""
    notes = _dense_chart(count=600, step=60, hold_every=2)
    result = _solve(notes, difficulty="NORMAL")
    counted = len(notes) - result.removed_count
    assert len(result.notes) == counted
    assert _hold_ratio_of(result.notes) >= TARGET_HOLD_RATIO["NORMAL"] - 1e-9
