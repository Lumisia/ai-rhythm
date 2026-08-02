"""6·7키 레인 변환 — C안.

위반 노트를 **삭제하지 않고 이동**한다. 삭제하면 음악 이벤트에 구멍이
생기고, 키음 구조에서는 그 소리가 아예 안 울린다.

시각은 절대 바뀌지 않는다. 레인과 노트 존재만 바꾼다.
"""

import dataclasses
from collections import deque
from dataclasses import dataclass

from chart_worker.postprocess.cost import (
    DEFAULT_WEIGHTS,
    CostWeights,
    PlacementContext,
    best_lane,
    placement_cost,
)
from chart_worker.postprocess.lane_rules import (
    ACCENT_STRENGTH,
    Hand,
    Rule,
    Violation,
    check_lane_rules,
    finger_of,
    hand_of,
)
from chart_worker.postprocess.lane_rules import FingerClass as Finger
from chart_worker.schema.note import Chart, NoteEvent
from chart_worker.schema.types import lane_semantics

MOVE_BUDGET = 0.15
"""moved_note_ratio 상한. 초안이다. 높으면 모델 출력과 우리 규칙이 많이
안 맞는다는 신호이고, LoRA 를 고려할 시점이라는 뜻이기도 하다."""

MAX_PASSES = 3
"""이동이 새 위반을 만들 수 있어 재검사한다."""

HAND_WINDOW = 8
"""손 편중을 재는 최근 노트 수."""

SHAPE_WINDOW = 4


@dataclass(frozen=True, slots=True)
class ConversionResult:
    notes: Chart
    moved_count: int
    deleted_count: int
    moved_note_ratio: float
    passes: int
    remaining_violations: tuple[Violation, ...]

    @property
    def budget_exhausted(self) -> bool:
        return self.deleted_count > 0


def _build_context(
    note: NoteEvent,
    *,
    key_mode: int,
    difficulty: str,
    last_time_by_lane: dict[int, int],
    occupied: set[int],
    held: set[int],
    previous_lane: int | None,
    hand_window: deque[Hand | None],
    shape_window: deque[tuple[int, ...]],
    row_is_downbeat: bool = False,
    row_is_accent: bool = False,
) -> PlacementContext:
    left = sum(1 for hand in hand_window if hand is Hand.LEFT)
    right = sum(1 for hand in hand_window if hand is Hand.RIGHT)
    return PlacementContext(
        key_mode=key_mode,
        difficulty=difficulty,
        last_time_by_lane=dict(last_time_by_lane),
        occupied_lanes=frozenset(occupied),
        held_lanes=frozenset(held),
        previous_lane=previous_lane,
        hand_counts=(left, right),
        recent_shapes=tuple(shape_window),
        row_is_downbeat=row_is_downbeat,
        row_is_accent=row_is_accent,
    )


def _one_pass(
    notes: Chart,
    *,
    key_mode: int,
    difficulty: str,
    weights: CostWeights,
    move_allowance: int,
) -> tuple[Chart, int]:
    """시각 순서대로 훑으며 위반 노트를 옮긴다.

    강한 노트부터 자리를 고른다. 약한 노트가 좋은 자리를 먼저 차지하면
    강한 타격이 밀려난다.
    """
    semantics = lane_semantics(key_mode)
    ordered = sorted(notes, key=lambda n: n.time_ms)
    placed: list[NoteEvent] = []
    last_time_by_lane: dict[int, int] = {}
    hand_window: deque[Hand | None] = deque(maxlen=HAND_WINDOW)
    shape_window: deque[tuple[int, ...]] = deque(maxlen=SHAPE_WINDOW)
    previous_lane: int | None = None
    moved = 0

    # 롱노트가 물고 있는 레인은 그 끝까지 비어 있지 않다. 시작 시각만 보면
    # 새 노트를 진행 중인 롱노트 위로 옮겨 칠 수 없는 채보가 된다.
    busy_until: dict[int, int] = {}

    index = 0
    while index < len(ordered):
        time_ms = ordered[index].time_ms
        row = [note for note in ordered[index:] if note.time_ms == time_ms]
        index += len(row)

        held = {lane for lane, until in busy_until.items() if time_ms < until}
        row_downbeat = any(note.is_downbeat for note in row)
        row_accent = row_downbeat or any(
            note.onset_strength is not None and note.onset_strength >= ACCENT_STRENGTH
            for note in row
        )
        occupied: set[int] = set()
        row_result: list[NoteEvent] = []
        for note in sorted(row, key=lambda n: -(n.onset_strength or 0.0)):
            context = _build_context(
                note,
                key_mode=key_mode,
                difficulty=difficulty,
                last_time_by_lane=last_time_by_lane,
                occupied=occupied,
                held=held,
                previous_lane=previous_lane,
                hand_window=hand_window,
                shape_window=shape_window,
                row_is_downbeat=row_downbeat,
                row_is_accent=row_accent,
            )
            lane = note.lane
            blocked = lane in occupied or lane in held
            if blocked or _breaks_a_rule(note, lane, context, weights):
                choice = best_lane(note, context, weights=weights)
                if choice is None:
                    choice = best_lane(note, context, weights=weights, preference="ANY")
                staying = (
                    float("inf") if blocked else placement_cost(note, lane, context, weights).total
                )
                # 레인 충돌은 예산과 무관하게 반드시 풀어야 한다.
                # 같은 시각 같은 레인, 진행 중인 롱노트 위는 칠 수 없다.
                affordable = moved < move_allowance or blocked
                if (
                    choice is not None
                    and choice[0] != lane
                    and choice[1].total < staying
                    and affordable
                ):
                    lane = choice[0]
                    moved += 1
            occupied.add(lane)
            placed_note = note if lane == note.lane else dataclasses.replace(note, lane=lane)
            row_result.append(placed_note)
            hand_window.append(hand_of(semantics[lane]))

        for placed_note in row_result:
            last_time_by_lane[placed_note.lane] = time_ms
            if placed_note.kind == "HOLD":
                busy_until[placed_note.lane] = time_ms + (placed_note.duration_ms or 0)
        shape_window.append(tuple(sorted(occupied)))
        previous_lane = min(occupied, key=lambda l: abs(l - (previous_lane or l)))
        placed.extend(row_result)

    return sorted(placed, key=lambda n: (n.time_ms, n.lane)), moved


def _breaks_a_rule(
    note: NoteEvent, lane: int, context: PlacementContext, weights: CostWeights
) -> bool:
    if weights.rule_violation <= 0:
        return False
    return placement_cost(note, lane, context, weights).terms["w8_rule_violation"] > 0


def _relieve_side_holds(
    notes: Chart, *, key_mode: int, difficulty: str, weights: CostWeights, move_allowance: int
) -> tuple[Chart, int]:
    """S3 — 사이드 HOLD 를 잡은 손의 노트를 반대 손으로 옮긴다.

    S1·S4·C3 는 한 시각에서 판정되지만 S3 는 **구간 밀도**라 한 노트만
    보고는 알 수 없다. 그래서 따로 돈다.
    """
    semantics = lane_semantics(key_mode)
    violations = [
        violation
        for violation in check_lane_rules(notes, key_mode=key_mode, difficulty=difficulty)
        if violation.rule is Rule.S3_SIDE_HOLD_SAME_HAND
    ]
    if not violations or move_allowance <= 0:
        return notes, 0

    busy: list[tuple[Hand, int, int]] = []
    for violation in violations:
        hold_lane = violation.lanes[0]
        hold = next(
            note for note in notes if note.time_ms == violation.time_ms and note.lane == hold_lane
        )
        hand = hand_of(semantics[hold_lane])
        if hand is not None:
            busy.append((hand, hold.time_ms, hold.time_ms + (hold.duration_ms or 0)))

    occupied_by_time: dict[int, set[int]] = {}
    for note in notes:
        occupied_by_time.setdefault(note.time_ms, set()).add(note.lane)

    moved = 0
    result: list[NoteEvent] = []
    # 약한 노트부터 옮긴다. 강한 타격은 모델이 고른 자리에 남긴다.
    order = sorted(range(len(notes)), key=lambda i: notes[i].onset_strength or 0.0)
    relocated: dict[int, int] = {}
    for position in order:
        note = notes[position]
        hand = hand_of(semantics[note.lane])
        if hand is None or note.kind == "HOLD" or moved >= move_allowance:
            continue
        if not any(h is hand and start <= note.time_ms <= end for h, start, end in busy):
            continue
        free = [
            lane
            for lane in range(key_mode)
            if lane not in occupied_by_time[note.time_ms]
            and finger_of(semantics[lane]) is Finger.MAIN
            and hand_of(semantics[lane]) is not hand
        ]
        if not free:
            continue
        target = min(free, key=lambda lane: abs(lane - note.origin_lane))
        occupied_by_time[note.time_ms].discard(note.lane)
        occupied_by_time[note.time_ms].add(target)
        relocated[position] = target
        moved += 1

    result = [
        dataclasses.replace(note, lane=relocated[position]) if position in relocated else note
        for position, note in enumerate(notes)
    ]
    return sorted(result, key=lambda n: (n.time_ms, n.lane)), moved


def convert_lanes(
    notes: Chart,
    *,
    key_mode: int,
    difficulty: str,
    weights: CostWeights = DEFAULT_WEIGHTS,
    budget: float = MOVE_BUDGET,
    max_passes: int = MAX_PASSES,
) -> ConversionResult:
    """규칙 위반 노트를 인접 MAIN 레인으로 옮긴다.

    Phase3 §7 알고리즘 6단계.
    """
    original_times = sorted(note.time_ms for note in notes)
    if not notes:
        return ConversionResult([], 0, 0, 0.0, 0, ())

    # 절사하면 짧은 채보에서 예산이 0 이 되어 고칠 수 있는 위반도 전부
    # 삭제로 떨어진다. 삭제는 음악 이벤트를 잃으므로 이동보다 나쁘다.
    allowance = round(len(notes) * budget)
    current = sorted(notes, key=lambda n: (n.time_ms, n.lane))
    moved = 0
    passes = 0

    for _ in range(max_passes):
        passes += 1
        current, _ = _one_pass(
            current,
            key_mode=key_mode,
            difficulty=difficulty,
            weights=weights,
            move_allowance=max(0, allowance - _moved_notes(current)),
        )
        current, _ = _relieve_side_holds(
            current,
            key_mode=key_mode,
            difficulty=difficulty,
            weights=weights,
            move_allowance=max(0, allowance - _moved_notes(current)),
        )
        if not check_lane_rules(current, key_mode=key_mode, difficulty=difficulty):
            break

    current, deleted = _drop_unfixable(current, key_mode=key_mode, difficulty=difficulty)
    remaining = check_lane_rules(current, key_mode=key_mode, difficulty=difficulty)

    _require_timing_invariant(original_times, current)
    moved = _moved_notes(current)
    return ConversionResult(
        notes=current,
        moved_count=moved,
        deleted_count=deleted,
        moved_note_ratio=round(moved / len(notes), 4),
        passes=passes,
        remaining_violations=tuple(remaining),
    )


def _moved_notes(notes: Chart) -> int:
    """원배치에서 벗어난 노트 수.

    연산 횟수를 세면 같은 노트를 여러 패스에서 옮길 때 중복 집계되어
    moved_note_ratio 가 부풀고 예산이 일찍 소진된다. 다시 제자리로 돌아온
    노트는 옮긴 것이 아니다.
    """
    return sum(1 for note in notes if note.lane != note.origin_lane)


def _violation_endpoints(notes: Chart, violation: Violation) -> list[NoteEvent]:
    """잭 위반에 얽힌 두 노트. 뒤 노트와 그 앞의 같은 레인 노트다."""
    lane = violation.lanes[0]
    later = next((n for n in notes if n.time_ms == violation.time_ms and n.lane == lane), None)
    earlier = max(
        (n for n in notes if n.lane == lane and n.time_ms < violation.time_ms),
        key=lambda n: n.time_ms,
        default=None,
    )
    return [note for note in (later, earlier) if note is not None]


def _drop_unfixable(notes: Chart, *, key_mode: int, difficulty: str) -> tuple[Chart, int]:
    """이동으로 못 고친 잭 위반을 최소한으로 지운다.

    위반 하나에 노트가 둘 얽혀 있고, 연타가 셋 이상이면 가운데 노트가 앞뒤
    두 위반에 **동시에** 얽힌다. 위반을 하나씩 보고 약한 쪽을 지우면 그
    가운데 노트를 놓쳐 두 번 지우게 된다. 가장 많은 위반에 걸린 노트를
    먼저 지우고, 같으면 약한 쪽을 지운다.
    """
    remaining = list(notes)
    deleted = 0
    while True:
        violations = [
            violation
            for violation in check_lane_rules(remaining, key_mode=key_mode, difficulty=difficulty)
            if violation.rule is Rule.S1_JACK_INTERVAL
        ]
        if not violations:
            return remaining, deleted

        blame: dict[int, int] = {}
        involved: dict[int, NoteEvent] = {}
        for violation in violations:
            for note in _violation_endpoints(remaining, violation):
                blame[id(note)] = blame.get(id(note), 0) + 1
                involved[id(note)] = note
        if not involved:
            return remaining, deleted

        victim = max(
            involved.values(),
            key=lambda n: (blame[id(n)], -(n.onset_strength or 0.0), -n.time_ms),
        )
        remaining = [note for note in remaining if note is not victim]
        deleted += 1


def _require_timing_invariant(original_times: list[int], notes: Chart) -> None:
    """시각 집합이 입력의 부분집합인지 본다.

    후처리가 시각을 만들거나 옮겼다면 코드 버그다. 조용히 넘기면
    Mapperatorinator 를 채택한 이유 자체가 무너진다.
    """
    allowed = set(original_times)
    stray = sorted({note.time_ms for note in notes} - allowed)
    if stray:
        raise ValueError(f"lane conversion invented note times: {stray[:5]}")
