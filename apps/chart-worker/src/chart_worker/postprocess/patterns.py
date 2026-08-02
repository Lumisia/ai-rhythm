"""구조 패턴 검출 — 레인 의미와 무관한 계열.

Phase3 §4 카탈로그의 4키 22종을 다룬다. 사이드·센터 계열(6·7키 추가분)은
laneSemantics 에 의존하므로 lane_rules 가 맡는다.

검출 결과는 세 곳에 쓰인다.
  1. chart-v1 의 metrics.patternEntropy
  2. 난이도별 패턴 해금표 집행 (Phase3 §8)
  3. 비용함수 w8 · w11
"""

import math
from collections import Counter
from dataclasses import dataclass
from enum import StrEnum
from itertools import pairwise

from chart_worker.schema.note import Chart, NoteEvent

GRACE_DIVISOR = 6.0
"""1/6 스냅보다 빠른 서로 다른 레인 연속은 화음처럼 들린다."""

BURST_DIVISOR = 4.0
"""1/4 스냅보다 빠른 짧은 스트림."""

STREAM_MIN_NOTES = 4
STAIRS_MIN_NOTES = 3
TRILL_MIN_NOTES = 3
BURST_MAX_NOTES = 8
DENIM_MIN_NOTES = 6
INVERSE_MIN_HOLDS = 4
ANCHOR_MIN_NOTES = 4
ANCHOR_GAP_RANGE = (0.4, 0.6)
"""1/2 박 근처. 손가락이 닻처럼 한 레인에 고정된다."""

INTERVAL_TOLERANCE = 0.25
"""연속 간격이 이 비율 안에서 흔들리면 같은 스트림으로 본다."""

SHIELD_GAP_DIVISOR = 2.0
INVERSE_GAP_MS = 30

MAX_RUN_GAP_BEATS = 1.0
"""연속 패턴이 이어지는 최대 간격.

트릴·계단·데님은 **끊기지 않는 흐름**이다. 화음 행을 걸러내고 남은
단노트만 이으면 10초 떨어진 두 노트나 화음이 관통한 구간도 하나로
붙어버린다. 행 인접성과 간격을 함께 봐야 한다.
"""


class PatternKind(StrEnum):
    JUMP = "JUMP"
    HAND = "HAND"
    QUAD = "QUAD"
    GRACE = "GRACE"

    STREAM = "STREAM"
    ROLL = "ROLL"
    BURST = "BURST"
    STAIRS = "STAIRS"
    JUMPSTREAM = "JUMPSTREAM"
    HANDSTREAM = "HANDSTREAM"
    CHORDSTREAM = "CHORDSTREAM"

    MINIJACK = "MINIJACK"
    JACK = "JACK"
    LONGJACK = "LONGJACK"
    ANCHOR = "ANCHOR"
    CHORDJACK = "CHORDJACK"

    TRILL_ONE_HANDED = "TRILL_ONE_HANDED"
    TRILL_TWO_HANDED = "TRILL_TWO_HANDED"
    JUMPTRILL = "JUMPTRILL"
    DENIM = "DENIM"

    HOLD = "HOLD"
    SHIELD = "SHIELD"
    REVERSE_SHIELD = "REVERSE_SHIELD"
    INVERSE = "INVERSE"


@dataclass(frozen=True, slots=True)
class PatternInstance:
    kind: PatternKind
    start_ms: int
    end_ms: int
    lanes: tuple[int, ...]
    size: int

    @property
    def duration_ms(self) -> int:
        return self.end_ms - self.start_ms


@dataclass(frozen=True, slots=True)
class Row:
    """같은 시각에 놓인 노트 묶음."""

    time_ms: int
    notes: tuple[NoteEvent, ...]

    @property
    def lanes(self) -> tuple[int, ...]:
        return tuple(note.lane for note in self.notes)

    @property
    def size(self) -> int:
        return len(self.notes)


def rows_of(notes: Chart) -> list[Row]:
    """노트를 시각별로 묶는다. 화음 판정의 기본 단위다."""
    grouped: dict[int, list[NoteEvent]] = {}
    for note in notes:
        grouped.setdefault(note.time_ms, []).append(note)
    return [
        Row(time_ms=time_ms, notes=tuple(sorted(grouped[time_ms], key=lambda n: n.lane)))
        for time_ms in sorted(grouped)
    ]


def _single_note_runs(rows: list[Row], *, beat_ms: float) -> list[list[Row]]:
    """화음에 끊기지 않고 간격도 벌어지지 않은 단노트 구간들."""
    limit = beat_ms * MAX_RUN_GAP_BEATS
    runs: list[list[Row]] = []
    current: list[Row] = []
    for row in rows:
        if row.size != 1:
            # 화음이 끼면 흐름이 끊긴다.
            if len(current) > 1:
                runs.append(current)
            current = []
            continue
        if current and row.time_ms - current[-1].time_ms > limit:
            if len(current) > 1:
                runs.append(current)
            current = []
        current.append(row)
    if len(current) > 1:
        runs.append(current)
    return runs


def _hand_of(lane: int, key_mode: int) -> int:
    """0 이 왼손, 1 이 오른손. 홀수 키의 가운데 레인은 오른손으로 센다."""
    return 0 if lane < key_mode // 2 else 1


def _close(a: float, b: float, tolerance: float = INTERVAL_TOLERANCE) -> bool:
    if a <= 0 or b <= 0:
        return False
    return abs(a - b) <= tolerance * max(a, b)


# --- 화음 -------------------------------------------------------------------


def detect_chords(rows: list[Row]) -> list[PatternInstance]:
    """동시 노트 수로 분류한다.

    5노트 이상은 QUAD 로 내되 size 에 실제 수를 남긴다. 해금표에 별도 행이
    없으므로 새 종류를 만들면 집행 단계에 구멍이 생긴다.
    """
    kinds = {2: PatternKind.JUMP, 3: PatternKind.HAND}
    found = []
    for row in rows:
        if row.size < 2:
            continue
        kind = kinds.get(row.size, PatternKind.QUAD)
        found.append(PatternInstance(kind, row.time_ms, row.time_ms, row.lanes, row.size))
    return found


def detect_grace(rows: list[Row], *, beat_ms: float) -> list[PatternInstance]:
    """1/6 스냅보다 빠른 서로 다른 레인 연속."""
    limit = beat_ms / GRACE_DIVISOR
    found = []
    for first, second in pairwise(rows):
        gap = second.time_ms - first.time_ms
        if 0 < gap < limit and not set(first.lanes) & set(second.lanes):
            found.append(
                PatternInstance(
                    PatternKind.GRACE,
                    first.time_ms,
                    second.time_ms,
                    first.lanes + second.lanes,
                    first.size + second.size,
                )
            )
    return found


# --- 잭 ---------------------------------------------------------------------


def _lane_runs(rows: list[Row]) -> list[tuple[int, list[int]]]:
    """레인별로, 중간에 다른 노트가 끼지 않는 연속 등장 구간."""
    runs: list[tuple[int, list[int]]] = []
    active: dict[int, list[int]] = {}
    for row in rows:
        lanes = set(row.lanes)
        for lane in list(active):
            if lane not in lanes:
                runs.append((lane, active.pop(lane)))
        for lane in lanes:
            active.setdefault(lane, []).append(row.time_ms)
    runs.extend(active.items())
    return runs


def detect_jacks(rows: list[Row]) -> list[PatternInstance]:
    """같은 레인 연속. 길이로 갈린다.

    긴 잭은 짧은 잭을 겸한다 — 4연타는 LONGJACK 이면서 JACK 이다.
    해금표가 Jack(3+)과 Longjack 을 따로 다루므로 둘 다 낸다.
    """
    found = []
    for lane, times in _lane_runs(rows):
        length = len(times)
        if length < 2:
            continue
        span = (times[0], times[-1])
        if length == 2:
            found.append(PatternInstance(PatternKind.MINIJACK, *span, (lane,), 2))
            continue
        found.append(PatternInstance(PatternKind.JACK, *span, (lane,), length))
        if length >= 4:
            found.append(PatternInstance(PatternKind.LONGJACK, *span, (lane,), length))
    return found


def detect_anchor(rows: list[Row], *, beat_ms: float) -> list[PatternInstance]:
    """한 레인이 1/2 박 간격으로 계속 등장하고 사이에 다른 레인이 낀다."""
    low, high = (beat_ms * ratio for ratio in ANCHOR_GAP_RANGE)
    by_lane: dict[int, list[int]] = {}
    for row in rows:
        for lane in row.lanes:
            by_lane.setdefault(lane, []).append(row.time_ms)
    other_times = {row.time_ms for row in rows if row.size >= 1}

    found = []
    for lane, times in by_lane.items():
        run = [times[0]]
        for previous, current in pairwise(times):
            gap = current - previous
            interleaved = any(
                previous < time_ms < current and time_ms not in (previous, current)
                for time_ms in other_times
            )
            if low <= gap <= high and interleaved:
                run.append(current)
                continue
            if len(run) >= ANCHOR_MIN_NOTES:
                found.append(
                    PatternInstance(PatternKind.ANCHOR, run[0], run[-1], (lane,), len(run))
                )
            run = [current]
        if len(run) >= ANCHOR_MIN_NOTES:
            found.append(PatternInstance(PatternKind.ANCHOR, run[0], run[-1], (lane,), len(run)))
    return found


def detect_chordjack(rows: list[Row]) -> list[PatternInstance]:
    """연속한 두 화음이 레인을 두 개 이상 공유한다."""
    found = []
    for first, second in pairwise(rows):
        if first.size < 2 or second.size < 2:
            continue
        shared = set(first.lanes) & set(second.lanes)
        if len(shared) >= 2:
            found.append(
                PatternInstance(
                    PatternKind.CHORDJACK,
                    first.time_ms,
                    second.time_ms,
                    tuple(sorted(shared)),
                    first.size + second.size,
                )
            )
    return found


# --- 스트림 -----------------------------------------------------------------


def _runs_of_even_intervals(rows: list[Row]) -> list[list[Row]]:
    """간격이 고르게 이어지는 구간으로 자른다."""
    if len(rows) < 2:
        return []
    runs: list[list[Row]] = []
    current = [rows[0], rows[1]]
    for previous, row in pairwise(rows[1:]):
        if _close(row.time_ms - previous.time_ms, current[-1].time_ms - current[-2].time_ms):
            current.append(row)
            continue
        runs.append(current)
        current = [previous, row]
    runs.append(current)
    return runs


def detect_streams(rows: list[Row], *, beat_ms: float, key_mode: int) -> list[PatternInstance]:
    """고른 간격의 연속. 섞인 화음 크기로 이름이 갈린다."""
    found = []
    for run in _runs_of_even_intervals(rows):
        if len(run) < STREAM_MIN_NOTES:
            continue
        interval = run[1].time_ms - run[0].time_ms
        lanes = tuple(lane for row in run for lane in row.lanes)
        span = (run[0].time_ms, run[-1].time_ms)
        sizes = {row.size for row in run}
        largest = max(sizes)

        if largest == 1:
            kind = PatternKind.STREAM
        elif largest == 2:
            kind = PatternKind.JUMPSTREAM
        elif largest == 3:
            kind = PatternKind.HANDSTREAM
        else:
            kind = PatternKind.CHORDSTREAM
        found.append(PatternInstance(kind, *span, lanes, len(lanes)))

        if len(sizes) > 1 and largest >= 3:
            found.append(PatternInstance(PatternKind.CHORDSTREAM, *span, lanes, len(lanes)))
        if largest == 1 and interval < beat_ms / BURST_DIVISOR and len(run) <= BURST_MAX_NOTES:
            found.append(PatternInstance(PatternKind.BURST, *span, lanes, len(run)))
        if largest == 1 and _is_roll(run, key_mode):
            found.append(PatternInstance(PatternKind.ROLL, *span, lanes, len(run)))
    return found


def _is_roll(run: list[Row], key_mode: int) -> bool:
    """모든 레인을 한 방향으로 훑는다."""
    lanes = [row.lanes[0] for row in run]
    if len(set(lanes)) < key_mode:
        return False
    steps = {b - a for a, b in pairwise(lanes)}
    return steps in ({1}, {-1}) or steps <= {1, 1 - key_mode} or steps <= {-1, key_mode - 1}


def detect_stairs(rows: list[Row], *, beat_ms: float) -> list[PatternInstance]:
    """레인이 한 칸씩 같은 방향으로 움직이는 단노트 연속."""
    found = []
    for singles in _single_note_runs(rows, beat_ms=beat_ms):
        run = [singles[0]]
        direction = 0
        for previous, row in pairwise(singles):
            step = row.lanes[0] - previous.lanes[0]
            if step in (1, -1) and (direction == 0 or step == direction):
                direction = step
                run.append(row)
                continue
            if len(run) >= STAIRS_MIN_NOTES:
                found.append(_stairs_instance(run))
            run = [previous, row] if step in (1, -1) else [row]
            direction = step if step in (1, -1) else 0
        if len(run) >= STAIRS_MIN_NOTES:
            found.append(_stairs_instance(run))
    return found


def _stairs_instance(run: list[Row]) -> PatternInstance:
    return PatternInstance(
        PatternKind.STAIRS,
        run[0].time_ms,
        run[-1].time_ms,
        tuple(row.lanes[0] for row in run),
        len(run),
    )


# --- 트릴 -------------------------------------------------------------------


def detect_trills(rows: list[Row], *, key_mode: int, beat_ms: float) -> list[PatternInstance]:
    """두 레인을 오가는 단노트 교대."""
    found = []
    for singles in _single_note_runs(rows, beat_ms=beat_ms):
        index = 0
        while index + TRILL_MIN_NOTES <= len(singles):
            first, second = singles[index].lanes[0], singles[index + 1].lanes[0]
            if first == second:
                index += 1
                continue
            end = index + 2
            while (
                end < len(singles) and singles[end].lanes[0] == (first, second)[end % 2 - index % 2]
            ):
                end += 1
            length = end - index
            if length >= TRILL_MIN_NOTES:
                same_hand = _hand_of(first, key_mode) == _hand_of(second, key_mode)
                kind = PatternKind.TRILL_ONE_HANDED if same_hand else PatternKind.TRILL_TWO_HANDED
                found.append(
                    PatternInstance(
                        kind,
                        singles[index].time_ms,
                        singles[end - 1].time_ms,
                        (first, second),
                        length,
                    )
                )
                index = end - 1
            else:
                index += 1
    return found


def detect_jumptrill(rows: list[Row]) -> list[PatternInstance]:
    """레인이 겹치지 않는 두 화음의 교대."""
    found = []
    index = 0
    while index + 2 < len(rows):
        window = rows[index : index + 3]
        if (
            all(row.size == 2 for row in window)
            and not (set(window[0].lanes) & set(window[1].lanes))
            and set(window[0].lanes) == set(window[2].lanes)
        ):
            found.append(
                PatternInstance(
                    PatternKind.JUMPTRILL,
                    window[0].time_ms,
                    window[2].time_ms,
                    window[0].lanes + window[1].lanes,
                    6,
                )
            )
            index += 2
            continue
        index += 1
    return found


def detect_denim(rows: list[Row], *, beat_ms: float) -> list[PatternInstance]:
    """홀수 레인과 짝수 레인이 번갈아 나오는 긴 단노트 연속."""
    found = []
    for singles in _single_note_runs(rows, beat_ms=beat_ms):
        run: list[Row] = []
        for row in singles:
            if run and row.lanes[0] % 2 == run[-1].lanes[0] % 2:
                if len(run) >= DENIM_MIN_NOTES and len({r.lanes[0] for r in run}) > 2:
                    found.append(_denim_instance(run))
                run = [row]
                continue
            run.append(row)
        if len(run) >= DENIM_MIN_NOTES and len({r.lanes[0] for r in run}) > 2:
            found.append(_denim_instance(run))
    return found


def _denim_instance(run: list[Row]) -> PatternInstance:
    return PatternInstance(
        PatternKind.DENIM,
        run[0].time_ms,
        run[-1].time_ms,
        tuple(row.lanes[0] for row in run),
        len(run),
    )


# --- 롱노트 -----------------------------------------------------------------


def detect_hold_patterns(notes: Chart, *, beat_ms: float) -> list[PatternInstance]:
    """롱노트와 그 주변 단노트."""
    holds = sorted(
        (note for note in notes if note.kind == "HOLD"), key=lambda n: (n.time_ms, n.lane)
    )
    found = [
        PatternInstance(
            PatternKind.HOLD, hold.time_ms, hold.time_ms + (hold.duration_ms or 0), (hold.lane,), 1
        )
        for hold in holds
    ]
    found.extend(_shields(notes, holds, beat_ms=beat_ms))
    found.extend(_inverse(holds))
    return found


def _shields(notes: Chart, holds: list[NoteEvent], *, beat_ms: float) -> list[PatternInstance]:
    limit = beat_ms / SHIELD_GAP_DIVISOR
    taps_by_lane: dict[int, list[int]] = {}
    for note in notes:
        if note.kind == "TAP":
            taps_by_lane.setdefault(note.lane, []).append(note.time_ms)

    found = []
    for hold in holds:
        end_ms = hold.time_ms + (hold.duration_ms or 0)
        for tap_ms in taps_by_lane.get(hold.lane, ()):
            if 0 < hold.time_ms - tap_ms <= limit:
                found.append(
                    PatternInstance(PatternKind.SHIELD, tap_ms, hold.time_ms, (hold.lane,), 2)
                )
            elif 0 < tap_ms - end_ms <= limit:
                found.append(
                    PatternInstance(PatternKind.REVERSE_SHIELD, end_ms, tap_ms, (hold.lane,), 2)
                )
    return found


def _inverse(holds: list[NoteEvent]) -> list[PatternInstance]:
    """앞 롱노트의 끝이 다음 노트 시작에 닿는 롱노트 벽."""
    found = []
    run = [holds[0]] if holds else []
    for previous, hold in pairwise(holds):
        end_ms = previous.time_ms + (previous.duration_ms or 0)
        if 0 <= hold.time_ms - end_ms <= INVERSE_GAP_MS:
            run.append(hold)
            continue
        if len(run) >= INVERSE_MIN_HOLDS:
            found.append(_inverse_instance(run))
        run = [hold]
    if len(run) >= INVERSE_MIN_HOLDS:
        found.append(_inverse_instance(run))
    return found


def _inverse_instance(run: list[NoteEvent]) -> PatternInstance:
    last = run[-1]
    return PatternInstance(
        PatternKind.INVERSE,
        run[0].time_ms,
        last.time_ms + (last.duration_ms or 0),
        tuple(hold.lane for hold in run),
        len(run),
    )


# --- 종합 -------------------------------------------------------------------


def detect_patterns(notes: Chart, *, key_mode: int, beat_ms: float) -> list[PatternInstance]:
    """구조 패턴을 전부 찾는다."""
    if beat_ms <= 0:
        raise ValueError("beat_ms must be positive")
    rows = rows_of(notes)
    found = [
        *detect_chords(rows),
        *detect_grace(rows, beat_ms=beat_ms),
        *detect_jacks(rows),
        *detect_anchor(rows, beat_ms=beat_ms),
        *detect_chordjack(rows),
        *detect_streams(rows, beat_ms=beat_ms, key_mode=key_mode),
        *detect_stairs(rows, beat_ms=beat_ms),
        *detect_trills(rows, key_mode=key_mode, beat_ms=beat_ms),
        *detect_jumptrill(rows),
        *detect_denim(rows, beat_ms=beat_ms),
        *detect_hold_patterns(notes, beat_ms=beat_ms),
    ]
    return sorted(found, key=lambda instance: (instance.start_ms, instance.kind))


def pattern_histogram(instances: list[PatternInstance]) -> dict[PatternKind, int]:
    counts = Counter(instance.kind for instance in instances)
    return {kind: counts[kind] for kind in PatternKind if counts[kind]}


def pattern_entropy(histogram: dict[PatternKind, int]) -> float:
    """패턴 종류 분포의 섀넌 엔트로피(bit).

    낮으면 같은 패턴만 반복한다는 뜻이다. chart-v1 의 metrics 에 실린다.
    """
    total = sum(histogram.values())
    if total <= 0:
        return 0.0
    return round(
        -sum(
            (count / total) * math.log2(count / total) for count in histogram.values() if count > 0
        ),
        4,
    )
