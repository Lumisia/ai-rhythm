"""난이도 solver — 목표 ★ 까지 노트를 솎는다.

**내리는 방향으로만 동작한다.** 없는 시각에 노트를 만들면 타이밍 불변
위반이다. 그래서 생성 단계가 목표보다 높은 채보를 만들어 재료를 남긴다.

레인 변환과 같은 틀이다 — 시각은 고정하고 노트 존재만 바꾼다.
HOLD → TAP 변환도 시작 시각을 유지하므로 불변 원칙을 통과한다.
"""

import dataclasses
from collections.abc import Iterator
from dataclasses import dataclass
from enum import StrEnum

from chart_worker.postprocess.patterns import rows_of
from chart_worker.rating.project_rating import TARGET_RATING, RatingMetrics, measure_rating
from chart_worker.schema.note import Chart, NoteEvent
from chart_worker.schema.types import DIFFICULTIES, TARGET_HOLD_RATIO

REMOVAL_BUDGET = 0.35
"""솎을 수 있는 노트 비율 상한. 초안이다.

다 써도 목표에 못 닿으면 채보를 살리고 실제 ★ 을 기록한다. 거짓 라벨보다
정직한 숫자가 낫다.
"""

RATING_TOLERANCE = 0.1
"""목표를 이만큼 밑돌 때까지 내리지 않는다. 과도하게 깎으면 밋밋해진다."""

DOWNBEAT_WEIGHT = 0.6
"""다운비트 노트를 빼면 마디의 뼈대가 무너진다."""

SECTION_WEIGHT: dict[str, float] = {"chorus": 0.5, "build": 0.3, "verse": 0.1}
"""곡 구조를 아는 경우에만 쓴다. Lyria 가 구조 JSON 을 준다."""

ISOLATION_WEIGHT = 0.8
"""주변에 노트가 없으면 빼는 순간 구멍이 된다."""

DEFAULT_STRENGTH = 0.5
"""onset 분석을 안 거친 노트의 대체값. 중간값이라 어느 쪽으로도 안 민다."""

MAX_JACK_RUN = 2
"""이보다 긴 같은 레인 연속의 초과분이 잭 축약 대상이다."""

REMEASURE_EVERY = 8
"""이만큼 솎을 때마다 실제 등급을 다시 잰다."""


class Operation(StrEnum):
    """약한 것부터. 앞선 연산이 음악을 덜 해친다."""

    CHORD_BREAK = "CHORD_BREAK"
    WEAK_ONSET = "WEAK_ONSET"
    SUBDIVISION = "SUBDIVISION"
    JACK_TRIM = "JACK_TRIM"
    HOLD_TO_TAP = "HOLD_TO_TAP"


OPERATION_ORDER: tuple[Operation, ...] = tuple(Operation)


@dataclass(frozen=True, slots=True)
class SolveResult:
    notes: Chart
    metrics: RatingMetrics
    target_rating: float
    removed_count: int
    removal_allowance: int
    converted_count: int
    removed_ratio: float
    operations: dict[str, int]

    @property
    def reached_target(self) -> bool:
        return self.metrics.rating <= self.target_rating + RATING_TOLERANCE

    @property
    def budget_exhausted(self) -> bool:
        return not self.reached_target and self.removed_count >= self.removal_allowance


def musical_cost(note: NoteEvent, *, isolation: float) -> float:
    """이 노트를 잃을 때 음악이 치르는 값.

    solver 는 이 값이 낮은 것부터 가져간다.
    """
    cost = note.onset_strength if note.onset_strength is not None else DEFAULT_STRENGTH
    if note.is_downbeat:
        cost += DOWNBEAT_WEIGHT
    cost += SECTION_WEIGHT.get(note.section or "", 0.0)
    return cost + ISOLATION_WEIGHT * isolation


def _isolation_by_note(notes: Chart, *, beat_ms: float) -> dict[int, float]:
    """앞뒤 이웃까지의 거리. 한 박 이상 떨어져 있으면 1.0 이다."""
    times = sorted({note.time_ms for note in notes})
    gap_at: dict[int, float] = {}
    for index, time_ms in enumerate(times):
        before = time_ms - times[index - 1] if index else beat_ms
        after = times[index + 1] - time_ms if index + 1 < len(times) else beat_ms
        gap_at[time_ms] = min(1.0, min(before, after) / beat_ms)
    return {id(note): gap_at[note.time_ms] for note in notes}


def _chord_break_candidates(notes: Chart) -> Iterator[NoteEvent]:
    """동시치기에서 가장 약한 노트. chord_ratio 가중치가 1.10 으로 가장 크다."""
    for row in rows_of(notes):
        if row.size < 2:
            continue
        weakest = min(row.notes, key=lambda n: n.onset_strength or DEFAULT_STRENGTH)
        yield weakest


def _weak_onset_candidates(notes: Chart) -> Iterator[NoteEvent]:
    ordered = sorted(notes, key=lambda n: n.onset_strength or DEFAULT_STRENGTH)
    yield from ordered


def _subdivision_candidates(notes: Chart) -> Iterator[NoteEvent]:
    """세부리듬부터. 16분 → 8분 → 정박 순으로 가져간다."""

    def rank(note: NoteEvent) -> float:
        fraction = note.beat_fraction
        if fraction is None:
            return 2.0
        offbeat = min(fraction % 0.5, 0.5 - (fraction % 0.5))
        on_eighth = min(abs(fraction - 0.5), fraction, abs(1.0 - fraction))
        if offbeat > 0.05:
            return 0.0  # 16분·24분 자리
        if on_eighth > 0.05:
            return 1.0  # 8분 뒷박
        return 3.0  # 정박은 마지막

    return iter(sorted((n for n in notes if n.beat_fraction is not None), key=rank))


def _jack_trim_candidates(notes: Chart) -> Iterator[NoteEvent]:
    """같은 레인 연속의 초과분. 오래된 것부터 남기고 뒤를 자른다."""
    runs: dict[int, list[NoteEvent]] = {}
    for row in rows_of(notes):
        lanes_in_row = {note.lane for note in row.notes}
        for lane in list(runs):
            if lane not in lanes_in_row:
                runs[lane] = []
        for note in row.notes:
            run = runs.setdefault(note.lane, [])
            run.append(note)
            if len(run) > MAX_JACK_RUN:
                yield note


def _hold_candidates(notes: Chart) -> Iterator[NoteEvent]:
    yield from (note for note in notes if note.kind == "HOLD")


def _hold_ratio(notes: Chart) -> float:
    """지금 채보의 롱노트 비율.

    삭제 연산도 롱노트를 가져가므로 변환 횟수만 세면 실제와 어긋난다.
    매번 실측한다.
    """
    if not notes:
        return 0.0
    return sum(1 for note in notes if note.kind == "HOLD") / len(notes)


_CANDIDATE_SOURCES = {
    Operation.CHORD_BREAK: _chord_break_candidates,
    Operation.WEAK_ONSET: _weak_onset_candidates,
    Operation.SUBDIVISION: _subdivision_candidates,
    Operation.JACK_TRIM: _jack_trim_candidates,
    Operation.HOLD_TO_TAP: _hold_candidates,
}


def _ranked_candidates(notes: Chart, operation: Operation, *, beat_ms: float) -> list[NoteEvent]:
    """이 연산이 건드릴 노트를 음악 비용이 낮은 순으로."""
    isolation = _isolation_by_note(notes, beat_ms=beat_ms)
    candidates = list(_CANDIDATE_SOURCES[operation](notes))
    return sorted(candidates, key=lambda n: musical_cost(n, isolation=isolation.get(id(n), 1.0)))


def solve_difficulty(
    notes: Chart,
    *,
    duration_ms: int,
    difficulty: str,
    beat_ms: float,
    budget: float = REMOVAL_BUDGET,
    tolerance: float = RATING_TOLERANCE,
) -> SolveResult:
    """목표 등급까지 솎는다.

    설계의 탐욕 알고리즘은 후보마다 Δrating 을 재서 고르지만, 노트 하나가
    바꾸는 등급이 1e-3 수준이라 1500노트 채보에서 전량 재측정이 수백만 번
    필요하다. 대신 **연산 등급 안에서 음악 비용 순으로 가져가고**, 몇 개마다
    실제 등급을 다시 재서 멈출 때를 정한다. 우선순위와 비용 기준은 그대로다.
    """
    if difficulty not in DIFFICULTIES:
        raise ValueError(f"unsupported difficulty: {difficulty}")
    if beat_ms <= 0:
        raise ValueError("beat_ms must be positive")

    target = TARGET_RATING[difficulty]
    current = list(notes)
    metrics = measure_rating(current, duration_ms)
    counts = dict.fromkeys((op.value for op in Operation), 0)
    removed = converted = 0
    allowance = int(len(notes) * budget)
    if not current:
        return SolveResult(
            notes=current,
            metrics=metrics,
            target_rating=target,
            removed_count=0,
            removal_allowance=allowance,
            converted_count=0,
            removed_ratio=0.0,
            operations=counts,
        )

    hold_floor = TARGET_HOLD_RATIO[difficulty]
    # 연산을 하나씩 소진하면 앞선 연산이 예산을 다 먹어 뒤가 굶는다. 잭
    # 축약은 max_jack 항을 줄이는 유일한 수단인데 그게 한 번도 안 돌 수
    # 있다. 라운드마다 모든 연산에 차례를 준다.
    while metrics.rating > target + tolerance:
        applied_this_round = 0
        for operation in OPERATION_ORDER:
            if metrics.rating <= target + tolerance:
                break
            # 예산은 **삭제**에만 걸린다. HOLD 를 TAP 으로 바꾸는 것은 노트를
            # 없애지 않으므로 삭제보다 순하다. 예산이 마르면 더 나쁜 수단만
            # 남기고 순한 수단을 막는 꼴이 된다.
            removes = operation is not Operation.HOLD_TO_TAP
            if removes and removed >= allowance:
                continue
            # 삭제 예산에서 뺐다고 상한까지 없애면 목표에 못 닿는 채보에서
            # 롱노트가 전멸한다. 등급 기여는 0.60*hold_ratio 뿐이라 다 없애도
            # 얼마 못 내리면서 노트 종류 하나를 통째로 지운다. 난이도별 목표
            # 비율이 바닥이다.
            if not removes and _hold_ratio(current) <= hold_floor:
                continue
            queue = _ranked_candidates(current, operation, beat_ms=beat_ms)
            for note in queue[:REMEASURE_EVERY]:
                if removes and removed >= allowance:
                    break
                if operation is Operation.HOLD_TO_TAP:
                    if _hold_ratio(current) <= hold_floor:
                        break
                    index = next((i for i, n in enumerate(current) if n is note), None)
                    if index is None:
                        continue
                    current[index] = dataclasses.replace(note, kind="TAP", duration_ms=None)
                    converted += 1
                else:
                    current = [n for n in current if n is not note]
                    removed += 1
                counts[operation.value] += 1
                applied_this_round += 1
            metrics = measure_rating(current, duration_ms)
        if applied_this_round == 0:
            break

    _require_timing_invariant(notes, current)
    return SolveResult(
        notes=current,
        metrics=metrics,
        target_rating=target,
        removed_count=removed,
        removal_allowance=allowance,
        converted_count=converted,
        removed_ratio=round(removed / len(notes), 4),
        operations=counts,
    )


def _require_timing_invariant(before: Chart, after: Chart) -> None:
    """솎기와 HOLD 변환은 시각을 만들지 않는다."""
    stray = sorted({n.time_ms for n in after} - {n.time_ms for n in before})
    if stray:
        raise ValueError(f"difficulty solver invented note times: {stray[:5]}")
