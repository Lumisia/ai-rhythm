"""배치 비용함수 — 레인 변환과 난이도 solver 가 공유한다.

두 문제가 같은 틀이다. "시각은 고정하고 레인과 노트 존재만 바꾼다."
그래서 비용함수도 하나다.

w9(원배치이탈)가 C안의 핵심이다. 모델이 만든 배치를 최대한 존중한다.
없으면 "그냥 다시 배치"가 되어 위반 노트를 지우는 A안보다 나을 게 없다.
"""

import math
from dataclasses import dataclass
from typing import Literal

from chart_worker.postprocess.lane_rules import (
    ACCENT_STRENGTH,
    BOTH_SIDES,
    CENTER_WITH_BOTH_SIDES_FROM,
    MAX_MAIN_WITH_CENTER_AND_BOTH_SIDES,
    BothSidesPolicy,
    FingerClass,
    Hand,
    finger_of,
    hand_of,
    jack_interval_ms,
)
from chart_worker.schema.note import NoteEvent
from chart_worker.schema.types import LaneSemantic, lane_semantics

INFEASIBLE = math.inf


@dataclass(frozen=True, slots=True)
class CostWeights:
    """Phase3 §7 의 w1~w11. 값은 초안이고 골든 테스트로 조정한다."""

    same_lane_repeat: float = 3.0
    """w1. 직전 같은 레인 노트와 간격이 짧을수록 크다."""

    hand_balance: float = 1.0
    """w2. 최근 창의 좌우 불균형."""

    travel: float = 0.6
    """w3. 직전 노트에서 손이 움직인 거리."""

    jack: float = 2.0
    """w4. 바로 앞 노트와 같은 레인."""

    chord_spread: float = 1.2
    """w5. 동시치기 시 손 벌어짐."""

    pattern_repeat: float = 0.8
    """w6. 직전 몇 개와 같은 모양."""

    band_mismatch: float = 0.3
    """w7. 저역은 왼쪽, 고역은 오른쪽. 약한 가중치다."""

    rule_violation: float = 12.0
    """w8. S1~S4 · C1~C3 위반. 크지만 유한하다 — 모든 후보가 위반이면
    가장 덜 나쁜 곳에 두고 재검사와 예산이 뒷정리한다."""

    origin_drift: float = 2.5
    """w9. origin_lane 에서 멀수록 크다. C안의 핵심."""

    target_pattern_reward: float = 0.0
    """w10. 구간별 목표 패턴 적합도(보상). 구간 분할이 아직 없어 0 이다."""

    recent_pattern_repeat: float = 0.0
    """w11. 직전 구간과 같은 패턴이면 페널티. w10 과 같은 이유로 0 이다."""


DEFAULT_WEIGHTS = CostWeights()


@dataclass(frozen=True, slots=True)
class PlacementContext:
    """한 노트를 어디 둘지 고를 때 필요한 주변 정보."""

    key_mode: int
    difficulty: str
    last_time_by_lane: dict[int, int]
    """이 노트보다 앞선 각 레인의 마지막 노트 시각."""

    occupied_lanes: frozenset[int] = frozenset()
    """같은 시각에 이미 찬 레인."""

    previous_lane: int | None = None
    """바로 앞 시각의 노트 레인. 여러 개면 가장 가까운 것."""

    hand_counts: tuple[int, int] = (0, 0)
    """최근 창의 (왼손, 오른손) 노트 수."""

    recent_shapes: tuple[tuple[int, ...], ...] = ()
    """최근 행들의 레인 모양. w6 이 쓴다."""

    row_is_downbeat: bool = False
    row_is_accent: bool = False
    """S4 는 행 전체의 악센트를 본다. 후보 노트 하나만 보면 같은 행의 다른
    노트가 들고 있는 다운비트를 놓친다."""

    @property
    def semantics(self) -> list[LaneSemantic]:
        return lane_semantics(self.key_mode)


@dataclass(frozen=True, slots=True)
class CostBreakdown:
    total: float
    terms: dict[str, float]


def _same_lane_repeat(note: NoteEvent, lane: int, context: PlacementContext) -> float:
    """직전 같은 레인 노트와의 간격을 손가락 한계로 나눈다."""
    last = context.last_time_by_lane.get(lane)
    if last is None:
        return 0.0
    limit = jack_interval_ms(context.semantics[lane], context.difficulty)
    gap = note.time_ms - last
    if gap >= limit:
        return 0.0
    return min(1.0, (limit - gap) / limit)


def _hand_balance(lane: int, context: PlacementContext) -> float:
    left, right = context.hand_counts
    if left + right == 0:
        # 창이 비어 있으면 균형에 대해 아는 것이 없다. 여기서 0 을 주지 않으면
        # 어느 손에 둬도 불균형 최대가 되고, 손이 없는 CENTER 만 공짜가 된다.
        return 0.0
    hand = hand_of(context.semantics[lane])
    if hand is Hand.LEFT:
        left += 1
    elif hand is Hand.RIGHT:
        right += 1
    return abs(left - right) / (left + right)


def _travel(lane: int, context: PlacementContext) -> float:
    if context.previous_lane is None or context.key_mode < 2:
        return 0.0
    return abs(lane - context.previous_lane) / (context.key_mode - 1)


def _jack(lane: int, context: PlacementContext) -> float:
    return 1.0 if lane == context.previous_lane else 0.0


def _chord_spread(lane: int, context: PlacementContext) -> float:
    if not context.occupied_lanes or context.key_mode < 2:
        return 0.0
    lanes = {*context.occupied_lanes, lane}
    return (max(lanes) - min(lanes)) / (context.key_mode - 1)


def _pattern_repeat(lane: int, context: PlacementContext) -> float:
    """직전 행들과 같은 모양이면 크다."""
    if not context.recent_shapes:
        return 0.0
    shape = tuple(sorted({*context.occupied_lanes, lane}))
    matches = sum(1 for recent in context.recent_shapes if recent == shape)
    return matches / len(context.recent_shapes)


def _band_mismatch(note: NoteEvent, lane: int, context: PlacementContext) -> float:
    """저역은 왼쪽, 고역은 오른쪽. 중역과 미분류는 비용이 없다."""
    if note.band not in ("LOW", "HIGH"):
        return 0.0
    hand = hand_of(context.semantics[lane])
    if hand is None:
        return 0.0
    wanted = Hand.LEFT if note.band == "LOW" else Hand.RIGHT
    return 0.0 if hand is wanted else 1.0


def _both_sides_allowed(note: NoteEvent, context: PlacementContext) -> bool:
    """S4 정책. lane_rules 의 판정과 같은 답을 내야 한다.

    난이도를 보지 않으면 EXPERT 에서 합법인 사이드 화음까지 옮기고
    이동 예산을 깎아먹는다.
    """
    policy = BOTH_SIDES[context.difficulty]
    if policy is BothSidesPolicy.FREE:
        return True
    downbeat = context.row_is_downbeat or note.is_downbeat
    if policy is BothSidesPolicy.DOWNBEAT_ONLY:
        return downbeat
    if policy is BothSidesPolicy.ACCENT_ONLY:
        strong = note.onset_strength is not None and note.onset_strength >= ACCENT_STRENGTH
        return downbeat or context.row_is_accent or strong
    return False


def _rule_violation(note: NoteEvent, lane: int, context: PlacementContext) -> float:
    """S1 위반 여부. S4·C3 는 같은 시각 구성에서 나온다."""
    violations = 0.0
    last = context.last_time_by_lane.get(lane)
    if last is not None:
        limit = jack_interval_ms(context.semantics[lane], context.difficulty)
        if note.time_ms - last < limit:
            violations += 1.0

    semantic = context.semantics[lane]
    others = {context.semantics[other] for other in context.occupied_lanes}
    if finger_of(semantic) is FingerClass.SIDE:
        opposite = (
            LaneSemantic.SIDE_RIGHT
            if semantic is LaneSemantic.SIDE_LEFT
            else LaneSemantic.SIDE_LEFT
        )
        if opposite in others and not _both_sides_allowed(note, context):
            violations += 1.0
    if semantic is LaneSemantic.CENTER and {
        LaneSemantic.SIDE_LEFT,
        LaneSemantic.SIDE_RIGHT,
    } <= others:
        mains = sum(1 for kind in others if finger_of(kind) is FingerClass.MAIN)
        if mains > MAX_MAIN_WITH_CENTER_AND_BOTH_SIDES or context.difficulty != CENTER_WITH_BOTH_SIDES_FROM:
            violations += 1.0
    return violations


def _origin_drift(note: NoteEvent, lane: int, context: PlacementContext) -> float:
    if context.key_mode < 2:
        return 0.0
    return abs(lane - note.origin_lane) / (context.key_mode - 1)


def placement_cost(
    note: NoteEvent,
    lane: int,
    context: PlacementContext,
    weights: CostWeights = DEFAULT_WEIGHTS,
) -> CostBreakdown:
    """노트를 이 레인에 둘 때의 비용."""
    if not 0 <= lane < context.key_mode:
        raise ValueError(f"lane {lane} is outside {context.key_mode}K")
    if lane in context.occupied_lanes:
        # 같은 시각 같은 레인에 두 노트를 둘 수 없다.
        return CostBreakdown(INFEASIBLE, {"occupied": INFEASIBLE})

    terms = {
        "w1_same_lane_repeat": weights.same_lane_repeat * _same_lane_repeat(note, lane, context),
        "w2_hand_balance": weights.hand_balance * _hand_balance(lane, context),
        "w3_travel": weights.travel * _travel(lane, context),
        "w4_jack": weights.jack * _jack(lane, context),
        "w5_chord_spread": weights.chord_spread * _chord_spread(lane, context),
        "w6_pattern_repeat": weights.pattern_repeat * _pattern_repeat(lane, context),
        "w7_band_mismatch": weights.band_mismatch * _band_mismatch(note, lane, context),
        "w8_rule_violation": weights.rule_violation * _rule_violation(note, lane, context),
        "w9_origin_drift": weights.origin_drift * _origin_drift(note, lane, context),
    }
    return CostBreakdown(total=sum(terms.values()), terms=terms)


LanePreference = Literal["MAIN_ONLY", "ANY"]


def candidate_lanes(
    note: NoteEvent,
    context: PlacementContext,
    *,
    preference: LanePreference = "MAIN_ONLY",
) -> list[int]:
    """이동 후보. 같은 시각에 비어있는 레인만.

    기본은 MAIN 만 후보로 둔다(Phase3 §7 3단계 — 사이드는 후보 제외).
    사이드로 옮기면 새끼손가락 문제를 다른 새끼손가락으로 옮기는 것뿐이다.
    가까운 레인부터 본다.
    """
    semantics = context.semantics
    lanes = [
        lane
        for lane in range(context.key_mode)
        if lane not in context.occupied_lanes
        and (preference == "ANY" or finger_of(semantics[lane]) is FingerClass.MAIN)
    ]
    return sorted(lanes, key=lambda lane: (abs(lane - note.origin_lane), lane))


def best_lane(
    note: NoteEvent,
    context: PlacementContext,
    *,
    weights: CostWeights = DEFAULT_WEIGHTS,
    preference: LanePreference = "MAIN_ONLY",
) -> tuple[int, CostBreakdown] | None:
    """비용이 가장 낮은 후보. 후보가 없거나 전부 불가능하면 None."""
    best: tuple[int, CostBreakdown] | None = None
    for lane in candidate_lanes(note, context, preference=preference):
        breakdown = placement_cost(note, lane, context, weights)
        if math.isinf(breakdown.total):
            continue
        if best is None or breakdown.total < best[1].total:
            best = (lane, breakdown)
    return best
