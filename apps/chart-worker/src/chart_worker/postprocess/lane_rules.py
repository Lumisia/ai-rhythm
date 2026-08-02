"""레인 규칙 — 밀도가 아니라 손가락 기준.

"사이드는 낮은 밀도"라는 규칙은 폐기했다. 밀도로 구분하면 사이드가
장식이 된다. DJMAX 에서 사이드는 저밀도 장식이 아니라 패턴 계열 하나의
축이다(Wrist Turns · Hamburgers · HEXAD Trills · FX Chords · ST/FX).

물리적으로 정확한 제약은 손가락별 연타 속도 상한이다.
"""

from dataclasses import dataclass
from enum import StrEnum
from itertools import pairwise

from chart_worker.postprocess.patterns import rows_of
from chart_worker.schema.note import Chart
from chart_worker.schema.types import DIFFICULTIES, LaneSemantic, lane_semantics


class FingerClass(StrEnum):
    SIDE = "SIDE"
    """새끼손가락. 빠른 잭이 약하다."""

    CENTER = "CENTER"
    """엄지. 중간."""

    MAIN = "MAIN"
    """검지·중지. 빠른 잭이 강하다."""


class Hand(StrEnum):
    LEFT = "LEFT"
    RIGHT = "RIGHT"


class Rule(StrEnum):
    S1_JACK_INTERVAL = "S1_JACK_INTERVAL"
    S3_SIDE_HOLD_SAME_HAND = "S3_SIDE_HOLD_SAME_HAND"
    S4_BOTH_SIDES = "S4_BOTH_SIDES"
    C3_CENTER_WITH_BOTH_SIDES = "C3_CENTER_WITH_BOTH_SIDES"


_FINGER_OF: dict[LaneSemantic, FingerClass] = {
    LaneSemantic.SIDE_LEFT: FingerClass.SIDE,
    LaneSemantic.SIDE_RIGHT: FingerClass.SIDE,
    LaneSemantic.CENTER: FingerClass.CENTER,
    LaneSemantic.MAIN_1: FingerClass.MAIN,
    LaneSemantic.MAIN_2: FingerClass.MAIN,
    LaneSemantic.MAIN_3: FingerClass.MAIN,
    LaneSemantic.MAIN_4: FingerClass.MAIN,
}

_HAND_OF: dict[LaneSemantic, Hand] = {
    LaneSemantic.SIDE_LEFT: Hand.LEFT,
    LaneSemantic.MAIN_1: Hand.LEFT,
    LaneSemantic.MAIN_2: Hand.LEFT,
    LaneSemantic.MAIN_3: Hand.RIGHT,
    LaneSemantic.MAIN_4: Hand.RIGHT,
    LaneSemantic.SIDE_RIGHT: Hand.RIGHT,
}
"""CENTER 는 엄지라 어느 손에도 고정되지 않으므로 빠진다."""

JACK_MIN_INTERVAL_MS: dict[str, dict[FingerClass, int]] = {
    "EASY": {FingerClass.SIDE: 500, FingerClass.CENTER: 400, FingerClass.MAIN: 250},
    "NORMAL": {FingerClass.SIDE: 350, FingerClass.CENTER: 280, FingerClass.MAIN: 170},
    "HARD": {FingerClass.SIDE: 250, FingerClass.CENTER: 200, FingerClass.MAIN: 120},
    "EXPERT": {FingerClass.SIDE: 180, FingerClass.CENTER: 140, FingerClass.MAIN: 85},
}
"""S1. BPM 150 기준으로 EASY 사이드는 4분음표 잭도 못 하고,
EXPERT 메인은 16분 이상이 자유롭다."""

SAME_HAND_NPS_DURING_SIDE_HOLD: dict[str, float] = {
    "EASY": 0.0,
    "NORMAL": 2.0,
    "HARD": 4.0,
    "EXPERT": 7.0,
}
"""S3. 사이드 HOLD 를 잡은 손이 동시에 처리할 수 있는 노트 밀도."""


class BothSidesPolicy(StrEnum):
    FORBIDDEN = "FORBIDDEN"
    DOWNBEAT_ONLY = "DOWNBEAT_ONLY"
    ACCENT_ONLY = "ACCENT_ONLY"
    FREE = "FREE"


BOTH_SIDES: dict[str, BothSidesPolicy] = {
    "EASY": BothSidesPolicy.FORBIDDEN,
    "NORMAL": BothSidesPolicy.DOWNBEAT_ONLY,
    "HARD": BothSidesPolicy.ACCENT_ONLY,
    "EXPERT": BothSidesPolicy.FREE,
}
"""S4."""

ACCENT_STRENGTH = 0.6
"""onset 강도가 이보다 크면 악센트로 본다. 정규화된 값이라 곡에 무관하다."""

CENTER_WITH_BOTH_SIDES_FROM = "EXPERT"
"""C3. Space + 양쪽 사이드 3동시는 EXPERT 에서만 허용한다."""

MAX_MAIN_WITH_CENTER_AND_BOTH_SIDES = 1
"""C3. 3동시에 일반키가 여럿 더 붙으면 어느 난이도에서도 못 친다."""


@dataclass(frozen=True, slots=True)
class Violation:
    rule: Rule
    time_ms: int
    lanes: tuple[int, ...]
    detail: str


def finger_of(semantic: LaneSemantic) -> FingerClass:
    return _FINGER_OF[semantic]


def hand_of(semantic: LaneSemantic) -> Hand | None:
    """CENTER 는 엄지라 어느 손에도 고정되지 않는다."""
    return _HAND_OF.get(semantic)


def jack_interval_ms(semantic: LaneSemantic, difficulty: str) -> int:
    return JACK_MIN_INTERVAL_MS[difficulty][finger_of(semantic)]


def _require_difficulty(difficulty: str) -> None:
    if difficulty not in DIFFICULTIES:
        raise ValueError(f"unsupported difficulty: {difficulty}")


def check_jack_intervals(
    notes: Chart, *, key_mode: int, difficulty: str
) -> list[Violation]:
    """S1 — 같은 레인 연타가 손가락 한계보다 빠른지 본다.

    잭에만 적용된다. 다른 레인 노트와는 무관하므로 밀도 제한이 아니다.
    간격은 시작 시각끼리 잰다 — 잭의 정의가 그렇고, 표의 ms 값도
    음표 단위(EXPERT MAIN 85ms = 176 BPM 의 16분)에서 나왔다.
    """
    _require_difficulty(difficulty)
    semantics = lane_semantics(key_mode)
    by_lane: dict[int, list[int]] = {}
    for note in sorted(notes, key=lambda n: n.time_ms):
        by_lane.setdefault(note.lane, []).append(note.time_ms)

    found = []
    for lane, times in by_lane.items():
        limit = jack_interval_ms(semantics[lane], difficulty)
        for previous, current in pairwise(times):
            gap = current - previous
            if gap < limit:
                found.append(
                    Violation(
                        Rule.S1_JACK_INTERVAL,
                        current,
                        (lane,),
                        f"{gap}ms gap on a {finger_of(semantics[lane]).value} lane "
                        f"needs at least {limit}ms at {difficulty}",
                    )
                )
    return found


def check_side_hold_density(
    notes: Chart, *, key_mode: int, difficulty: str
) -> list[Violation]:
    """S3 — 사이드 HOLD 를 잡은 손이 같은 시간에 얼마나 치는지 본다."""
    _require_difficulty(difficulty)
    semantics = lane_semantics(key_mode)
    limit = SAME_HAND_NPS_DURING_SIDE_HOLD[difficulty]

    found = []
    for hold in notes:
        if hold.kind != "HOLD" or finger_of(semantics[hold.lane]) is not FingerClass.SIDE:
            continue
        hand = hand_of(semantics[hold.lane])
        end_ms = hold.time_ms + (hold.duration_ms or 0)
        span_sec = max((end_ms - hold.time_ms) / 1000.0, 1e-9)
        same_hand = [
            note
            for note in notes
            if note is not hold
            and hold.time_ms <= note.time_ms <= end_ms
            and hand_of(semantics[note.lane]) is hand
        ]
        nps = len(same_hand) / span_sec
        if same_hand and nps > limit:
            found.append(
                Violation(
                    Rule.S3_SIDE_HOLD_SAME_HAND,
                    hold.time_ms,
                    (hold.lane, *sorted({note.lane for note in same_hand})),
                    f"{nps:.2f} NPS on the {hand.value if hand else '?'} hand during a "
                    f"side hold exceeds {limit} at {difficulty}",
                )
            )
    return found


def check_both_sides(notes: Chart, *, key_mode: int, difficulty: str) -> list[Violation]:
    """S4 — 양쪽 사이드를 동시에 누르는 순간을 본다."""
    _require_difficulty(difficulty)
    semantics = lane_semantics(key_mode)
    policy = BOTH_SIDES[difficulty]
    if policy is BothSidesPolicy.FREE:
        return []

    found = []
    for row in rows_of(notes):
        side_lanes = {
            semantics[note.lane]
            for note in row.notes
            if semantics[note.lane]
            in (LaneSemantic.SIDE_LEFT, LaneSemantic.SIDE_RIGHT)
        }
        if len(side_lanes) < 2:
            continue
        if policy is BothSidesPolicy.DOWNBEAT_ONLY and _is_downbeat(row):
            continue
        if policy is BothSidesPolicy.ACCENT_ONLY and _is_accent(row):
            continue
        found.append(
            Violation(
                Rule.S4_BOTH_SIDES,
                row.time_ms,
                row.lanes,
                f"both sides at once is {policy.value.lower()} at {difficulty}",
            )
        )
    return found


def _is_downbeat(row) -> bool:
    return any(note.is_downbeat for note in row.notes)


def _is_accent(row) -> bool:
    if _is_downbeat(row):
        return True
    return any(
        note.onset_strength is not None and note.onset_strength >= ACCENT_STRENGTH
        for note in row.notes
    )


def check_center_combinations(
    notes: Chart, *, key_mode: int, difficulty: str
) -> list[Violation]:
    """C3 — Space 와 양쪽 사이드가 겹치는 순간을 본다."""
    _require_difficulty(difficulty)
    semantics = lane_semantics(key_mode)
    if LaneSemantic.CENTER not in semantics:
        return []
    allowed = difficulty == CENTER_WITH_BOTH_SIDES_FROM

    found = []
    for row in rows_of(notes):
        kinds = [semantics[note.lane] for note in row.notes]
        if LaneSemantic.CENTER not in kinds:
            continue
        if not {LaneSemantic.SIDE_LEFT, LaneSemantic.SIDE_RIGHT} <= set(kinds):
            continue
        mains = sum(1 for kind in kinds if finger_of(kind) is FingerClass.MAIN)
        if mains > MAX_MAIN_WITH_CENTER_AND_BOTH_SIDES:
            # 엄지와 양쪽 새끼가 이미 묶인 위에 일반키가 여럿 더 붙는다.
            found.append(
                Violation(
                    Rule.C3_CENTER_WITH_BOTH_SIDES,
                    row.time_ms,
                    row.lanes,
                    f"center + both sides + {mains} main notes is not playable",
                )
            )
        elif not allowed:
            found.append(
                Violation(
                    Rule.C3_CENTER_WITH_BOTH_SIDES,
                    row.time_ms,
                    row.lanes,
                    f"center with both sides is only allowed at "
                    f"{CENTER_WITH_BOTH_SIDES_FROM}, not {difficulty}",
                )
            )
    return found


def check_lane_rules(notes: Chart, *, key_mode: int, difficulty: str) -> list[Violation]:
    """검사 가능한 레인 규칙을 전부 본다.

    S2(사이드 롱노트 우선)와 C1(Space 다운비트 우선)은 **배치 유도**라
    여기서 검사하지 않는다. 비용함수의 보상 항으로 들어간다.
    """
    found = [
        *check_jack_intervals(notes, key_mode=key_mode, difficulty=difficulty),
        *check_side_hold_density(notes, key_mode=key_mode, difficulty=difficulty),
        *check_both_sides(notes, key_mode=key_mode, difficulty=difficulty),
        *check_center_combinations(notes, key_mode=key_mode, difficulty=difficulty),
    ]
    return sorted(found, key=lambda violation: (violation.time_ms, violation.rule))
