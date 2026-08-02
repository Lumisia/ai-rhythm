"""난이도별 구조 패턴 허용과 8마디 쿼터."""

from enum import StrEnum

from chart_worker.postprocess.patterns import PatternInstance, PatternKind
from chart_worker.schema.types import DIFFICULTIES

BARS_PER_SEGMENT = 8
BEATS_PER_BAR = 4


class Allow(StrEnum):
    FULL = "FULL"
    LIMITED = "LIMITED"
    FORBIDDEN = "FORBIDDEN"


_FULL: dict[str, frozenset[PatternKind]] = {
    "EASY": frozenset({PatternKind.STREAM}),
    "NORMAL": frozenset(
        {
            PatternKind.STREAM,
            PatternKind.STAIRS,
            PatternKind.ROLL,
            PatternKind.JUMP,
            PatternKind.HOLD,
        }
    ),
    "HARD": frozenset(
        {
            PatternKind.STREAM,
            PatternKind.STAIRS,
            PatternKind.ROLL,
            PatternKind.JUMP,
            PatternKind.TRILL_TWO_HANDED,
            PatternKind.MINIJACK,
            PatternKind.HOLD,
            PatternKind.SHIELD,
            PatternKind.REVERSE_SHIELD,
            PatternKind.HAND,
            PatternKind.JUMPSTREAM,
            PatternKind.ANCHOR,
        }
    ),
    "EXPERT": frozenset(
        {
            PatternKind.STREAM,
            PatternKind.STAIRS,
            PatternKind.ROLL,
            PatternKind.JUMP,
            PatternKind.TRILL_TWO_HANDED,
            PatternKind.MINIJACK,
            PatternKind.HOLD,
            PatternKind.SHIELD,
            PatternKind.REVERSE_SHIELD,
            PatternKind.HAND,
            PatternKind.JUMPSTREAM,
            PatternKind.ANCHOR,
            PatternKind.TRILL_ONE_HANDED,
            PatternKind.BURST,
            PatternKind.HANDSTREAM,
            PatternKind.CHORDSTREAM,
            PatternKind.JACK,
        }
    ),
}

_LIMITED: dict[str, frozenset[PatternKind]] = {
    "EASY": frozenset({PatternKind.STAIRS, PatternKind.JUMP, PatternKind.HOLD}),
    "NORMAL": frozenset(
        {
            PatternKind.TRILL_TWO_HANDED,
            PatternKind.MINIJACK,
            PatternKind.SHIELD,
            PatternKind.REVERSE_SHIELD,
            PatternKind.HAND,
            PatternKind.JUMPSTREAM,
            PatternKind.ANCHOR,
        }
    ),
    "HARD": frozenset({PatternKind.TRILL_ONE_HANDED, PatternKind.BURST}),
    "EXPERT": frozenset(
        {
            PatternKind.LONGJACK,
            PatternKind.QUAD,
            PatternKind.JUMPTRILL,
            PatternKind.CHORDJACK,
            PatternKind.DENIM,
            PatternKind.GRACE,
            PatternKind.INVERSE,
        }
    ),
}

_FORBIDDEN: dict[str, frozenset[PatternKind]] = {
    "EASY": frozenset(
        {
            PatternKind.HAND,
            PatternKind.QUAD,
            PatternKind.GRACE,
            PatternKind.ROLL,
            PatternKind.BURST,
            PatternKind.JUMPSTREAM,
            PatternKind.HANDSTREAM,
            PatternKind.CHORDSTREAM,
            PatternKind.MINIJACK,
            PatternKind.JACK,
            PatternKind.LONGJACK,
            PatternKind.ANCHOR,
            PatternKind.CHORDJACK,
            PatternKind.TRILL_ONE_HANDED,
            PatternKind.TRILL_TWO_HANDED,
            PatternKind.JUMPTRILL,
            PatternKind.DENIM,
            PatternKind.SHIELD,
            PatternKind.REVERSE_SHIELD,
            PatternKind.INVERSE,
        }
    ),
    "NORMAL": frozenset(
        {
            PatternKind.QUAD,
            PatternKind.GRACE,
            PatternKind.BURST,
            PatternKind.HANDSTREAM,
            PatternKind.CHORDSTREAM,
            PatternKind.JACK,
            PatternKind.LONGJACK,
            PatternKind.CHORDJACK,
            PatternKind.TRILL_ONE_HANDED,
            PatternKind.JUMPTRILL,
            PatternKind.DENIM,
            PatternKind.INVERSE,
        }
    ),
    "HARD": frozenset(
        {
            PatternKind.QUAD,
            PatternKind.GRACE,
            PatternKind.HANDSTREAM,
            PatternKind.CHORDSTREAM,
            PatternKind.JACK,
            PatternKind.LONGJACK,
            PatternKind.CHORDJACK,
            PatternKind.JUMPTRILL,
            PatternKind.DENIM,
            PatternKind.INVERSE,
        }
    ),
    "EXPERT": frozenset(),
}

_QUOTAS: dict[tuple[str, PatternKind], int] = {
    ("EASY", PatternKind.STAIRS): 2,
    ("EASY", PatternKind.JUMP): 4,
    ("EASY", PatternKind.HOLD): 4,
    ("NORMAL", PatternKind.TRILL_TWO_HANDED): 2,
    ("NORMAL", PatternKind.MINIJACK): 2,
    ("NORMAL", PatternKind.SHIELD): 2,
    ("NORMAL", PatternKind.HAND): 3,
    ("NORMAL", PatternKind.JUMPSTREAM): 1,
    ("NORMAL", PatternKind.ANCHOR): 2,
    ("HARD", PatternKind.TRILL_ONE_HANDED): 2,
    ("HARD", PatternKind.BURST): 2,
    ("EXPERT", PatternKind.LONGJACK): 2,
    ("EXPERT", PatternKind.QUAD): 2,
    ("EXPERT", PatternKind.JUMPTRILL): 2,
    ("EXPERT", PatternKind.CHORDJACK): 2,
    ("EXPERT", PatternKind.DENIM): 2,
    ("EXPERT", PatternKind.GRACE): 4,
    ("EXPERT", PatternKind.INVERSE): 2,
}


def _check_policy() -> None:
    expected = set(PatternKind)
    for difficulty in DIFFICULTIES:
        groups = (_FULL[difficulty], _LIMITED[difficulty], _FORBIDDEN[difficulty])
        if set().union(*groups) != expected or any(
            first & second for index, first in enumerate(groups) for second in groups[index + 1 :]
        ):
            raise RuntimeError(f"incomplete pattern policy for {difficulty}")


_check_policy()


def allowance_of(kind: PatternKind, difficulty: str) -> Allow:
    if difficulty not in DIFFICULTIES:
        raise ValueError(f"unsupported difficulty: {difficulty}")
    if kind in _FULL[difficulty]:
        return Allow.FULL
    if kind in _LIMITED[difficulty]:
        return Allow.LIMITED
    if kind in _FORBIDDEN[difficulty]:
        return Allow.FORBIDDEN
    raise RuntimeError(f"pattern policy is missing {difficulty}/{kind}")


def _quota_kind(kind: PatternKind) -> PatternKind:
    return PatternKind.SHIELD if kind is PatternKind.REVERSE_SHIELD else kind


def quota_excesses(
    instances: list[PatternInstance],
    *,
    difficulty: str,
    beat_ms: float,
) -> tuple[PatternInstance, ...]:
    if difficulty not in DIFFICULTIES:
        raise ValueError(f"unsupported difficulty: {difficulty}")
    if beat_ms <= 0:
        raise ValueError("beat_ms must be positive")

    segment_ms = beat_ms * BEATS_PER_BAR * BARS_PER_SEGMENT
    counts: dict[tuple[int, PatternKind], int] = {}
    excesses = []
    for instance in sorted(instances, key=lambda item: (item.start_ms, item.kind)):
        if allowance_of(instance.kind, difficulty) is not Allow.LIMITED:
            continue
        kind = _quota_kind(instance.kind)
        key = (int(instance.start_ms // segment_ms), kind)
        counts[key] = counts.get(key, 0) + 1
        quota = _QUOTAS[(difficulty, kind)]
        if counts[key] > quota:
            excesses.append(instance)
    return tuple(excesses)
