"""차트의 시각 레인 의미와 실제 입력 손가락 역할을 연결한다."""

from enum import StrEnum

from chart_worker.schema.types import LaneSemantic


class ErgonomicRole(StrEnum):
    """실제 홈 포지션 입력에서 레인이 담당하는 손가락 역할."""

    MAIN = "MAIN"
    CENTER = "CENTER"


_ERGONOMIC_ROLES: dict[int, tuple[ErgonomicRole, ...]] = {
    4: (ErgonomicRole.MAIN,) * 4,
    6: (ErgonomicRole.MAIN,) * 6,
    7: (
        ErgonomicRole.MAIN,
        ErgonomicRole.MAIN,
        ErgonomicRole.MAIN,
        ErgonomicRole.CENTER,
        ErgonomicRole.MAIN,
        ErgonomicRole.MAIN,
        ErgonomicRole.MAIN,
    ),
}


class Hand(StrEnum):
    LEFT = "LEFT"
    RIGHT = "RIGHT"


_HAND_OF: dict[LaneSemantic, Hand] = {
    LaneSemantic.SIDE_LEFT: Hand.LEFT,
    LaneSemantic.MAIN_1: Hand.LEFT,
    LaneSemantic.MAIN_2: Hand.LEFT,
    LaneSemantic.MAIN_3: Hand.RIGHT,
    LaneSemantic.MAIN_4: Hand.RIGHT,
    LaneSemantic.SIDE_RIGHT: Hand.RIGHT,
}
"""CENTER는 엄지라 어느 손에도 고정되지 않으므로 빠진다."""


def ergonomic_roles(key_mode: int) -> tuple[ErgonomicRole, ...]:
    try:
        return _ERGONOMIC_ROLES[key_mode]
    except KeyError:
        raise ValueError(f"unsupported key_mode: {key_mode}") from None


def hand_of(semantic: LaneSemantic) -> Hand | None:
    """CENTER는 엄지라 어느 손에도 고정되지 않는다."""
    return _HAND_OF.get(semantic)
