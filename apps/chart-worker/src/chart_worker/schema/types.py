from enum import StrEnum
from typing import Literal

KeyMode = Literal[4, 6, 7]
Difficulty = Literal["EASY", "NORMAL", "HARD", "EXPERT"]

KEY_MODES: tuple[int, ...] = (4, 6, 7)
DIFFICULTIES: tuple[str, ...] = ("EASY", "NORMAL", "HARD", "EXPERT")


class LaneSemantic(StrEnum):
    MAIN_1 = "MAIN_1"
    MAIN_2 = "MAIN_2"
    MAIN_3 = "MAIN_3"
    MAIN_4 = "MAIN_4"
    CENTER = "CENTER"
    SIDE_LEFT = "SIDE_LEFT"
    SIDE_RIGHT = "SIDE_RIGHT"


_LAYOUTS: dict[int, list[LaneSemantic]] = {
    4: [
        LaneSemantic.MAIN_1,
        LaneSemantic.MAIN_2,
        LaneSemantic.MAIN_3,
        LaneSemantic.MAIN_4,
    ],
    6: [
        LaneSemantic.SIDE_LEFT,
        LaneSemantic.MAIN_1,
        LaneSemantic.MAIN_2,
        LaneSemantic.MAIN_3,
        LaneSemantic.MAIN_4,
        LaneSemantic.SIDE_RIGHT,
    ],
    7: [
        LaneSemantic.SIDE_LEFT,
        LaneSemantic.MAIN_1,
        LaneSemantic.MAIN_2,
        LaneSemantic.CENTER,
        LaneSemantic.MAIN_3,
        LaneSemantic.MAIN_4,
        LaneSemantic.SIDE_RIGHT,
    ],
}


def lane_semantics(key_mode: int) -> list[LaneSemantic]:
    try:
        return list(_LAYOUTS[key_mode])
    except KeyError:
        raise ValueError(f"unsupported key_mode: {key_mode}") from None
