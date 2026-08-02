"""Mapperatorinator 생성 파라미터.

Mapperatorinator 는 노트를 만드는 유일한 도구다. Beat This! · librosa ·
Demucs 는 절대 노트를 만들지 않는다.
"""

from dataclasses import dataclass, field
from pathlib import Path

from chart_worker.schema.note import coerce_int
from chart_worker.schema.types import DIFFICULTIES, KEY_MODES

GAMEMODE_MANIA = 3
PRECISION = "fp16"
"""v32.yaml 기본값은 bf16 이다. Turing(sm_75)은 bf16 을 지원하지 않는다."""

MANIA_COLUMN_TEMPERATURE = 0.8
"""레인 선택 샘플링 온도. 패턴 다양성에 직결된다."""

DEFAULT_YEAR = 2023
YEAR_RANGE = (2007, 2024)

REQUESTED_STAR: dict[str, float] = {
    "EASY": 2.0,
    "NORMAL": 3.0,
    "HARD": 4.0,
    "EXPERT": 5.0,
}
"""solver 목표(1.5 · 2.6 · 3.8 · 5.0)보다 높다.

solver 는 내리는 방향으로만 동작한다. 없는 시각에 노트를 만들면 타이밍
불변 위반이므로, 솎을 재료가 처음부터 남아 있어야 한다.
"""

HOLD_NOTE_RATIO: dict[str, float] = {
    "EASY": 0.10,
    "NORMAL": 0.15,
    "HARD": 0.20,
    "EXPERT": 0.25,
}

# style/clean 은 존재하지 않는 태그다. 아래는 전부 실존 mania 태그다.
DESCRIPTORS: dict[str, tuple[str, ...]] = {
    "EASY": ("expression/simple", "style/mixed rice"),
    "NORMAL": ("style/mixed rice", "style/jumpstream"),
    "HARD": ("style/jumpstream", "style/handstream", "style/LN coordination"),
    "EXPERT": ("style/handstream", "style/chordstream", "tech/technical hybrid"),
}

NEGATIVE_DESCRIPTORS: dict[str, tuple[str, ...]] = {
    "EASY": ("style/chordjack", "style/longjack"),
    "NORMAL": ("style/chordjack", "skillset/speedjack"),
    "HARD": ("style/dump", "style/chordjack", "skillset/wristjack"),
    "EXPERT": ("style/dump", "style/longjack", "style/o2jam"),
}


@dataclass(frozen=True, slots=True)
class GenerationRequest:
    audio_path: Path
    key_mode: int
    difficulty: str
    year: int = DEFAULT_YEAR
    seed: int | None = None
    timing_osu_path: Path | None = None
    """Beat This! 격자를 담은 참조 .osu. 있으면 in_context=[TIMING] 을 켠다."""

    cfg_scale: float = 1.0
    descriptors: tuple[str, ...] = ()
    negative_descriptors: tuple[str, ...] = ()
    requested_star: float = field(default=0.0)
    hold_note_ratio: float = field(default=0.0)
    duration_ms: int = 0
    """표준화 단계가 이미 재놨다. fake 생성기와 start/end 지정에 쓴다."""

    def __post_init__(self) -> None:
        object.__setattr__(self, "key_mode", coerce_int(self.key_mode, "key_mode"))
        object.__setattr__(self, "year", coerce_int(self.year, "year"))
        object.__setattr__(self, "duration_ms", coerce_int(self.duration_ms, "duration_ms"))
        if self.key_mode not in KEY_MODES:
            raise ValueError(f"unsupported key_mode: {self.key_mode}")
        if self.difficulty not in DIFFICULTIES:
            raise ValueError(f"unsupported difficulty: {self.difficulty}")
        if not YEAR_RANGE[0] <= self.year <= YEAR_RANGE[1]:
            raise ValueError(f"year must be within {YEAR_RANGE}, got {self.year}")

        if not self.descriptors:
            object.__setattr__(self, "descriptors", DESCRIPTORS[self.difficulty])
        if not self.negative_descriptors:
            object.__setattr__(self, "negative_descriptors", NEGATIVE_DESCRIPTORS[self.difficulty])
        if self.requested_star <= 0:
            object.__setattr__(self, "requested_star", REQUESTED_STAR[self.difficulty])
        if self.hold_note_ratio <= 0:
            object.__setattr__(self, "hold_note_ratio", HOLD_NOTE_RATIO[self.difficulty])

        # cfg_scale > 1 이면 classifier-free guidance 가 두 목록을 짝지어 쓴다.
        # 개수가 다르면 조용히 어긋난 조건으로 생성된다.
        if self.cfg_scale > 1.0 and len(self.descriptors) != len(self.negative_descriptors):
            raise ValueError(
                "cfg_scale > 1 requires the same number of descriptors and "
                f"negative_descriptors ({len(self.descriptors)} vs "
                f"{len(self.negative_descriptors)})"
            )
