"""Mapperatorinator 직접 생성 파라미터."""

from dataclasses import dataclass, field
from pathlib import Path

from chart_worker.schema.note import coerce_int
from chart_worker.schema.types import DIFFICULTIES, KEY_MODES

GAMEMODE_MANIA = 3
PRECISION = "fp16"
"""v32.yaml 기본값은 bf16 이다. Turing(sm_75)은 bf16 을 지원하지 않는다."""

DEFAULT_YEAR = 2023
YEAR_RANGE = (2007, 2024)

REQUESTED_STAR: dict[str, float] = {
    "EASY": 1.0,
    "NORMAL": 1.5,
    "HARD": 2.0,
    "EXPERT": 3.0,
}
"""기존 시험곡의 과생성을 반영한 직접 요청값.

생성 뒤 난이도 solver로 노트를 삭제하지 않으므로 요청 단계에서만 밀도를
조절한다. 결과 난이도는 보고서와 수동 플레이로 확인한다.
"""

# style/clean 은 존재하지 않는 태그다. 아래는 전부 실존 mania 태그다.
DESCRIPTORS: dict[str, tuple[str, ...]] = {
    "EASY": ("expression/simple",),
    "NORMAL": ("style/mixed rice",),
    "HARD": ("style/mixed rice", "streams/bursts"),
    "EXPERT": ("style/mixed rice", "skillset/streams", "skillset/tech"),
}


@dataclass(frozen=True, slots=True)
class TimingGenerationRequest:
    audio_path: Path
    duration_ms: int
    year: int = DEFAULT_YEAR
    seed: int | None = None
    super_timing: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "year", coerce_int(self.year, "year"))
        object.__setattr__(self, "duration_ms", coerce_int(self.duration_ms, "duration_ms"))
        if self.duration_ms <= 0:
            raise ValueError("duration_ms must be positive")
        if not YEAR_RANGE[0] <= self.year <= YEAR_RANGE[1]:
            raise ValueError(f"year must be within {YEAR_RANGE}, got {self.year}")


@dataclass(frozen=True, slots=True)
class GenerationRequest:
    audio_path: Path
    key_mode: int
    difficulty: str
    timing_reference_path: Path = Path("timing-reference.osu")
    year: int = DEFAULT_YEAR
    seed: int | None = None
    cfg_scale: float = 1.0
    descriptors: tuple[str, ...] = ()
    requested_star: float = field(default=0.0)
    duration_ms: int = 0
    partial_start_ms: int | None = None
    partial_end_ms: int | None = None
    add_to_beatmap: bool = False
    """표준화 단계가 이미 재놨다. fake 생성기와 start/end 지정에 쓴다."""

    def __post_init__(self) -> None:
        object.__setattr__(self, "key_mode", coerce_int(self.key_mode, "key_mode"))
        object.__setattr__(self, "year", coerce_int(self.year, "year"))
        object.__setattr__(self, "duration_ms", coerce_int(self.duration_ms, "duration_ms"))
        if self.partial_start_ms is not None:
            object.__setattr__(
                self,
                "partial_start_ms",
                coerce_int(self.partial_start_ms, "partial_start_ms"),
            )
        if self.partial_end_ms is not None:
            object.__setattr__(
                self,
                "partial_end_ms",
                coerce_int(self.partial_end_ms, "partial_end_ms"),
            )
        if self.key_mode not in KEY_MODES:
            raise ValueError(f"unsupported key_mode: {self.key_mode}")
        if self.difficulty not in DIFFICULTIES:
            raise ValueError(f"unsupported difficulty: {self.difficulty}")
        if self.duration_ms <= 0:
            raise ValueError("duration_ms must be positive")
        partial_values = (self.partial_start_ms, self.partial_end_ms)
        if (partial_values[0] is None) != (partial_values[1] is None):
            raise ValueError("partial start and end must be provided together")
        if partial_values[0] is not None:
            start_ms = partial_values[0]
            end_ms = partial_values[1]
            assert end_ms is not None
            if not 0 <= start_ms < end_ms <= self.duration_ms:
                raise ValueError("partial range must be within canonical audio duration")
            if not self.add_to_beatmap:
                raise ValueError("partial generation requires add_to_beatmap")
        elif self.add_to_beatmap:
            raise ValueError("add_to_beatmap requires a partial range")
        if not YEAR_RANGE[0] <= self.year <= YEAR_RANGE[1]:
            raise ValueError(f"year must be within {YEAR_RANGE}, got {self.year}")

        if not self.descriptors:
            object.__setattr__(self, "descriptors", DESCRIPTORS[self.difficulty])
        if self.cfg_scale != 1.0:
            raise ValueError("direct generation requires cfg_scale=1.0")
        if self.requested_star <= 0:
            object.__setattr__(self, "requested_star", REQUESTED_STAR[self.difficulty])
