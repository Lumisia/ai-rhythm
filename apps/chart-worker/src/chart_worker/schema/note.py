import numbers
from dataclasses import dataclass, field
from typing import Any, Literal


def coerce_int(value: Any, field_name: str) -> int:
    """정수형 값을 파이썬 int 로 정규화한다.

    numpy 정수(np.int32 · np.int64 등)를 받아들인다. 분석 단계가
    numpy 스칼라를 그대로 넘기더라도 노트 모델이 거부하지 않게 하기 위함이다.
    bool 과 실수는 정수로 보지 않는다.
    """
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        # 이 패키지는 입력 검증 실패를 ValueError 로 통일한다. TypeError 로 바꾸면
        # 호출부가 두 예외를 모두 처리해야 하므로 TRY004 를 의도적으로 무시한다.
        raise ValueError(f"{field_name} must be an integer")  # noqa: TRY004
    return int(value)


@dataclass(frozen=True, slots=True)
class NoteEvent:
    time_ms: int
    lane: int
    kind: Literal["TAP", "HOLD"] = "TAP"
    duration_ms: int | None = None
    onset_strength: float | None = None
    band: str | None = None
    is_downbeat: bool = False
    beat_fraction: float | None = None
    section: str | None = None
    origin_lane: int = field(default=-1)

    def __post_init__(self) -> None:
        object.__setattr__(self, "time_ms", coerce_int(self.time_ms, "time_ms"))
        object.__setattr__(self, "lane", coerce_int(self.lane, "lane"))
        object.__setattr__(self, "origin_lane", coerce_int(self.origin_lane, "origin_lane"))
        if self.duration_ms is not None:
            object.__setattr__(
                self, "duration_ms", coerce_int(self.duration_ms, "duration_ms")
            )
        if self.time_ms < 0:
            raise ValueError("time_ms must be non-negative")
        if self.lane < 0:
            raise ValueError("lane must be non-negative")
        if self.kind not in ("TAP", "HOLD"):
            raise ValueError(f"unsupported note kind: {self.kind}")
        if self.kind == "HOLD":
            if self.duration_ms is None or self.duration_ms <= 0:
                raise ValueError("duration_ms must be positive for a HOLD note")
        elif self.duration_ms is not None:
            raise ValueError("duration_ms must be None for a TAP note")
        if self.origin_lane == -1:
            object.__setattr__(self, "origin_lane", self.lane)
        elif self.origin_lane < 0:
            raise ValueError("origin_lane must be non-negative")


Chart = list[NoteEvent]
