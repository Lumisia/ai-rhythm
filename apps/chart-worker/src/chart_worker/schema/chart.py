"""chart-v1 직렬화 모델.

후처리 결과를 프론트가 그대로 읽는 JSON 문서로 굳힌다.
파이썬 필드는 snake_case, JSON 키는 camelCase 다.
"""

from typing import Annotated, Any, Literal, Self
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator
from pydantic.alias_generators import to_camel

from chart_worker.schema.note import Chart
from chart_worker.schema.types import Difficulty, LaneSemantic, lane_semantics

SCHEMA_VERSION = 1

Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
BpmSource = Literal["BEAT_THIS", "MAPPERATORINATOR", "MANUAL"]


class _CamelModel(BaseModel):
    """JSON 은 camelCase, 파이썬은 snake_case 를 쓰는 공통 설정.

    extra="forbid" 는 오타 난 키를 조용히 버리지 않게 한다.
    frozen=True 는 문서가 만들어진 뒤 바뀌지 않게 한다.
    """

    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True,
        extra="forbid",
        frozen=True,
    )


class BpmEvent(_CamelModel):
    time_ms: int = Field(ge=0)
    bpm: float = Field(gt=0)


class ChartNote(_CamelModel):
    """직렬화된 노트.

    JSON 키는 `type` 이지만 파이썬 이름은 `NoteEvent.kind` 와 맞춘다.
    필드명을 `type` 으로 두면 내장 이름을 가린다.
    """

    id: int = Field(ge=1)
    lane: int = Field(ge=0)
    time_ms: int = Field(ge=0)
    kind: Literal["TAP", "HOLD"] = Field(alias="type")
    duration_ms: int | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def _check_duration(self) -> Self:
        if self.kind == "HOLD" and self.duration_ms is None:
            raise ValueError("durationMs is required for a HOLD note")
        if self.kind == "TAP" and self.duration_ms is not None:
            raise ValueError("durationMs must be absent for a TAP note")
        return self

    @property
    def end_ms(self) -> int:
        return self.time_ms + (self.duration_ms or 0)


class ChartMetrics(_CamelModel):
    """곡 선택 화면과 분석이 함께 쓰는 측정값.

    후처리 파생 필드에 기본값을 주지 않는다. 0.0 을 기본값으로 두면
    "아직 계산하지 않음"과 "계산했더니 0"을 구분할 수 없다.
    """

    note_count: int = Field(ge=0)
    hold_count: int = Field(ge=0)
    avg_nps: float = Field(ge=0)
    p95_nps: float = Field(ge=0)
    peak_nps: float = Field(ge=0)
    chord_ratio: float = Field(ge=0, le=1)
    max_jack: int = Field(ge=0)
    project_rating: float = Field(ge=0)
    project_tier: Difficulty
    pattern_entropy: float = Field(ge=0)
    drum_coverage: float = Field(ge=0, le=1)
    drum_precision: float = Field(ge=0, le=1)
    mean_abs_err_ms: float = Field(ge=0)
    side_note_ratio: float = Field(ge=0, le=1)
    side_hold_ratio: float = Field(ge=0, le=1)
    moved_note_ratio: float = Field(ge=0, le=1)


class GeneratorInfo(_CamelModel):
    name: str = Field(min_length=1)
    version: str = Field(min_length=1)
    analysis_version: str = Field(min_length=1)
    postprocess_version: str = Field(min_length=1)
    seed: int


def notes_to_chart_notes(notes: Chart) -> list[ChartNote]:
    """중간표현 노트를 직렬화 노트로 바꾼다. id 는 정렬 후 1부터 매긴다."""
    ordered = sorted(notes, key=lambda note: (note.time_ms, note.lane))
    return [
        ChartNote(
            id=index,
            lane=note.lane,
            time_ms=note.time_ms,
            kind=note.kind,
            duration_ms=note.duration_ms,
        )
        for index, note in enumerate(ordered, start=1)
    ]


class ChartDocument(_CamelModel):
    schema_version: Literal[1] = SCHEMA_VERSION
    chart_id: UUID
    song_version_id: UUID
    game_audio_asset_id: UUID
    audio_sha256: Sha256

    key_mode: Literal[4, 6, 7]
    difficulty: Difficulty
    lane_semantics: list[LaneSemantic]

    offset_ms: int
    duration_ms: int = Field(gt=0)
    bpm_events: list[BpmEvent] = Field(min_length=1)
    bpm_source: BpmSource

    notes: list[ChartNote]
    auto_play_onsets: list[int]
    metrics: ChartMetrics
    generator: GeneratorInfo

    @model_validator(mode="after")
    def _check_document(self) -> Self:
        self._check_lane_semantics()
        self._check_notes()
        self._check_bpm_events()
        self._check_auto_play_onsets()
        self._check_metrics()
        return self

    def _check_lane_semantics(self) -> None:
        # 프론트는 laneSemantics 만 보고 키 바인딩을 정한다. 배열이 틀리면
        # keyMode 가 맞아도 조작키 자체가 틀린다.
        expected = lane_semantics(self.key_mode)
        if self.lane_semantics != expected:
            names = [semantic.value for semantic in expected]
            raise ValueError(f"laneSemantics must be {names} for {self.key_mode}K")

    def _check_notes(self) -> None:
        seen_ids: set[int] = set()
        previous_key: tuple[int, int] | None = None
        hold_end_by_lane: dict[int, int] = {}
        for note in self.notes:
            if note.lane >= self.key_mode:
                raise ValueError(f"note lane {note.lane} is outside {self.key_mode}K")
            if note.id in seen_ids:
                raise ValueError(f"duplicate note id {note.id}")
            seen_ids.add(note.id)
            key = (note.time_ms, note.lane)
            if previous_key is not None and key < previous_key:
                raise ValueError("notes must be sorted by (timeMs, lane)")
            previous_key = key
            # measure_rating 과 같은 경계를 쓴다. 두 층이 어긋나면
            # 스키마를 통과한 채보가 측정 단계에서 터진다.
            if note.time_ms >= self.duration_ms:
                raise ValueError("note timeMs must be less than durationMs")
            if note.end_ms > self.duration_ms:
                raise ValueError("HOLD end time must not exceed durationMs")
            # 진행 중인 롱노트 위에는 무엇도 놓을 수 없다. HOLD 끼리만
            # 검사하면 그 안으로 들어간 TAP 이 검증을 통과해 칠 수 없는
            # 채보가 배포된다.
            if note.time_ms < hold_end_by_lane.get(note.lane, -1):
                raise ValueError(f"note overlaps a HOLD in lane {note.lane}")
            if note.kind == "HOLD":
                hold_end_by_lane[note.lane] = note.end_ms

    def _check_bpm_events(self) -> None:
        if self.bpm_events[0].time_ms != 0:
            raise ValueError("bpmEvents must start at timeMs 0")
        times = [event.time_ms for event in self.bpm_events]
        if times != sorted(set(times)):
            raise ValueError("bpmEvents must be sorted by timeMs without duplicates")

    def _check_auto_play_onsets(self) -> None:
        if self.auto_play_onsets != sorted(set(self.auto_play_onsets)):
            raise ValueError("autoPlayOnsets must be sorted without duplicates")
        if self.auto_play_onsets and not 0 <= self.auto_play_onsets[0]:
            raise ValueError("autoPlayOnsets must be non-negative")
        if self.auto_play_onsets and self.auto_play_onsets[-1] >= self.duration_ms:
            raise ValueError("autoPlayOnsets must be less than durationMs")

    def _check_metrics(self) -> None:
        # 파이프라인 조립 단계가 다른 채보의 측정값을 끼워 넣는 배선 실수를 잡는다.
        if self.metrics.note_count != len(self.notes):
            raise ValueError("metrics noteCount must match the number of notes")
        holds = sum(1 for note in self.notes if note.kind == "HOLD")
        if self.metrics.hold_count != holds:
            raise ValueError("metrics holdCount must match the number of HOLD notes")

    def to_json(self, *, indent: int | None = None) -> str:
        """camelCase JSON 문자열로 직렬화한다.

        `model_dump_json()` 을 직접 부르면 by_alias 를 빠뜨려 snake_case
        키가 새어나간다. 직렬화는 항상 이 함수를 거친다.
        """
        return self.model_dump_json(by_alias=True, indent=indent)


def chart_json_schema() -> dict[str, Any]:
    """chart-v1.schema.json 으로 배포할 JSON Schema 를 만든다."""
    return ChartDocument.model_json_schema(by_alias=True)
