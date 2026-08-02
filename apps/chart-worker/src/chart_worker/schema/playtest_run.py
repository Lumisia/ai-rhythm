"""로컬 플레이테스터 실행 디렉터리의 진입 계약."""

from datetime import datetime
from pathlib import PurePosixPath
from typing import Annotated, Literal, Self
from uuid import UUID

from pydantic import AfterValidator, Field, model_validator

from chart_worker.schema.chart import CamelModel, Sha256
from chart_worker.schema.types import DIFFICULTIES, KEY_MODES, Difficulty


def safe_relative_path(value: str) -> str:
    path = PurePosixPath(value.replace("\\", "/"))
    if (
        not value
        or path == PurePosixPath(".")
        or path.is_absolute()
        or ".." in path.parts
        or ":" in value
    ):
        raise ValueError("path must be a safe relative path")
    return path.as_posix()


SafeRelativePath = Annotated[str, AfterValidator(safe_relative_path)]


class AudioFileRef(CamelModel):
    path: SafeRelativePath
    sha256: Sha256


class RunChartRef(AudioFileRef):
    key_mode: Literal[4, 6, 7]
    difficulty: Difficulty


class RunAudioRefs(CamelModel):
    game: AudioFileRef
    no_drums: AudioFileRef | None = None
    keys: AudioFileRef | None = None


class PlaytestRunManifest(CamelModel):
    version: Literal[1] = 1
    run_id: UUID
    title: str = Field(min_length=1)
    generated_at: datetime
    worker_version: str = Field(min_length=1)
    audio: RunAudioRefs
    charts: list[RunChartRef]
    keysound_manifest_path: SafeRelativePath | None = None
    generation_report_path: SafeRelativePath

    @model_validator(mode="after")
    def _check_run(self) -> Self:
        combinations = [(chart.key_mode, chart.difficulty) for chart in self.charts]
        if len(combinations) != len(set(combinations)):
            raise ValueError("duplicate chart combination")
        expected = {(key_mode, difficulty) for key_mode in KEY_MODES for difficulty in DIFFICULTIES}
        if set(combinations) != expected:
            raise ValueError("run must contain exactly 12 chart combinations")

        keysound_parts = (
            self.audio.no_drums is not None,
            self.audio.keys is not None,
            self.keysound_manifest_path is not None,
        )
        if any(keysound_parts) and not all(keysound_parts):
            raise ValueError("keysound references must all be present or absent")
        return self
