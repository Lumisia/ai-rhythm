from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class WorkerConfig(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    chart_generator: Literal["mapperatorinator", "fake"] = "mapperatorinator"
    mapperatorinator_home: Path | None = None
    mapperatorinator_python: Path | None = None
    mapperatorinator_precision: Literal["fp16", "bf16"] = "fp16"
    mapperatorinator_hold_state_mode: Literal[
        "full_scan", "incremental_verify", "incremental"
    ] = "incremental"
    mapperatorinator_write_generation_telemetry: bool = False
    mapperatorinator_backend: Literal["oneshot", "song_session"] = "oneshot"
    mapperatorinator_model_root: Path | None = None
    mapperatorinator_model_revision: str | None = None
    mapperatorinator_tail_repairs: Literal[2] = 2
    mapperatorinator_checkpoint_interval_windows: Literal[8] = 8
    mapperatorinator_protocol_max_line_bytes: Literal[1_048_576] = 1_048_576
    mapperatorinator_resident_startup_timeout_sec: float = Field(
        default=1800.0,
        gt=0.0,
        le=1800.0,
    )
    mapperatorinator_resident_invocation_timeout_sec: float = Field(
        default=10800.0,
        gt=0.0,
        le=10800.0,
    )
    mapperatorinator_resident_close_timeout_sec: float = Field(
        default=5.0,
        gt=0.0,
        le=30.0,
    )
    difficulty_selector_mode: Literal["CURRENT", "SHADOW_V2", "V2"] = "V2"
    difficulty_shadow_challenger_enabled: bool = False
    difficulty_family_compiler_shadow_enabled: bool = False
    difficulty_family_resolution_enabled: bool = True
    boundary_policy_mode: Literal[
        "SHADOW",
        "EXPERIMENTAL_ENFORCED",
        "HIGH_CONFIDENCE_ENFORCED",
    ] = "HIGH_CONFIDENCE_ENFORCED"
    beat_this_enabled: bool = True
    beat_this_checkpoint: str = "final0"
    beat_this_device: Literal["cpu", "cuda"] = "cpu"
    beat_this_float16: bool = False
    ffmpeg_bin: Path = Path("ffmpeg")
    ffmpeg_shared_bin_dir: Path | None = None
    storage_local_root: Path = Path(".data/storage")

    @field_validator("mapperatorinator_model_root")
    @classmethod
    def _require_absolute_model_root(cls, value: Path | None) -> Path | None:
        if value is not None and not value.is_absolute():
            raise ValueError("mapperatorinator_model_root must be absolute")
        return value

    @field_validator("mapperatorinator_model_revision", mode="before")
    @classmethod
    def _require_immutable_model_revision(cls, value: object) -> object:
        if value is None:
            return None
        if type(value) is not str or len(value) != 40:
            raise ValueError("mapperatorinator_model_revision must be a 40-character SHA")
        if any(character not in "0123456789abcdef" for character in value):
            raise ValueError("mapperatorinator_model_revision must be lowercase hexadecimal")
        return value

    @model_validator(mode="after")
    def _validate_song_session_contract(self) -> "WorkerConfig":
        if self.mapperatorinator_backend != "song_session":
            return self
        if self.mapperatorinator_hold_state_mode != "incremental":
            raise ValueError("song_session requires mapperatorinator_hold_state_mode=incremental")
        if self.mapperatorinator_home is None or not self.mapperatorinator_home.is_absolute():
            raise ValueError("song_session requires absolute mapperatorinator_home")
        if self.mapperatorinator_python is None or not self.mapperatorinator_python.is_absolute():
            raise ValueError("song_session requires absolute mapperatorinator_python")
        if self.mapperatorinator_model_root is None:
            raise ValueError("song_session requires mapperatorinator_model_root")
        if self.mapperatorinator_model_revision is None:
            raise ValueError("song_session requires mapperatorinator_model_revision")
        return self

    @property
    def ffprobe_bin(self) -> Path:
        """ffprobe 는 ffmpeg 옆에 같은 확장자로 설치된다.

        따로 설정하게 두면 두 값이 서로 다른 빌드를 가리키는 사고가 난다.
        static 빌드와 shared 빌드가 섞이면 원인을 찾기 어렵다.
        """
        return self.ffmpeg_bin.with_name(f"ffprobe{self.ffmpeg_bin.suffix}")


def load_config() -> WorkerConfig:
    return WorkerConfig()
