from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class WorkerConfig(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    chart_generator: Literal["mapperatorinator", "fake"] = "fake"
    mapperatorinator_home: Path | None = None
    mapperatorinator_python: Path | None = None
    mapperatorinator_precision: str = "fp16"
    ffmpeg_bin: Path = Path("ffmpeg")
    ffmpeg_shared_bin_dir: Path | None = None
    storage_local_root: Path = Path(".data/storage")


def load_config() -> WorkerConfig:
    return WorkerConfig()
