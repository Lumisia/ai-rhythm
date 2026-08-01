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

    @property
    def ffprobe_bin(self) -> Path:
        """ffprobe 는 ffmpeg 옆에 같은 확장자로 설치된다.

        따로 설정하게 두면 두 값이 서로 다른 빌드를 가리키는 사고가 난다.
        static 빌드와 shared 빌드가 섞이면 원인을 찾기 어렵다.
        """
        return self.ffmpeg_bin.with_name(f"ffprobe{self.ffmpeg_bin.suffix}")


def load_config() -> WorkerConfig:
    return WorkerConfig()
