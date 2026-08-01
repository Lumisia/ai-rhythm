"""분석 단계의 오디오 로딩.

soundfile(libsndfile)로 읽는다. torchaudio 를 쓰지 않는다.

beat_this 의 load_audio 는 torchaudio 를 먼저 쓰고 실패하면 예외를 삼키고
soundfile 로 넘어간다. torchaudio 는 .wav 외 형식에 ffmpeg(torchcodec)가
필요한데 GAME_AUDIO 는 FLAC 이다. libsndfile 은 FLAC 을 자체 지원하므로
직접 읽으면 ffmpeg 의존이 아예 사라진다. 예외를 삼키는 폴백에 기대지 않는다.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from chart_worker.errors import ErrorCode, WorkerError


@dataclass(frozen=True, slots=True)
class AudioSignal:
    samples: np.ndarray
    """(프레임, 채널) 모양의 float64 배열. 모노도 2차원으로 맞춘다."""

    sample_rate_hz: int

    @property
    def frame_count(self) -> int:
        return int(self.samples.shape[0])

    @property
    def channels(self) -> int:
        return int(self.samples.shape[1])

    @property
    def duration_ms(self) -> int:
        return round(self.frame_count * 1000 / self.sample_rate_hz)

    def to_mono(self) -> np.ndarray:
        """채널 평균. 1차원 float64 배열을 돌려준다."""
        return self.samples.mean(axis=1)


def load_audio(path: Path) -> AudioSignal:
    """오디오 파일을 float64 로 읽는다.

    실패는 AUDIO_INVALID 로 바꾼다. 분석 단계가 읽지 못하는 파일은
    다시 시도해도 읽지 못한다.
    """
    try:
        import soundfile as sf
    except ImportError as error:  # pragma: no cover - 설치 문제
        raise WorkerError(
            ErrorCode.CHART_ANALYSIS_FAILED,
            "soundfile is not installed; install the 'analysis' extra",
        ) from error

    try:
        samples, sample_rate = sf.read(str(path), dtype="float64", always_2d=True)
    except Exception as error:
        raise WorkerError(
            ErrorCode.AUDIO_INVALID,
            f"could not read audio: {error}",
            context={"path": str(path)},
        ) from error

    if samples.shape[0] == 0:
        raise WorkerError(
            ErrorCode.AUDIO_INVALID,
            "audio file has no frames",
            context={"path": str(path)},
        )
    return AudioSignal(samples=samples, sample_rate_hz=int(sample_rate))
