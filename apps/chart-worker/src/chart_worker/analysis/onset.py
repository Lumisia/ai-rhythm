"""생성 뒤 명시적으로 실행하는 librosa onset 진단 유틸리티.

진단은 원본 채보를 설명할 뿐 노트를 선택·삭제하거나 timing을 바꾸지 않는다.
"""

import dataclasses
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from chart_worker.analysis.activity import AudioActivity, build_audio_activity
from chart_worker.analysis.audio_io import AudioSignal, load_audio
from chart_worker.analysis.beat import BeatGrid
from chart_worker.errors import ErrorCode, WorkerError
from chart_worker.schema.note import Chart

Band = Literal["LOW", "MID", "HIGH"]
BAND_NAMES: tuple[Band, ...] = ("LOW", "MID", "HIGH")

BAND_EDGES_HZ: tuple[float, float] = (250.0, 2000.0)
"""킥·베이스 / 스네어·보컬 / 하이햇·심벌의 통상 경계."""

HOP_LENGTH = 512
"""48 kHz 에서 프레임 간격 10.67 ms. 판정 창 ±22 ms 보다 촘촘하다."""

N_FFT = 2048
"""48 kHz 에서 42.7 ms. 이 폭만큼 어택이 뒤로 번진다."""

N_MELS = 128
NORMALIZATION_PERCENTILE = 99.0
"""최대값으로 나누면 한 번의 큰 타격이 나머지를 전부 눌러버린다."""

DOWNBEAT_WINDOW_MS = 50
"""Phase3 의 키음 어택 스냅 창과 같은 값. 창을 두 개 두면 판정이 갈린다."""


def normalize_envelope(
    values: np.ndarray, *, percentile: float = NORMALIZATION_PERCENTILE
) -> np.ndarray:
    """분위수로 나누고 1.0 에서 자른다.

    원값은 곡마다 스케일이 다르다. solver 는 한 곡 안에서 비교하므로
    절대값이 필요 없고, 정규화해야 임계값을 곡에 무관하게 쓸 수 있다.
    """
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return array
    scale = float(np.percentile(array, percentile))
    if scale <= 0:
        return np.zeros_like(array)
    return np.clip(array / scale, 0.0, 1.0)


@dataclass(frozen=True, slots=True)
class OnsetAnalysis:
    sample_rate_hz: int
    hop_length: int
    strength: np.ndarray
    """(프레임,) 0~1 로 정규화된 종합 onset 강도."""

    band_strength: np.ndarray
    """(3, 프레임) 대역별 0~1 강도. 행 순서는 BAND_NAMES 와 같다."""

    onset_ms: tuple[int, ...]
    n_fft: int = N_FFT
    activity: AudioActivity | None = None

    @property
    def frame_count(self) -> int:
        return int(self.strength.size)

    @property
    def frame_ms(self) -> float:
        return self.hop_length * 1000.0 / self.sample_rate_hz

    def frame_of(self, time_ms: int) -> int:
        """시각에 가장 가까운 프레임 번호. 범위를 벗어나면 양끝으로 자른다."""
        if self.frame_count == 0:
            raise ValueError("onset analysis has no frames")
        index = round(time_ms / self.frame_ms)
        return max(0, min(self.frame_count - 1, index))

    def window_of(self, time_ms: int) -> slice:
        """타격 시각을 덮는 프레임 구간. 앞뒤가 비대칭이다.

        spectral flux 는 프레임 차분이라 어택 **직후**에 뜨고, STFT 창
        폭(n_fft)만큼 뒤로 번진다. 절대 앞서지 않는다. 단일 프레임을
        읽으면 가장 강한 타격이 강도 0 으로 나와 solver 순위가 뒤집힌다.
        """
        center = self.frame_of(time_ms)
        ahead = max(1, -(-self.n_fft // self.hop_length))
        return slice(max(0, center - 1), min(self.frame_count, center + ahead + 1))

    def strength_at(self, time_ms: int) -> float:
        return float(self.strength[self.window_of(time_ms)].max())

    def band_at(self, time_ms: int) -> Band:
        """그 시각에 가장 두드러진 대역."""
        window = self.band_strength[:, self.window_of(time_ms)]
        return BAND_NAMES[int(np.argmax(window.max(axis=1)))]


OnsetBackend = Callable[[np.ndarray, int], OnsetAnalysis]


def analyze_onsets(signal: AudioSignal, *, backend: OnsetBackend) -> OnsetAnalysis:
    try:
        return backend(signal.to_mono(), signal.sample_rate_hz)
    except Exception as error:
        raise WorkerError(
            ErrorCode.CHART_ANALYSIS_FAILED,
            f"onset analysis failed: {error}",
        ) from error


def analyze_canonical_audio(path: Path) -> OnsetAnalysis:
    """Analyze the exact normalized audio later referenced by the manifest."""
    return analyze_onsets(load_audio(path), backend=librosa_backend())


def mel_band_channels(
    sample_rate_hz: int,
    *,
    n_mels: int = N_MELS,
    edges_hz: tuple[float, float] = BAND_EDGES_HZ,
) -> list[int]:
    """대역 경계 Hz 를 mel 밴드 인덱스로 옮긴다."""
    import librosa

    centers = librosa.mel_frequencies(n_mels=n_mels + 2, fmin=0.0, fmax=sample_rate_hz / 2)
    bounds = [int(np.searchsorted(centers, hz)) for hz in edges_hz]
    channels = [0, *bounds, n_mels]
    if sorted(set(channels)) != channels:
        raise ValueError(f"band edges collapse into the same mel bin: {channels}")
    return channels


def librosa_backend(
    *,
    hop_length: int = HOP_LENGTH,
    n_fft: int = N_FFT,
    n_mels: int = N_MELS,
    edges_hz: tuple[float, float] = BAND_EDGES_HZ,
) -> OnsetBackend:
    """librosa 로 onset 포락선과 대역 강도를 낸다.

    librosa 기본 22050 Hz 로 리샘플하지 않는다. 모든 도구가 같은
    시간축을 봐야 한다.
    """
    import librosa

    def run(mono: np.ndarray, sample_rate_hz: int) -> OnsetAnalysis:
        channels = mel_band_channels(sample_rate_hz, n_mels=n_mels, edges_hz=edges_hz)
        aggregate = librosa.onset.onset_strength(
            y=mono, sr=sample_rate_hz, hop_length=hop_length, n_fft=n_fft, n_mels=n_mels
        )
        bands = librosa.onset.onset_strength_multi(
            y=mono,
            sr=sample_rate_hz,
            hop_length=hop_length,
            n_fft=n_fft,
            n_mels=n_mels,
            channels=channels,
        )
        peaks = librosa.onset.onset_detect(
            onset_envelope=aggregate,
            sr=sample_rate_hz,
            hop_length=hop_length,
            units="frames",
            # flux 피크는 어택보다 늦게 뜬다. 직전 에너지 최소점으로 되돌려야
            # 키음 어택 스냅이 밀린 소리를 내지 않는다. 실측 평균 오차 6.9 -> 4.1 ms.
            backtrack=True,
        )
        rms = librosa.feature.rms(
            y=mono,
            frame_length=n_fft,
            hop_length=hop_length,
        )[0]
        rms_db = librosa.amplitude_to_db(rms, ref=1.0)
        frame_count = min(aggregate.size, bands.shape[1], rms_db.size)
        aggregate = aggregate[:frame_count]
        bands = bands[:, :frame_count]
        rms_db = rms_db[:frame_count]
        peaks = np.asarray(peaks, dtype=np.int64)
        peaks = peaks[(peaks >= 0) & (peaks < frame_count)]
        frame_ms = hop_length * 1000.0 / sample_rate_hz
        normalized_strength = normalize_envelope(aggregate)
        activity = build_audio_activity(
            rms_db=rms_db,
            normalized_strength=normalized_strength,
            onset_frames=peaks,
            frame_ms=frame_ms,
            n_fft=n_fft,
            hop_length=hop_length,
        )
        return OnsetAnalysis(
            sample_rate_hz=int(sample_rate_hz),
            hop_length=hop_length,
            strength=normalized_strength,
            # 대역마다 mel bin 수가 달라 원값 argmax 는 고역으로 쏠린다.
            band_strength=np.vstack([normalize_envelope(band) for band in bands]),
            onset_ms=tuple(np.round(np.asarray(peaks) * frame_ms).astype(np.int64).tolist()),
            n_fft=n_fft,
            activity=activity,
        )

    return run


def _is_downbeat(time_ms: int, downbeat_ms: tuple[int, ...], window_ms: int) -> bool:
    if not downbeat_ms:
        return False
    times = np.asarray(downbeat_ms)
    return bool(np.abs(times - time_ms).min() <= window_ms)


def _beat_fraction(time_ms: int, beat_ms: tuple[int, ...]) -> float | None:
    """비트 안에서의 상대 위치. 0.0 이 정박, 0.5 가 8분 뒷박이다."""
    if len(beat_ms) < 2:
        return None
    times = np.asarray(beat_ms)
    index = int(np.searchsorted(times, time_ms, side="right")) - 1
    if index < 0:
        index = 0
    elif index >= times.size - 1:
        index = times.size - 2
    start, end = int(times[index]), int(times[index + 1])
    if end <= start:
        return None
    return round((time_ms - start) / (end - start) % 1.0, 6)


def annotate_notes(
    notes: Chart,
    *,
    onsets: OnsetAnalysis,
    grid: BeatGrid,
    downbeat_window_ms: int = DOWNBEAT_WINDOW_MS,
) -> Chart:
    """노트에 분석 메타데이터를 채운다.

    `time_ms` 를 건드리지 않는다. 타이밍 불변 원칙은 후처리만이 아니라
    분석 주석에도 적용된다.
    """
    downbeat_ms = grid.downbeat_ms
    return [
        dataclasses.replace(
            note,
            onset_strength=onsets.strength_at(note.time_ms),
            band=onsets.band_at(note.time_ms),
            is_downbeat=_is_downbeat(note.time_ms, downbeat_ms, downbeat_window_ms),
            beat_fraction=_beat_fraction(note.time_ms, grid.beat_ms),
        )
        for note in notes
    ]
