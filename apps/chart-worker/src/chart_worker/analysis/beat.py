"""선택적 진단 백엔드의 비트 격자 유틸리티."""

from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from chart_worker.analysis.audio_io import AudioSignal
from chart_worker.errors import ErrorCode, WorkerError
from chart_worker.schema.chart import BpmEvent

BeatBackend = Callable[[np.ndarray, int], tuple[np.ndarray, np.ndarray]]
"""(모노 신호, 샘플레이트) -> (비트 초, 다운비트 초)."""

MUSICAL_FLOOR_SEC = 0.2
"""300 BPM. 이보다 짧은 간격은 어떤 템포에서도 비트 간격이 아니다."""

DEDUPE_RATIO = 0.5
"""중앙값 간격의 절반보다 가까운 비트는 중복 검출로 본다."""

DRIFT_TOLERANCE_PCT = 2.0
"""전반·후반 BPM 차이가 이보다 크면 단일 템포로 보기 어렵다."""


@dataclass(frozen=True, slots=True)
class BeatGrid:
    beat_ms: tuple[int, ...]
    downbeat_indices: tuple[int, ...]
    """다운비트가 놓인 beat_ms 의 인덱스. 다운비트는 항상 비트의 부분집합이다."""

    bpm: float
    beats_per_bar: int | None
    bpm_drift_pct: float
    raw_beat_count: int
    dropped_beat_count: int
    residual_rms_ms: float
    residual_max_ms: float

    @property
    def is_constant_tempo(self) -> bool:
        return self.bpm_drift_pct <= DRIFT_TOLERANCE_PCT

    @property
    def downbeat_ms(self) -> tuple[int, ...]:
        return tuple(self.beat_ms[index] for index in self.downbeat_indices)


def robust_interval_sec(beat_sec: np.ndarray, *, floor_sec: float = MUSICAL_FLOOR_SEC) -> float:
    """중복 검출을 뺀 비트 간격의 중앙값.

    임계값을 상수로 박지 않기 위한 1차 추정이다. 곡 템포에 따라
    중복 판정 기준이 따라가야 한다.
    """
    gaps = np.diff(np.asarray(beat_sec, dtype=np.float64))
    musical = gaps[gaps >= floor_sec]
    if musical.size == 0:
        raise ValueError("no musically plausible beat interval found")
    return float(np.median(musical))


def dedupe_beats(beat_sec: np.ndarray, *, min_gap_sec: float) -> np.ndarray:
    """너무 가까이 붙은 비트를 앞의 것만 남기고 버린다."""
    times = np.asarray(beat_sec, dtype=np.float64)
    if times.size == 0:
        return times
    kept = [float(times[0])]
    for value in times[1:]:
        if float(value) - kept[-1] >= min_gap_sec:
            kept.append(float(value))
    return np.array(kept, dtype=np.float64)


def fit_bpm(beat_sec: np.ndarray) -> tuple[float, np.ndarray]:
    """비트 번호 대 시각을 최소제곱으로 적합해 BPM 과 잔차를 낸다.

    인접 간격으로 BPM 을 내지 않는 이유: Beat This! 프레임 격자가
    20ms(50 Hz)라 간격이 720·740·760 처럼 양자화된다. 국소 창으로 재면
    82 BPM 곡이 80.0~83.3 을 오가며 없는 템포 변화를 수십 개 만든다.
    회귀는 양자화를 평균낸다.
    """
    times = np.asarray(beat_sec, dtype=np.float64)
    if times.size < 2:
        raise ValueError("at least two beats are required to fit a tempo")
    index = np.arange(times.size, dtype=np.float64)
    slope, intercept = np.polyfit(index, times, 1)
    if slope <= 0:
        raise ValueError("fitted beat interval is not positive")
    return 60.0 / float(slope), times - (slope * index + intercept)


def snap_downbeats(
    beat_sec: np.ndarray,
    downbeat_sec: np.ndarray,
    *,
    tolerance_sec: float,
) -> tuple[int, ...]:
    """다운비트를 정제된 비트 격자의 인덱스로 옮긴다.

    비트와 다운비트를 각각 중복 제거하면 같은 중복 쌍에서 서로 다른 쪽이
    남아 다운비트가 비트 집합에 없게 된다. 그래서 스냅으로 푼다.
    """
    beats = np.asarray(beat_sec, dtype=np.float64)
    if beats.size == 0:
        return ()
    matched: set[int] = set()
    for value in np.asarray(downbeat_sec, dtype=np.float64):
        index = int(np.argmin(np.abs(beats - value)))
        if abs(float(beats[index]) - float(value)) <= tolerance_sec:
            matched.add(index)
    return tuple(sorted(matched))


def _beats_per_bar(downbeat_indices: tuple[int, ...]) -> int | None:
    if len(downbeat_indices) < 2:
        return None
    gaps = np.diff(np.array(downbeat_indices))
    most_common, _ = Counter(int(gap) for gap in gaps).most_common(1)[0]
    return most_common if most_common > 0 else None


def _drift_pct(beat_sec: np.ndarray) -> float:
    """전반·후반을 따로 적합해 템포 드리프트를 잰다."""
    if beat_sec.size < 8:
        return 0.0
    half = beat_sec.size // 2
    first, _ = fit_bpm(beat_sec[:half])
    second, _ = fit_bpm(beat_sec[half:])
    return abs(second - first) / first * 100.0


def build_beat_grid(beat_sec: np.ndarray, downbeat_sec: np.ndarray) -> BeatGrid:
    """Beat This! 원본 출력을 정제된 비트 격자로 만든다."""
    raw = np.asarray(beat_sec, dtype=np.float64)
    if raw.size < 2:
        raise ValueError("at least two beats are required to build a grid")

    interval = robust_interval_sec(raw)
    clean = dedupe_beats(raw, min_gap_sec=DEDUPE_RATIO * interval)
    bpm, residual = fit_bpm(clean)
    downbeat_indices = snap_downbeats(clean, downbeat_sec, tolerance_sec=DEDUPE_RATIO * interval)
    return BeatGrid(
        # tolist() 는 numpy 스칼라가 아니라 파이썬 int 를 준다.
        beat_ms=tuple(np.round(clean * 1000).astype(np.int64).tolist()),
        downbeat_indices=downbeat_indices,
        bpm=round(bpm, 3),
        beats_per_bar=_beats_per_bar(downbeat_indices),
        bpm_drift_pct=round(_drift_pct(clean), 3),
        raw_beat_count=int(raw.size),
        dropped_beat_count=int(raw.size - clean.size),
        residual_rms_ms=round(float(np.sqrt((residual**2).mean())) * 1000, 3),
        residual_max_ms=round(float(np.abs(residual).max()) * 1000, 3),
    )


def bpm_events_of(grid: BeatGrid) -> list[BpmEvent]:
    """chart-v1 의 bpmEvents.

    지금은 항상 하나만 낸다. 20ms 프레임 격자 때문에 진짜 템포 변화와
    양자화 흔들림을 가르는 임계값은 실제로 템포가 바뀌는 곡 없이는 정할 수
    없다. 추정으로 박으면 없는 변화를 만들어낸다. 드리프트는 BeatGrid 에
    기록해 두고 구간 분할은 미룬다.
    """
    from chart_worker.analysis.timing import bpm_events_of as timing_bpm_events_of
    from chart_worker.analysis.timing import fit_piecewise_timing

    return timing_bpm_events_of(fit_piecewise_timing(grid))


def analyze_beats(signal: AudioSignal, *, backend: BeatBackend) -> BeatGrid:
    """오디오에서 비트 격자를 뽑는다."""
    try:
        beat_sec, downbeat_sec = backend(signal.to_mono(), signal.sample_rate_hz)
    except Exception as error:
        raise WorkerError(
            ErrorCode.CHART_ANALYSIS_FAILED,
            f"beat tracking failed: {error}",
        ) from error
    try:
        return build_beat_grid(np.asarray(beat_sec), np.asarray(downbeat_sec))
    except ValueError as error:
        raise WorkerError(
            ErrorCode.CHART_ANALYSIS_FAILED,
            f"beat grid is unusable: {error}",
            context={"beat_count": int(np.asarray(beat_sec).size)},
        ) from error
