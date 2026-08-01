"""실제 Beat This! 로 도는 비트 추적 검증.

정답 BPM 을 아는 클릭 트랙을 numpy 로 만든다. 저장소에 바이너리도,
ffmpeg 도 필요 없다.
"""

import numpy as np
import pytest

from chart_worker.analysis.audio_io import AudioSignal, load_audio
from chart_worker.analysis.beat import analyze_beats, beat_this_backend, bpm_events_of

pytestmark = pytest.mark.beat_this

pytest.importorskip("beat_this", reason="analysis extra is not installed")
soundfile = pytest.importorskip("soundfile", reason="analysis extra is not installed")

SAMPLE_RATE = 48_000
BPM = 120.0
BEATS_PER_BAR = 4
BAR_COUNT = 8


def _click_track() -> np.ndarray:
    """120 BPM 4/4 클릭. 다운비트는 낮고 길게, 나머지는 높고 짧게."""
    interval = 60.0 / BPM
    total = int(BEATS_PER_BAR * BAR_COUNT * interval * SAMPLE_RATE)
    track = np.zeros(total, dtype=np.float64)

    for beat in range(BEATS_PER_BAR * BAR_COUNT):
        downbeat = beat % BEATS_PER_BAR == 0
        frequency, length = (60.0, 0.18) if downbeat else (1000.0, 0.05)
        start = int(beat * interval * SAMPLE_RATE)
        span = np.arange(int(length * SAMPLE_RATE))
        envelope = np.exp(-span / (0.02 * SAMPLE_RATE))
        click = np.sin(2 * np.pi * frequency * span / SAMPLE_RATE) * envelope
        end = min(start + span.size, total)
        track[start:end] += click[: end - start] * (0.9 if downbeat else 0.5)
    return track


@pytest.fixture(scope="module")
def backend():
    return beat_this_backend(checkpoint="final0", device="cpu")


@pytest.fixture(scope="module")
def click_signal(tmp_path_factory):
    path = tmp_path_factory.mktemp("beat") / "click.flac"
    track = _click_track()
    soundfile.write(str(path), np.column_stack([track, track]), SAMPLE_RATE)
    return load_audio(path)


def test_click_track_loads_as_expected(click_signal):
    assert click_signal.sample_rate_hz == SAMPLE_RATE
    assert click_signal.channels == 2
    assert click_signal.duration_ms == pytest.approx(16_000, abs=50)


def test_recovers_the_known_tempo(click_signal, backend):
    grid = analyze_beats(click_signal, backend=backend)
    assert grid.bpm == pytest.approx(BPM, rel=0.02)
    assert grid.is_constant_tempo


def test_beat_count_matches_the_click_count(click_signal, backend):
    grid = analyze_beats(click_signal, backend=backend)
    assert len(grid.beat_ms) == pytest.approx(BEATS_PER_BAR * BAR_COUNT, abs=2)


def test_grid_is_monotonic_and_deduplicated(click_signal, backend):
    grid = analyze_beats(click_signal, backend=backend)
    gaps = np.diff(grid.beat_ms)
    assert (gaps > 0).all()
    assert gaps.min() > 200, "300 BPM 보다 빠른 간격이 남으면 중복 검출이다"


def test_downbeats_are_a_subset_of_beats(click_signal, backend):
    grid = analyze_beats(click_signal, backend=backend)
    assert set(grid.downbeat_ms) <= set(grid.beat_ms)
    assert all(0 <= index < len(grid.beat_ms) for index in grid.downbeat_indices)


def test_bpm_events_are_usable_by_chart_v1(click_signal, backend):
    events = bpm_events_of(analyze_beats(click_signal, backend=backend))
    assert events[0].time_ms == 0
    assert events[0].bpm > 0


def test_flac_analysis_needs_no_ffmpeg(click_signal, backend):
    """soundfile 로 읽어 텐서를 넘기므로 torchcodec 경로를 타지 않는다."""
    assert isinstance(click_signal, AudioSignal)
    assert analyze_beats(click_signal, backend=backend).beat_ms
