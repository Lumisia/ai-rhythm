import numpy as np
import pytest

from chart_worker.analysis.audio_io import AudioSignal
from chart_worker.analysis.beat import (
    BeatGrid,
    analyze_beats,
    bpm_events_of,
    build_beat_grid,
    dedupe_beats,
    fit_bpm,
    robust_interval_sec,
    snap_downbeats,
)
from chart_worker.errors import ErrorCode, WorkerError

BPM = 82.0
INTERVAL = 60.0 / BPM


def _beats(count=64, bpm=BPM, start=0.12):
    return start + np.arange(count) * (60.0 / bpm)


def _quantized(times, frame_sec=0.02):
    """Beat This! 의 50 Hz 프레임 격자를 흉내낸다."""
    return np.round(np.asarray(times) / frame_sec) * frame_sec


def _signal(seconds=2.0, sample_rate=48_000):
    frames = int(seconds * sample_rate)
    return AudioSignal(samples=np.zeros((frames, 2)), sample_rate_hz=sample_rate)


def test_robust_interval_ignores_duplicate_detections():
    beats = np.sort(np.concatenate([_beats(32), _beats(4) + 0.08]))
    assert robust_interval_sec(beats) == pytest.approx(INTERVAL, abs=1e-9)


def test_robust_interval_rejects_a_run_of_impossible_gaps():
    with pytest.raises(ValueError, match="musically plausible"):
        robust_interval_sec(np.array([0.0, 0.05, 0.1, 0.15]))


def test_dedupe_keeps_the_first_of_a_close_pair():
    beats = np.array([0.0, 0.08, 0.73, 1.46, 1.50])
    kept = dedupe_beats(beats, min_gap_sec=0.5 * INTERVAL)
    assert kept.tolist() == [0.0, 0.73, 1.46]


def test_dedupe_leaves_a_clean_grid_untouched():
    beats = _beats(16)
    assert dedupe_beats(beats, min_gap_sec=0.5 * INTERVAL).tolist() == beats.tolist()


def test_dedupe_handles_an_empty_input():
    assert dedupe_beats(np.array([]), min_gap_sec=0.3).size == 0


def test_regression_beats_the_frame_grid_quantization():
    """국소 간격으로 재면 없는 템포 변화가 생긴다. 회귀는 양자화를 평균낸다."""
    quantized = _quantized(_beats(240))
    gaps = np.diff(quantized)
    assert len(set(np.round(gaps, 4))) > 1, "양자화가 간격을 흔들어야 한다"

    bpm, residual = fit_bpm(quantized)
    assert bpm == pytest.approx(BPM, abs=0.02)
    assert np.abs(residual).max() < 0.02


def test_fit_bpm_needs_two_beats():
    with pytest.raises(ValueError, match="at least two beats"):
        fit_bpm(np.array([1.0]))


def test_fit_bpm_rejects_a_non_increasing_grid():
    with pytest.raises(ValueError, match="not positive"):
        fit_bpm(np.array([2.0, 1.0, 0.0]))


def test_downbeats_snap_onto_the_cleaned_grid():
    """따로 중복 제거하면 다운비트가 비트 집합 밖으로 나간다."""
    beats = _beats(16)
    downbeats = beats[::4] + 0.01
    indices = snap_downbeats(beats, downbeats, tolerance_sec=0.5 * INTERVAL)
    assert indices == (0, 4, 8, 12)


def test_downbeats_outside_the_tolerance_are_dropped():
    beats = _beats(8)
    downbeats = np.array([beats[0], 999.0])
    assert snap_downbeats(beats, downbeats, tolerance_sec=0.3) == (0,)


def test_duplicate_downbeats_collapse_to_one_index():
    beats = _beats(8)
    downbeats = np.array([beats[4] - 0.01, beats[4] + 0.01])
    assert snap_downbeats(beats, downbeats, tolerance_sec=0.3) == (4,)


def test_snapping_an_empty_grid_is_empty():
    assert snap_downbeats(np.array([]), np.array([1.0]), tolerance_sec=0.3) == ()


def test_grid_matches_the_measured_track_shape():
    """실측: 272 -> 240 비트, 60마디 x 4비트, 82 BPM."""
    clean = _quantized(_beats(240))
    noisy = np.sort(np.concatenate([clean, clean[:32] + 0.08]))
    grid = build_beat_grid(noisy, clean[::4])

    assert grid.raw_beat_count == 272
    assert grid.dropped_beat_count == 32
    assert len(grid.beat_ms) == 240
    assert grid.bpm == pytest.approx(BPM, abs=0.02)
    assert grid.beats_per_bar == 4
    assert len(grid.downbeat_indices) == 60
    assert grid.downbeat_indices[:3] == (0, 4, 8)
    assert grid.is_constant_tempo


def test_grid_reports_residuals_in_milliseconds():
    grid = build_beat_grid(_quantized(_beats(64)), _beats(64)[::4])
    assert 0 < grid.residual_max_ms < 20
    assert grid.residual_rms_ms <= grid.residual_max_ms


def test_grid_times_are_plain_ints():
    grid = build_beat_grid(_beats(8), _beats(8)[::4])
    assert all(type(value) is int for value in grid.beat_ms)


def test_downbeat_ms_is_always_a_subset_of_beat_ms():
    grid = build_beat_grid(_quantized(_beats(32)), _quantized(_beats(32))[::4] + 0.005)
    assert set(grid.downbeat_ms) <= set(grid.beat_ms)


def test_grid_flags_a_tempo_change():
    slow = _beats(32, bpm=80.0)
    fast = slow[-1] + np.arange(1, 33) * (60.0 / 100.0)
    grid = build_beat_grid(np.concatenate([slow, fast]), np.array([slow[0]]))
    assert grid.bpm_drift_pct > 2.0
    assert not grid.is_constant_tempo


def test_beats_per_bar_is_unknown_without_downbeats():
    grid = build_beat_grid(_beats(8), np.array([]))
    assert grid.beats_per_bar is None
    assert grid.downbeat_indices == ()


def test_grid_needs_two_beats():
    with pytest.raises(ValueError, match="at least two beats"):
        build_beat_grid(np.array([1.0]), np.array([]))


def test_bpm_events_start_at_zero_and_keep_later_tempo_changes():
    first = np.arange(32) * 0.5
    second = first[-1] + np.arange(1, 33) * 0.6
    grid = build_beat_grid(np.concatenate([first, second]), np.concatenate([first, second])[::4])
    events = bpm_events_of(grid)
    assert len(events) == 2
    assert events[0].time_ms == 0
    assert events[0].bpm == pytest.approx(120.0, abs=0.05)
    assert events[1].time_ms == grid.beat_ms[32]
    assert events[1].bpm == pytest.approx(100.0, abs=0.05)


def test_analyze_beats_passes_mono_audio_to_the_backend():
    seen: dict[str, object] = {}

    def backend(signal, sample_rate_hz):
        seen["ndim"] = signal.ndim
        seen["rate"] = sample_rate_hz
        return _beats(16), _beats(16)[::4]

    grid = analyze_beats(_signal(), backend=backend)
    assert seen == {"ndim": 1, "rate": 48_000}
    assert isinstance(grid, BeatGrid)


def test_backend_failure_becomes_a_worker_error():
    def backend(signal, sample_rate_hz):
        raise RuntimeError("model exploded")

    with pytest.raises(WorkerError) as caught:
        analyze_beats(_signal(), backend=backend)
    assert caught.value.code is ErrorCode.CHART_ANALYSIS_FAILED
    assert caught.value.retryable is True


def test_too_few_beats_becomes_a_worker_error():
    def backend(signal, sample_rate_hz):
        return np.array([1.0]), np.array([])

    with pytest.raises(WorkerError) as caught:
        analyze_beats(_signal(), backend=backend)
    assert caught.value.code is ErrorCode.CHART_ANALYSIS_FAILED
    assert caught.value.context == {"beat_count": 1}
