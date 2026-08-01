import numpy as np
import pytest

from chart_worker.analysis.audio_io import AudioSignal, load_audio
from chart_worker.errors import ErrorCode, WorkerError

soundfile = pytest.importorskip("soundfile", reason="analysis extra is not installed")


def _write(path, seconds=1.0, sample_rate=48_000, channels=2, subtype="PCM_16"):
    frames = int(seconds * sample_rate)
    tone = np.sin(2 * np.pi * 440 * np.arange(frames) / sample_rate) * 0.5
    data = np.column_stack([tone] * channels) if channels > 1 else tone
    soundfile.write(str(path), data, sample_rate, subtype=subtype)
    return path


def test_reads_a_flac_without_ffmpeg(tmp_path):
    """libsndfile 이 FLAC 을 자체 지원한다. torchcodec 이 필요 없다."""
    signal = load_audio(_write(tmp_path / "a.flac"))
    assert signal.sample_rate_hz == 48_000
    assert signal.channels == 2
    assert signal.frame_count == 48_000
    assert signal.duration_ms == 1000


def test_mono_is_still_two_dimensional(tmp_path):
    signal = load_audio(_write(tmp_path / "mono.flac", channels=1))
    assert signal.samples.ndim == 2
    assert signal.channels == 1


def test_to_mono_averages_channels():
    samples = np.array([[1.0, 3.0], [0.0, 2.0]])
    mono = AudioSignal(samples=samples, sample_rate_hz=48_000).to_mono()
    assert mono.tolist() == [2.0, 1.0]
    assert mono.ndim == 1


def test_samples_are_float64(tmp_path):
    signal = load_audio(_write(tmp_path / "a.flac"))
    assert signal.samples.dtype == np.float64


def test_duration_rounds_to_the_nearest_millisecond():
    signal = AudioSignal(samples=np.zeros((48_025, 2)), sample_rate_hz=48_000)
    assert signal.duration_ms == 1001


def test_missing_file_is_invalid_audio(tmp_path):
    with pytest.raises(WorkerError) as caught:
        load_audio(tmp_path / "nope.flac")
    assert caught.value.code is ErrorCode.AUDIO_INVALID
    assert caught.value.disposition.value == "FINAL"


def test_garbage_file_is_invalid_audio(tmp_path):
    broken = tmp_path / "broken.flac"
    broken.write_bytes(b"not audio")
    with pytest.raises(WorkerError) as caught:
        load_audio(broken)
    assert caught.value.code is ErrorCode.AUDIO_INVALID


def test_empty_file_is_invalid_audio(tmp_path):
    empty = tmp_path / "empty.flac"
    soundfile.write(str(empty), np.zeros((0, 2)), 48_000)
    with pytest.raises(WorkerError) as caught:
        load_audio(empty)
    assert caught.value.code is ErrorCode.AUDIO_INVALID
