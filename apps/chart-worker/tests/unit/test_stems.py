import numpy as np
import pytest

from chart_worker.analysis.audio_io import AudioSignal
from chart_worker.analysis.stems import (
    MODEL_SAMPLE_RATE_HZ,
    RECONSTRUCTION_TOLERANCE,
    Stems,
    separate_stems,
)
from chart_worker.errors import ErrorCode, WorkerError

SAMPLE_RATE = 48_000
FRAMES = SAMPLE_RATE  # 1초


def _signal(frames=FRAMES, channels=2, sample_rate=SAMPLE_RATE):
    rng = np.random.default_rng(0)
    return AudioSignal(
        samples=rng.standard_normal((frames, channels)).astype(np.float32) * 0.1,
        sample_rate_hz=sample_rate,
    )


def _half_backend(model_mix, sample_rate_hz):
    """믹스의 절반을 드럼이라고 답하는 백엔드."""
    return model_mix * 0.5


def test_backend_sees_the_model_sample_rate():
    seen = {}

    def backend(model_mix, sample_rate_hz):
        seen["rate"] = sample_rate_hz
        seen["frames"] = model_mix.shape[0]
        return model_mix * 0.5

    separate_stems(_signal(), backend=backend)
    assert seen["rate"] == MODEL_SAMPLE_RATE_HZ
    assert seen["frames"] == pytest.approx(FRAMES * 44_100 / 48_000, rel=0.01)


def test_stems_come_back_at_the_source_rate_and_length():
    signal = _signal()
    stems = separate_stems(signal, backend=_half_backend)
    assert stems.sample_rate_hz == SAMPLE_RATE
    assert stems.frame_count == FRAMES
    assert stems.no_drums.shape == signal.samples.shape


def test_drums_plus_no_drums_reproduces_the_mix():
    """감산 키음의 전제. 모든 노트를 친 플레이어가 원곡 그대로를 들어야 한다."""
    signal = _signal()
    stems = separate_stems(signal, backend=_half_backend)
    error = float(np.abs(stems.drums + stems.no_drums - signal.samples).max())
    # float32 반올림 한 자리(약 3e-08) 수준. 실측 htdemucs 4스템 합의
    # 복원 오차 3.3e-03 보다 다섯 자릿수 작다.
    assert error <= RECONSTRUCTION_TOLERANCE


def test_no_drums_keeps_what_the_model_failed_to_attribute():
    """네 스템 합이 아니라 원본 빼기 드럼이라 잔차가 사라지지 않는다."""
    signal = _signal()

    def lossy(model_mix, sample_rate_hz):
        # 실제 htdemucs 처럼 일부를 어느 스템에도 넣지 못한 상황
        return model_mix * 0.3

    stems = separate_stems(signal, backend=lossy)
    assert float(np.sqrt((stems.no_drums**2).mean())) > 0
    assert float(np.abs(stems.drums + stems.no_drums - signal.samples).max()) <= (
        RECONSTRUCTION_TOLERANCE
    )


def test_no_resampling_when_the_rates_already_match():
    signal = _signal(frames=44_100, sample_rate=MODEL_SAMPLE_RATE_HZ)
    stems = separate_stems(signal, backend=_half_backend)
    assert stems.frame_count == 44_100
    assert np.allclose(stems.drums, signal.samples * 0.5)


@pytest.mark.parametrize("delta", [-137, 241])
def test_length_drift_from_resampling_is_corrected(delta):
    signal = _signal()

    def drifting(model_mix, sample_rate_hz):
        drums = model_mix * 0.5
        if delta < 0:
            return drums[:delta]
        return np.vstack([drums, np.zeros((delta, drums.shape[1]), dtype=drums.dtype)])

    stems = separate_stems(signal, backend=drifting)
    assert stems.frame_count == FRAMES
    assert stems.no_drums.shape[0] == FRAMES


def test_backend_failure_becomes_a_worker_error():
    def backend(model_mix, sample_rate_hz):
        raise RuntimeError("out of memory")

    with pytest.raises(WorkerError) as caught:
        separate_stems(_signal(), backend=backend)
    assert caught.value.code is ErrorCode.STEMS_SEPARATION_FAILED
    assert caught.value.retryable is True


@pytest.mark.parametrize(
    "bad",
    [
        lambda mix, sr: mix[:, :1],
        lambda mix, sr: mix[:, 0],
    ],
)
def test_a_wrongly_shaped_stem_is_rejected(bad):
    with pytest.raises(WorkerError) as caught:
        separate_stems(_signal(), backend=bad)
    assert caught.value.code is ErrorCode.STEMS_SEPARATION_FAILED


def test_stems_expose_audio_signals_for_downstream_analysis():
    stems = separate_stems(_signal(), backend=_half_backend)
    drums = stems.drums_signal()
    assert isinstance(drums, AudioSignal)
    assert drums.sample_rate_hz == SAMPLE_RATE
    assert drums.duration_ms == 1000
    assert stems.no_drums_signal().frame_count == FRAMES


def test_reconstruction_guard_rejects_a_broken_stem_set():
    signal = _signal()
    broken = Stems(
        drums=signal.samples * 0.5,
        no_drums=signal.samples * 0.9,
        sample_rate_hz=SAMPLE_RATE,
        model_name="fake",
    )
    from chart_worker.analysis.stems import _require_reconstruction

    with pytest.raises(WorkerError, match="differ from the mix"):
        _require_reconstruction(broken, signal.samples)
