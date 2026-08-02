"""Demucs 스템 분리 — 감산 키음의 재료.

BGM 은 no_drums 로 계속 재생하고, 키음은 drums 조각을 노트를 칠 때만
울린다. 놓치면 그 드럼이 안 난다.
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from chart_worker.analysis.audio_io import AudioSignal
from chart_worker.errors import ErrorCode, WorkerError

DEMUCS_MODEL = "htdemucs"
DRUM_SOURCE = "drums"
MODEL_SAMPLE_RATE_HZ = 44_100
"""htdemucs 가 학습된 샘플레이트. GAME_AUDIO 의 48 kHz 와 다르다."""

RESAMPLE_QUALITY = "VHQ"
RECONSTRUCTION_TOLERANCE = 1e-6
"""drums + no_drums 가 원본과 어긋나면 감산 키음의 전제가 깨진다."""

StemBackend = Callable[[np.ndarray, int], np.ndarray]
"""(모델 샘플레이트의 믹스, 샘플레이트) -> 같은 모양의 드럼 스템."""


@dataclass(frozen=True, slots=True)
class Stems:
    drums: np.ndarray
    """(프레임, 채널). 원본 샘플레이트로 되돌린 드럼."""

    no_drums: np.ndarray
    """원본에서 드럼을 뺀 나머지. 분리 잔차가 여기 남는다."""

    sample_rate_hz: int
    model_name: str

    @property
    def frame_count(self) -> int:
        return int(self.drums.shape[0])

    def drums_signal(self) -> AudioSignal:
        return AudioSignal(samples=self.drums, sample_rate_hz=self.sample_rate_hz)

    def no_drums_signal(self) -> AudioSignal:
        return AudioSignal(samples=self.no_drums, sample_rate_hz=self.sample_rate_hz)


def _fit_length(samples: np.ndarray, frames: int) -> np.ndarray:
    """리샘플 왕복에서 생긴 길이 차를 원본에 맞춘다."""
    if samples.shape[0] == frames:
        return samples
    if samples.shape[0] > frames:
        return samples[:frames]
    pad = np.zeros((frames - samples.shape[0], samples.shape[1]), dtype=samples.dtype)
    return np.vstack([samples, pad])


def _resample(samples: np.ndarray, source_hz: int, target_hz: int) -> np.ndarray:
    if source_hz == target_hz:
        return samples
    import soxr

    return soxr.resample(samples, source_hz, target_hz, quality=RESAMPLE_QUALITY)


def separate_stems(
    signal: AudioSignal,
    *,
    backend: StemBackend,
    model_name: str = DEMUCS_MODEL,
    model_sample_rate_hz: int = MODEL_SAMPLE_RATE_HZ,
) -> Stems:
    """드럼과 나머지로 가른다.

    나머지를 bass + other + vocals 로 합치지 않는다. htdemucs 의 네 스템은
    원본으로 정확히 복원되지 않아(실측 오차 rms 3.3e-03) 합치면 그 잔차가
    사라진다. `원본 - drums` 로 만들면 **drums + no_drums 가 원본과 같아져**
    모든 노트를 친 플레이어가 원곡 그대로를 듣는다. 분리 실패분도 BGM 에
    남아 계속 들린다.
    """
    mix = np.asarray(signal.samples, dtype=np.float32)
    frames = mix.shape[0]

    model_mix = _resample(mix, signal.sample_rate_hz, model_sample_rate_hz)
    try:
        drums_at_model_rate = backend(model_mix, model_sample_rate_hz)
    except Exception as error:
        raise WorkerError(
            ErrorCode.STEMS_SEPARATION_FAILED,
            f"demucs failed: {error}",
            context={"model": model_name},
        ) from error

    drums = np.asarray(drums_at_model_rate, dtype=np.float32)
    if drums.ndim != 2 or drums.shape[1] != mix.shape[1]:
        raise WorkerError(
            ErrorCode.STEMS_SEPARATION_FAILED,
            f"drum stem has shape {drums.shape}, expected (frames, {mix.shape[1]})",
        )

    drums = _fit_length(_resample(drums, model_sample_rate_hz, signal.sample_rate_hz), frames)
    stems = Stems(
        drums=drums,
        no_drums=mix - drums,
        sample_rate_hz=signal.sample_rate_hz,
        model_name=model_name,
    )
    _require_reconstruction(stems, mix)
    return stems


def _require_reconstruction(stems: Stems, mix: np.ndarray) -> None:
    """감산 키음의 전제를 확인한다."""
    error = float(np.abs(stems.drums + stems.no_drums - mix).max())
    if error > RECONSTRUCTION_TOLERANCE:
        raise WorkerError(
            ErrorCode.STEMS_SEPARATION_FAILED,
            f"drums + no_drums differ from the mix by {error}",
            context={"max_abs_error": error},
        )


def demucs_backend(
    *,
    model_name: str = DEMUCS_MODEL,
    device: str = "cpu",
    overlap: float = 0.25,
    shifts: int = 1,
) -> StemBackend:
    """htdemucs 로 드럼 스템만 뽑는다.

    apply_model 에 텐서를 직접 넘긴다. demucs 의 파일 입출력을 쓰면
    torchaudio 를 거쳐 ffmpeg 의존이 생긴다.
    """
    import torch
    from demucs.apply import apply_model
    from demucs.pretrained import get_model

    model = get_model(model_name)
    model.eval()
    try:
        drum_index = list(model.sources).index(DRUM_SOURCE)
    except ValueError:
        raise ValueError(f"{model_name} has no '{DRUM_SOURCE}' source: {model.sources}") from None

    def run(model_mix: np.ndarray, sample_rate_hz: int) -> np.ndarray:
        # demucs 는 (배치, 채널, 프레임) 을 받는다.
        tensor = torch.from_numpy(np.ascontiguousarray(model_mix.T)).unsqueeze(0)
        with torch.no_grad():
            separated = apply_model(
                model,
                tensor,
                device=device,
                split=True,
                overlap=overlap,
                shifts=shifts,
                progress=False,
            )
        return separated[0, drum_index].numpy().T

    return run
