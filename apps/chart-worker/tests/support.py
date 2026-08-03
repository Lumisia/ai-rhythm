from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

import numpy as np
import soundfile as sf

from chart_worker.audio.normalize import NormalizedAudio
from chart_worker.config import WorkerConfig
from chart_worker.generation.fake import FakeGenerator
from chart_worker.hashing import sha256_file
from chart_worker.pipeline import PipelineDependencies
from chart_worker.stages.types import PreparedAudio


def _write_prepared_audio(
    run_dir: Path,
    samples: np.ndarray,
    *,
    sample_rate_hz: int,
) -> PreparedAudio:
    audio_path = run_dir / "audio" / "game.flac"
    audio_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(audio_path), samples, sample_rate_hz, format="FLAC", subtype="PCM_16")
    duration_ms = round(len(samples) * 1_000 / sample_rate_hz)
    return PreparedAudio(
        NormalizedAudio(
            path=audio_path,
            profile_version="audio-profile-v1",
            sha256=sha256_file(audio_path),
            duration_ms=duration_ms,
            sample_rate_hz=sample_rate_hz,
            channels=2,
            source_duration_ms=duration_ms,
            trimmed_ms=0,
            gain_db=0.0,
            achieved_lufs=-14.0,
            achieved_true_peak_dbtp=-1.0,
            shortfall_lu=0.0,
            limited_by="LOUDNESS",
        )
    )


def _dependencies(samples: np.ndarray, sample_rate_hz: int) -> PipelineDependencies:
    def prepare(source: Path, run_dir: Path, config: WorkerConfig) -> PreparedAudio:
        del source, config
        return _write_prepared_audio(
            run_dir,
            samples,
            sample_rate_hz=sample_rate_hz,
        )

    return PipelineDependencies(
        config=WorkerConfig(),
        prepare=prepare,
        select_generator=lambda _name, _config: FakeGenerator(),
        now=lambda: datetime(2026, 8, 2, tzinfo=UTC),
        new_run_id=lambda: UUID("00000000-0000-0000-0000-000000000007"),
    )


def fake_dependencies() -> PipelineDependencies:
    return _dependencies(np.zeros((96_000, 2)), 48_000)


def contract_fixture_dependencies() -> PipelineDependencies:
    sample_rate_hz = 48_000
    duration_ms = 8_000
    frames = np.arange(sample_rate_hz * duration_ms // 1_000, dtype=np.float64)
    samples = np.column_stack(
        (
            0.08 * np.sin(2.0 * np.pi * 220.0 * frames / sample_rate_hz),
            0.08 * np.sin(2.0 * np.pi * 330.0 * frames / sample_rate_hz),
        )
    )
    return _dependencies(samples, sample_rate_hz)
