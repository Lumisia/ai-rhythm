from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

import numpy as np
import soundfile as sf

from chart_worker.analysis.audio_io import AudioSignal
from chart_worker.analysis.beat import BeatGrid
from chart_worker.analysis.onset import OnsetAnalysis
from chart_worker.audio.normalize import NormalizedAudio
from chart_worker.config import WorkerConfig
from chart_worker.generation.timing_osu import beat_grid_to_timing_osu
from chart_worker.hashing import sha256_file
from chart_worker.pipeline import PipelineDependencies
from chart_worker.stages.types import AnalysisStageResult


def fake_analysis(source: Path, run_dir: Path, config: WorkerConfig) -> AnalysisStageResult:
    del config
    audio_path = run_dir / "audio" / "game.flac"
    audio_path.parent.mkdir(parents=True, exist_ok=True)
    samples = np.zeros((96_000, 2))
    sf.write(str(audio_path), samples, 48_000, format="FLAC", subtype="PCM_16")
    grid = BeatGrid(
        beat_ms=(0, 500, 1_000, 1_500),
        downbeat_indices=(0,),
        bpm=120.0,
        beats_per_bar=4,
        bpm_drift_pct=0.0,
        raw_beat_count=4,
        dropped_beat_count=0,
        residual_rms_ms=0.0,
        residual_max_ms=0.0,
    )
    timing_path = run_dir / "analysis" / "timing.osu"
    timing_path.parent.mkdir(parents=True, exist_ok=True)
    timing_path.write_text(
        beat_grid_to_timing_osu(
            grid,
            audio_filename=audio_path.name,
            title=source.stem,
        ),
        encoding="utf-8",
    )
    return AnalysisStageResult(
        normalized=NormalizedAudio(
            path=audio_path,
            profile_version="audio-profile-v1",
            sha256=sha256_file(audio_path),
            duration_ms=2_000,
            sample_rate_hz=48_000,
            channels=2,
            source_duration_ms=2_000,
            trimmed_ms=0,
            gain_db=0.0,
            achieved_lufs=-14.0,
            achieved_true_peak_dbtp=-1.0,
            shortfall_lu=0.0,
            limited_by="LOUDNESS",
        ),
        signal=AudioSignal(samples, 48_000),
        beat_grid=grid,
        onsets=OnsetAnalysis(
            sample_rate_hz=48_000,
            hop_length=512,
            strength=np.ones(200),
            band_strength=np.ones((3, 200)),
            onset_ms=(500, 1_000, 1_500),
        ),
        timing_osu_path=timing_path,
    )


def fake_dependencies() -> PipelineDependencies:
    return PipelineDependencies(
        config=WorkerConfig(),
        analysis=fake_analysis,
        now=lambda: datetime(2026, 8, 2, tzinfo=UTC),
        new_run_id=lambda: UUID("00000000-0000-0000-0000-000000000007"),
    )
