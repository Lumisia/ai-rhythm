from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

import numpy as np
import pytest

from chart_worker.analysis.audio_io import AudioSignal
from chart_worker.analysis.beat import BeatGrid
from chart_worker.analysis.onset import OnsetAnalysis
from chart_worker.audio.normalize import NormalizedAudio
from chart_worker.config import WorkerConfig
from chart_worker.generation.timing_osu import beat_grid_to_timing_osu
from chart_worker.hashing import sha256_file
from chart_worker.pipeline import PipelineDependencies, PipelineOptions, run_pipeline
from chart_worker.schema.playtest_run import PlaytestRunManifest
from chart_worker.stages.types import AnalysisStageResult


def _fake_analysis(source: Path, run_dir: Path, config: WorkerConfig) -> AnalysisStageResult:
    del config
    audio_path = run_dir / "audio" / "game.flac"
    audio_path.parent.mkdir(parents=True, exist_ok=True)
    audio_path.write_bytes(b"normalized-game-audio")
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
        signal=AudioSignal(np.zeros((96_000, 2)), 48_000),
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


def _dependencies() -> PipelineDependencies:
    return PipelineDependencies(
        config=WorkerConfig(),
        analysis=_fake_analysis,
        now=lambda: datetime(2026, 8, 2, tzinfo=UTC),
        new_run_id=lambda: UUID("00000000-0000-0000-0000-000000000007"),
    )


def test_fake_pipeline_writes_playtest_manifest(tmp_path: Path):
    source = tmp_path / "fixture.wav"
    source.write_bytes(b"source")
    output_dir = tmp_path / "run"

    result = run_pipeline(
        PipelineOptions(
            source=source,
            output_dir=output_dir,
            title="fixture",
            generator="fake",
            keysounds=False,
            seed=7,
        ),
        dependencies=_dependencies(),
    )

    manifest = PlaytestRunManifest.model_validate_json(
        result.manifest_path.read_text(encoding="utf-8")
    )
    assert len(manifest.charts) == 12
    assert len({(chart.key_mode, chart.difficulty) for chart in manifest.charts}) == 12
    assert all((output_dir / chart.path).is_file() for chart in manifest.charts)
    assert len(result.raw_osu_paths) == 12
    assert set(result.elapsed_ms_by_stage) == {"analysis", "generation", "stems", "postprocess"}
    assert (output_dir / manifest.generation_report_path).is_file()


def test_pipeline_refuses_a_nonempty_output_without_removing_it(tmp_path: Path):
    source = tmp_path / "fixture.wav"
    source.write_bytes(b"source")
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    marker = output_dir / "keep.txt"
    marker.write_text("keep", encoding="utf-8")

    with pytest.raises(ValueError, match="not empty"):
        run_pipeline(
            PipelineOptions(source=source, output_dir=output_dir, title="fixture"),
            dependencies=_dependencies(),
        )

    assert marker.read_text(encoding="utf-8") == "keep"
