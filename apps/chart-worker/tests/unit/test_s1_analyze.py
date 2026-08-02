from pathlib import Path

import numpy as np

from chart_worker.analysis.audio_io import AudioSignal
from chart_worker.analysis.onset import OnsetAnalysis
from chart_worker.analysis.timing import TimingSource
from chart_worker.audio.normalize import NormalizedAudio
from chart_worker.config import WorkerConfig
from chart_worker.stages.s1_analyze import run_analysis

SHA = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"


def test_run_analysis_writes_normalized_audio_and_timing_osu(tmp_path: Path):
    source = tmp_path / "입력 곡.wav"
    source.write_bytes(b"source")
    calls = []

    def normalizer(source_path, target_path, *, config, run):
        calls.append((source_path, target_path, config, run))
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_bytes(b"normalized")
        return NormalizedAudio(
            path=target_path,
            profile_version="audio-profile-v1",
            sha256=SHA,
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
        )

    signal = AudioSignal(np.zeros((96_000, 2)), 48_000)

    def audio_loader(path):
        assert path == tmp_path / "audio" / "game.flac"
        return signal

    def beat_backend(samples, sample_rate):
        assert samples.shape == (96_000,)
        assert sample_rate == 48_000
        return np.array([0.0, 0.5, 1.0, 1.5]), np.array([0.0, 1.0])

    def onset_backend(samples, sample_rate):
        return OnsetAnalysis(
            sample_rate_hz=sample_rate,
            hop_length=512,
            strength=np.array([0.0, 1.0]),
            band_strength=np.zeros((3, 2)),
            onset_ms=(500,),
        )

    config = WorkerConfig(ffmpeg_bin=Path("ffmpeg"))
    result = run_analysis(
        source,
        tmp_path,
        config=config,
        run=None,
        normalizer=normalizer,
        audio_loader=audio_loader,
        beat_backend=beat_backend,
        onset_backend=onset_backend,
    )

    assert len(calls) == 1
    assert result.normalized.path == tmp_path / "audio" / "game.flac"
    assert result.signal is signal
    assert result.beat_grid.bpm == 120.0
    assert result.onsets.onset_ms == (500,)
    assert result.timing_candidate.source is TimingSource.BEAT_THIS_PIECEWISE
    assert result.timing_candidate.points[0].time_ms == 0
    assert result.timing_quality_report_path == tmp_path / "analysis" / "timing-quality-v1.json"
    timing = result.timing_osu_path.read_text(encoding="utf-8")
    assert timing.startswith("osu file format v14")
    assert "AudioFilename: game.flac" in timing
    assert "Title:입력 곡" in timing
