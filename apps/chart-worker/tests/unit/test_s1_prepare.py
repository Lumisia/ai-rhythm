from pathlib import Path

from chart_worker.audio.normalize import NormalizedAudio
from chart_worker.config import WorkerConfig
from chart_worker.stages.s1_prepare import run_prepare

SHA = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"


def test_prepare_only_normalizes_the_source_audio(tmp_path: Path):
    source = tmp_path / "source.wav"
    source.write_bytes(b"source")
    calls = []

    def normalizer(source_path, target_path, *, config, run):
        calls.append((source_path, target_path, config, run))
        target_path.parent.mkdir(parents=True)
        target_path.write_bytes(b"normalized")
        return NormalizedAudio(
            path=target_path,
            profile_version="audio-profile-v2",
            sha256=SHA,
            duration_ms=2_000,
            sample_rate_hz=48_000,
            channels=2,
            source_duration_ms=2_100,
            trimmed_ms=100,
            gain_db=0.0,
            achieved_lufs=-14.0,
            achieved_true_peak_dbtp=-1.0,
            shortfall_lu=0.0,
            limited_by="LOUDNESS",
        )

    config = WorkerConfig()
    marker = object()
    prepared = run_prepare(
        source,
        tmp_path,
        config=config,
        run=marker,
        normalizer=normalizer,
    )

    assert prepared.normalized.path == tmp_path / "audio" / "game.flac"
    assert calls == [
        (source, tmp_path / "audio" / "game.flac", config, marker),
    ]
    assert not (tmp_path / "analysis").exists()
