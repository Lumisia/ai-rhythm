from pathlib import Path

import numpy as np

from chart_worker.analysis.audio_io import AudioSignal, load_audio
from chart_worker.analysis.beat import BeatGrid
from chart_worker.analysis.onset import OnsetAnalysis
from chart_worker.audio.normalize import NormalizedAudio
from chart_worker.stages.s3_stems import run_stems
from chart_worker.stages.types import AnalysisStageResult

SHA = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"


def _analysis(tmp_path: Path) -> AnalysisStageResult:
    audio_path = tmp_path / "audio" / "game.flac"
    audio_path.parent.mkdir(parents=True)
    audio_path.write_bytes(b"game")
    signal = AudioSignal(np.zeros((4_800, 2), dtype=np.float64), 48_000)
    return AnalysisStageResult(
        normalized=NormalizedAudio(
            audio_path,
            "audio-profile-v1",
            SHA,
            100,
            48_000,
            2,
            100,
            0,
            0.0,
            -14.0,
            -1.0,
            0.0,
            "LOUDNESS",
        ),
        signal=signal,
        beat_grid=BeatGrid((0, 50), (0,), 1_200.0, 4, 0.0, 2, 0, 0.0, 0.0),
        onsets=OnsetAnalysis(48_000, 512, np.ones(2), np.ones((3, 2)), (25,)),
        timing_osu_path=tmp_path / "analysis" / "timing.osu",
    )


def test_disabled_stems_only_returns_game_audio(tmp_path: Path):
    result = run_stems(_analysis(tmp_path), tmp_path, enabled=False)

    assert result.game_ref.path == "audio/game.flac"
    assert result.no_drums_ref is None
    assert result.keys_ref is None
    assert result.drum_onsets == ()
    assert result.keysound_manifest is None
    assert result.keysound_manifest_path is None


def test_enabled_stems_writes_two_lossless_files_and_drum_onsets(tmp_path: Path):
    analysis = _analysis(tmp_path)

    def stem_backend(model_mix, sample_rate):
        assert sample_rate == 44_100
        return np.full_like(model_mix, 0.1, dtype=np.float32)

    def onset_backend(mono, sample_rate):
        assert sample_rate == 48_000
        return OnsetAnalysis(
            sample_rate_hz=sample_rate,
            hop_length=512,
            strength=np.array([0.0, 1.0]),
            band_strength=np.zeros((3, 2)),
            onset_ms=(70, 20, 70),
        )

    result = run_stems(
        analysis,
        tmp_path,
        enabled=True,
        stem_backend=stem_backend,
        onset_backend=onset_backend,
    )

    assert result.no_drums_ref is not None
    assert result.keys_ref is not None
    no_drums_path = tmp_path / result.no_drums_ref.path
    keys_path = tmp_path / result.keys_ref.path
    assert no_drums_path.is_file()
    assert keys_path.is_file()
    assert load_audio(no_drums_path).frame_count == analysis.signal.frame_count
    assert load_audio(keys_path).frame_count == analysis.signal.frame_count
    assert result.drum_onsets == (20, 70)
    assert result.keysound_manifest is not None
    assert result.keysound_manifest.drum_onsets == [20, 70]
    assert result.keysound_manifest_path == tmp_path / "keysound-manifest.json"
    assert '"drumOnsets"' in result.keysound_manifest_path.read_text(encoding="utf-8")
