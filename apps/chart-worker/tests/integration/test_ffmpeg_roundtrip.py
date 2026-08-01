"""실제 FFmpeg 로 도는 표준화 왕복 검증.

테스트 오디오를 ffmpeg 로 합성한다. 저장소에 바이너리를 넣지 않고도
트림과 게인이 실제로 먹었는지 확인할 수 있다.
"""

import shutil
from pathlib import Path

import pytest

from chart_worker.audio import profile
from chart_worker.audio.normalize import normalize_audio
from chart_worker.audio.runner import run_command
from chart_worker.config import WorkerConfig
from chart_worker.errors import ErrorCode, WorkerError

pytestmark = pytest.mark.ffmpeg


@pytest.fixture(scope="module")
def config():
    settings = WorkerConfig(ffmpeg_bin=Path("ffmpeg"), ffmpeg_shared_bin_dir=None)
    for binary in (settings.ffmpeg_bin, settings.ffprobe_bin):
        if shutil.which(str(binary)) is None:
            pytest.skip(f"{binary} is not installed")
    return settings


def _synthesize(config: WorkerConfig, target: Path, source_filter: str, filters: str) -> Path:
    run_command(
        [
            str(config.ffmpeg_bin),
            "-hide_banner",
            "-v",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            source_filter,
            "-af",
            filters,
            "-ac",
            "2",
            str(target),
        ]
    )
    return target


@pytest.fixture(scope="module")
def quiet_source(tmp_path_factory, config):
    """앞 무음 2초 + 440Hz 사인 3초, -20 dBFS, 44.1 kHz.

    44.1 kHz 로 만들어 48 kHz 리샘플까지 함께 확인한다.
    """
    target = tmp_path_factory.mktemp("audio") / "source.wav"
    return _synthesize(
        config,
        target,
        "sine=frequency=440:duration=3:sample_rate=44100",
        "adelay=2000,volume=-20dB",
    )


def test_roundtrip_trims_silence_and_pins_the_profile(config, quiet_source, tmp_path):
    result = normalize_audio(quiet_source, tmp_path / "game.flac", config=config)

    assert result.sample_rate_hz == profile.SAMPLE_RATE_HZ
    assert result.channels == profile.CHANNELS
    assert result.profile_version == profile.PROFILE_VERSION
    assert result.source_duration_ms == pytest.approx(5000, abs=50)
    assert result.duration_ms == pytest.approx(3000, abs=150)
    assert result.trimmed_ms == pytest.approx(2000, abs=150)


def test_roundtrip_reaches_the_target_loudness(config, quiet_source, tmp_path):
    result = normalize_audio(quiet_source, tmp_path / "game.flac", config=config)

    assert result.gain_db > 0
    assert result.limited_by == "LOUDNESS"
    assert result.achieved_lufs == pytest.approx(profile.TARGET_LUFS, abs=0.01)
    assert result.shortfall_lu == 0.0


def test_output_is_byte_identical_across_runs(config, quiet_source, tmp_path):
    """-bitexact 가 빠지면 vendor string 때문에 여기서 깨진다."""
    first = normalize_audio(quiet_source, tmp_path / "a.flac", config=config)
    second = normalize_audio(quiet_source, tmp_path / "b.flac", config=config)

    assert first.sha256 == second.sha256
    assert len(first.sha256) == 64
    assert (tmp_path / "a.flac").read_bytes() == (tmp_path / "b.flac").read_bytes()


def test_output_really_is_flac(config, quiet_source, tmp_path):
    target = normalize_audio(quiet_source, tmp_path / "game.flac", config=config).path
    assert target.read_bytes()[:4] == b"fLaC"


def test_silent_input_is_rejected(config, tmp_path):
    silent = _synthesize(
        config,
        tmp_path / "silent.wav",
        "anullsrc=r=44100:cl=mono:duration=2",
        "acopy",
    )
    with pytest.raises(WorkerError) as caught:
        normalize_audio(silent, tmp_path / "game.flac", config=config)
    assert caught.value.code is ErrorCode.AUDIO_SILENT


def test_non_audio_input_is_rejected(config, tmp_path):
    broken = tmp_path / "broken.mp3"
    broken.write_bytes(b"not audio at all")
    with pytest.raises(WorkerError) as caught:
        normalize_audio(broken, tmp_path / "game.flac", config=config)
    assert caught.value.code is ErrorCode.AUDIO_INVALID
