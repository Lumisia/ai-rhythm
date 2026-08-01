from pathlib import Path

import pytest

from chart_worker.config import load_config


def test_defaults_to_fake_generator(monkeypatch):
    monkeypatch.delenv("CHART_GENERATOR", raising=False)
    assert load_config().chart_generator == "fake"


def test_reads_mapperatorinator_paths(monkeypatch):
    monkeypatch.setenv("CHART_GENERATOR", "mapperatorinator")
    monkeypatch.setenv("MAPPERATORINATOR_HOME", r"C:\Users\PC\Mapperatorinator")
    cfg = load_config()
    assert cfg.mapperatorinator_home == Path(r"C:\Users\PC\Mapperatorinator")


def test_precision_defaults_to_fp16(monkeypatch):
    monkeypatch.delenv("MAPPERATORINATOR_PRECISION", raising=False)
    assert load_config().mapperatorinator_precision == "fp16"


@pytest.mark.parametrize(
    ("ffmpeg_bin", "expected"),
    [
        (r"C:\ffmpeg\bin\ffmpeg.exe", Path(r"C:\ffmpeg\bin\ffprobe.exe")),
        ("ffmpeg", Path("ffprobe")),
    ],
)
def test_ffprobe_is_derived_from_ffmpeg(monkeypatch, ffmpeg_bin, expected):
    """따로 설정하게 두면 두 값이 서로 다른 빌드를 가리키는 사고가 난다."""
    monkeypatch.setenv("FFMPEG_BIN", ffmpeg_bin)
    assert load_config().ffprobe_bin == expected
