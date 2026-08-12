from pathlib import Path

import pytest
from pydantic import ValidationError

from chart_worker.config import WorkerConfig, load_config


def test_defaults_to_mapperatorinator_generator(monkeypatch):
    monkeypatch.delenv("CHART_GENERATOR", raising=False)
    assert load_config().chart_generator == "mapperatorinator"


def test_reads_mapperatorinator_paths(monkeypatch):
    monkeypatch.setenv("CHART_GENERATOR", "mapperatorinator")
    monkeypatch.setenv("MAPPERATORINATOR_HOME", r"C:\Users\PC\Mapperatorinator")
    cfg = load_config()
    assert cfg.mapperatorinator_home == Path(r"C:\Users\PC\Mapperatorinator")


def test_precision_defaults_to_fp16(monkeypatch):
    monkeypatch.delenv("MAPPERATORINATOR_PRECISION", raising=False)
    assert load_config().mapperatorinator_precision == "fp16"


@pytest.mark.parametrize("precision", ["fp16", "bf16"])
def test_supported_mapperatorinator_precisions_are_accepted(precision):
    assert WorkerConfig(mapperatorinator_precision=precision).mapperatorinator_precision == precision


def test_unknown_mapperatorinator_precision_is_rejected():
    with pytest.raises(ValidationError):
        WorkerConfig(mapperatorinator_precision="fp32")


def test_difficulty_selector_defaults_to_v2_after_regression_gate_passes(monkeypatch):
    monkeypatch.delenv("DIFFICULTY_SELECTOR_MODE", raising=False)

    assert load_config().difficulty_selector_mode == "V2"


@pytest.mark.parametrize("mode", ["CURRENT", "SHADOW_V2", "V2"])
def test_supported_difficulty_selector_modes_are_accepted(mode):
    assert WorkerConfig(difficulty_selector_mode=mode).difficulty_selector_mode == mode


def test_unknown_difficulty_selector_mode_is_rejected():
    with pytest.raises(ValidationError):
        WorkerConfig(difficulty_selector_mode="AUTO")


def test_boundary_policy_defaults_to_shadow(monkeypatch):
    monkeypatch.delenv("BOUNDARY_POLICY_MODE", raising=False)

    assert load_config().boundary_policy_mode == "SHADOW"


@pytest.mark.parametrize("mode", ["SHADOW", "EXPERIMENTAL_ENFORCED"])
def test_supported_boundary_policy_modes_are_accepted(mode):
    assert WorkerConfig(boundary_policy_mode=mode).boundary_policy_mode == mode


def test_unknown_boundary_policy_mode_is_rejected():
    with pytest.raises(ValidationError):
        WorkerConfig(boundary_policy_mode="ACTIVE")


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
