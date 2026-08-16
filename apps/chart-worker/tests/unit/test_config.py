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


def test_hold_state_mode_defaults_to_incremental(monkeypatch):
    monkeypatch.delenv("MAPPERATORINATOR_HOLD_STATE_MODE", raising=False)

    assert load_config().mapperatorinator_hold_state_mode == "incremental"


@pytest.mark.parametrize("mode", ["full_scan", "incremental_verify", "incremental"])
def test_supported_hold_state_modes_are_accepted_from_environment(monkeypatch, mode):
    monkeypatch.setenv("MAPPERATORINATOR_HOLD_STATE_MODE", mode)

    assert load_config().mapperatorinator_hold_state_mode == mode


def test_unknown_hold_state_mode_is_rejected():
    with pytest.raises(ValidationError):
        WorkerConfig(mapperatorinator_hold_state_mode="fast")


def test_resident_inference_defaults_are_fail_closed():
    config = WorkerConfig()

    assert config.mapperatorinator_backend == "oneshot"
    assert config.mapperatorinator_write_generation_telemetry is False
    assert config.mapperatorinator_tail_repairs == 2
    assert config.mapperatorinator_checkpoint_interval_windows == 8
    assert config.mapperatorinator_protocol_max_line_bytes == 1_048_576
    assert config.mapperatorinator_resident_startup_timeout_sec == 1800.0
    assert config.mapperatorinator_resident_invocation_timeout_sec == 10800.0
    assert config.mapperatorinator_resident_close_timeout_sec == 5.0
    assert config.mapperatorinator_model_root is None
    assert config.mapperatorinator_model_revision is None


def test_generation_telemetry_can_be_enabled_explicitly_from_environment(monkeypatch):
    monkeypatch.setenv("MAPPERATORINATOR_WRITE_GENERATION_TELEMETRY", "true")

    assert load_config().mapperatorinator_write_generation_telemetry is True


def test_song_session_requires_incremental_mode_and_immutable_model_identity():
    with pytest.raises(ValidationError):
        WorkerConfig(mapperatorinator_backend="song_session")

    config = WorkerConfig(
        mapperatorinator_backend="song_session",
        mapperatorinator_hold_state_mode="incremental",
        mapperatorinator_home=Path(r"C:\Mapperatorinator"),
        mapperatorinator_python=Path(r"C:\Mapperatorinator\.venv\Scripts\python.exe"),
        mapperatorinator_model_root=Path(r"C:\models\mapperatorinator-v32"),
        mapperatorinator_model_revision="74f22583400d259bf424819e11027c17933efe54",
    )

    assert config.mapperatorinator_backend == "song_session"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("mapperatorinator_backend", "daemon"),
        ("mapperatorinator_tail_repairs", 3),
        ("mapperatorinator_checkpoint_interval_windows", 4),
        ("mapperatorinator_protocol_max_line_bytes", 2_000_000),
        ("mapperatorinator_model_revision", "main"),
        ("mapperatorinator_resident_startup_timeout_sec", 0.0),
        ("mapperatorinator_resident_invocation_timeout_sec", 10_800.1),
        ("mapperatorinator_resident_close_timeout_sec", 30.1),
    ],
)
def test_resident_inference_contract_rejects_floating_or_changed_values(field, value):
    with pytest.raises(ValidationError):
        WorkerConfig(**{field: value})


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
