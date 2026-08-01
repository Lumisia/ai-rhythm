import hashlib
import json
from pathlib import Path

import pytest

from chart_worker.audio import profile
from chart_worker.audio.normalize import normalize_audio
from chart_worker.audio.runner import CommandError, CommandResult
from chart_worker.config import WorkerConfig
from chart_worker.errors import Disposition, ErrorCode, WorkerError

FLAC_BYTES = b"fLaC-not-really-but-hashable"
FLAC_SHA = hashlib.sha256(FLAC_BYTES).hexdigest()


def _probe_json(*, duration_ts=144_000, sample_rate=48_000, channels=2, codec="flac"):
    return json.dumps(
        {
            "streams": [
                {
                    "codec_type": "audio",
                    "codec_name": codec,
                    "sample_rate": str(sample_rate),
                    "channels": channels,
                    "time_base": f"1/{sample_rate}",
                    "duration_ts": duration_ts,
                }
            ],
            "format": {},
        }
    )


def _loudnorm_stderr(input_i="-27.61", input_tp="-20.0"):
    payload = {
        "input_i": input_i,
        "input_tp": input_tp,
        "input_lra": "8.00",
        "input_thresh": "-37.00",
        "output_i": "-14.00",
        "normalization_type": "dynamic",
    }
    return f"frame= 100 time=00:00:03.00\n[Parsed_loudnorm_1 @ 0]\n{json.dumps(payload)}\n"


class FakeFfmpeg:
    """argv 모양으로 어떤 단계인지 판별하는 가짜 러너."""

    def __init__(
        self,
        *,
        source_json=None,
        target_json=None,
        measure_stderr=None,
        fail_on=None,
        write_output=True,
    ):
        self.source_json = source_json or _probe_json(duration_ts=44_100 * 5, sample_rate=44_100)
        self.target_json = target_json or _probe_json()
        self.measure_stderr = measure_stderr if measure_stderr is not None else _loudnorm_stderr()
        self.fail_on = fail_on
        self.write_output = write_output
        self.calls: list[list[str]] = []

    def _stage(self, argv: list[str]) -> str:
        if "ffprobe" in Path(argv[0]).name:
            return "probe_target" if Path(argv[-1]).suffix == ".flac" else "probe_source"
        return "measure" if argv[-3:] == ["-f", "null", "-"] else "encode"

    def __call__(self, argv: list[str]) -> CommandResult:
        self.calls.append(list(argv))
        stage = self._stage(argv)
        if stage == self.fail_on:
            raise CommandError(argv, "exited with 1", returncode=1, stderr="boom")
        if stage == "probe_source":
            return CommandResult(argv, 0, self.source_json, "")
        if stage == "probe_target":
            return CommandResult(argv, 0, self.target_json, "")
        if stage == "measure":
            return CommandResult(argv, 0, "", self.measure_stderr)
        if self.write_output:
            Path(argv[-1]).write_bytes(FLAC_BYTES)
        return CommandResult(argv, 0, "", "")

    def command_for(self, stage: str) -> list[str]:
        return next(argv for argv in self.calls if self._stage(argv) == stage)


@pytest.fixture
def config():
    return WorkerConfig(ffmpeg_bin=Path("ffmpeg"), ffmpeg_shared_bin_dir=None)


@pytest.fixture
def paths(tmp_path):
    source = tmp_path / "in.mp3"
    source.write_bytes(b"source")
    return source, tmp_path / "out" / "game.flac"


def test_returns_the_measured_profile(config, paths):
    source, target = paths
    result = normalize_audio(source, target, config=config, run=FakeFfmpeg())
    assert result.path == target
    assert result.profile_version == profile.PROFILE_VERSION
    assert result.sha256 == FLAC_SHA
    assert (result.duration_ms, result.sample_rate_hz, result.channels) == (3000, 48000, 2)
    assert result.source_duration_ms == 5000
    assert result.trimmed_ms == 2000


def test_reports_the_gain_plan(config, paths):
    source, target = paths
    result = normalize_audio(source, target, config=config, run=FakeFfmpeg())
    assert result.gain_db == pytest.approx(13.61)
    assert result.achieved_lufs == pytest.approx(-14.0)
    assert result.shortfall_lu == 0.0
    assert result.limited_by == "LOUDNESS"


def test_passes_the_computed_gain_to_the_encoder(config, paths):
    source, target = paths
    fake = FakeFfmpeg()
    normalize_audio(source, target, config=config, run=fake)
    encode = fake.command_for("encode")
    assert "volume=13.610000dB" in encode[encode.index("-af") + 1]


def test_creates_the_target_directory(config, paths):
    source, target = paths
    assert not target.parent.exists()
    normalize_audio(source, target, config=config, run=FakeFfmpeg())
    assert target.exists()


def test_shortfall_is_reported_on_peaky_material(config, paths):
    source, target = paths
    fake = FakeFfmpeg(measure_stderr=_loudnorm_stderr(input_tp="-4.47"))
    result = normalize_audio(source, target, config=config, run=fake)
    assert result.limited_by == "TRUE_PEAK"
    assert result.shortfall_lu == pytest.approx(10.14)


def test_rejects_an_input_longer_than_the_defence_line(config, paths):
    source, target = paths
    long_source = _probe_json(duration_ts=48_000 * 601)
    fake = FakeFfmpeg(source_json=long_source)
    with pytest.raises(WorkerError) as caught:
        normalize_audio(source, target, config=config, run=fake)
    assert caught.value.code is ErrorCode.AUDIO_TOO_LONG
    assert source.exists(), "입력 파일은 우리 것이 아니다"
    assert [argv for argv in fake.calls if fake._stage(argv) == "measure"] == []


def test_rejects_output_longer_than_the_profile_limit_and_removes_it(config, paths):
    source, target = paths
    fake = FakeFfmpeg(target_json=_probe_json(duration_ts=48_000 * 185))
    with pytest.raises(WorkerError) as caught:
        normalize_audio(source, target, config=config, run=fake)
    assert caught.value.code is ErrorCode.AUDIO_TOO_LONG
    assert not target.exists(), "남겨두면 다음 실행이 유효한 산출물로 오인한다"


@pytest.mark.parametrize("stderr", [_loudnorm_stderr(input_i="-inf"), _loudnorm_stderr("-60.0")])
def test_silence_is_detected_from_the_measurement(config, paths, stderr):
    source, target = paths
    with pytest.raises(WorkerError) as caught:
        normalize_audio(source, target, config=config, run=FakeFfmpeg(measure_stderr=stderr))
    assert caught.value.code is ErrorCode.AUDIO_SILENT


def test_measurement_without_json_means_the_trim_took_everything(config, paths):
    """loudnorm 은 프레임을 하나라도 받으면 JSON 을 찍는다."""
    source, target = paths
    fake = FakeFfmpeg(measure_stderr="frame=    0 time=00:00:00.00\n")
    with pytest.raises(WorkerError) as caught:
        normalize_audio(source, target, config=config, run=fake)
    assert caught.value.code is ErrorCode.AUDIO_SILENT


@pytest.mark.parametrize(
    ("fail_on", "expected"),
    [
        ("probe_source", ErrorCode.AUDIO_INVALID),
        ("probe_target", ErrorCode.AUDIO_INVALID),
        ("measure", ErrorCode.AUDIO_NORMALIZATION_FAILED),
        ("encode", ErrorCode.AUDIO_NORMALIZATION_FAILED),
    ],
)
def test_each_stage_maps_to_its_documented_error_code(config, paths, fail_on, expected):
    source, target = paths
    with pytest.raises(WorkerError) as caught:
        normalize_audio(source, target, config=config, run=FakeFfmpeg(fail_on=fail_on))
    assert caught.value.code is expected


def test_probe_failure_is_final_but_ffmpeg_failure_is_retryable(config, paths):
    """코드를 잘못 붙이면 재시도 정책이 뒤집힌다."""
    source, target = paths
    with pytest.raises(WorkerError) as invalid:
        normalize_audio(source, target, config=config, run=FakeFfmpeg(fail_on="probe_source"))
    with pytest.raises(WorkerError) as failed:
        normalize_audio(source, target, config=config, run=FakeFfmpeg(fail_on="encode"))
    assert invalid.value.disposition is Disposition.FINAL
    assert failed.value.retryable is True


def test_failed_encoding_leaves_no_partial_file(config, paths):
    source, target = paths
    target.parent.mkdir(parents=True)
    target.write_bytes(b"stale")
    with pytest.raises(WorkerError):
        normalize_audio(source, target, config=config, run=FakeFfmpeg(fail_on="encode"))
    assert not target.exists()


def test_unparsable_probe_output_is_invalid_audio(config, paths):
    source, target = paths
    fake = FakeFfmpeg(source_json='{"streams": []}')
    with pytest.raises(WorkerError) as caught:
        normalize_audio(source, target, config=config, run=fake)
    assert caught.value.code is ErrorCode.AUDIO_INVALID


@pytest.mark.parametrize(
    "target_json",
    [
        _probe_json(sample_rate=44_100),
        _probe_json(channels=1),
        _probe_json(codec="mp3"),
    ],
)
def test_output_that_misses_the_profile_is_rejected_and_removed(config, paths, target_json):
    """명령행에 -ar 을 박아도 확인은 별개다."""
    source, target = paths
    with pytest.raises(WorkerError) as caught:
        normalize_audio(source, target, config=config, run=FakeFfmpeg(target_json=target_json))
    assert caught.value.code is ErrorCode.AUDIO_NORMALIZATION_FAILED
    assert not target.exists()
