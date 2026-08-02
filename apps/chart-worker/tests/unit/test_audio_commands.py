from pathlib import Path

import pytest

from chart_worker.audio import commands, profile


def _pairs(argv: list[str]) -> dict[str, str]:
    """`-flag value` 쌍을 뽑는다. 중복 플래그는 마지막 값이 남는다."""
    return {
        argv[index]: argv[index + 1]
        for index in range(len(argv) - 1)
        if argv[index].startswith("-")
    }


def _filter_chain(argv: list[str]) -> str:
    return argv[argv.index("-af") + 1]


FFMPEG = Path("C:/ffmpeg/bin/ffmpeg.exe")
SOURCE = Path("in.mp3")
TARGET = Path("out.flac")


def test_silence_trim_filter_uses_documented_values():
    chain = commands.silence_trim_filter()
    assert chain == (
        "silenceremove=start_periods=1:start_duration=0:start_threshold=-60dB:detection=rms"
    )


def test_silence_trim_filter_writes_thresholds_without_trailing_zeros():
    assert "start_threshold=-42dB" in commands.silence_trim_filter(-42.0)
    assert "start_threshold=-42.5dB" in commands.silence_trim_filter(-42.5)


def test_both_passes_share_the_same_trim_arguments():
    """측정과 인코딩의 트림이 다르면 측정값이 실제 신호를 설명하지 못한다."""
    trim = commands.silence_trim_filter()
    measure = _filter_chain(commands.measure_command(FFMPEG, SOURCE))
    normalize = _filter_chain(commands.normalize_command(FFMPEG, SOURCE, TARGET, gain_db=-3.0))
    assert measure.startswith(f"{trim},")
    assert normalize.startswith(f"{trim},")


def test_measure_command_only_measures():
    argv = commands.measure_command(FFMPEG, SOURCE)
    assert argv[0] == str(FFMPEG)
    assert argv[-3:] == ["-f", "null", "-"]
    assert "print_format=json" in _filter_chain(argv)
    assert "loudnorm" in _filter_chain(argv)
    assert _pairs(argv)["-map"] == "0:a:0"
    # 측정 JSON 은 info 레벨로 나온다. error 로 낮추면 아무것도 못 읽는다.
    assert _pairs(argv)["-v"] == "info"


def test_measure_command_targets_the_documented_profile():
    chain = _filter_chain(commands.measure_command(FFMPEG, SOURCE))
    assert "I=-14" in chain
    assert "TP=-1" in chain


def test_normalize_command_pins_the_audio_profile():
    argv = commands.normalize_command(FFMPEG, SOURCE, TARGET, gain_db=-3.25)
    pairs = _pairs(argv)
    assert pairs["-ar"] == str(profile.SAMPLE_RATE_HZ)
    assert pairs["-ac"] == str(profile.CHANNELS)
    assert pairs["-sample_fmt"] == profile.SAMPLE_FORMAT
    assert pairs["-c:a"] == profile.AUDIO_CODEC
    assert pairs["-compression_level"] == str(profile.FLAC_COMPRESSION_LEVEL)
    assert argv[-1] == str(TARGET)


def test_normalize_command_is_reproducible_byte_for_byte():
    """-bitexact 가 없으면 vendor string 때문에 sha256 이 매번 달라진다."""
    argv = commands.normalize_command(FFMPEG, SOURCE, TARGET, gain_db=0.0)
    assert "-bitexact" in argv
    assert _pairs(argv)["-map_metadata"] == "-1"


def test_normalize_command_never_uses_loudnorm():
    """loudnorm 이 섞이면 dynamic 전환 위험이 되살아난다."""
    chain = _filter_chain(commands.normalize_command(FFMPEG, SOURCE, TARGET, gain_db=1.0))
    assert "loudnorm" not in chain
    assert "volume=" in chain


@pytest.mark.parametrize(
    ("gain_db", "expected"),
    [(-3.25, "volume=-3.250000dB"), (0.0, "volume=0.000000dB"), (2.5, "volume=2.500000dB")],
)
def test_normalize_command_writes_gain_with_fixed_precision(gain_db, expected):
    assert expected in _filter_chain(
        commands.normalize_command(FFMPEG, SOURCE, TARGET, gain_db=gain_db)
    )


def test_probe_command_asks_for_streams_and_format_as_json():
    argv = commands.probe_command(Path("ffprobe"), SOURCE)
    assert argv[0] == "ffprobe"
    assert "-show_streams" in argv
    assert "-show_format" in argv
    assert _pairs(argv)["-print_format"] == "json"
    assert argv[-1] == str(SOURCE)


@pytest.mark.parametrize(
    ("ffmpeg_bin", "expected"),
    [
        (Path("ffmpeg"), Path("ffprobe")),
        (Path("C:/ffmpeg/bin/ffmpeg.exe"), Path("C:/ffmpeg/bin/ffprobe.exe")),
        (Path("/usr/local/bin/ffmpeg"), Path("/usr/local/bin/ffprobe")),
    ],
)
def test_ffprobe_sits_next_to_ffmpeg(ffmpeg_bin, expected):
    assert commands.ffprobe_for(ffmpeg_bin) == expected
