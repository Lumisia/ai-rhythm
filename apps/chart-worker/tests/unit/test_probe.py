import json

import pytest

from chart_worker.audio.probe import parse_probe_json


def _payload(**stream_overrides):
    stream = {
        "codec_type": "audio",
        "codec_name": "flac",
        "sample_rate": "48000",
        "channels": 2,
        "time_base": "1/48000",
        "duration_ts": 144_000,
        "duration": "3.001000",
    }
    stream.update(stream_overrides)
    return {"streams": [stream], "format": {"duration": "3.005000"}}


def test_prefers_the_exact_sample_count_over_container_duration():
    """format.duration 은 근삿값이다. 경계 노트가 어긋난다."""
    probe = parse_probe_json(_payload())
    assert probe.duration_ms == 3000
    assert probe.duration_is_exact is True


def test_reads_codec_rate_and_channels():
    probe = parse_probe_json(_payload())
    assert (probe.codec_name, probe.sample_rate_hz, probe.channels) == ("flac", 48000, 2)


def test_accepts_a_json_string():
    assert parse_probe_json(json.dumps(_payload())).duration_ms == 3000


def test_non_integer_tick_count_is_rounded_not_truncated():
    probe = parse_probe_json(_payload(time_base="1/44100", duration_ts=44_150))
    assert probe.duration_ms == 1001


def test_skips_non_audio_streams():
    payload = _payload()
    payload["streams"].insert(0, {"codec_type": "video", "codec_name": "mjpeg"})
    assert parse_probe_json(payload).codec_name == "flac"


@pytest.mark.parametrize(
    "overrides",
    [
        {"duration_ts": None, "time_base": None},
        {"duration_ts": "not-a-number"},
        {"time_base": "1/0"},
        {"time_base": 48000},
        {"duration_ts": -1},
    ],
)
def test_falls_back_to_stream_duration_when_ticks_are_unusable(overrides):
    probe = parse_probe_json(_payload(**overrides))
    assert probe.duration_ms == 3001
    assert probe.duration_is_exact is False


def test_falls_back_to_format_duration_last():
    payload = _payload(duration_ts=None, time_base=None, duration=None)
    assert parse_probe_json(payload).duration_ms == 3005


def test_rejects_a_file_with_no_audio_stream():
    with pytest.raises(ValueError, match="no audio stream"):
        parse_probe_json({"streams": [{"codec_type": "video"}], "format": {}})


def test_rejects_output_without_any_duration():
    payload = {
        "streams": [
            {
                "codec_type": "audio",
                "codec_name": "flac",
                "sample_rate": "48000",
                "channels": 2,
            }
        ],
        "format": {},
    }
    with pytest.raises(ValueError, match="no usable duration"):
        parse_probe_json(payload)


@pytest.mark.parametrize(
    "overrides",
    [{"sample_rate": None}, {"channels": None}, {"codec_name": None}],
)
def test_rejects_an_incomplete_audio_stream(overrides):
    payload = _payload()
    for key, value in overrides.items():
        if value is None:
            del payload["streams"][0][key]
    with pytest.raises(ValueError, match="incomplete"):
        parse_probe_json(payload)


@pytest.mark.parametrize("overrides", [{"sample_rate": "0"}, {"channels": 0}])
def test_rejects_a_non_positive_rate_or_channel_count(overrides):
    with pytest.raises(ValueError, match="non-positive"):
        parse_probe_json(_payload(**overrides))


@pytest.mark.parametrize("payload", ["not json", "[]", '"text"'])
def test_rejects_output_that_is_not_a_json_object(payload):
    with pytest.raises(ValueError, match="ffprobe output"):
        parse_probe_json(payload)
