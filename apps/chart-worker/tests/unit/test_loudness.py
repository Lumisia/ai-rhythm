import pytest

from chart_worker.audio.loudness import (
    LoudnessMeasurement,
    compute_gain_db,
    is_silent,
    parse_loudnorm_json,
)

STDERR = """\
[Parsed_silenceremove_0 @ 000001] chunk
frame=  100 fps=0.0 q=-0.0 size=N/A time=00:00:02.00 bitrate=N/A speed=  50x
[Parsed_loudnorm_1 @ 000002]
{
\t"input_i" : "-27.61",
\t"input_tp" : "-4.47",
\t"input_lra" : "18.06",
\t"input_thresh" : "-39.20",
\t"output_i" : "-14.00",
\t"output_tp" : "-1.00",
\t"output_lra" : "18.10",
\t"output_thresh" : "-25.59",
\t"normalization_type" : "dynamic",
\t"target_offset" : "0.58"
}
"""


def test_parses_measurement_from_noisy_stderr():
    measurement = parse_loudnorm_json(STDERR)
    assert measurement == LoudnessMeasurement(-27.61, -4.47, 18.06, -39.20)


def test_uses_the_last_block_when_several_are_printed():
    first = STDERR.replace("-27.61", "-99.00")
    measurement = parse_loudnorm_json(first + STDERR)
    assert measurement.input_i == -27.61


def test_skips_json_blocks_that_are_not_the_measurement():
    noise = '{"progress" : "continue"}\n'
    assert parse_loudnorm_json(STDERR + noise).input_i == -27.61


def test_accepts_negative_infinity_from_a_silent_file():
    silent = STDERR.replace('"-27.61"', '"-inf"').replace('"-4.47"', '"-inf"')
    measurement = parse_loudnorm_json(silent)
    assert measurement.input_i == float("-inf")
    assert is_silent(measurement)


@pytest.mark.parametrize(
    "stderr",
    ["", "no json here", '{"input_i" : "-27.61"}'],
)
def test_rejects_output_without_a_measurement_block(stderr):
    with pytest.raises(ValueError, match="not found"):
        parse_loudnorm_json(stderr)


def test_rejects_non_numeric_measurement_values():
    broken = STDERR.replace('"-27.61"', '"loud"')
    with pytest.raises(ValueError, match="input_i"):
        parse_loudnorm_json(broken)


@pytest.mark.parametrize("input_i", [-60.0, -70.0, float("-inf")])
def test_silence_is_detected_at_or_below_the_threshold(input_i):
    assert is_silent(LoudnessMeasurement(input_i, -90.0, 0.0, -70.0))


def test_audible_file_is_not_silent():
    assert not is_silent(LoudnessMeasurement(-59.9, -30.0, 4.0, -70.0))


def test_gain_reaches_the_target_when_headroom_allows():
    # 피크 여유 19 dB > 필요한 게인 13.61 dB 이므로 라우드니스가 제한한다.
    plan = compute_gain_db(LoudnessMeasurement(-27.61, -20.0, 8.0, -37.0))
    assert plan.gain_db == pytest.approx(13.61)
    assert plan.achieved_lufs == pytest.approx(-14.0)
    assert plan.achieved_true_peak_dbtp == pytest.approx(-6.39)
    assert plan.shortfall_lu == 0.0
    assert plan.limited_by == "LOUDNESS"


def test_true_peak_caps_the_gain_on_peaky_material():
    """피크가 심하면 목표에 미달한다. 리미터를 걸지 않는 대가다."""
    plan = compute_gain_db(LoudnessMeasurement(-27.61, -4.47, 18.06, -39.20))
    assert plan.gain_db == pytest.approx(3.47)
    assert plan.achieved_true_peak_dbtp == pytest.approx(-1.0)
    assert plan.achieved_lufs == pytest.approx(-24.14)
    assert plan.shortfall_lu == pytest.approx(10.14)
    assert plan.limited_by == "TRUE_PEAK"


def test_loud_input_is_attenuated():
    plan = compute_gain_db(LoudnessMeasurement(-8.0, -0.2, 5.0, -18.0))
    assert plan.gain_db == pytest.approx(-6.0)
    assert plan.achieved_lufs == pytest.approx(-14.0)
    assert plan.achieved_true_peak_dbtp == pytest.approx(-6.2)
    assert plan.shortfall_lu == 0.0
    assert plan.limited_by == "LOUDNESS"


def test_shortfall_is_never_negative_when_overshooting_is_impossible():
    plan = compute_gain_db(LoudnessMeasurement(-14.0, -20.0, 5.0, -24.0))
    assert plan.gain_db == pytest.approx(0.0)
    assert plan.shortfall_lu == 0.0


def test_custom_targets_are_honoured():
    plan = compute_gain_db(
        LoudnessMeasurement(-20.0, -10.0, 5.0, -30.0),
        target_lufs=-16.0,
        target_true_peak_dbtp=-2.0,
    )
    assert plan.gain_db == pytest.approx(4.0)
    assert plan.achieved_lufs == pytest.approx(-16.0)


@pytest.mark.parametrize(
    "measurement",
    [
        LoudnessMeasurement(float("-inf"), -10.0, 5.0, -70.0),
        LoudnessMeasurement(-20.0, float("-inf"), 5.0, -30.0),
    ],
)
def test_rejects_planning_gain_from_a_silent_measurement(measurement):
    with pytest.raises(ValueError, match="non-finite"):
        compute_gain_db(measurement)
