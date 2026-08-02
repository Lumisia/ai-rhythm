import json

import numpy as np
import pytest

from chart_worker.analysis.beat import build_beat_grid
from chart_worker.analysis.timing import (
    MatchMetrics,
    ReferenceChart,
    TimingPoint,
    _bps_are_mergeable,
    _exceeds_error_limits,
    _sse_improvement_is_sufficient,
    evaluate_reference,
    fit_piecewise_timing,
    load_reference_onsets,
    match_times,
    project_beats,
)


def _write_reference(tmp_path, payload):
    path = tmp_path / "reference-onsets-v1.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _valid_reference_payload():
    return {
        "version": 1,
        "charts": [
            {
                "keyMode": 4,
                "difficulty": "NORMAL",
                "sections": [
                    {"id": "verse", "onsetMs": [100, 200, 300]},
                    {"id": "chorus", "onsetMs": [1000, 1100, 1200]},
                ],
            }
        ],
    }


def test_reference_gate_uses_unweighted_section_macro_f1_and_ignores_notes_outside_sections(
    tmp_path,
):
    references = load_reference_onsets(_write_reference(tmp_path, _valid_reference_payload()))

    quality = evaluate_reference(
        references[(4, "NORMAL")],
        (0, 100, 200, 1000, 1100, 2000),
    )

    assert quality is not None
    assert quality.macro_f1_20ms == pytest.approx(0.8)
    assert quality.phase_abs_ms == 0.0
    assert quality.p95_abs_ms == 0.0
    assert quality.passes is True


def test_reference_phase_is_absolute_signed_median_and_does_not_treat_jitter_as_offset():
    reference = ReferenceChart(
        key_mode=4,
        difficulty="NORMAL",
        sections={"verse": (100, 200, 300, 400)},
    )

    quality = evaluate_reference(reference, (120, 180, 320, 380))

    assert quality is not None
    assert quality.macro_f1_20ms == 1.0
    assert quality.phase_abs_ms == 0.0
    assert quality.p95_abs_ms == 20.0
    assert quality.passes is True


def test_reference_phase_rejects_a_constant_offset_above_fifteen_ms():
    reference = ReferenceChart(
        key_mode=4,
        difficulty="NORMAL",
        sections={"verse": (100, 200, 300, 400, 500)},
    )

    quality = evaluate_reference(reference, (120, 220, 320, 420, 480))

    assert quality is not None
    assert quality.macro_f1_20ms == 1.0
    assert quality.phase_abs_ms == 20.0
    assert quality.p95_abs_ms == 20.0
    assert quality.passes is False


def test_reference_p95_uses_fifty_ms_pairs_and_rejects_tail_errors_above_thirty_ms():
    reference = ReferenceChart(
        key_mode=4,
        difficulty="NORMAL",
        sections={"song": (100, 200, 300, 400, 500, 600, 700, 800, 900, 1000)},
    )

    quality = evaluate_reference(
        reference,
        (100, 200, 300, 400, 540, 600, 740, 800, 900, 1000),
    )

    assert quality is not None
    assert quality.macro_f1_20ms == pytest.approx(0.8)
    assert quality.phase_abs_ms == 0.0
    assert quality.p95_abs_ms == 40.0
    assert quality.passes is False


def test_missing_reference_is_reported_as_unavailable():
    assert evaluate_reference(None, (100, 200)) is None


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda payload: payload.update(version=2), "version"),
        (
            lambda payload: payload["charts"].append(dict(payload["charts"][0])),
            "duplicate chart",
        ),
        (
            lambda payload: payload["charts"][0]["sections"].append(
                {"id": "verse", "onsetMs": [1300]}
            ),
            "duplicate section",
        ),
        (
            lambda payload: payload["charts"][0]["sections"][0].update(onsetMs=[]),
            "empty",
        ),
        (
            lambda payload: payload["charts"][0]["sections"][0].update(
                onsetMs=[100, -1, 300]
            ),
            "non-negative",
        ),
        (
            lambda payload: payload["charts"][0]["sections"][0].update(
                onsetMs=[100, 300, 200]
            ),
            "sorted",
        ),
    ],
)
def test_reference_loader_rejects_invalid_contracts(tmp_path, mutate, message):
    payload = _valid_reference_payload()
    mutate(payload)

    with pytest.raises(ValueError, match=message):
        load_reference_onsets(_write_reference(tmp_path, payload))


def test_piecewise_fit_splits_a_real_tempo_change_at_a_downbeat():
    first = np.arange(32) * 0.5
    second = first[-1] + np.arange(1, 33) * 0.6
    beats = np.concatenate([first, second])
    grid = build_beat_grid(beats, beats[::4])

    points = fit_piecewise_timing(grid)

    assert len(points) == 2
    assert points[1].start_beat_index == 32
    assert points[0].bpm == pytest.approx(120.0, abs=0.05)
    assert points[1].bpm == pytest.approx(100.0, abs=0.05)


def test_match_times_is_one_to_one_and_reports_f1_at_the_window():
    result = match_times((100, 101, 300), (100, 300), window_ms=20)

    assert result.precision == pytest.approx(2 / 3)
    assert result.recall == 1.0
    assert result.f1 == pytest.approx(0.8)


def test_piecewise_fit_rejects_a_tempo_change_with_only_seven_beats_before_it():
    first = np.arange(7) * 0.5
    second = first[-1] + np.arange(1, 10) * 0.6
    beats = np.concatenate([first, second])
    grid = build_beat_grid(beats, beats[[0, 7]])

    assert len(fit_piecewise_timing(grid)) == 1


def test_piecewise_fit_accepts_a_tempo_change_with_exactly_eight_beats_per_side():
    first = np.arange(8) * 0.5
    second = first[-1] + np.arange(1, 9) * 0.6
    beats = np.concatenate([first, second])
    grid = build_beat_grid(beats, beats[[0, 8]])

    points = fit_piecewise_timing(grid)

    assert len(points) == 2
    assert points[1].start_beat_index == 8


def test_error_limits_allow_exact_30ms_and_50ms_but_reject_values_above_either_limit():
    assert not _exceeds_error_limits(30.0, 50.0)
    assert _exceeds_error_limits(np.nextafter(30.0, np.inf), 50.0)
    assert _exceeds_error_limits(30.0, np.nextafter(50.0, np.inf))


def test_sse_improvement_accepts_exactly_35_percent_and_rejects_less():
    assert _sse_improvement_is_sufficient(100.0, 65.0)
    assert not _sse_improvement_is_sufficient(100.0, np.nextafter(65.0, np.inf))


def test_piecewise_fit_stops_at_32_timing_points():
    beat_times: list[float] = []
    time_sec = 0.0
    for segment_index in range(33):
        interval_sec = 0.9 - segment_index * 0.018
        for _ in range(8):
            beat_times.append(time_sec)
            time_sec += interval_sec
    beats = np.array(beat_times)
    grid = build_beat_grid(beats, beats[::8])

    assert len(fit_piecewise_timing(grid)) == 32


def test_bpm_merge_threshold_is_strictly_less_than_half_a_percent():
    assert _bps_are_mergeable(120.0, 120.599_999)
    assert not _bps_are_mergeable(120.0, 120.6)


def test_project_beats_hands_off_at_the_next_timing_point_and_stops_at_end():
    points = (
        TimingPoint(time_ms=100, bpm=120.0, meter=4, start_beat_index=0),
        TimingPoint(time_ms=1_100, bpm=60.0, meter=4, start_beat_index=2),
    )

    assert project_beats(points, end_ms=2_300) == (100, 600, 1_100, 2_100)


def test_match_metrics_aggregates_signed_and_absolute_errors():
    result = MatchMetrics.from_pairs(
        ((0, 0), (105, 100), (130, 100)),
        predicted_count=3,
        reference_count=3,
    )

    assert result.precision == result.recall == result.f1 == 1.0
    assert result.median_signed_ms == 5.0
    assert result.p95_abs_ms == pytest.approx(27.5)
    assert result.p99_abs_ms == pytest.approx(29.5)
    assert result.max_abs_ms == 30.0


def test_empty_match_metrics_have_zero_quality_and_error_aggregates():
    result = MatchMetrics.from_pairs((), predicted_count=0, reference_count=3)

    assert result.precision == result.recall == result.f1 == 0.0
    assert result.median_signed_ms == result.p95_abs_ms == result.p99_abs_ms == result.max_abs_ms == 0.0
