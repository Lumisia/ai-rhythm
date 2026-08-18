from __future__ import annotations

import numpy as np

from chart_worker.analysis.activity import AudioActivity
from chart_worker.analysis.coverage_jury import measure_local_gap_evidence
from chart_worker.analysis.onset import OnsetAnalysis


def _analysis(
    *,
    duration_ms: int,
    strengths: dict[int, float],
    active_ranges: tuple[tuple[int, int], ...],
) -> OnsetAnalysis:
    frame_ms = 100.0
    frame_count = duration_ms // 100 + 1
    envelope = np.zeros(frame_count, dtype=np.float64)
    rms = np.full(frame_count, -80.0, dtype=np.float64)
    for start_ms, end_ms in active_ranges:
        rms[start_ms // 100 : end_ms // 100] = -10.0
    for time_ms, value in strengths.items():
        envelope[time_ms // 100] = value
    onset_ms = tuple(sorted(strengths))
    return OnsetAnalysis(
        sample_rate_hz=1_000,
        hop_length=100,
        strength=envelope,
        band_strength=np.zeros((3, frame_count), dtype=np.float64),
        onset_ms=onset_ms,
        n_fft=100,
        activity=AudioActivity(
            frame_ms=frame_ms,
            rms_db=rms,
            floor_db=-20.0,
            active_onset_ms=onset_ms,
        ),
    )


def test_local_evidence_exposes_a_phrase_hidden_by_the_song_global_threshold() -> None:
    outside = {time_ms: 0.9 for time_ms in range(1_000, 11_000, 500)}
    target = {
        21_000: 0.9,
        22_000: 0.9,
        23_000: 0.9,
        24_000: 0.4,
        25_000: 0.4,
        26_000: 0.4,
        27_000: 0.4,
    }
    analysis = _analysis(
        duration_ms=40_000,
        strengths={**outside, **target},
        active_ranges=((0, 12_000), (19_000, 29_000)),
    )

    observed = measure_local_gap_evidence(analysis, start_ms=20_000, end_ms=28_000)

    assert observed.active_frame_ratio == 1.0
    assert observed.active_onset_count == 7
    assert observed.global_strong_attack_count == 3
    assert observed.local_strong_attack_count == 7
    assert observed.local_threshold is not None
    assert observed.local_threshold < observed.global_threshold
    assert observed.to_report()["policyState"] == "OBSERVATION_ONLY"


def test_quiet_break_has_no_local_attacks_or_active_frames() -> None:
    analysis = _analysis(
        duration_ms=30_000,
        strengths={1_000: 0.8, 29_000: 0.8},
        active_ranges=((0, 5_000), (25_000, 30_000)),
    )

    observed = measure_local_gap_evidence(analysis, start_ms=10_000, end_ms=20_000)

    assert observed.active_frame_ratio == 0.0
    assert observed.active_onset_count == 0
    assert observed.local_strong_attack_count == 0
    assert observed.local_threshold is None


def test_isolated_stinger_is_recorded_but_not_promoted_to_a_phrase() -> None:
    analysis = _analysis(
        duration_ms=30_000,
        strengths={15_000: 1.0},
        active_ranges=((14_900, 15_100),),
    )

    observed = measure_local_gap_evidence(analysis, start_ms=10_000, end_ms=20_000)

    assert observed.active_onset_count == 1
    assert observed.local_strong_attack_count == 1
    assert observed.active_frame_ratio < 0.05


def test_neighboring_activity_is_separate_from_target_activity() -> None:
    analysis = _analysis(
        duration_ms=30_000,
        strengths={5_000: 0.8, 25_000: 0.8},
        active_ranges=((2_000, 8_000), (22_000, 28_000)),
    )

    observed = measure_local_gap_evidence(analysis, start_ms=10_000, end_ms=20_000)

    assert observed.active_frame_ratio == 0.0
    assert observed.neighboring_activity_ratio is not None
    assert observed.neighboring_activity_ratio > 0.0


def test_missing_activity_is_unavailable_not_fabricated_zero_evidence() -> None:
    analysis = _analysis(
        duration_ms=10_000,
        strengths={5_000: 0.8},
        active_ranges=((0, 10_000),),
    )
    analysis = OnsetAnalysis(
        sample_rate_hz=analysis.sample_rate_hz,
        hop_length=analysis.hop_length,
        strength=analysis.strength,
        band_strength=analysis.band_strength,
        onset_ms=analysis.onset_ms,
        n_fft=analysis.n_fft,
        activity=None,
    )

    observed = measure_local_gap_evidence(analysis, start_ms=1_000, end_ms=9_000)

    assert observed.active_frame_ratio is None
    assert observed.neighboring_activity_ratio is None


def test_leading_gap_does_not_average_a_nonexistent_left_neighbor_as_silence() -> None:
    analysis = _analysis(
        duration_ms=10_000,
        strengths={6_000: 0.8},
        active_ranges=((5_000, 10_000),),
    )

    observed = measure_local_gap_evidence(analysis, start_ms=0, end_ms=5_000)

    assert observed.neighboring_activity_ratio == 1.0


def test_trailing_gap_does_not_average_a_nonexistent_right_neighbor_as_silence() -> None:
    analysis = _analysis(
        duration_ms=10_000,
        strengths={4_000: 0.8},
        active_ranges=((0, 5_000),),
    )

    observed = measure_local_gap_evidence(analysis, start_ms=5_000, end_ms=10_000)

    assert observed.neighboring_activity_ratio == 1.0
