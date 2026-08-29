from dataclasses import replace

import numpy as np

from chart_worker.analysis.activity import AudioActivity
from chart_worker.analysis.intro_anchor import classify_intro_anchor
from chart_worker.analysis.leading_silence import (
    DEFAULT_THRESHOLDS_DB,
    LeadingSilenceObservation,
    LeadingThresholdCandidate,
)
from chart_worker.analysis.onset import OnsetAnalysis
from chart_worker.generation.osu_parser import OsuBpmEvent


def _analysis(
    *,
    onset_ms: tuple[int, ...],
    active_onset_ms: tuple[int, ...],
    aggregate: tuple[float, ...],
    bands: tuple[tuple[float, ...], ...],
    duration_ms: int = 5_000,
    quiet: bool = False,
) -> OnsetAnalysis:
    frame_ms = 10.0
    frame_count = duration_ms // 10 + 1
    strength = np.zeros(frame_count)
    band_strength = np.zeros((3, frame_count))
    for index, onset in enumerate(onset_ms):
        frame = round(onset / frame_ms)
        strength[frame] = aggregate[index]
        band_strength[:, frame] = np.asarray(bands[index])
    return OnsetAnalysis(
        sample_rate_hz=1_000,
        hop_length=10,
        strength=strength,
        band_strength=band_strength,
        onset_ms=onset_ms,
        n_fft=10,
        activity=AudioActivity(
            frame_ms=frame_ms,
            rms_db=np.full(frame_count, -80.0 if quiet else -20.0),
            floor_db=-40.0,
            active_onset_ms=active_onset_ms,
        ),
    )


def test_single_broadband_attack_with_pulse_continuation_is_confirmed():
    """Ignite 실측: 21 ms 강한 첫 타격 뒤 약 140 BPM 간격이 이어진다."""
    result = classify_intro_anchor(
        (OsuBpmEvent(405, 140.0),),
        _analysis(
            onset_ms=(21, 437, 864, 1_291, 1_728),
            active_onset_ms=(21,),
            aggregate=(1.0, 0.41, 0.52, 0.44, 0.41),
            bands=(
                (1.0, 1.0, 0.995),
                (0.28, 0.30, 0.41),
                (0.49, 0.27, 0.57),
                (0.78, 0.33, 0.44),
                (0.45, 0.19, 0.47),
            ),
        ),
        duration_ms=5_000,
    )

    assert result.status == "CONFIRMED"
    assert result.anchor_ms == 21
    assert result.anchor_grid_ms == 0
    assert result.grid_distance_ms == 21
    assert result.pulse_continuation_matches >= 2
    assert result.prominent_band_count == 3
    assert result.supported_pulse_ms == (0, 429, 857, 1_286, 1_714)


def test_lone_off_grid_fx_stays_uncertain_instead_of_forcing_recovery():
    result = classify_intro_anchor(
        (OsuBpmEvent(4_000, 120.0),),
        _analysis(
            onset_ms=(120,),
            active_onset_ms=(120,),
            aggregate=(1.0,),
            bands=((1.0, 1.0, 1.0),),
        ),
        duration_ms=5_000,
    )

    assert result.status == "UNCERTAIN"
    assert result.anchor_ms == 120
    assert result.grid_distance_ms > 70
    assert result.pulse_continuation_matches == 0


def test_quiet_leading_region_is_non_rhythmic():
    result = classify_intro_anchor(
        (OsuBpmEvent(4_000, 120.0),),
        _analysis(
            onset_ms=(120, 620),
            active_onset_ms=(),
            aggregate=(1.0, 0.8),
            bands=((1.0, 1.0, 1.0), (0.8, 0.8, 0.8)),
            quiet=True,
        ),
        duration_ms=5_000,
    )

    assert result.status == "NON_RHYTHMIC"
    assert result.anchor_ms is None


def test_earliest_confirmed_anchor_wins_over_an_earlier_off_grid_fx():
    result = classify_intro_anchor(
        (OsuBpmEvent(2_000, 120.0),),
        _analysis(
            onset_ms=(120, 500, 1_000, 1_500),
            active_onset_ms=(120, 500),
            aggregate=(1.0, 0.95, 0.7, 0.7),
            bands=(
                (1.0, 0.1, 0.1),
                (0.9, 0.9, 0.9),
                (0.5, 0.5, 0.5),
                (0.5, 0.5, 0.5),
            ),
        ),
        duration_ms=5_000,
    )

    assert result.status == "CONFIRMED"
    assert result.anchor_ms == 500
    assert result.anchor_grid_ms == 500


def test_locally_audible_raw_onset_can_support_a_pickup_anchor() -> None:
    """v9 #7 축약: active-only 필터가 224ms grid support를 버리면 안 된다."""

    result = classify_intro_anchor(
        (OsuBpmEvent(442, 130.0),),
        _analysis(
            onset_ms=(117, 224, 459, 683, 896, 1_024),
            active_onset_ms=(117, 896),
            aggregate=(0.76, 0.41, 0.77, 0.88, 0.64, 0.83),
            bands=(
                (0.42, 0.70, 0.85),
                (0.31, 0.53, 0.53),
                (0.27, 0.41, 0.91),
                (0.49, 0.90, 0.85),
                (0.35, 0.50, 0.76),
                (0.15, 0.67, 1.00),
            ),
        ),
        duration_ms=5_000,
    )

    assert result.status == "CONFIRMED"
    assert result.anchor_ms == 117
    assert result.anchor_grid_ms == 211
    assert result.supported_pulse_ms == (211, 442, 673, 903)


def test_supported_pulses_follow_a_local_bpm_change() -> None:
    result = classify_intro_anchor(
        (OsuBpmEvent(500, 120.0), OsuBpmEvent(1_000, 240.0)),
        _analysis(
            onset_ms=(0, 250, 500, 750, 1_000, 1_125),
            active_onset_ms=(0, 250, 500, 750, 1_000, 1_125),
            aggregate=(1.0, 0.8, 0.8, 0.8, 0.8, 0.8),
            bands=((1.0, 1.0, 1.0),) * 6,
        ),
        duration_ms=5_000,
    )

    assert result.status == "CONFIRMED"
    assert result.supported_pulse_ms == (0, 250, 500, 750, 1_000, 1_125)


def test_anchor_after_a_bpm_change_uses_the_local_timing_point_phase() -> None:
    result = classify_intro_anchor(
        (OsuBpmEvent(0, 120.0), OsuBpmEvent(1_000, 240.0)),
        _analysis(
            onset_ms=(120, 1_125, 1_250, 1_375, 1_500),
            active_onset_ms=(120, 1_125, 1_250, 1_375, 1_500),
            aggregate=(0.2, 1.0, 0.8, 0.8, 0.8),
            bands=(
                (0.2, 0.1, 0.1),
                (1.0, 1.0, 1.0),
                (0.8, 0.8, 0.8),
                (0.8, 0.8, 0.8),
                (0.8, 0.8, 0.8),
            ),
        ),
        duration_ms=5_000,
    )

    assert result.status == "CONFIRMED"
    assert result.anchor_ms == 1_125
    assert result.anchor_grid_ms == 1_125
    assert result.grid_distance_ms == 0


def test_supported_pulses_reset_phase_at_a_non_aligned_tempo_change() -> None:
    result = classify_intro_anchor(
        (OsuBpmEvent(0, 120.0), OsuBpmEvent(875, 60.0)),
        _analysis(
            onset_ms=(0, 250, 500, 750, 875, 1_375, 1_875),
            active_onset_ms=(0, 250, 500, 750, 875, 1_375, 1_875),
            aggregate=(1.0, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8),
            bands=((1.0, 1.0, 1.0),) * 7,
        ),
        duration_ms=5_000,
    )

    assert result.status == "CONFIRMED"
    assert result.supported_pulse_ms[:7] == (
        0,
        250,
        500,
        750,
        875,
        1_375,
        1_875,
    )


def test_zero_origin_timing_still_searches_the_early_musical_phrase() -> None:
    result = classify_intro_anchor(
        (OsuBpmEvent(0, 120.0),),
        _analysis(
            onset_ms=(21, 500, 1_000, 1_500, 2_000),
            active_onset_ms=(21, 500, 1_000, 1_500, 2_000),
            aggregate=(1.0, 0.8, 0.8, 0.8, 0.8),
            bands=((1.0, 1.0, 1.0),) * 5,
        ),
        duration_ms=5_000,
    )

    assert result.status == "CONFIRMED"
    assert result.anchor_ms == 21
    assert result.anchor_grid_ms == 0
    assert result.supported_pulse_ms == (0, 500, 1_000, 1_500, 2_000)


def test_rhythm_after_the_early_beat_horizon_is_not_relabelled_as_intro() -> None:
    result = classify_intro_anchor(
        (OsuBpmEvent(0, 120.0),),
        _analysis(
            onset_ms=(4_500, 5_000, 5_500, 6_000),
            active_onset_ms=(4_500, 5_000, 5_500, 6_000),
            aggregate=(1.0, 0.8, 0.8, 0.8),
            bands=((1.0, 1.0, 1.0),) * 4,
            duration_ms=8_000,
        ),
        duration_ms=8_000,
    )

    assert result.status == "NON_RHYTHMIC"
    assert result.anchor_ms is None


def test_flat_regular_timestamps_without_audio_activity_stay_uncertain() -> None:
    result = classify_intro_anchor(
        (OsuBpmEvent(0, 120.0),),
        OnsetAnalysis(
            sample_rate_hz=1_000,
            hop_length=100,
            strength=np.zeros(21),
            band_strength=np.zeros((3, 21)),
            onset_ms=tuple(range(250, 2_000, 250)),
        ),
        duration_ms=2_000,
    )

    assert result.status == "UNCERTAIN"
    assert result.anchor_ms == 250


def test_confirmed_long_leading_silence_extends_the_beat_search_origin() -> None:
    analysis = _analysis(
        onset_ms=(5_000, 5_500, 6_000, 6_500),
        active_onset_ms=(5_000, 5_500, 6_000, 6_500),
        aggregate=(1.0, 0.8, 0.8, 0.8),
        bands=((1.0, 1.0, 1.0),) * 4,
        duration_ms=10_000,
    )
    analysis = replace(
        analysis,
        leading_silence=LeadingSilenceObservation(
            version="leading-silence-observation-v1",
            duration_ms=10_000,
            frame_ms=20,
            channel_count=2,
            candidates=tuple(
                LeadingThresholdCandidate(rms_db, peak_db, 5_000, 5_000)
                for rms_db, peak_db in DEFAULT_THRESHOLDS_DB
            ),
            candidate_spread_ms=0,
            first_onset_ms=5_000,
        ),
    )

    result = classify_intro_anchor(
        (OsuBpmEvent(0, 120.0),),
        analysis,
        duration_ms=10_000,
    )

    assert result.status == "CONFIRMED"
    assert result.anchor_ms == 5_000


def test_supported_region_pulses_extend_beyond_the_confirmation_prefix() -> None:
    rows = tuple(range(0, 3_001, 250))
    result = classify_intro_anchor(
        (OsuBpmEvent(0, 120.0),),
        _analysis(
            onset_ms=rows,
            active_onset_ms=rows,
            aggregate=(1.0,) * len(rows),
            bands=((1.0, 1.0, 1.0),) * len(rows),
        ),
        duration_ms=5_000,
    )

    assert result.status == "CONFIRMED"
    assert result.pulse_continuation_matches == 4
    assert result.pulse_continuation_opportunities == 4
    assert result.supported_pulse_ms[-1] == 3_000
