import numpy as np

from chart_worker.analysis.activity import AudioActivity
from chart_worker.analysis.intro_anchor import classify_intro_anchor
from chart_worker.analysis.onset import OnsetAnalysis
from chart_worker.generation.osu_parser import OsuBpmEvent


def _analysis(
    *,
    onset_ms: tuple[int, ...],
    active_onset_ms: tuple[int, ...],
    aggregate: tuple[float, ...],
    bands: tuple[tuple[float, ...], ...],
    duration_ms: int = 5_000,
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
            rms_db=np.full(frame_count, -20.0),
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
