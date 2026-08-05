import numpy as np

from chart_worker.analysis.grid_alignment import (
    TempoCandidateMetrics,
    measure_note_grid_alignment,
    measure_tempo_candidates,
)
from chart_worker.analysis.onset import OnsetAnalysis
from chart_worker.generation.osu_parser import OsuBpmEvent


def test_tempo_candidate_metrics_report_has_complete_stable_camel_case_schema():
    metrics = TempoCandidateMetrics(
        base_pulse_support=0.51,
        half_pulse_support=0.72,
        double_pulse_support=0.33,
        base_supported_pulses=17,
        half_supported_pulses=19,
        double_supported_pulses=13,
        pulse_best_alternative="HALF",
        pulse_alternative_margin=0.21,
        base_periodicity_support=0.41,
        half_periodicity_support=0.56,
        double_periodicity_support=0.12,
        periodicity_frame_count=321,
        periodicity_best_alternative="HALF",
        periodicity_margin=0.15,
        evidence_agrees=True,
        evidence_status="SUFFICIENT",
    )

    assert metrics.to_report() == {
        "basePulseSupport": 0.51,
        "halfPulseSupport": 0.72,
        "doublePulseSupport": 0.33,
        "baseSupportedPulses": 17,
        "halfSupportedPulses": 19,
        "doubleSupportedPulses": 13,
        "pulseBestAlternative": "HALF",
        "pulseAlternativeMargin": 0.21,
        "basePeriodicitySupport": 0.41,
        "halfPeriodicitySupport": 0.56,
        "doublePeriodicitySupport": 0.12,
        "periodicityFrameCount": 321,
        "periodicityBestAlternative": "HALF",
        "periodicityMargin": 0.15,
        "evidenceAgrees": True,
        "evidenceStatus": "SUFFICIENT",
    }


def _analysis_with_second_pulses(*, seconds: int) -> OnsetAnalysis:
    """A hand-authored envelope: 1.0 at every 1,000ms and zero at 500ms."""
    strength = np.zeros(seconds * 10 + 1)
    strength[::10] = 1.0
    return OnsetAnalysis(
        sample_rate_hz=1_000,
        hop_length=100,
        strength=strength,
        band_strength=np.zeros((3, strength.size)),
        onset_ms=(),
        n_fft=1,
    )


def test_measure_note_grid_alignment_reports_clean_rows_on_supported_divisors():
    metrics = measure_note_grid_alignment(
        (0, 125, 250, 375, 500),
        (OsuBpmEvent(0, 120.0),),
    )

    assert metrics.clean_rate == 1.0
    assert metrics.absolute_p95_beats == 0.0


def test_measure_note_grid_alignment_counts_rows_before_the_first_event_as_non_clean():
    metrics = measure_note_grid_alignment(
        (0, 250),
        (OsuBpmEvent(250, 120.0),),
    )

    assert metrics.unique_row_count == 2
    assert metrics.clean_row_count == 1
    assert metrics.clean_rate == 0.5


def test_measure_tempo_candidates_prefers_half_with_pulse_and_periodicity_axes():
    metrics = measure_tempo_candidates(
        (OsuBpmEvent(0, 120.0),),
        _analysis_with_second_pulses(seconds=40),
    )

    assert metrics.half_pulse_support == 1.0
    assert metrics.base_pulse_support == 0.5
    assert metrics.pulse_best_alternative == "HALF"
    assert metrics.periodicity_best_alternative == "HALF"
    assert metrics.evidence_agrees is True


def test_measure_tempo_candidates_marks_short_or_unusable_evidence_insufficient():
    short = measure_tempo_candidates(
        (OsuBpmEvent(0, 120.0),),
        _analysis_with_second_pulses(seconds=8),
    )
    silent = measure_tempo_candidates(
        (OsuBpmEvent(0, 120.0),),
        OnsetAnalysis(
            sample_rate_hz=1_000,
            hop_length=100,
            strength=np.zeros(401),
            band_strength=np.zeros((3, 401)),
            onset_ms=(),
            n_fft=1,
        ),
    )

    assert short.evidence_status == "INSUFFICIENT"
    assert silent.evidence_status == "INSUFFICIENT"


def test_measure_tempo_candidates_rejects_constant_nonzero_periodicity():
    constant = measure_tempo_candidates(
        (OsuBpmEvent(0, 120.0),),
        OnsetAnalysis(
            sample_rate_hz=1_000,
            hop_length=100,
            strength=np.full(401, 0.4),
            band_strength=np.zeros((3, 401)),
            onset_ms=(),
            n_fft=1,
        ),
    )

    assert constant.evidence_status == "INSUFFICIENT"


def test_measure_tempo_candidates_does_not_select_a_tied_alternative():
    strength = np.zeros(801)
    strength[::20] = 0.6
    strength[5::20] = 0.9
    strength[15::20] = 0.9
    tied = measure_tempo_candidates(
        (OsuBpmEvent(0, 120.0),),
        OnsetAnalysis(
            sample_rate_hz=1_000,
            hop_length=50,
            strength=strength,
            band_strength=np.zeros((3, strength.size)),
            onset_ms=(),
            n_fft=1,
        ),
    )

    assert tied.pulse_best_alternative is None


def test_measure_tempo_candidates_clips_negative_phase_to_audio_and_keeps_four_cycles():
    strength = np.zeros(41)
    strength[5::10] = 1.0
    metrics = measure_tempo_candidates(
        (OsuBpmEvent(-500, 120.0),),
        OnsetAnalysis(
            sample_rate_hz=1_000,
            hop_length=100,
            strength=strength,
            band_strength=np.zeros((3, strength.size)),
            onset_ms=(),
            n_fft=1,
        ),
    )

    assert metrics.base_pulse_support == 0.5
    assert metrics.half_pulse_support == 1.0
    assert metrics.periodicity_frame_count == 40


def test_measure_tempo_candidates_keeps_pulses_inside_piecewise_timing_boundaries():
    strength = np.zeros(81)
    strength[::5] = 1.0
    strength[40::10] = 1.0
    metrics = measure_tempo_candidates(
        (OsuBpmEvent(0, 120.0), OsuBpmEvent(4_000, 60.0)),
        OnsetAnalysis(
            sample_rate_hz=1_000,
            hop_length=100,
            strength=strength,
            band_strength=np.zeros((3, strength.size)),
            onset_ms=(),
            n_fft=1,
        ),
    )

    assert metrics.base_pulse_support == 1.0


def test_measure_tempo_candidates_rejects_deterministic_nonperiodic_evidence():
    analysis = OnsetAnalysis(
        sample_rate_hz=1_000,
        hop_length=100,
        strength=np.random.default_rng(7).random(401),
        band_strength=np.zeros((3, 401)),
        onset_ms=(),
        n_fft=1,
    )

    metrics = measure_tempo_candidates((OsuBpmEvent(0, 120.0),), analysis)

    assert metrics.evidence_status == "INSUFFICIENT"
