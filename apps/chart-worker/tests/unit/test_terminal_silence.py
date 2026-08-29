from __future__ import annotations

import numpy as np

from chart_worker.analysis.audio_io import AudioSignal
from chart_worker.analysis.terminal_silence import (
    DEFAULT_THRESHOLDS_DB,
    TerminalSilenceObservation,
    TerminalThresholdCandidate,
    consensus_terminal_boundary_ms,
    observe_terminal_silence,
)

SAMPLE_RATE = 1_000
FRAME_MS = 20


def _signal(samples: np.ndarray) -> AudioSignal:
    array = np.asarray(samples, dtype=np.float64)
    if array.ndim == 1:
        array = array[:, None]
    return AudioSignal(samples=array, sample_rate_hz=SAMPLE_RATE)


def test_mono_absolute_silence_suffix_is_observed_without_a_verdict() -> None:
    samples = np.concatenate([np.full(600, 0.25), np.zeros(400)])
    observation = observe_terminal_silence(_signal(samples), frame_ms=FRAME_MS, last_onset_ms=580)

    assert observation.duration_ms == 1_000
    assert observation.frame_ms == FRAME_MS
    assert observation.channel_count == 1
    assert observation.last_onset_ms == 580
    assert len(observation.candidates) == len(DEFAULT_THRESHOLDS_DB)
    assert {candidate.suffix_start_ms for candidate in observation.candidates} == {600}
    assert {candidate.suffix_duration_ms for candidate in observation.candidates} == {400}
    assert observation.candidate_spread_ms == 0
    assert observation.to_report() == {
        "version": "terminal-silence-observation-v1",
        "durationMs": 1_000,
        "frameMs": 20,
        "channelCount": 1,
        "candidates": [
            {
                "rmsDb": rms_db,
                "peakDb": peak_db,
                "suffixStartMs": 600,
                "suffixDurationMs": 400,
            }
            for rms_db, peak_db in DEFAULT_THRESHOLDS_DB
        ],
        "candidateSpreadMs": 0,
        "lastOnsetMs": 580,
        "policyState": "OBSERVATION_ONLY",
        "mutatesGeneration": False,
    }


def test_stereo_any_channel_energy_prevents_an_early_silence_boundary() -> None:
    left = np.concatenate([np.full(600, 0.25), np.zeros(400)])
    right = np.concatenate([np.full(600, 0.25), np.full(300, 0.02), np.zeros(100)])
    observation = observe_terminal_silence(
        _signal(np.column_stack([left, right])), frame_ms=FRAME_MS
    )

    assert observation.channel_count == 2
    assert {candidate.suffix_start_ms for candidate in observation.candidates} == {900}


def test_only_the_terminal_suffix_is_observed_not_an_internal_break() -> None:
    samples = np.concatenate([np.full(300, 0.25), np.zeros(200), np.full(300, 0.25), np.zeros(200)])
    observation = observe_terminal_silence(_signal(samples), frame_ms=FRAME_MS)

    assert {candidate.suffix_start_ms for candidate in observation.candidates} == {800}


def test_a_late_stinger_resets_the_terminal_suffix_start() -> None:
    samples = np.concatenate([np.full(600, 0.25), np.zeros(200), np.full(20, 0.5), np.zeros(180)])
    observation = observe_terminal_silence(_signal(samples), frame_ms=FRAME_MS)

    assert {candidate.suffix_start_ms for candidate in observation.candidates} == {820}


def test_a_long_fade_exposes_threshold_disagreement_instead_of_hiding_it() -> None:
    main = np.full(600, 0.25)
    fade = np.geomspace(0.03, 1e-5, 400)
    observation = observe_terminal_silence(_signal(np.concatenate([main, fade])), frame_ms=FRAME_MS)

    starts = [
        candidate.suffix_start_ms
        for candidate in observation.candidates
        if candidate.suffix_start_ms is not None
    ]
    assert len(set(starts)) > 1
    assert observation.candidate_spread_ms == max(starts) - min(starts)


def test_complete_silence_and_constant_signal_are_explicit_controls() -> None:
    silent = observe_terminal_silence(_signal(np.zeros(1_000)), frame_ms=FRAME_MS)
    constant = observe_terminal_silence(_signal(np.full(1_000, 0.25)), frame_ms=FRAME_MS)

    assert {candidate.suffix_start_ms for candidate in silent.candidates} == {0}
    assert all(candidate.suffix_start_ms is None for candidate in constant.candidates)
    assert constant.candidate_spread_ms is None


def test_short_codec_padding_is_measured_but_not_classified() -> None:
    samples = np.concatenate([np.full(980, 0.25), np.zeros(20)])
    observation = observe_terminal_silence(_signal(samples), frame_ms=FRAME_MS)

    assert {candidate.suffix_duration_ms for candidate in observation.candidates} == {20}
    assert not hasattr(observation, "status")


def test_three_thresholds_must_agree_on_a_long_terminal_suffix() -> None:
    samples = np.concatenate([np.full(1_000, 0.25), np.zeros(4_000)])
    observation = observe_terminal_silence(
        _signal(samples),
        frame_ms=FRAME_MS,
        last_onset_ms=980,
    )

    assert consensus_terminal_boundary_ms(observation) == 1_000


def test_short_padding_and_threshold_disagreement_are_not_enforceable() -> None:
    padding = observe_terminal_silence(
        _signal(np.concatenate([np.full(980, 0.25), np.zeros(20)])),
        frame_ms=FRAME_MS,
        last_onset_ms=960,
    )
    fade = observe_terminal_silence(
        _signal(np.concatenate([np.full(1_000, 0.25), np.geomspace(0.03, 1e-5, 4_000)])),
        frame_ms=FRAME_MS,
        last_onset_ms=4_500,
    )

    assert consensus_terminal_boundary_ms(padding) is None
    assert consensus_terminal_boundary_ms(fade) is None


def test_consensus_uses_the_common_silent_intersection_not_equal_start_times() -> None:
    observation = TerminalSilenceObservation(
        version="terminal-silence-observation-v1",
        duration_ms=345_229,
        frame_ms=20,
        channel_count=2,
        candidates=tuple(
            TerminalThresholdCandidate(
                rms_db=rms_db,
                peak_db=peak_db,
                suffix_start_ms=start_ms,
                suffix_duration_ms=345_229 - start_ms,
            )
            for (rms_db, peak_db), start_ms in zip(
                DEFAULT_THRESHOLDS_DB,
                (318_700, 318_320, 318_180),
                strict=True,
            )
        ),
        candidate_spread_ms=520,
        last_onset_ms=316_853,
    )

    assert consensus_terminal_boundary_ms(observation) == 318_700


def test_common_silent_intersection_still_rejects_late_onset_and_short_suffix() -> None:
    def observation(*, duration_ms: int, latest_start_ms: int, last_onset_ms: int):
        starts = (latest_start_ms, latest_start_ms - 200, latest_start_ms - 400)
        return TerminalSilenceObservation(
            version="terminal-silence-observation-v1",
            duration_ms=duration_ms,
            frame_ms=20,
            channel_count=2,
            candidates=tuple(
                TerminalThresholdCandidate(
                    rms_db=rms_db,
                    peak_db=peak_db,
                    suffix_start_ms=start_ms,
                    suffix_duration_ms=duration_ms - start_ms,
                )
                for (rms_db, peak_db), start_ms in zip(
                    DEFAULT_THRESHOLDS_DB,
                    starts,
                    strict=True,
                )
            ),
            candidate_spread_ms=400,
            last_onset_ms=last_onset_ms,
        )

    assert (
        consensus_terminal_boundary_ms(
            observation(
                duration_ms=10_000,
                latest_start_ms=6_000,
                last_onset_ms=6_100,
            )
        )
        is None
    )
    assert (
        consensus_terminal_boundary_ms(
            observation(
                duration_ms=10_000,
                latest_start_ms=7_001,
                last_onset_ms=7_000,
            )
        )
        is None
    )


def test_stable_threshold_crossings_accept_a_near_minimum_strict_suffix() -> None:
    """Catch falling back to full duration for a stable 3-second terminal region."""
    observation = TerminalSilenceObservation(
        version="terminal-silence-observation-v1",
        duration_ms=156_012,
        frame_ms=20,
        channel_count=2,
        candidates=tuple(
            TerminalThresholdCandidate(
                rms_db=rms_db,
                peak_db=peak_db,
                suffix_start_ms=start_ms,
                suffix_duration_ms=156_012 - start_ms,
            )
            for (rms_db, peak_db), start_ms in zip(
                DEFAULT_THRESHOLDS_DB,
                (153_140, 153_040, 152_960),
                strict=True,
            )
        ),
        candidate_spread_ms=180,
        last_onset_ms=152_341,
    )

    assert consensus_terminal_boundary_ms(observation) == 153_140


def test_broad_fade_crossings_do_not_borrow_duration_from_the_lenient_detector() -> None:
    """Catch treating a wide threshold disagreement as stable terminal silence."""
    observation = TerminalSilenceObservation(
        version="terminal-silence-observation-v1",
        duration_ms=10_000,
        frame_ms=20,
        channel_count=2,
        candidates=tuple(
            TerminalThresholdCandidate(
                rms_db=rms_db,
                peak_db=peak_db,
                suffix_start_ms=start_ms,
                suffix_duration_ms=10_000 - start_ms,
            )
            for (rms_db, peak_db), start_ms in zip(
                DEFAULT_THRESHOLDS_DB,
                (7_000, 5_000, 2_000),
                strict=True,
            )
        ),
        candidate_spread_ms=5_000,
        last_onset_ms=1_000,
    )

    assert consensus_terminal_boundary_ms(observation) is None


def test_consensus_rejects_a_spread_field_that_does_not_match_candidates() -> None:
    """Catch trusting report metadata that disagrees with the measured starts."""
    observation = TerminalSilenceObservation(
        version="terminal-silence-observation-v1",
        duration_ms=10_000,
        frame_ms=20,
        channel_count=2,
        candidates=tuple(
            TerminalThresholdCandidate(
                rms_db=rms_db,
                peak_db=peak_db,
                suffix_start_ms=start_ms,
                suffix_duration_ms=10_000 - start_ms,
            )
            for (rms_db, peak_db), start_ms in zip(
                DEFAULT_THRESHOLDS_DB,
                (6_900, 6_800, 6_700),
                strict=True,
            )
        ),
        candidate_spread_ms=0,
        last_onset_ms=6_000,
    )

    assert consensus_terminal_boundary_ms(observation) is None
