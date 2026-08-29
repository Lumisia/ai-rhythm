from __future__ import annotations

import numpy as np

from chart_worker.analysis.audio_io import AudioSignal
from chart_worker.analysis.leading_silence import (
    DEFAULT_THRESHOLDS_DB,
    LeadingSilenceObservation,
    LeadingThresholdCandidate,
    consensus_leading_boundary_ms,
    observe_leading_silence,
)

SAMPLE_RATE = 1_000
FRAME_MS = 20


def _signal(samples: np.ndarray) -> AudioSignal:
    array = np.asarray(samples, dtype=np.float64)
    if array.ndim == 1:
        array = array[:, None]
    return AudioSignal(samples=array, sample_rate_hz=SAMPLE_RATE)


def test_observes_only_the_silent_prefix_not_an_internal_break() -> None:
    samples = np.concatenate(
        [np.zeros(4_000), np.full(1_000, 0.25), np.zeros(2_000)]
    )

    observation = observe_leading_silence(
        _signal(samples), frame_ms=FRAME_MS, first_onset_ms=4_020
    )

    assert observation.duration_ms == 7_000
    assert observation.channel_count == 1
    assert {candidate.prefix_end_ms for candidate in observation.candidates} == {4_000}
    assert {candidate.prefix_duration_ms for candidate in observation.candidates} == {
        4_000
    }
    assert observation.first_onset_ms == 4_020


def test_stereo_energy_in_either_channel_ends_the_silent_prefix() -> None:
    left = np.concatenate([np.zeros(4_000), np.full(1_000, 0.25)])
    right = np.concatenate(
        [np.zeros(3_000), np.full(1_000, 0.02), np.full(1_000, 0.25)]
    )

    observation = observe_leading_silence(
        _signal(np.column_stack([left, right])), frame_ms=FRAME_MS
    )

    assert observation.channel_count == 2
    assert {candidate.prefix_end_ms for candidate in observation.candidates} == {3_000}


def test_consensus_uses_the_common_prefix_and_requires_three_seconds() -> None:
    long = LeadingSilenceObservation(
        version="leading-silence-observation-v1",
        duration_ms=10_000,
        frame_ms=20,
        channel_count=2,
        candidates=tuple(
            LeadingThresholdCandidate(
                rms_db=rms_db,
                peak_db=peak_db,
                prefix_end_ms=end_ms,
                prefix_duration_ms=end_ms,
            )
            for (rms_db, peak_db), end_ms in zip(
                DEFAULT_THRESHOLDS_DB,
                (3_600, 3_800, 4_000),
                strict=True,
            )
        ),
        candidate_spread_ms=400,
        first_onset_ms=3_700,
    )
    short = LeadingSilenceObservation(
        version="leading-silence-observation-v1",
        duration_ms=10_000,
        frame_ms=20,
        channel_count=1,
        candidates=tuple(
            LeadingThresholdCandidate(rms_db, peak_db, 2_999, 2_999)
            for rms_db, peak_db in DEFAULT_THRESHOLDS_DB
        ),
        candidate_spread_ms=0,
        first_onset_ms=3_000,
    )

    assert consensus_leading_boundary_ms(long) == 3_600
    assert consensus_leading_boundary_ms(short) is None


def test_onset_inside_the_common_silent_prefix_invalidates_consensus() -> None:
    observation = LeadingSilenceObservation(
        version="leading-silence-observation-v1",
        duration_ms=10_000,
        frame_ms=20,
        channel_count=1,
        candidates=tuple(
            LeadingThresholdCandidate(rms_db, peak_db, 4_000, 4_000)
            for rms_db, peak_db in DEFAULT_THRESHOLDS_DB
        ),
        candidate_spread_ms=0,
        first_onset_ms=3_900,
    )

    assert consensus_leading_boundary_ms(observation) is None


def test_fade_in_exposes_threshold_disagreement() -> None:
    fade = np.geomspace(1e-5, 0.03, 4_000)
    observation = observe_leading_silence(
        _signal(np.concatenate([fade, np.full(1_000, 0.25)])),
        frame_ms=FRAME_MS,
    )

    ends = [
        candidate.prefix_end_ms
        for candidate in observation.candidates
        if candidate.prefix_end_ms is not None
    ]
    assert len(set(ends)) > 1
    assert observation.candidate_spread_ms == max(ends) - min(ends)
