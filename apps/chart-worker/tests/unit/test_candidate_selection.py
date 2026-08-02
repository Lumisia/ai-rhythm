import dataclasses

import numpy as np
import pytest

from chart_worker.analysis.onset import OnsetAnalysis
from chart_worker.analysis.timing import TimingCandidate, TimingPoint, TimingSource, TimingStatus
from chart_worker.generation.candidate_selection import (
    CandidateQuality,
    longest_active_bar_gap,
    needs_retry,
    rank_candidate,
    select_candidate_index,
)


def passing_quality(**overrides) -> CandidateQuality:
    base = CandidateQuality(
        long_gap_bars=0.0,
        rating_error=0.0,
        removed_ratio=0.0,
        drum_precision=0.9,
        playability_passes=1,
        hold_ratio_error=0.0,
        reference_pass=None,
    )
    return dataclasses.replace(base, **overrides)


@pytest.mark.parametrize(
    ("changes", "expected"),
    [
        ({"long_gap_bars": 2.0}, False),
        ({"long_gap_bars": 2.0001}, True),
        ({"rating_error": 0.3499}, False),
        ({"rating_error": 0.35}, True),
        ({"removed_ratio": 0.45}, False),
        ({"removed_ratio": 0.4501}, True),
        ({"playability_passes": 7}, False),
        ({"playability_passes": 8}, True),
    ],
)
def test_structural_retry_boundaries(changes, expected):
    assert needs_retry(passing_quality(**changes), difficulty="NORMAL") is expected


def test_missing_drum_metric_does_not_force_a_retry():
    assert not needs_retry(passing_quality(drum_precision=None), difficulty="EXPERT")


def test_expert_drum_precision_below_seventy_percent_requests_a_retry():
    assert needs_retry(passing_quality(drum_precision=0.6999), difficulty="EXPERT")
    assert not needs_retry(passing_quality(drum_precision=0.7), difficulty="EXPERT")


def test_easy_does_not_use_drum_precision_as_a_gate():
    assert not needs_retry(passing_quality(drum_precision=0.1), difficulty="EASY")


def test_human_reference_failure_requests_a_retry_for_every_difficulty():
    assert needs_retry(passing_quality(reference_pass=False), difficulty="EASY")


def test_rank_is_lexicographic_in_the_approved_order():
    passing = passing_quality(rating_error=0.34, removed_ratio=0.44)
    structurally_failed = passing_quality(rating_error=0.01, reference_pass=False)
    assert rank_candidate(passing, difficulty="NORMAL") < rank_candidate(
        structurally_failed, difficulty="NORMAL"
    )

    lower_rating_error = passing_quality(rating_error=0.1, removed_ratio=0.4)
    lower_removal = passing_quality(rating_error=0.2, removed_ratio=0.0)
    assert rank_candidate(lower_rating_error, difficulty="NORMAL") < rank_candidate(
        lower_removal, difficulty="NORMAL"
    )

    high_drum = passing_quality(drum_precision=0.9, hold_ratio_error=0.2)
    low_drum = passing_quality(drum_precision=0.8, hold_ratio_error=0.0)
    assert rank_candidate(high_drum, difficulty="EXPERT") < rank_candidate(
        low_drum, difficulty="EXPERT"
    )


def test_candidate_selection_breaks_exact_ties_by_original_attempt_order():
    qualities = (passing_quality(), passing_quality(), passing_quality())
    assert select_candidate_index(qualities, difficulty="NORMAL") == 0


def test_active_bar_gap_counts_variable_length_bars_not_elapsed_constant_bars():
    points = (
        TimingPoint(0, 120.0, 4, 0),
        TimingPoint(4_000, 60.0, 3, 8),
    )
    timing = TimingCandidate(
        source=TimingSource.BEAT_THIS_PIECEWISE,
        points=points,
        projected_beat_ms=(
            0,
            500,
            1_000,
            1_500,
            2_000,
            2_500,
            3_000,
            3_500,
            4_000,
            5_000,
            6_000,
            7_000,
            8_000,
            9_000,
            10_000,
            11_000,
        ),
        f1_20ms=1.0,
        f1_50ms=1.0,
        p95_abs_ms=0.0,
        status=TimingStatus.PASS,
        reasons=(),
    )
    strengths = np.zeros(121)
    strengths[1] = 1.0
    strengths[101] = 1.0
    onsets = OnsetAnalysis(
        sample_rate_hz=1_000,
        hop_length=100,
        strength=strengths,
        band_strength=np.zeros((3, 121)),
        onset_ms=(100, 10_100),
        n_fft=100,
    )

    assert longest_active_bar_gap(
        onsets=onsets,
        timing=timing,
        duration_ms=12_000,
        note_times=(100, 10_100),
    ) == 3.0


def test_onset_at_the_exact_seventy_fifth_percentile_makes_a_bar_active():
    timing = TimingCandidate(
        source=TimingSource.BEAT_THIS_PIECEWISE,
        points=(TimingPoint(0, 60.0, 1, 0),),
        projected_beat_ms=(0, 1_000, 2_000, 3_000),
        f1_20ms=1.0,
        f1_50ms=1.0,
        p95_abs_ms=0.0,
        status=TimingStatus.PASS,
        reasons=(),
    )
    strengths = np.zeros(41)
    strengths[[1, 11, 31]] = 1.0
    onsets = OnsetAnalysis(
        sample_rate_hz=1_000,
        hop_length=100,
        strength=strengths,
        band_strength=np.zeros((3, 41)),
        onset_ms=(100, 1_100, 3_100),
        n_fft=100,
    )

    # p75 is exactly 1.0. The equality boundary keeps bars 0, 1, and 3 active,
    # so the empty bar between the latter pair is observable.
    assert longest_active_bar_gap(
        onsets=onsets,
        timing=timing,
        duration_ms=4_000,
        note_times=(100, 1_100, 3_100),
    ) == 1.0


def test_active_but_note_empty_bars_remain_one_contiguous_gap():
    timing = TimingCandidate(
        source=TimingSource.BEAT_THIS_PIECEWISE,
        points=(TimingPoint(0, 60.0, 1, 0),),
        projected_beat_ms=(0, 1_000, 2_000, 3_000),
        f1_20ms=1.0,
        f1_50ms=1.0,
        p95_abs_ms=0.0,
        status=TimingStatus.PASS,
        reasons=(),
    )
    strengths = np.zeros(41)
    strengths[[1, 11, 21, 31]] = 1.0
    onsets = OnsetAnalysis(
        sample_rate_hz=1_000,
        hop_length=100,
        strength=strengths,
        band_strength=np.zeros((3, 41)),
        onset_ms=(100, 1_100, 2_100, 3_100),
        n_fft=100,
    )

    assert longest_active_bar_gap(
        onsets=onsets,
        timing=timing,
        duration_ms=4_000,
        note_times=(100, 3_100),
    ) == 2.0
