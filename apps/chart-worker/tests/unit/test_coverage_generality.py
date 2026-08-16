from __future__ import annotations

import numpy as np
import pytest

from chart_worker.analysis.activity import AudioActivity
from chart_worker.analysis.coverage_opportunity import (
    CoverageKind,
    classify_coverage_interval,
)
from chart_worker.analysis.onset import OnsetAnalysis
from chart_worker.analysis.song_context import LocalTempoMap
from chart_worker.generation.osu_parser import OsuBpmEvent


def _strong_analysis(duration_ms: int, onset_ms: tuple[int, ...]) -> OnsetAnalysis:
    frame_ms = 100.0
    frame_count = duration_ms // 100 + 2
    strength = np.zeros(frame_count)
    for time_ms in onset_ms:
        strength[round(time_ms / frame_ms)] = 1.0
    return OnsetAnalysis(
        sample_rate_hz=1_000,
        hop_length=100,
        strength=strength,
        band_strength=np.zeros((3, frame_count)),
        onset_ms=onset_ms,
        n_fft=100,
        activity=AudioActivity(
            frame_ms=frame_ms,
            rms_db=np.full(frame_count, -10.0),
            floor_db=-20.0,
            active_onset_ms=onset_ms,
        ),
    )


@pytest.mark.parametrize(
    ("bpm", "end_ms", "expected_beats"),
    [
        (60.0, 16_000, 16.0),
        (120.0, 8_000, 16.0),
        (180.0, 5_333, 15.999),
        # The 4-second safety floor intentionally requires 20 beats at 300 BPM.
        (300.0, 4_000, 20.0),
    ],
)
def test_equivalent_phrase_gaps_are_actionable_across_bpm(
    bpm: float, end_ms: int, expected_beats: float
):
    step = max(100, end_ms // 10)
    onset_ms = tuple(range(step, end_ms, step))
    analysis = _strong_analysis(end_ms + 1_000, onset_ms)

    result = classify_coverage_interval(
        [],
        analysis,
        LocalTempoMap((OsuBpmEvent(0, bpm),)),
        start_ms=0,
        end_ms=end_ms,
        difficulty="EXPERT",
    )

    assert result.kind is CoverageKind.ATTACK_REQUIRED
    assert result.beat_count == pytest.approx(expected_beats, abs=0.002)


@pytest.mark.parametrize("duration_ms", [60_000, 180_000, 360_000, 600_000])
def test_song_length_does_not_change_same_local_interval_decision(duration_ms: int):
    onset_ms = tuple(range(1_000, 9_000, 1_000))
    result = classify_coverage_interval(
        [],
        _strong_analysis(duration_ms, onset_ms),
        LocalTempoMap((OsuBpmEvent(0, 120.0),)),
        start_ms=0,
        end_ms=8_000,
        difficulty="HARD",
    )

    assert result.kind is CoverageKind.ATTACK_REQUIRED
    assert result.beat_count == 16.0


def test_tempo_preserving_time_scale_keeps_decision():
    slow = classify_coverage_interval(
        [],
        _strong_analysis(10_000, tuple(range(1_000, 9_000, 1_000))),
        LocalTempoMap((OsuBpmEvent(0, 120.0),)),
        start_ms=0,
        end_ms=8_000,
        difficulty="HARD",
    )
    fast = classify_coverage_interval(
        [],
        _strong_analysis(6_000, tuple(range(500, 4_500, 500))),
        LocalTempoMap((OsuBpmEvent(0, 240.0),)),
        start_ms=0,
        end_ms=4_000,
        difficulty="HARD",
    )

    assert slow.kind is fast.kind is CoverageKind.ATTACK_REQUIRED
    assert slow.beat_count == fast.beat_count == 16.0


def test_missing_tempo_for_free_time_audio_is_review_evidence_only():
    result = classify_coverage_interval(
        [],
        _strong_analysis(20_000, tuple(range(1_000, 17_000, 1_000))),
        None,
        start_ms=0,
        end_ms=16_000,
        difficulty="EXPERT",
    )

    assert result.kind is CoverageKind.INSUFFICIENT_EVIDENCE
    assert result.beat_count is None


def test_leading_silence_shift_preserves_local_phrase_evidence():
    base_onsets = tuple(range(1_000, 9_000, 1_000))
    shifted_onsets = tuple(time_ms + 10_000 for time_ms in base_onsets)
    base = classify_coverage_interval(
        [],
        _strong_analysis(10_000, base_onsets),
        LocalTempoMap((OsuBpmEvent(0, 120.0),)),
        start_ms=0,
        end_ms=8_000,
        difficulty="HARD",
    )
    shifted = classify_coverage_interval(
        [],
        _strong_analysis(20_000, shifted_onsets),
        LocalTempoMap((OsuBpmEvent(10_000, 120.0),)),
        start_ms=10_000,
        end_ms=18_000,
        difficulty="HARD",
    )

    assert base.kind is shifted.kind is CoverageKind.ATTACK_REQUIRED
    assert base.beat_count == shifted.beat_count == 16.0
