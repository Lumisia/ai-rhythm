from __future__ import annotations

import numpy as np
import pytest

from chart_worker.analysis.activity import AudioActivity
from chart_worker.analysis.coverage_opportunity import (
    CoverageKind,
    classify_coverage_interval,
)
from chart_worker.analysis.onset import OnsetAnalysis, normalize_envelope
from chart_worker.analysis.song_context import LocalTempoMap
from chart_worker.generation.osu_parser import OsuBpmEvent
from chart_worker.schema.note import NoteEvent


def _analysis(
    *,
    duration_ms: int,
    onset_strengths: dict[int, float],
    active: bool = True,
    active_ranges: tuple[tuple[int, int], ...] | None = None,
    frame_ms: int = 1_000,
) -> OnsetAnalysis:
    frame_count = duration_ms // frame_ms + 2
    strength = np.zeros(frame_count, dtype=np.float64)
    for time_ms, value in onset_strengths.items():
        strength[time_ms // frame_ms] = value
    onset_ms = tuple(sorted(onset_strengths))
    rms = np.full(frame_count, -10.0 if active else -80.0)
    if active_ranges is not None:
        rms.fill(-80.0)
        for start_ms, end_ms in active_ranges:
            rms[start_ms // frame_ms : end_ms // frame_ms] = -10.0
    return OnsetAnalysis(
        sample_rate_hz=1_000,
        hop_length=frame_ms,
        strength=strength,
        band_strength=np.zeros((3, frame_count), dtype=np.float64),
        onset_ms=onset_ms,
        n_fft=frame_ms,
        activity=AudioActivity(
            frame_ms=float(frame_ms),
            rms_db=rms,
            floor_db=-20.0,
            active_onset_ms=onset_ms if active else (),
        ),
    )


def _tempo(*events: tuple[int, float]) -> LocalTempoMap:
    return LocalTempoMap(tuple(OsuBpmEvent(time_ms, bpm) for time_ms, bpm in events))


def test_classifies_active_low_attack_interval_covered_by_hold_as_sustain():
    analysis = _analysis(
        duration_ms=30_000,
        onset_strengths={
            1_000: 1.0,
            **{time_ms: 0.1 for time_ms in range(5_000, 21_000, 1_000)},
        },
    )

    result = classify_coverage_interval(
        [NoteEvent(4_000, 0, kind="HOLD", duration_ms=18_000)],
        analysis,
        _tempo((0, 120.0)),
        start_ms=4_000,
        end_ms=22_000,
        difficulty="EASY",
    )

    assert result.kind is CoverageKind.SUSTAIN_REPRESENTABLE
    assert result.hold_occupancy_ratio == 1.0
    assert result.strong_attack_count == 0
    assert result.beat_count == pytest.approx(36.0)


def test_hold_does_not_hide_repeated_strong_attacks():
    analysis = _analysis(
        duration_ms=30_000,
        onset_strengths={time_ms: 0.9 for time_ms in range(5_000, 21_000, 1_000)},
    )

    result = classify_coverage_interval(
        [NoteEvent(4_000, 0, kind="HOLD", duration_ms=18_000)],
        analysis,
        _tempo((0, 120.0)),
        start_ms=4_000,
        end_ms=22_000,
        difficulty="EXPERT",
    )

    assert result.kind is CoverageKind.ATTACK_REQUIRED
    assert result.strong_attack_count == 16


def test_repeated_strong_attacks_without_hold_require_attack_coverage():
    analysis = _analysis(
        duration_ms=30_000,
        onset_strengths={time_ms: 0.8 for time_ms in range(5_000, 21_000, 1_000)},
    )

    result = classify_coverage_interval(
        [],
        analysis,
        _tempo((0, 120.0)),
        start_ms=4_000,
        end_ms=22_000,
        difficulty="HARD",
    )

    assert result.kind is CoverageKind.ATTACK_REQUIRED
    assert result.hold_occupancy_ratio == 0.0


def test_strong_audio_attacks_are_not_vetoed_by_a_corrupted_tempo_integral():
    """Independent audio evidence must survive a pathological timing candidate.

    A very low local BPM can make a long, clearly active phrase integrate to
    fewer than 16 beats.  That is evidence against the timing authority, not
    evidence that the audible attacks disappeared.
    """

    analysis = _analysis(
        duration_ms=40_000,
        onset_strengths={time_ms: 0.9 for time_ms in range(5_000, 31_000, 1_000)},
    )

    result = classify_coverage_interval(
        [],
        analysis,
        _tempo((0, 4.3)),
        start_ms=4_000,
        end_ms=32_000,
        difficulty="HARD",
    )

    assert result.beat_count is not None
    assert result.beat_count < 16.0
    assert result.active_frame_ratio >= 0.35
    assert result.strong_attack_count >= 4
    assert result.kind is CoverageKind.ATTACK_REQUIRED


def test_song23_like_local_phrase_is_actionable_when_global_peak_count_misses_by_one():
    """Catch the source classifier discarding a locally clear EXPERT phrase.

    The target has seven active onsets, but only three reach the song-global
    saturated threshold.  Continuous target/neighbor RMS and three locally
    strong attacks are the independent corroboration observed in the frozen
    23번 4K EXPERT gap.
    """
    analysis = _analysis(
        duration_ms=40_000,
        onset_strengths={
            **{time_ms: 1.0 for time_ms in range(1_000, 11_000, 1_000)},
            16_000: 1.0,
            17_000: 1.0,
            21_000: 1.0,
            22_000: 1.0,
            23_000: 1.0,
            24_000: 0.95,
            25_000: 0.95,
            26_000: 0.95,
            27_000: 0.95,
            31_000: 1.0,
            32_000: 1.0,
        },
        active_ranges=((12_000, 36_000),),
        frame_ms=100,
    )

    result = classify_coverage_interval(
        [],
        analysis,
        _tempo((0, 120.0)),
        start_ms=20_000,
        end_ms=28_000,
        difficulty="EXPERT",
    )

    assert result.strong_attack_count == 3
    assert result.active_onset_count == 7
    assert result.kind is CoverageKind.ATTACK_REQUIRED
    assert result.to_report()["attackEvidenceScope"] == "LOCAL_CORROBORATED"


def test_song20_like_local_phrase_is_actionable_when_normal_global_count_misses():
    """Catch requiring six song-global peaks despite five local phrase peaks."""
    analysis = _analysis(
        duration_ms=30_000,
        onset_strengths={
            **{time_ms: 1.0 for time_ms in range(1_000, 6_000, 1_000)},
            11_000: 1.0,
            12_000: 1.0,
            13_000: 1.0,
            14_000: 1.0,
            15_000: 0.95,
            16_000: 0.9,
            17_000: 0.9,
            18_000: 0.9,
            21_000: 0.9,
            22_000: 0.9,
        },
        active_ranges=((7_000, 25_000),),
        frame_ms=100,
    )

    result = classify_coverage_interval(
        [],
        analysis,
        _tempo((0, 96.0), (15_000, 174.0)),
        start_ms=10_000,
        end_ms=19_000,
        difficulty="NORMAL",
    )

    assert result.strong_attack_count == 4
    assert result.active_onset_count == 8
    assert result.kind is CoverageKind.ATTACK_REQUIRED
    assert result.to_report()["localStrongAttackCount"] >= 5


def test_local_peak_route_rejects_quiet_neighbors_and_hold_occupied_phrase():
    """Catch local peak counts alone authorizing notes in breaks or under HOLDs."""
    analysis = _analysis(
        duration_ms=40_000,
        onset_strengths={
            **{time_ms: 1.0 for time_ms in range(1_000, 11_000, 1_000)},
            21_000: 1.0,
            22_000: 1.0,
            23_000: 1.0,
            24_000: 0.4,
            25_000: 0.4,
            26_000: 0.4,
            27_000: 0.4,
        },
        active_ranges=((20_000, 28_000),),
        frame_ms=100,
    )
    kwargs = {
        "onset_analysis": analysis,
        "tempo_map": _tempo((0, 120.0)),
        "start_ms": 20_000,
        "end_ms": 28_000,
        "difficulty": "EXPERT",
    }

    isolated = classify_coverage_interval(notes=[], **kwargs)
    sustained = classify_coverage_interval(
        notes=[NoteEvent(19_000, 0, kind="HOLD", duration_ms=10_000)],
        **kwargs,
    )

    assert isolated.kind is CoverageKind.UNCERTAIN
    assert sustained.kind is CoverageKind.SUSTAIN_COVERED


def test_missing_tempo_is_uncertain_while_positive_quiet_evidence_is_rest():
    with_activity = _analysis(
        duration_ms=30_000,
        onset_strengths={time_ms: 0.9 for time_ms in range(5_000, 21_000, 1_000)},
    )
    without_activity = _analysis(
        duration_ms=30_000,
        onset_strengths={time_ms: 0.9 for time_ms in range(5_000, 21_000, 1_000)},
        active=False,
    )

    no_tempo = classify_coverage_interval(
        [],
        with_activity,
        None,
        start_ms=4_000,
        end_ms=22_000,
        difficulty="HARD",
    )
    no_activity = classify_coverage_interval(
        [],
        without_activity,
        _tempo((0, 120.0)),
        start_ms=4_000,
        end_ms=22_000,
        difficulty="HARD",
    )

    assert no_tempo.kind is CoverageKind.UNCERTAIN
    assert no_tempo.evidence_confidence == "INSUFFICIENT"
    assert no_tempo.actionable is False
    assert no_activity.kind is CoverageKind.MUSICAL_REST_OR_SIMPLIFICATION
    assert no_activity.evidence_confidence == "SUFFICIENT"
    assert no_activity.actionable is False


def test_active_sustain_without_detected_onsets_is_not_mislabeled_as_rest():
    """Catch a soft sustained phrase being treated as proven musical silence."""
    analysis = _analysis(
        duration_ms=30_000,
        onset_strengths={},
        active=True,
    )
    kwargs = {
        "onset_analysis": analysis,
        "tempo_map": _tempo((0, 120.0)),
        "start_ms": 4_000,
        "end_ms": 22_000,
        "difficulty": "EASY",
    }

    uncovered = classify_coverage_interval(notes=[], **kwargs)
    sustained = classify_coverage_interval(
        notes=[NoteEvent(4_000, 0, kind="HOLD", duration_ms=18_000)],
        **kwargs,
    )

    assert uncovered.kind is CoverageKind.UNCERTAIN
    assert sustained.kind is CoverageKind.SUSTAIN_COVERED


def test_integrates_variable_bpm_inside_interval():
    analysis = _analysis(
        duration_ms=25_000,
        onset_strengths={time_ms: 0.9 for time_ms in range(1_000, 20_000, 1_000)},
    )

    result = classify_coverage_interval(
        [],
        analysis,
        _tempo((0, 60.0), (10_000, 180.0)),
        start_ms=0,
        end_ms=20_000,
        difficulty="HARD",
    )

    assert result.beat_count == pytest.approx(40.0)


def test_song_relative_attack_classification_is_gain_invariant():
    raw = np.array([0.0, 0.2, 0.4, 0.8, 0.3], dtype=np.float64)
    assert normalize_envelope(raw).tolist() == pytest.approx(
        normalize_envelope(raw * 0.01).tolist()
    )

    first = _analysis(
        duration_ms=30_000,
        onset_strengths={time_ms: value for time_ms, value in zip(
            range(1_000, 21_000, 1_000),
            normalize_envelope(np.linspace(0.2, 1.0, 20)),
            strict=True,
        )},
    )
    second = _analysis(
        duration_ms=30_000,
        onset_strengths={time_ms: value for time_ms, value in zip(
            range(1_000, 21_000, 1_000),
            normalize_envelope(np.linspace(0.2, 1.0, 20) * 0.01),
            strict=True,
        )},
    )

    kwargs = {
        "notes": [],
        "tempo_map": _tempo((0, 120.0)),
        "start_ms": 0,
        "end_ms": 20_000,
        "difficulty": "HARD",
    }
    assert classify_coverage_interval(onset_analysis=first, **kwargs).kind is (
        classify_coverage_interval(onset_analysis=second, **kwargs).kind
    )


def test_easy_never_requires_less_attack_evidence_than_expert():
    analysis = _analysis(
        duration_ms=30_000,
        onset_strengths={
            1_000: 1.0,
            5_000: 0.9,
            8_000: 0.9,
            11_000: 0.9,
            14_000: 0.9,
            17_000: 0.9,
        },
    )
    kwargs = {
        "notes": [],
        "onset_analysis": analysis,
        "tempo_map": _tempo((0, 120.0)),
        "start_ms": 4_000,
        "end_ms": 20_000,
    }

    easy = classify_coverage_interval(difficulty="EASY", **kwargs)
    expert = classify_coverage_interval(difficulty="EXPERT", **kwargs)

    assert easy.kind is CoverageKind.INSUFFICIENT_EVIDENCE
    assert expert.kind is CoverageKind.ATTACK_REQUIRED


def test_union_hold_occupancy_does_not_double_count_overlapping_lanes():
    analysis = _analysis(
        duration_ms=30_000,
        onset_strengths={
            1_000: 1.0,
            **{time_ms: 0.1 for time_ms in range(5_000, 21_000, 1_000)},
        },
    )

    result = classify_coverage_interval(
        [
            NoteEvent(4_000, 0, kind="HOLD", duration_ms=18_000),
            NoteEvent(5_000, 1, kind="HOLD", duration_ms=10_000),
        ],
        analysis,
        _tempo((0, 120.0)),
        start_ms=4_000,
        end_ms=22_000,
        difficulty="EASY",
    )

    assert result.hold_occupancy_ratio == 1.0
