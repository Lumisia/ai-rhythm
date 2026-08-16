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
) -> OnsetAnalysis:
    frame_count = duration_ms // 1_000 + 2
    strength = np.zeros(frame_count, dtype=np.float64)
    for time_ms, value in onset_strengths.items():
        strength[time_ms // 1_000] = value
    onset_ms = tuple(sorted(onset_strengths))
    return OnsetAnalysis(
        sample_rate_hz=1_000,
        hop_length=1_000,
        strength=strength,
        band_strength=np.zeros((3, frame_count), dtype=np.float64),
        onset_ms=onset_ms,
        n_fft=1_000,
        activity=AudioActivity(
            frame_ms=1_000.0,
            rms_db=np.full(frame_count, -10.0 if active else -80.0),
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


def test_missing_tempo_or_activity_is_insufficient_not_hard_failure():
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

    assert no_tempo.kind is CoverageKind.INSUFFICIENT_EVIDENCE
    assert no_tempo.evidence_confidence == "INSUFFICIENT"
    assert no_activity.kind is CoverageKind.INSUFFICIENT_EVIDENCE


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
