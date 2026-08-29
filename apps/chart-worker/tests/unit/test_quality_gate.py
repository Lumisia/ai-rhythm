from dataclasses import replace
from importlib import import_module
from pathlib import Path

import numpy as np
import pytest

from chart_worker.analysis.activity import AudioActivity
from chart_worker.analysis.onset import OnsetAnalysis
from chart_worker.generation.mapperatorinator import GeneratedChart
from chart_worker.generation.osu_parser import OsuBpmEvent
from chart_worker.schema.note import NoteEvent
from chart_worker.stages.types import SongTimingAuthority
from chart_worker.validation.timing_integrity import (
    TimingIntegrityAssessment,
    TimingIntegrityStatus,
)


def _gate():
    return import_module("chart_worker.validation.quality_gate")


def _authority(
    events: tuple[OsuBpmEvent, ...] = (OsuBpmEvent(0, 120.0),),
) -> SongTimingAuthority:
    return SongTimingAuthority(
        reference_path=Path("timing-reference.osu"),
        sha256="reference",
        audio_sha256="audio",
        bpm_events=events,
        generator_name="test",
        seed=0,
        mode="STANDARD",
        attempt_count=1,
    )


def _analysis(
    onsets: tuple[int, ...],
    *,
    activity: AudioActivity | None = None,
    strengths: dict[int, float] | None = None,
) -> OnsetAnalysis:
    frame_count = max(601, max(onsets, default=0) // 100 + 2)
    strength = np.zeros(frame_count)
    resolved_strengths = (
        strengths if strengths is not None else {time_ms: 1.0 for time_ms in onsets}
    )
    for time_ms, value in resolved_strengths.items():
        strength[min(frame_count - 1, round(time_ms / 100))] = value
    return OnsetAnalysis(
        sample_rate_hz=1_000,
        hop_length=100,
        strength=strength,
        band_strength=np.zeros((3, frame_count)),
        onset_ms=onsets,
        activity=activity,
    )


def _chart(
    rows: tuple[int, ...],
    *,
    key_mode: int = 4,
    events: tuple[OsuBpmEvent, ...] = (OsuBpmEvent(0, 120.0),),
    lane: int = 0,
) -> GeneratedChart:
    return GeneratedChart(
        notes=[NoteEvent(row, lane) for row in rows],
        key_mode=key_mode,
        osu_text="",
        generator_name="test",
        seed=0,
        bpm_events=events,
    )


def _evaluate(
    chart: GeneratedChart,
    onsets: tuple[int, ...],
    *,
    duration_ms: int,
    authority: SongTimingAuthority | None = None,
    difficulty: str = "EASY",
    activity: AudioActivity | None = None,
    boundary_policy_mode: str = "SHADOW",
):
    return _gate().evaluate_chart_candidate(
        chart,
        authority or _authority(),
        _analysis(onsets, activity=activity),
        requested_key_mode=4,
        requested_difficulty=difficulty,
        duration_ms=duration_ms,
        boundary_policy_mode=boundary_policy_mode,
    )


def _decision(result, axis: str):
    return result.decision(_gate().GateAxis(axis))


def test_acceptance_keeps_shared_section_profile_and_report():
    rows = (1_000, 16_000, 31_000, 46_000)
    result = _evaluate(_chart(rows), rows, duration_ms=60_000)

    assert result.profile is not None
    assert result.profile.active_section_mask == (True, True, True, True)
    assert _decision(result, "PATTERN").action is _gate().GateAction.PASS
    report = result.to_report()
    assert report["qualityProfile"] == result.profile.to_report()


@pytest.mark.parametrize(
    ("status", "action", "reason"),
    [
        (
            TimingIntegrityStatus.NEEDS_CORROBORATION,
            "REVIEW",
            "TIMING_AUTHORITY_NEEDS_CORROBORATION",
        ),
        (
            TimingIntegrityStatus.DAMAGED,
            "RETRY_MAP",
            "TIMING_AUTHORITY_DAMAGED",
        ),
    ],
)
def test_timing_integrity_uncertainty_is_visible_in_chart_acceptance(
    status,
    action,
    reason,
):
    authority = replace(
        _authority(),
        timing_integrity=TimingIntegrityAssessment(
            status=status,
            reasons=("FIXTURE",),
            islands=(),
        ),
    )

    decision = _gate()._timing_identity_decision(_chart((500,)), authority)

    assert decision.action.value == action
    assert decision.reasons == (reason,)


def test_high_confidence_terminal_silence_rejects_a_hold_releasing_deep_in_silence():
    terminal = import_module("chart_worker.analysis.terminal_silence")
    activity = AudioActivity(
        frame_ms=100.0,
        rms_db=np.array([-10.0] * 201 + [-80.0] * 100),
        floor_db=-60.0,
        active_onset_ms=tuple(range(500, 20_001, 500)),
    )
    observation = terminal.TerminalSilenceObservation(
        version="terminal-silence-observation-v1",
        duration_ms=30_000,
        frame_ms=100,
        channel_count=2,
        candidates=tuple(
            terminal.TerminalThresholdCandidate(
                rms_db=rms_db,
                peak_db=peak_db,
                suffix_start_ms=20_000,
                suffix_duration_ms=10_000,
            )
            for rms_db, peak_db in terminal.DEFAULT_THRESHOLDS_DB
        ),
        candidate_spread_ms=0,
        last_onset_ms=20_000,
    )
    analysis = _analysis(activity.active_onset_ms, activity=activity)
    analysis = replace(analysis, terminal_silence=observation)
    chart = GeneratedChart(
        notes=[NoteEvent(19_000, 0, kind="HOLD", duration_ms=5_000)],
        key_mode=4,
        osu_text="",
        generator_name="terminal-hold-fixture",
        seed=0,
        bpm_events=(OsuBpmEvent(0, 120.0),),
    )

    result = _gate().evaluate_chart_candidate(
        chart,
        _authority(),
        analysis,
        requested_key_mode=4,
        requested_difficulty="EASY",
        duration_ms=30_000,
        boundary_policy_mode="HIGH_CONFIDENCE_ENFORCED",
    )

    assert _decision(result, "SONG_BOUNDS").action is _gate().GateAction.RETRY_MAP
    assert _decision(result, "SONG_BOUNDS").reasons == ("HOLD_END_AFTER_RELEASE",)


def test_terminal_cut_accepts_upstream_ten_ms_hold_serialization_tolerance():
    terminal = import_module("chart_worker.analysis.terminal_silence")
    activity = AudioActivity(
        frame_ms=100.0,
        rms_db=np.array([-10.0] * 201 + [-80.0] * 100),
        floor_db=-60.0,
        active_onset_ms=tuple(range(500, 20_001, 500)),
    )
    observation = terminal.TerminalSilenceObservation(
        version="terminal-silence-observation-v1",
        duration_ms=30_000,
        frame_ms=100,
        channel_count=2,
        candidates=tuple(
            terminal.TerminalThresholdCandidate(
                rms_db=rms_db,
                peak_db=peak_db,
                suffix_start_ms=20_000,
                suffix_duration_ms=10_000,
            )
            for rms_db, peak_db in terminal.DEFAULT_THRESHOLDS_DB
        ),
        candidate_spread_ms=0,
        last_onset_ms=20_000,
    )
    analysis = replace(
        _analysis(activity.active_onset_ms, activity=activity),
        terminal_silence=observation,
    )
    chart = GeneratedChart(
        notes=[NoteEvent(19_000, 0, kind="HOLD", duration_ms=1_010)],
        key_mode=4,
        osu_text="",
        generator_name="terminal-hold-tolerance-fixture",
        seed=0,
        bpm_events=(OsuBpmEvent(0, 120.0),),
    )

    result = _gate().evaluate_chart_candidate(
        chart,
        _authority(),
        analysis,
        requested_key_mode=4,
        requested_difficulty="EASY",
        duration_ms=30_000,
        boundary_policy_mode="HIGH_CONFIDENCE_ENFORCED",
    )

    assert _decision(result, "SONG_BOUNDS").action is _gate().GateAction.PASS


def test_acceptance_uses_every_local_bpm_segment_for_hold_beats():
    events = (OsuBpmEvent(0, 120.0), OsuBpmEvent(1_000, 240.0))
    chart = GeneratedChart(
        notes=[NoteEvent(500, 0, kind="HOLD", duration_ms=1_000)],
        key_mode=4,
        osu_text="",
        generator_name="test",
        seed=0,
        bpm_events=events,
    )

    result = _evaluate(
        chart,
        (500,),
        duration_ms=2_000,
        authority=_authority(events),
    )

    assert result.profile is not None
    assert result.profile.difficulty_vector_v2.mean_hold_beats == pytest.approx(3.0)


def test_structural_retry_is_not_hidden_by_profile_construction():
    result = _evaluate(
        _chart((1_000,), lane=4),
        (1_000,),
        duration_ms=2_000,
    )

    assert result.action is _gate().GateAction.RETRY_MAP
    assert result.profile is None
    assert _decision(result, "STRUCTURE").action is _gate().GateAction.RETRY_MAP
    assert _decision(result, "SONG_BOUNDS").action is _gate().GateAction.PASS
    assert _decision(result, "SONG_BOUNDS").reasons == (
        "NOT_EVALUATED_STRUCTURE_INVALID",
    )
    assert _decision(result, "PATTERN").reasons == (
        "PROFILE_UNAVAILABLE_STRUCTURE_INVALID",
    )
    assert result.to_report()["qualityProfile"] is None


def test_onsets_without_activity_evidence_review_instead_of_retrying():
    result = _evaluate(
        _chart((30_000,)),
        tuple(range(1_000, 21_000, 1_000)) + (30_000,),
        duration_ms=40_000,
    )

    assert result.action is _gate().GateAction.REVIEW
    assert _decision(result, "COVERAGE").action is _gate().GateAction.REVIEW
    assert _decision(result, "COVERAGE").reasons == (
        "UNCERTAIN_COVERAGE_EVIDENCE_LEADING_GAP",
    )


def test_quality_gate_uses_musical_coverage_horizon_for_trailing_gap():
    activity = AudioActivity(
        frame_ms=1_000.0,
        rms_db=np.array([-80.0] * 70 + [-10.0] * 6 + [-80.0] * 12),
        floor_db=-60.0,
        active_onset_ms=tuple(range(71_000, 85_000, 1_000)),
    )
    onsets = (0, 70_000, *range(71_000, 85_000, 1_000))

    result = _evaluate(
        _chart((0, 70_000)),
        onsets,
        duration_ms=88_000,
        activity=activity,
        boundary_policy_mode="EXPERIMENTAL_ENFORCED",
    )

    assert _decision(result, "COVERAGE").action is _gate().GateAction.RETRY_MAP
    assert _decision(result, "COVERAGE").reasons == (
        "ATTACK_REQUIRED_TRAILING_GAP",
    )
    assert result.timing.coverage_gaps[0].end_ms == 84_000


def test_song_bounds_reject_hold_end_beyond_completion_horizon():
    activity = AudioActivity(
        frame_ms=1_000.0,
        rms_db=np.array([-10.0] * 80 + [-80.0] * 20),
        floor_db=-60.0,
        active_onset_ms=(79_000, 80_000),
    )
    chart = GeneratedChart(
        notes=[NoteEvent(79_000, 0, kind="HOLD", duration_ms=16_000)],
        key_mode=4,
        osu_text="",
        generator_name="test",
        seed=0,
        bpm_events=(OsuBpmEvent(0, 120.0),),
    )

    result = _evaluate(
        chart,
        (79_000, 80_000),
        duration_ms=100_000,
        activity=activity,
        boundary_policy_mode="EXPERIMENTAL_ENFORCED",
    )

    assert result.action is _gate().GateAction.RETRY_MAP
    assert _decision(result, "STRUCTURE").action is _gate().GateAction.PASS
    assert _decision(result, "SONG_BOUNDS").action is _gate().GateAction.RETRY_MAP
    assert _decision(result, "SONG_BOUNDS").reasons == (
        "HOLD_END_AFTER_RELEASE",
    )


@pytest.mark.parametrize(
    ("rows", "onsets", "duration_ms", "reason"),
    [
        (
            (0, 20_000, 40_000),
            tuple(range(21_000, 40_000, 1_000)),
            60_000,
            "UNCERTAIN_COVERAGE_EVIDENCE_MIDDLE_GAP",
        ),
        (
            (0, 10_000),
            tuple(range(11_000, 31_000, 1_000)),
            40_000,
            "UNCERTAIN_COVERAGE_EVIDENCE_TRAILING_GAP",
        ),
    ],
)
def test_active_gap_positions_have_stable_reasons(rows, onsets, duration_ms, reason):
    result = _evaluate(_chart(rows), onsets, duration_ms=duration_ms)

    assert _decision(result, "COVERAGE").reasons == (reason,)


def test_missing_onset_evidence_requires_review_not_retry():
    result = _evaluate(_chart((500,)), (), duration_ms=1_000)

    assert result.action is _gate().GateAction.REVIEW
    assert _decision(result, "TIMING_ALIGNMENT").action is _gate().GateAction.REVIEW
    assert "LOW_OVERALL_ONSET_SUPPORT" in _decision(result, "TIMING_ALIGNMENT").reasons


def test_global_sixty_millisecond_corruption_retries_timing_alignment():
    rows = tuple(range(1_000, 17_000, 1_000))
    result = _evaluate(_chart(rows), tuple(row - 60 for row in rows), duration_ms=20_000)

    assert result.timing.overall.precision_50 < 0.70
    assert result.timing.overall.absolute_p95_ms >= 60
    assert _decision(result, "TIMING_ALIGNMENT").action is _gate().GateAction.RETRY_MAP
    assert _decision(result, "TIMING_ALIGNMENT").reasons == ("OVERALL_TIMING_MISALIGNED",)


def test_sparse_onset_support_with_aligned_global_phase_is_advisory():
    rows = tuple(range(1_000, 61_000, 1_000))
    onsets = tuple(row for index, row in enumerate(rows) if index % 3 != 0)
    result = _evaluate(_chart(rows), onsets, duration_ms=61_000)

    assert result.timing.overall.precision_50 < 0.70
    assert result.timing.overall.absolute_p95_ms >= 60
    assert abs(result.timing.overall.signed_median_ms or 0.0) < 20
    decision = _decision(result, "TIMING_ALIGNMENT")
    assert decision.action is _gate().GateAction.PASS
    assert "OVERALL_TIMING_WEAK_SUPPORT" in decision.reasons


def test_single_section_onset_shift_is_advisory_at_map_stage():
    rows = tuple(range(1_000, 60_000, 1_000))
    # At 120 BPM the production diagnostic now uses 32-beat (16-second)
    # sections. Corrupt one complete interior section.
    corrupted = set(range(33_000, 48_000, 1_000))
    onsets = tuple(row - 60 if row in corrupted else row for row in rows)
    result = _evaluate(_chart(rows), onsets, duration_ms=60_000)

    section = result.timing.sections[2]
    assert section.metrics.precision_50 < 0.60
    assert section.metrics.absolute_p95_ms >= 60
    decision = _decision(result, "TIMING_ALIGNMENT")
    assert decision.action is _gate().GateAction.PASS
    assert "SECTION_TIMING_WEAK_SUPPORT" in decision.reasons
    assert "SECTION_PHASE_DELTA" in decision.reasons


def test_phase_only_section_drift_is_advisory_without_retry():
    first_rows = tuple(range(1_000, 9_000, 1_000))
    second_rows = tuple(range(31_000, 39_000, 1_000))
    rows = first_rows + second_rows
    onsets = tuple(row + 5 for row in first_rows) + tuple(row - 50 for row in second_rows)
    result = _evaluate(_chart(rows), onsets, duration_ms=60_000)

    assert result.action is _gate().GateAction.PASS
    assert _decision(result, "TIMING_ALIGNMENT").action is _gate().GateAction.PASS
    assert "SECTION_PHASE_DELTA" in _decision(result, "TIMING_ALIGNMENT").reasons


@pytest.mark.parametrize(
    ("shifted_rows", "offset"),
    [
        pytest.param(
            (1_000, 2_000, 3_000, 4_000, 5_000), 55, id="precision-only"
        ),
        pytest.param((1_000, 2_000), 60, id="p95-only"),
    ],
)
def test_one_weak_overall_signal_is_advisory_not_blocking(shifted_rows, offset):
    rows = tuple(range(1_000, 15_000, 1_000))
    result = _evaluate(
        _chart(rows),
        tuple(row - offset if row in shifted_rows else row for row in rows),
        duration_ms=15_000,
    )

    decision = _decision(result, "TIMING_ALIGNMENT")
    assert decision.action is _gate().GateAction.PASS
    assert "OVERALL_TIMING_WEAK_SUPPORT" in decision.reasons


def test_section_p95_only_support_is_advisory_without_phase_reason():
    rows = (
        tuple(range(1_000, 10_000, 1_000))
        + tuple(range(16_000, 30_000, 1_000))
        + tuple(range(31_000, 45_000, 1_000))
    )
    first_section_offsets = {1_000: 60, 2_000: 60, 3_000: 60}
    result = _evaluate(
        _chart(rows),
        tuple(row - first_section_offsets.get(row, 0) for row in rows),
        duration_ms=45_000,
    )

    decision = _decision(result, "TIMING_ALIGNMENT")
    assert decision.action is _gate().GateAction.PASS
    assert "SECTION_TIMING_WEAK_SUPPORT" in decision.reasons
    assert "SECTION_PHASE_DELTA" not in decision.reasons


def test_low_section_precision_without_phase_delta_is_advisory():
    rows = (
        tuple(range(1_000, 9_000, 1_000))
        + tuple(range(16_000, 30_000, 1_000))
        + tuple(range(31_000, 45_000, 1_000))
    )
    offsets = {
        row: 55 if index % 2 == 0 else -55
        for index, row in enumerate(range(1_000, 9_000, 1_000))
    }
    result = _evaluate(
        _chart(rows),
        tuple(row - offsets.get(row, 0) for row in rows),
        duration_ms=45_000,
    )

    decision = _decision(result, "TIMING_ALIGNMENT")
    assert decision.action is _gate().GateAction.PASS
    assert "SECTION_TIMING_WEAK_SUPPORT" in decision.reasons
    assert "SECTION_PHASE_DELTA" not in decision.reasons


def test_correlated_section_onset_errors_without_phase_drift_are_advisory():
    first_section = tuple(range(1_000, 9_000, 1_000))
    later_sections = tuple(range(16_000, 45_000, 1_000))
    rows = first_section + later_sections
    offsets = {
        row: 100 if index % 2 == 0 else -100
        for index, row in enumerate(first_section)
    }
    result = _evaluate(
        _chart(rows),
        tuple(row - offsets.get(row, 0) for row in rows),
        duration_ms=45_000,
    )

    section = result.timing.sections[0]
    assert section.status == "REVIEW"
    assert section.metrics.precision_50 < 0.60
    assert section.metrics.absolute_p95_ms >= 60
    assert abs(section.phase_delta_ms or 0) <= 25
    decision = _decision(result, "TIMING_ALIGNMENT")
    assert decision.action is _gate().GateAction.PASS
    assert "SECTION_TIMING_WEAK_SUPPORT" in decision.reasons


def test_whole_beat_section_phase_is_metrically_equivalent_not_misaligned():
    events = (OsuBpmEvent(190, 150.0),)  # 400ms per beat
    first_section = tuple(1_390 + 1_200 * index for index in range(9))
    later_sections = tuple(range(16_190, 44_191, 400))
    rows = first_section + later_sections
    onsets = tuple(row - 390 for row in first_section) + later_sections
    result = _evaluate(
        _chart(rows, events=events),
        onsets,
        duration_ms=45_000,
        authority=_authority(events),
    )

    section = result.timing.sections[0]
    assert section.metrics.precision_50 < 0.60
    assert section.phase_delta_ms == 390.0
    decision = _decision(result, "TIMING_ALIGNMENT")
    assert decision.action is _gate().GateAction.PASS
    assert "SECTION_TIMING_WEAK_SUPPORT" in decision.reasons
    assert "SECTION_PHASE_DELTA" not in decision.reasons


@pytest.mark.parametrize("active_onset_ms", [(), (1_000, 2_000)])
def test_missing_or_low_active_onset_support_requires_review(active_onset_ms):
    rows = tuple(range(1_000, 9_000, 1_000))
    result = _evaluate(
        _chart(rows),
        rows,
        duration_ms=15_000,
        activity=AudioActivity(
            frame_ms=1_000.0,
            rms_db=np.full(15, -10.0),
            floor_db=-20.0,
            active_onset_ms=active_onset_ms,
        ),
    )

    assert result.action is _gate().GateAction.REVIEW
    assert _decision(result, "TIMING_ALIGNMENT").reasons == ("LOW_ACTIVE_ONSET_SUPPORT",)


def test_missing_activity_does_not_treat_a_short_aligned_chart_as_low_active_support():
    rows = tuple(range(1_000, 8_000, 1_000))
    result = _evaluate(_chart(rows), rows, duration_ms=15_000)

    decision = _decision(result, "TIMING_ALIGNMENT")
    assert decision.action is _gate().GateAction.PASS
    assert decision.reasons == ("SECTION_TIMING_INSUFFICIENT",)


def test_quiet_coverage_gap_is_advisory_with_a_position_specific_reason():
    result = _evaluate(
        _chart((0, *range(31_000, 39_000, 1_000))),
        (0, *range(1_000, 21_000, 1_000), *range(31_000, 39_000, 1_000)),
        duration_ms=40_000,
        activity=AudioActivity(
            frame_ms=1_000.0,
            rms_db=np.array([-30.0] * 30 + [-10.0] * 10),
            floor_db=-20.0,
            active_onset_ms=tuple(range(31_000, 39_000, 1_000)),
        ),
    )

    decision = _decision(result, "COVERAGE")
    assert decision.action is _gate().GateAction.PASS
    assert decision.reasons == ("QUIET_POST_FIRST_GAP",)
    assert result.action is _gate().GateAction.PASS


def test_sustain_representable_gap_requires_review_not_full_map_retry():
    onsets = (1_000, *range(5_000, 21_000, 1_000), 30_000)
    analysis = _analysis(
        onsets,
        strengths={1_000: 1.0, 30_000: 1.0, **{time: 0.1 for time in onsets[1:-1]}},
        activity=AudioActivity(
            frame_ms=1_000.0,
            rms_db=np.full(42, -10.0),
            floor_db=-20.0,
            active_onset_ms=onsets,
        ),
    )
    chart = GeneratedChart(
        notes=[
            NoteEvent(4_000, 0, kind="HOLD", duration_ms=26_000),
            NoteEvent(30_000, 0),
        ],
        key_mode=4,
        osu_text="",
        generator_name="test",
        seed=0,
        bpm_events=(OsuBpmEvent(0, 120.0),),
    )

    result = _gate().evaluate_chart_candidate(
        chart,
        _authority(),
        analysis,
        requested_key_mode=4,
        requested_difficulty="EASY",
        duration_ms=40_000,
    )

    decision = _decision(result, "COVERAGE")
    assert decision.action is _gate().GateAction.REVIEW
    assert "SUSTAIN_COVERED_POST_FIRST_GAP" in decision.reasons


def test_repeated_strong_attacks_under_a_fully_occupying_hold_retry_map():
    onsets = tuple(range(5_000, 21_000, 1_000))
    analysis = _analysis(
        onsets,
        strengths={time: 0.9 for time in onsets},
        activity=AudioActivity(
            frame_ms=1_000.0,
            rms_db=np.full(42, -10.0),
            floor_db=-20.0,
            active_onset_ms=onsets,
        ),
    )
    chart = GeneratedChart(
        notes=[
            NoteEvent(4_000, 0, kind="HOLD", duration_ms=26_000),
            NoteEvent(30_000, 0),
        ],
        key_mode=4,
        osu_text="",
        generator_name="test",
        seed=0,
        bpm_events=(OsuBpmEvent(0, 120.0),),
    )

    result = _gate().evaluate_chart_candidate(
        chart,
        _authority(),
        analysis,
        requested_key_mode=4,
        requested_difficulty="EXPERT",
        duration_ms=40_000,
    )

    decision = _decision(result, "COVERAGE")
    assert decision.action is _gate().GateAction.RETRY_MAP
    assert "ATTACK_REQUIRED_POST_FIRST_GAP" in decision.reasons


def test_near_active_quiet_trailing_gap_requires_review_without_retrying_the_map():
    rows = tuple(range(0, 91_000, 1_000))
    trailing_onsets = tuple(range(91_000, 100_000, 1_000))
    result = _evaluate(
        _chart(rows),
        (*rows, *trailing_onsets),
        duration_ms=100_000,
        activity=AudioActivity(
            frame_ms=1_000.0,
            rms_db=np.array([-10.0] * 90 + [-10.0] * 3 + [-80.0] * 7),
            floor_db=-20.0,
            active_onset_ms=(*rows, *trailing_onsets[:6]),
        ),
    )

    gap = result.timing.quiet_coverage_gaps[-1]
    assert gap.position == "TRAILING"
    assert gap.active_onset_count == 6
    assert gap.active_frame_ratio == 0.3
    decision = _decision(result, "COVERAGE")
    assert decision.action is _gate().GateAction.REVIEW
    assert decision.reasons == (
        "QUIET_TRAILING_GAP",
        "QUIET_TRAILING_GAP_NEAR_ACTIVE_THRESHOLD",
    )
    assert result.action is _gate().GateAction.REVIEW


def test_clearly_quiet_trailing_gap_remains_advisory_pass():
    rows = tuple(range(0, 91_000, 1_000))
    trailing_onsets = tuple(range(91_000, 100_000, 1_000))
    result = _evaluate(
        _chart(rows),
        (*rows, *trailing_onsets),
        duration_ms=100_000,
        activity=AudioActivity(
            frame_ms=1_000.0,
            rms_db=np.array([-10.0] * 90 + [-10.0] + [-80.0] * 9),
            floor_db=-20.0,
            active_onset_ms=(*rows, *trailing_onsets[:2]),
        ),
    )

    decision = _decision(result, "COVERAGE")
    assert decision.action is _gate().GateAction.PASS
    assert decision.reasons == ("QUIET_TRAILING_GAP",)


def test_insufficient_tail_section_cannot_force_a_map_retry():
    aligned = tuple(range(1_000, 41_000, 1_000))
    tail = (90_500, 92_500, 94_500, 96_500, 98_500)
    rows = aligned + tail
    onsets = aligned + tuple(row - 800 for row in tail)

    result = _evaluate(_chart(rows), onsets, duration_ms=100_000)

    tail_section = result.timing.sections[-1]
    assert tail_section.status == "INSUFFICIENT"
    assert tail_section.metrics.precision_50 < 0.60
    decision = _decision(result, "TIMING_ALIGNMENT")
    assert decision.action is _gate().GateAction.PASS
    assert decision.reasons == (
        "OVERALL_TIMING_WEAK_SUPPORT",
        "SECTION_TIMING_INSUFFICIENT",
    )


def test_timing_identity_mismatch_retries_with_a_machine_reason():
    result = _evaluate(
        _chart((500,), events=(OsuBpmEvent(0, 121.0),)),
        (500,),
        duration_ms=1_000,
    )

    assert _decision(result, "TIMING_IDENTITY").action is _gate().GateAction.RETRY_MAP
    assert _decision(result, "TIMING_IDENTITY").reasons == ("TIMING_REFERENCE_MISMATCH",)


def test_structural_invalidity_is_a_machine_retry_reason_not_exception_prose():
    result = _evaluate(_chart((500,), lane=4), (500,), duration_ms=1_000)

    decision = _decision(result, "STRUCTURE")
    assert decision.action is _gate().GateAction.RETRY_MAP
    assert decision.reasons == ("STRUCTURE_INVALID",)


def test_duplicate_structure_failure_preserves_machine_context():
    result = _evaluate(
        _chart((500, 500), lane=1),
        (500,),
        duration_ms=1_000,
    )

    assert result.structure_error == {
        "reasonCode": "DUPLICATE_NOTE",
        "context": {"lane": 1, "timeMs": 500, "noteKind": "TAP"},
    }
    assert result.to_report()["structureError"] == result.structure_error


@pytest.mark.parametrize(
    ("rows", "expected_action", "expected_reason"),
    [
        (
            (281, 781, 1_281, 1_781, 2_281, 2_781, 3_281, 3_781, 4_281, 4_781),
            "RETRY_MAP",
            "NOTE_GRID_MISALIGNED",
        ),
        (
            (500, 1_000, 1_500, 2_000, 2_500, 3_000, 3_500, 4_000, 4_500, 5_000, 5_281, 5_781),
            "REVIEW",
            "NOTE_GRID_WEAK_SUPPORT",
        ),
        (
            (500, 1_000, 1_500, 2_000, 2_500, 3_000, 3_500, 4_000, 4_500, 4_520, 5_020, 5_520),
            "REVIEW",
            "NOTE_GRID_WEAK_SUPPORT",
        ),
    ],
)
def test_note_grid_requires_both_weak_signals_for_retry(rows, expected_action, expected_reason):
    result = _evaluate(_chart(rows), rows, duration_ms=15_000)

    decision = _decision(result, "TIMING_ALIGNMENT")
    assert decision.action.value == expected_action
    assert expected_reason in decision.reasons


def test_axes_remain_independent_and_retry_beats_review_and_pass():
    retry = _evaluate(
        _chart((30_000,), events=(OsuBpmEvent(0, 121.0),)),
        tuple(range(1_000, 21_000, 1_000)) + (30_000,),
        duration_ms=40_000,
    )
    review = _evaluate(_chart((500,)), (), duration_ms=1_000)
    passed = _evaluate(
        _chart(tuple(range(1_000, 9_000, 1_000))),
        tuple(range(1_000, 9_000, 1_000)),
        duration_ms=15_000,
    )

    assert retry.action is _gate().GateAction.RETRY_MAP
    assert retry.decisions and {decision.axis.value for decision in retry.decisions} == {
        "STRUCTURE",
        "SONG_BOUNDS",
        "TIMING_IDENTITY",
        "TIMING_ALIGNMENT",
        "COVERAGE",
        "PATTERN",
    }
    assert review.action is _gate().GateAction.REVIEW
    assert passed.action is _gate().GateAction.PASS


def test_candidate_disposition_separates_hard_reject_from_repairable_quality():
    structural = _evaluate(
        _chart((1_000,), lane=4),
        (1_000,),
        duration_ms=2_000,
    )
    coverage = _evaluate(
        _chart((30_000,)),
        tuple(range(1_000, 21_000, 1_000)) + (30_000,),
        duration_ms=40_000,
    )

    assert structural.disposition is _gate().CandidateDisposition.HARD_REJECT
    assert structural.repair_eligible is False
    assert coverage.disposition is _gate().CandidateDisposition.REVIEW
    assert coverage.repair_eligible is False
    assert coverage.repair_axes == ()


def test_quality_defect_without_a_supported_repair_axis_is_preserved_as_unresolved():
    result = _evaluate(
        _chart(tuple(range(1_000, 9_000, 1_000))),
        tuple(range(1_000, 9_000, 1_000)),
        duration_ms=15_000,
    )
    decisions = tuple(
        replace(
            decision,
            action=_gate().GateAction.RETRY_MAP,
            reasons=("PATTERN_DENSITY_OUT_OF_RANGE",),
        )
        if decision.axis is _gate().GateAxis.PATTERN
        else decision
        for decision in result.decisions
    )
    pattern_retry = replace(
        result,
        action=_gate().GateAction.RETRY_MAP,
        decisions=decisions,
    )

    assert pattern_retry.disposition is _gate().CandidateDisposition.QUALITY_DEFECT
    assert pattern_retry.repair_eligible is False
    assert pattern_retry.repair_axes == ()
    report = pattern_retry.to_report()
    assert report["candidateDisposition"] == "QUALITY_DEFECT"
    assert report["repairEligible"] is False
    assert report["repairAxes"] == []


def test_candidate_disposition_keeps_review_and_admit_distinct():
    review = _evaluate(_chart((500,)), (), duration_ms=1_000)
    passed = _evaluate(
        _chart(tuple(range(1_000, 9_000, 1_000))),
        tuple(range(1_000, 9_000, 1_000)),
        duration_ms=15_000,
    )

    assert review.disposition is _gate().CandidateDisposition.REVIEW
    assert passed.disposition is _gate().CandidateDisposition.ADMIT
    assert passed.to_report()["candidateDisposition"] == "ADMIT"


def test_invalid_difficulty_is_rejected_before_evaluation():
    with pytest.raises(ValueError, match="difficulty"):
        _evaluate(_chart((500,)), (500,), duration_ms=1_000, difficulty="UNKNOWN")


def test_report_is_complete_and_stable_without_exception_text():
    result = _evaluate(
        _chart((500,), lane=4),
        (500,),
        duration_ms=1_000,
    )

    report = result.to_report()
    assert report["action"] == "RETRY_MAP"
    assert report["decisions"]["STRUCTURE"] == {
        "action": "RETRY_MAP",
        "reasons": ["STRUCTURE_INVALID"],
    }
    assert report["timing"] == result.timing.to_report()
    assert report["structureError"]["reasonCode"] == "NOTE_LANE_OUT_OF_RANGE"
    assert set(report["noteGrid"]) == {
        "uniqueRowCount",
        "cleanRowCount",
        "cleanRate",
        "absoluteP95Beats",
    }
    assert "lane 4" not in repr(report)
