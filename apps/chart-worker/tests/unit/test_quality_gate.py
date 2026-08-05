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
    onsets: tuple[int, ...], *, activity: AudioActivity | None = None
) -> OnsetAnalysis:
    return OnsetAnalysis(
        sample_rate_hz=1_000,
        hop_length=100,
        strength=np.zeros(601),
        band_strength=np.zeros((3, 601)),
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
):
    return _gate().evaluate_chart_candidate(
        chart,
        authority or _authority(),
        _analysis(onsets, activity=activity),
        requested_key_mode=4,
        requested_difficulty=difficulty,
        duration_ms=duration_ms,
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


def test_structural_retry_is_not_hidden_by_profile_construction():
    result = _evaluate(
        _chart((1_000,), lane=4),
        (1_000,),
        duration_ms=2_000,
    )

    assert result.action is _gate().GateAction.RETRY_MAP
    assert result.profile is None
    assert _decision(result, "STRUCTURE").action is _gate().GateAction.RETRY_MAP
    assert _decision(result, "PATTERN").reasons == (
        "PROFILE_UNAVAILABLE_STRUCTURE_INVALID",
    )
    assert result.to_report()["qualityProfile"] is None


def test_active_leading_coverage_gap_retries_the_map():
    result = _evaluate(
        _chart((30_000,)),
        tuple(range(1_000, 21_000, 1_000)) + (30_000,),
        duration_ms=40_000,
    )

    assert result.action is _gate().GateAction.RETRY_MAP
    assert _decision(result, "COVERAGE").action is _gate().GateAction.RETRY_MAP
    assert _decision(result, "COVERAGE").reasons == ("ACTIVE_LEADING_GAP",)


@pytest.mark.parametrize(
    ("rows", "onsets", "duration_ms", "reason"),
    [
        ((0, 20_000, 40_000), tuple(range(21_000, 40_000, 1_000)), 60_000, "ACTIVE_MIDDLE_GAP"),
        ((0, 10_000), tuple(range(11_000, 31_000, 1_000)), 40_000, "ACTIVE_TRAILING_GAP"),
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


def test_single_fifteen_second_section_corruption_retries_timing_alignment():
    rows = tuple(range(1_000, 60_000, 1_000))
    corrupted = set(range(31_000, 39_000, 1_000))
    onsets = tuple(row - 60 if row in corrupted else row for row in rows)
    result = _evaluate(_chart(rows), onsets, duration_ms=60_000)

    section = result.timing.sections[2]
    assert section.metrics.precision_50 < 0.60
    assert section.metrics.absolute_p95_ms >= 60
    assert _decision(result, "TIMING_ALIGNMENT").reasons == ("SECTION_TIMING_MISALIGNED",)


def test_phase_only_section_drift_requires_review_without_retry():
    first_rows = tuple(range(1_000, 9_000, 1_000))
    second_rows = tuple(range(31_000, 39_000, 1_000))
    rows = first_rows + second_rows
    onsets = tuple(row + 5 for row in first_rows) + tuple(row - 50 for row in second_rows)
    result = _evaluate(_chart(rows), onsets, duration_ms=60_000)

    assert result.action is _gate().GateAction.REVIEW
    assert _decision(result, "TIMING_ALIGNMENT").action is _gate().GateAction.REVIEW
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
    assert decision.reasons == ("QUIET_LEADING_GAP",)
    assert result.action is _gate().GateAction.PASS


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
        "TIMING_IDENTITY",
        "TIMING_ALIGNMENT",
        "COVERAGE",
        "PATTERN",
    }
    assert review.action is _gate().GateAction.REVIEW
    assert passed.action is _gate().GateAction.PASS


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
    assert set(report["noteGrid"]) == {
        "uniqueRowCount",
        "cleanRowCount",
        "cleanRate",
        "absoluteP95Beats",
    }
    assert "lane 4" not in repr(report)
