import numpy as np
import pytest

from chart_worker.analysis.activity import AudioActivity
from chart_worker.analysis.coverage_opportunity import CoverageKind
from chart_worker.analysis.onset import OnsetAnalysis
from chart_worker.analysis.timing_diagnostics import diagnose_chart_timing
from chart_worker.generation.osu_parser import OsuBpmEvent
from chart_worker.schema.note import NoteEvent


def tap(time_ms: int, lane: int = 0) -> NoteEvent:
    return NoteEvent(time_ms=time_ms, lane=lane)


def _coverage_analysis(
    onset_strengths: dict[int, float], *, duration_ms: int = 40_000
) -> OnsetAnalysis:
    strength = np.zeros(duration_ms // 1_000 + 2)
    for time_ms, value in onset_strengths.items():
        strength[time_ms // 1_000] = value
    onset_ms = tuple(sorted(onset_strengths))
    return OnsetAnalysis(
        sample_rate_hz=1_000,
        hop_length=1_000,
        strength=strength,
        band_strength=np.zeros((3, strength.size)),
        onset_ms=onset_ms,
        n_fft=1_000,
        activity=AudioActivity(
            frame_ms=1_000.0,
            rms_db=np.full(strength.size, -10.0),
            floor_db=-20.0,
            active_onset_ms=onset_ms,
        ),
    )


def test_diagnoses_unique_rows_in_fifteen_second_sections():
    notes = [
        tap(100, 0),
        tap(100, 1),
        tap(1_100),
        tap(31_000),
        tap(32_000),
    ]
    result = diagnose_chart_timing(
        notes,
        (110, 1_140, 31_080, 32_200),
        duration_ms=60_000,
    )

    assert result.onset_count == 4
    assert result.overall.row_count == 4
    assert result.overall.precision_20 == 0.25
    assert result.overall.precision_50 == 0.5
    assert result.overall.signed_median_ms == -60.0
    assert result.overall.absolute_p95_ms == 182.0
    assert [section.status for section in result.sections] == [
        "INSUFFICIENT",
        "INSUFFICIENT",
        "INSUFFICIENT",
        "INSUFFICIENT",
    ]
    assert result.status == "REVIEW"
    assert result.first_note_time_ms == 100
    assert result.max_gap_ms == 29_900


def test_passes_well_aligned_rows_without_let_chords_inflate_the_score():
    rows = tuple(range(1_000, 17_000, 1_000))
    notes = [note for row in rows for note in (tap(row, 0), tap(row, 1))]
    result = diagnose_chart_timing(
        notes,
        tuple(row + 10 for row in rows),
        duration_ms=60_000,
    )

    assert result.overall.row_count == 16
    assert result.overall.precision_20 == 1.0
    assert result.sections[0].status == "PASS"
    assert result.sections[1].status == "INSUFFICIENT"
    assert result.status == "PASS"


def test_reports_one_to_one_matches_without_reusing_one_audio_onset():
    result = diagnose_chart_timing(
        [tap(980), tap(1_020)],
        (1_000,),
        duration_ms=2_000,
    )

    assert result.overall.precision_50 == 1.0
    assert result.overall.matched_count_50 == 1
    assert result.overall.matched_precision_50 == 0.5
    assert result.overall.matched_recall_50 == 1.0
    assert result.overall.matched_f1_50 == 0.666667
    assert result.overall.onset_reuse_inflation_50 == 0.5


def test_sparse_easy_chart_is_not_rejected_for_intentionally_low_recall():
    rows = tuple(range(1_000, 9_000, 1_000))
    onsets = tuple(range(500, 9_000, 500))

    result = diagnose_chart_timing(
        [tap(row) for row in rows],
        onsets,
        duration_ms=9_000,
    )

    assert result.overall.matched_precision_50 == 1.0
    assert result.overall.matched_recall_50 == 0.470588
    assert result.status == "PASS"


def test_reports_absolute_p99_for_a_globally_shifted_chart():
    rows = tuple(range(1_000, 9_000, 1_000))

    result = diagnose_chart_timing(
        [tap(row) for row in rows],
        tuple(row - 60 for row in rows),
        duration_ms=15_000,
    )

    assert result.overall.signed_median_ms == 60.0
    assert result.overall.absolute_p99_ms == 60.0


def test_labels_deleted_active_front_as_a_leading_coverage_gap():
    result = diagnose_chart_timing(
        [tap(30_000)],
        tuple(range(1_000, 21_000, 1_000)) + (30_000,),
        duration_ms=40_000,
    )

    assert result.coverage_gaps[0].position == "LEADING"
    assert result.coverage_gaps[0].to_report()["position"] == "LEADING"


def test_labels_gap_after_zero_ms_first_row_as_post_first_not_leading():
    result = diagnose_chart_timing(
        [tap(0), tap(12_745), tap(13_000)],
        (0, *range(1_000, 12_000, 1_000), 12_745, 13_000),
        duration_ms=20_000,
    )

    assert result.coverage_gaps[0].start_ms == 0
    assert result.coverage_gaps[0].end_ms == 12_745
    assert result.coverage_gaps[0].position == "POST_FIRST"
    assert result.coverage_gaps[0].to_report()["position"] == "POST_FIRST"


def test_reports_section_phase_delta_and_gap_positions_at_boundaries():
    first_rows = tuple(range(1_000, 9_000, 1_000))
    second_rows = tuple(range(16_000, 24_000, 1_000))
    phase_result = diagnose_chart_timing(
        [tap(row) for row in first_rows + second_rows],
        first_rows + tuple(row - 40 for row in second_rows),
        duration_ms=30_000,
    )
    gap_result = diagnose_chart_timing(
        [tap(time_ms) for time_ms in (0, 20_000, 40_000, 60_000)],
        tuple(range(1_000, 20_000, 1_000))
        + tuple(range(21_000, 40_000, 1_000))
        + tuple(range(41_000, 60_000, 1_000))
        + (0, 20_000, 40_000, 60_000),
        duration_ms=60_000,
    )

    assert [section.phase_delta_ms for section in phase_result.sections] == [-20.0, 20.0]
    assert [section.to_report()["phaseDeltaMs"] for section in phase_result.sections] == [
        -20.0,
        20.0,
    ]
    assert [gap.position for gap in gap_result.coverage_gaps] == [
        "POST_FIRST",
        "MIDDLE",
        "TRAILING",
    ]
    assert [gap.to_report()["position"] for gap in gap_result.coverage_gaps] == [
        "POST_FIRST",
        "MIDDLE",
        "TRAILING",
    ]


def test_marks_section_phase_drift_for_review_even_when_every_row_is_within_50ms():
    first_rows = tuple(range(1_000, 9_000, 1_000))
    second_rows = tuple(range(31_000, 39_000, 1_000))
    rows = first_rows + second_rows
    onsets = tuple(row + 5 for row in first_rows) + tuple(row - 50 for row in second_rows)

    result = diagnose_chart_timing(
        [tap(row) for row in rows],
        onsets,
        duration_ms=60_000,
    )

    assert result.overall.precision_50 == 1.0
    assert result.overall.signed_median_ms == 22.5
    assert [section.status for section in result.sections] == [
        "REVIEW",
        "INSUFFICIENT",
        "REVIEW",
        "INSUFFICIENT",
    ]
    assert result.status == "REVIEW"


def test_reports_insufficient_when_rows_or_onsets_are_missing():
    no_rows = diagnose_chart_timing([], (100,), duration_ms=1_000)
    no_onsets = diagnose_chart_timing([tap(100)], (), duration_ms=1_000)

    assert no_rows.status == "INSUFFICIENT"
    assert no_rows.overall.row_count == 0
    assert no_onsets.status == "INSUFFICIENT"
    assert no_onsets.overall.precision_50 is None


def test_marks_long_gap_with_active_onsets_for_review():
    onset_ms = (0, *range(1_000, 21_000, 1_000), 30_838)
    result = diagnose_chart_timing(
        [tap(0), tap(30_838)],
        onset_ms,
        duration_ms=60_000,
    )

    assert result.status == "REVIEW"
    assert [gap.to_report() for gap in result.coverage_gaps] == [
        {
            "startMs": 0,
            "endMs": 30_838,
            "durationMs": 30_838,
            "onsetCount": 20,
            "activeOnsetCount": 20,
            "activeFrameRatio": 1.0,
            "position": "POST_FIRST",
            "rowSpanStartMs": 0,
            "rowSpanDurationMs": 30_838,
            "unoccupiedStartMs": 0,
            "unoccupiedDurationMs": 30_838,
        }
    ]


def test_marks_only_sustained_active_gap_for_review():
    activity = AudioActivity(
        frame_ms=1_000.0,
        rms_db=np.array([-10.0] * 40),
        floor_db=-20.0,
        active_onset_ms=tuple(range(1_000, 21_000, 1_000)),
    )
    result = diagnose_chart_timing(
        [tap(0), tap(30_000)],
        (0, *range(1_000, 21_000, 1_000), 30_000),
        duration_ms=40_000,
        activity=activity,
    )

    assert result.status == "REVIEW"
    assert len(result.coverage_gaps) == 1
    assert result.quiet_coverage_gaps == ()
    assert result.coverage_gaps[0].active_onset_count == 20
    assert result.coverage_gaps[0].active_frame_ratio == 1.0


def test_reports_hold_covered_low_attack_gap_as_sustain_opportunity():
    analysis = _coverage_analysis(
        {
            1_000: 1.0,
            **{time_ms: 0.1 for time_ms in range(5_000, 21_000, 1_000)},
            30_000: 1.0,
        }
    )
    result = diagnose_chart_timing(
        [
            NoteEvent(4_000, 0, kind="HOLD", duration_ms=26_000),
            tap(30_000),
        ],
        analysis.onset_ms,
        duration_ms=40_000,
        bpm_events=(OsuBpmEvent(0, 120.0),),
        activity=analysis.activity,
        onset_analysis=analysis,
        difficulty="EASY",
    )

    gap = next(gap for gap in result.quiet_coverage_gaps if gap.start_ms == 4_000)
    assert gap.opportunity is not None
    assert gap.opportunity.kind is CoverageKind.SUSTAIN_REPRESENTABLE
    assert gap.to_report()["opportunity"]["holdOccupancyRatio"] == 1.0


def test_reports_fully_hold_covered_strong_attacks_as_attack_required():
    analysis = _coverage_analysis(
        {time_ms: 0.9 for time_ms in range(5_000, 21_000, 1_000)}
    )
    result = diagnose_chart_timing(
        [
            NoteEvent(4_000, 0, kind="HOLD", duration_ms=26_000),
            tap(30_000),
        ],
        analysis.onset_ms,
        duration_ms=40_000,
        bpm_events=(OsuBpmEvent(0, 120.0),),
        activity=analysis.activity,
        onset_analysis=analysis,
        difficulty="EXPERT",
    )

    gap = next(gap for gap in result.coverage_gaps if gap.start_ms == 4_000)
    assert gap.opportunity is not None
    assert gap.opportunity.kind is CoverageKind.ATTACK_REQUIRED
    assert gap.opportunity.actionable is True
    assert gap.opportunity.attack_evidence_scope == "GLOBAL"
    assert gap.unoccupied_start_ms == 30_000
    assert gap.unoccupied_duration_ms == 0


def test_hold_tail_uses_unoccupied_span_and_keeps_uncertain_evidence_non_actionable():
    analysis = _coverage_analysis(
        {
            184_500: 0.4,
            185_500: 0.4,
            187_000: 0.4,
            188_000: 0.4,
        },
        duration_ms=190_000,
    )
    result = diagnose_chart_timing(
        [
            NoteEvent(181_316, 0, kind="HOLD", duration_ms=2_681),
            NoteEvent(183_997, 1, kind="HOLD", duration_ms=2_560),
            tap(188_880),
        ],
        analysis.onset_ms,
        duration_ms=190_000,
        coverage_end_ms=188_880,
        bpm_events=(OsuBpmEvent(0, 120.0),),
        activity=analysis.activity,
        onset_analysis=analysis,
        difficulty="EASY",
    )

    assert result.coverage_gaps == ()
    gap = next(
        gap
        for gap in result.uncertain_coverage_gaps
        if gap.row_span_start_ms == 183_997
    )
    assert gap.unoccupied_start_ms == 186_557
    assert gap.end_ms == 188_880
    assert gap.unoccupied_duration_ms == 2_323
    assert gap.to_report()["rowSpanDurationMs"] == 4_883
    assert gap.to_report()["unoccupiedDurationMs"] == 2_323


def test_locally_corroborated_gap_is_made_actionable_only_at_the_source_classifier():
    """Catch diagnostics discarding a source-authorized local phrase."""
    strengths = {
        **{time_ms: 0.9 for time_ms in range(1_000, 11_000, 500)},
        21_000: 0.9,
        22_000: 0.9,
        23_000: 0.9,
        24_000: 0.4,
        25_000: 0.4,
        26_000: 0.4,
        27_000: 0.4,
    }
    strength = np.zeros(402, dtype=np.float64)
    for time_ms, value in strengths.items():
        strength[time_ms // 100] = value
    onset_ms = tuple(sorted(strengths))
    analysis = OnsetAnalysis(
        sample_rate_hz=1_000,
        hop_length=100,
        strength=strength,
        band_strength=np.zeros((3, strength.size), dtype=np.float64),
        onset_ms=onset_ms,
        n_fft=100,
        activity=AudioActivity(
            frame_ms=100.0,
            rms_db=np.full(strength.size, -10.0),
            floor_db=-20.0,
            active_onset_ms=onset_ms,
        ),
    )

    result = diagnose_chart_timing(
        [tap(20_000), tap(28_000)],
        analysis.onset_ms,
        duration_ms=40_000,
        bpm_events=(OsuBpmEvent(0, 120.0),),
        activity=analysis.activity,
        onset_analysis=analysis,
        difficulty="EXPERT",
    )

    gap = next(
        gap
        for gap in result.coverage_gaps
        if gap.start_ms == 20_000 and gap.end_ms == 28_000
    )
    assert gap.opportunity is not None
    assert gap.opportunity.kind is CoverageKind.ATTACK_REQUIRED
    assert gap.opportunity.actionable is True
    assert gap.opportunity.attack_evidence_scope == "LOCAL_CORROBORATED"
    assert gap.to_report()["localAudioEvidence"] == {
        "version": "coverage-jury-local-evidence-v1",
        "startMs": 20_000,
        "endMs": 28_000,
        "durationMs": 8_000,
        "activeFrameRatio": 1.0,
        "activeOnsetCount": 7,
        "globalStrongAttackCount": 3,
        "localStrongAttackCount": 7,
        "globalThreshold": 0.9,
        "localThreshold": 0.4,
        "neighboringActivityRatio": 1.0,
        "policyState": "OBSERVATION_ONLY",
        "mutatesGeneration": False,
    }


def test_reports_quiet_gap_without_forcing_review():
    activity = AudioActivity(
        frame_ms=1_000.0,
        rms_db=np.array([-30.0] * 30 + [-10.0] * 10),
        floor_db=-20.0,
        active_onset_ms=(),
    )
    result = diagnose_chart_timing(
        [tap(0), tap(30_000)],
        (0, *range(1_000, 21_000, 1_000), 30_000),
        duration_ms=40_000,
        activity=activity,
    )

    assert result.coverage_gaps == ()
    assert len(result.quiet_coverage_gaps) == 1
    assert result.quiet_coverage_gaps[0].onset_count == 20
    assert result.quiet_coverage_gaps[0].active_onset_count == 0
    assert result.quiet_coverage_gaps[0].active_frame_ratio == 0.0
    assert result.status == "PASS"


def test_coverage_horizon_prevents_silent_tail_from_diluting_active_gap():
    activity = AudioActivity(
        frame_ms=1_000.0,
        rms_db=np.array(
            [-80.0] * 70 + [-10.0] * 6 + [-80.0] * 12,
        ),
        floor_db=-60.0,
        active_onset_ms=tuple(range(71_000, 85_000, 1_000)),
    )
    onsets = (0, 70_000, *range(71_000, 85_000, 1_000))

    full_duration = diagnose_chart_timing(
        [tap(0), tap(70_000)],
        onsets,
        duration_ms=88_000,
        activity=activity,
    )
    musical_horizon = diagnose_chart_timing(
        [tap(0), tap(70_000)],
        onsets,
        duration_ms=88_000,
        coverage_end_ms=84_000,
        activity=activity,
    )

    assert full_duration.coverage_gaps == ()
    assert full_duration.quiet_coverage_gaps[0].active_frame_ratio == pytest.approx(
        6 / 18,
        abs=1e-6,
    )
    assert musical_horizon.quiet_coverage_gaps == ()
    assert musical_horizon.coverage_gaps[0].position == "TRAILING"
    assert musical_horizon.coverage_gaps[0].end_ms == 84_000
    assert musical_horizon.coverage_gaps[0].active_frame_ratio == pytest.approx(
        6 / 14,
        abs=1e-6,
    )


def test_serializes_stable_camel_case_report_fields():
    result = diagnose_chart_timing(
        [tap(row) for row in range(1_000, 9_000, 1_000)],
        tuple(range(1_000, 9_000, 1_000)),
        duration_ms=30_000,
    )

    assert result.to_report() == {
        "status": "PASS",
        "onsetCount": 8,
        "activeOnsetCount": 8,
        "firstNoteTimeMs": 1000,
        "maxGapMs": 1000,
        "coverageGaps": [],
        "quietCoverageGaps": [],
        "uncertainCoverageGaps": [],
        "maxRowSpanGapMs": 1000,
        "overall": {
            "rowCount": 8,
            "precision20": 1.0,
            "precision50": 1.0,
            "signedMedianMs": 0.0,
            "absoluteP95Ms": 0.0,
            "absoluteP99Ms": 0.0,
            "matchedCount50": 8,
            "matchedPrecision50": 1.0,
            "matchedRecall50": 1.0,
            "matchedF150": 1.0,
            "onsetReuseInflation50": 0.0,
        },
        "sections": [
            {
                "startMs": 0,
                "endMs": 15000,
                "status": "PASS",
                "rowCount": 8,
                "precision20": 1.0,
                "precision50": 1.0,
                "signedMedianMs": 0.0,
                "absoluteP95Ms": 0.0,
                "absoluteP99Ms": 0.0,
                "matchedCount50": 8,
                "matchedPrecision50": 1.0,
                "matchedRecall50": 1.0,
                "matchedF150": 1.0,
                "onsetReuseInflation50": 0.0,
                "phaseDeltaMs": 0.0,
            },
            {
                "startMs": 15000,
                "endMs": 30000,
                "status": "INSUFFICIENT",
                "rowCount": 0,
                "precision20": None,
                "precision50": None,
                "signedMedianMs": None,
                "absoluteP95Ms": None,
                "absoluteP99Ms": None,
                "matchedCount50": 0,
                "matchedPrecision50": None,
                "matchedRecall50": None,
                "matchedF150": None,
                "onsetReuseInflation50": None,
                "phaseDeltaMs": None,
            }
        ],
    }


def test_uses_beat_normalized_sections_with_slow_tempo_cap():
    result = diagnose_chart_timing(
        [],
        (),
        duration_ms=65_000,
        bpm_events=(OsuBpmEvent(time_ms=0, bpm=60.0),),
    )

    assert [(section.start_ms, section.end_ms) for section in result.sections] == [
        (0, 30_000),
        (30_000, 60_000),
        (60_000, 65_000),
    ]


def test_uses_thirty_two_beat_sections_at_typical_tempo():
    result = diagnose_chart_timing(
        [],
        (),
        duration_ms=33_000,
        bpm_events=(OsuBpmEvent(time_ms=0, bpm=120.0),),
    )

    assert [(section.start_ms, section.end_ms) for section in result.sections] == [
        (0, 16_000),
        (16_000, 32_000),
        (32_000, 33_000),
    ]


def test_uses_beat_normalized_sections_with_fast_tempo_floor():
    result = diagnose_chart_timing(
        [],
        (),
        duration_ms=17_000,
        bpm_events=(OsuBpmEvent(time_ms=0, bpm=300.0),),
    )

    assert [(section.start_ms, section.end_ms) for section in result.sections] == [
        (0, 8_000),
        (8_000, 16_000),
        (16_000, 17_000),
    ]


def test_integrates_local_bpm_changes_when_building_sections():
    result = diagnose_chart_timing(
        [],
        (),
        duration_ms=40_000,
        bpm_events=(
            OsuBpmEvent(time_ms=0, bpm=60.0),
            OsuBpmEvent(time_ms=10_000, bpm=180.0),
        ),
    )

    assert [(section.start_ms, section.end_ms) for section in result.sections] == [
        (0, 17_333),
        (17_333, 28_000),
        (28_000, 38_667),
        (38_667, 40_000),
    ]
