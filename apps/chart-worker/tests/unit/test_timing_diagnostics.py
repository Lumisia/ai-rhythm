import numpy as np

from chart_worker.analysis.activity import AudioActivity
from chart_worker.analysis.timing_diagnostics import diagnose_chart_timing
from chart_worker.schema.note import NoteEvent


def tap(time_ms: int, lane: int = 0) -> NoteEvent:
    return NoteEvent(time_ms=time_ms, lane=lane)


def test_diagnoses_unique_rows_in_thirty_second_sections():
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
    assert [section.status for section in result.sections] == ["REVIEW", "REVIEW"]
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
        "overall": {
            "rowCount": 8,
            "precision20": 1.0,
            "precision50": 1.0,
            "signedMedianMs": 0.0,
            "absoluteP95Ms": 0.0,
        },
        "sections": [
            {
                "startMs": 0,
                "endMs": 30000,
                "status": "PASS",
                "rowCount": 8,
                "precision20": 1.0,
                "precision50": 1.0,
                "signedMedianMs": 0.0,
                "absoluteP95Ms": 0.0,
            }
        ],
    }
