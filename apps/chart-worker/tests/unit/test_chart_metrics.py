import pytest

from chart_worker.report.alignment import AlignmentReport
from chart_worker.report.chart_metrics import build_chart_metrics
from chart_worker.schema.note import NoteEvent


def test_build_chart_metrics_counts_side_notes_holds_and_alignment():
    notes = [
        NoteEvent(100, 0),
        NoteEvent(200, 5, kind="HOLD", duration_ms=300),
    ]
    alignment = AlignmentReport(((100, 102),), (900,), 0.5, 0.5, 2.0)
    metrics = build_chart_metrics(
        notes,
        duration_ms=2_000,
        key_mode=6,
        beat_ms=500.0,
        alignment=alignment,
        moved_note_ratio=0.25,
    )

    assert metrics.note_count == 2
    assert metrics.hold_count == 1
    assert metrics.side_note_ratio == 1.0
    assert metrics.side_hold_ratio == 0.5
    assert metrics.drum_coverage == 0.5
    assert metrics.drum_precision == 0.5
    assert metrics.mean_abs_err_ms == 2.0
    assert metrics.moved_note_ratio == 0.25


def test_build_chart_metrics_reports_zero_side_ratios_for_four_keys():
    metrics = build_chart_metrics(
        [NoteEvent(100, 0, kind="HOLD", duration_ms=100)],
        duration_ms=1_000,
        key_mode=4,
        beat_ms=500.0,
        alignment=AlignmentReport((), (), 0.0, 0.0, 0.0),
        moved_note_ratio=0.0,
    )
    assert metrics.side_note_ratio == 0.0
    assert metrics.side_hold_ratio == 0.0


@pytest.mark.parametrize(
    ("key_mode", "beat_ms", "moved_note_ratio", "message"),
    [(5, 500.0, 0.0, "key_mode"), (4, 0.0, 0.0, "beat_ms"), (4, 500.0, 1.1, "ratio")],
)
def test_build_chart_metrics_rejects_invalid_inputs(key_mode, beat_ms, moved_note_ratio, message):
    with pytest.raises(ValueError, match=message):
        build_chart_metrics(
            [],
            duration_ms=1_000,
            key_mode=key_mode,
            beat_ms=beat_ms,
            alignment=AlignmentReport((), (), 0.0, 0.0, 0.0),
            moved_note_ratio=moved_note_ratio,
        )
