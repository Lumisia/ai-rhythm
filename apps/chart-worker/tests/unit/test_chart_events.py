import pytest

from chart_worker.analysis.chart_events import ChartEventIndex
from chart_worker.schema.note import NoteEvent


def test_rows_are_independent_of_note_input_order():
    forward = [
        NoteEvent(100, 0),
        NoteEvent(100, 2),
        NoteEvent(300, 1),
    ]

    assert ChartEventIndex.build(
        forward, 4, 1_000
    ).rows == ChartEventIndex.build(list(reversed(forward)), 4, 1_000).rows


def test_has_other_row_between_uses_a_strict_open_interval():
    index = ChartEventIndex.build(
        [NoteEvent(100, 0), NoteEvent(200, 1), NoteEvent(300, 0)],
        4,
        1_000,
    )

    assert index.has_other_row_between(100, 300, excluded_lanes={0}) is True
    assert index.has_other_row_between(100, 200, excluded_lanes={0}) is False


def test_hold_intervals_and_section_rows_are_precomputed():
    index = ChartEventIndex.build(
        [
            NoteEvent(100, 2, kind="HOLD", duration_ms=500),
            NoteEvent(450, 1),
            NoteEvent(850, 0),
        ],
        4,
        1_000,
        section_ms=400,
    )

    assert [(hold.lane, hold.start_ms, hold.end_ms) for hold in index.holds] == [
        (2, 100, 600)
    ]
    assert index.section_rows == ((0,), (1,), (2,))


@pytest.mark.parametrize(
    "notes, message",
    [
        ([NoteEvent(100, 4)], "lane"),
        ([NoteEvent(1_000, 0)], "duration"),
        (
            [NoteEvent(800, 0, kind="HOLD", duration_ms=300)],
            "HOLD end",
        ),
        ([NoteEvent(100, 0), NoteEvent(100, 0)], "duplicate"),
    ],
)
def test_rejects_invalid_chart_events(notes, message):
    with pytest.raises(ValueError, match=message):
        ChartEventIndex.build(notes, 4, 1_000)
