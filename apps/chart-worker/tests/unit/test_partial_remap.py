from chart_worker.analysis.timing_diagnostics import TimingCoverageGap
from chart_worker.generation.osu_parser import OsuBpmEvent
from chart_worker.generation.partial_remap import build_partial_remap_window
from chart_worker.schema.note import NoteEvent


def gap(start_ms: int, end_ms: int, position: str = "MIDDLE") -> TimingCoverageGap:
    return TimingCoverageGap(
        start_ms=start_ms,
        end_ms=end_ms,
        onset_count=20,
        active_onset_count=20,
        active_frame_ratio=1.0,
        position=position,  # type: ignore[arg-type]
    )


def test_window_uses_the_local_bpm_at_each_coverage_boundary():
    result = build_partial_remap_window(
        [],
        (gap(32_000, 40_000),),
        (
            OsuBpmEvent(time_ms=0, bpm=120.0),
            OsuBpmEvent(time_ms=30_000, bpm=150.0),
        ),
        duration_ms=60_000,
    )

    assert result is not None
    assert (result.start_ms, result.end_ms) == (30_400, 41_600)


def test_window_clamps_leading_and_trailing_gaps_to_audio_bounds():
    leading = build_partial_remap_window(
        [],
        (gap(0, 9_000, "LEADING"),),
        (OsuBpmEvent(time_ms=0, bpm=120.0),),
        duration_ms=60_000,
    )
    trailing = build_partial_remap_window(
        [],
        (gap(50_000, 60_000, "TRAILING"),),
        (OsuBpmEvent(time_ms=0, bpm=120.0),),
        duration_ms=60_000,
    )

    assert leading is not None and leading.start_ms == 0
    assert trailing is not None and trailing.end_ms == 60_000


def test_window_expands_until_no_existing_hold_crosses_either_boundary():
    notes = [
        NoteEvent(time_ms=28_000, lane=0, kind="HOLD", duration_ms=4_000),
        NoteEvent(time_ms=41_000, lane=1, kind="HOLD", duration_ms=4_000),
        NoteEvent(time_ms=44_000, lane=2, kind="HOLD", duration_ms=3_000),
    ]

    result = build_partial_remap_window(
        notes,
        (gap(32_000, 40_000),),
        (OsuBpmEvent(time_ms=0, bpm=120.0),),
        duration_ms=60_000,
    )

    assert result is not None
    assert (result.start_ms, result.end_ms) == (28_000, 47_000)


def test_window_declines_a_repair_that_would_replace_most_of_the_song():
    assert (
        build_partial_remap_window(
            [],
            (gap(5_000, 55_000),),
            (OsuBpmEvent(time_ms=0, bpm=120.0),),
            duration_ms=60_000,
        )
        is None
    )
