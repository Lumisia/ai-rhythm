import pytest

from chart_worker.generation.mapperatorinator import GeneratedChart
from chart_worker.generation.osu_parser import OsuBpmEvent
from chart_worker.schema.note import NoteEvent
from chart_worker.validation.generated_chart import (
    GeneratedChartValidationError,
    validate_generated_chart,
)


def tap(time_ms: int, lane: int) -> NoteEvent:
    return NoteEvent(time_ms=time_ms, lane=lane)


def hold(time_ms: int, lane: int, duration_ms: int) -> NoteEvent:
    return NoteEvent(time_ms=time_ms, lane=lane, kind="HOLD", duration_ms=duration_ms)


def generated(
    notes: list[NoteEvent],
    *,
    key_mode: int = 4,
    timing: tuple[OsuBpmEvent, ...] = (OsuBpmEvent(time_ms=0, bpm=120.0),),
) -> GeneratedChart:
    return GeneratedChart(
        notes=notes,
        key_mode=key_mode,
        osu_text="osu file format v14\n",
        generator_name="test",
        seed=0,
        bpm_events=timing,
    )


def test_accepts_zero_millisecond_rows_and_a_note_before_audio_end():
    chart = generated([tap(0, 0), tap(9_999, 3)])
    validate_generated_chart(chart, key_mode=4, duration_ms=10_000)


def test_rejects_duplicate_lane_and_time():
    chart = generated([tap(0, 1), tap(0, 1)])
    with pytest.raises(
        GeneratedChartValidationError, match=r"duplicate.*lane 1.*0ms"
    ) as captured:
        validate_generated_chart(chart, key_mode=4, duration_ms=10_000)

    assert captured.value.reason_code == "DUPLICATE_NOTE"
    assert captured.value.context == {
        "lane": 1,
        "timeMs": 0,
        "noteKind": "TAP",
    }


def test_allows_a_chord_across_different_lanes():
    chart = generated([tap(0, 0), tap(0, 1), tap(0, 2), tap(0, 3)])
    validate_generated_chart(chart, key_mode=4, duration_ms=10_000)


def test_rejects_a_note_inside_an_active_hold_in_the_same_lane():
    chart = generated([hold(100, 1, 500), tap(400, 1)])
    with pytest.raises(
        GeneratedChartValidationError, match=r"overlap.*lane 1"
    ) as captured:
        validate_generated_chart(chart, key_mode=4, duration_ms=10_000)

    assert captured.value.reason_code == "HOLD_OVERLAP"
    assert captured.value.context == {
        "lane": 1,
        "timeMs": 400,
        "noteKind": "TAP",
        "holdEndMs": 600,
    }


def test_allows_the_next_note_exactly_when_a_hold_ends():
    chart = generated([hold(100, 1, 500), tap(600, 1)])
    validate_generated_chart(chart, key_mode=4, duration_ms=10_000)


@pytest.mark.parametrize(
    ("notes", "message"),
    [
        ([tap(10_000, 0)], "duration"),
        ([tap(10_001, 0)], "duration"),
        ([hold(9_800, 0, 300)], "duration"),
        ([tap(100, 4)], "lane 4"),
    ],
)
def test_rejects_notes_outside_the_canonical_chart_bounds(notes, message):
    with pytest.raises(GeneratedChartValidationError, match=message):
        validate_generated_chart(generated(notes), key_mode=4, duration_ms=10_000)


def test_rejects_a_generated_key_mode_that_differs_from_the_request():
    with pytest.raises(GeneratedChartValidationError, match=r"requested 4K.*generated 6K"):
        validate_generated_chart(
            generated([tap(100, 0)], key_mode=6),
            key_mode=4,
            duration_ms=10_000,
        )


def test_rejects_a_chart_without_timing_events():
    with pytest.raises(GeneratedChartValidationError, match="timing"):
        validate_generated_chart(
            generated([tap(100, 0)], timing=()),
            key_mode=4,
            duration_ms=10_000,
        )


def test_rejects_a_chart_without_notes():
    with pytest.raises(GeneratedChartValidationError, match="no notes"):
        validate_generated_chart(
            generated([]),
            key_mode=4,
            duration_ms=10_000,
        )


def test_rejects_duplicate_or_unsorted_timing_events():
    duplicate = (
        OsuBpmEvent(time_ms=0, bpm=120.0),
        OsuBpmEvent(time_ms=0, bpm=130.0),
    )
    unsorted = (
        OsuBpmEvent(time_ms=1_000, bpm=120.0),
        OsuBpmEvent(time_ms=0, bpm=130.0),
    )

    for timing in (duplicate, unsorted):
        with pytest.raises(GeneratedChartValidationError, match="timing"):
            validate_generated_chart(
                generated([tap(100, 0)], timing=timing),
                key_mode=4,
                duration_ms=10_000,
            )


@pytest.mark.parametrize("bpm", [0.0, -120.0, float("inf"), float("nan")])
def test_rejects_non_positive_or_non_finite_bpm(bpm):
    with pytest.raises(GeneratedChartValidationError, match="bpm"):
        validate_generated_chart(
            generated([tap(100, 0)], timing=(OsuBpmEvent(time_ms=0, bpm=bpm),)),
            key_mode=4,
            duration_ms=10_000,
        )
