import pytest

from chart_worker.generation.osu_parser import OsuBpmEvent, parse_osu_mania
from chart_worker.generation.osu_writer import notes_to_osu_mania, timing_to_osu_mania
from chart_worker.schema.note import NoteEvent


@pytest.mark.parametrize("key_mode", [4, 6, 7])
def test_osu_writer_round_trips_taps_holds_key_mode_and_bpm(key_mode):
    notes = [
        NoteEvent(500, 0),
        NoteEvent(1_000, key_mode - 1, kind="HOLD", duration_ms=250),
    ]
    text = notes_to_osu_mania(
        notes,
        key_mode=key_mode,
        bpm=120.0,
        offset_ms=100,
        audio_filename="game.flac",
        title="声の行く先",
    )
    beatmap = parse_osu_mania(text)

    assert beatmap.key_mode == key_mode
    assert beatmap.notes == notes
    assert [(event.time_ms, event.bpm) for event in beatmap.bpm_events] == [(100, 120.0)]
    assert "Title:声の行く先" in text


def test_osu_writer_rejects_a_non_positive_bpm():
    with pytest.raises(ValueError, match="bpm"):
        notes_to_osu_mania(
            [], key_mode=4, bpm=0.0, offset_ms=0, audio_filename="game.flac", title="x"
        )


def test_timing_reference_serializes_every_bpm_event():
    text = timing_to_osu_mania(
        (OsuBpmEvent(0, 120.0), OsuBpmEvent(10_000, 150.0)),
        audio_filename="game.flac",
        title="fixture",
    )
    parsed = parse_osu_mania(text)

    assert parsed.notes == []
    assert [(event.time_ms, event.bpm) for event in parsed.bpm_events] == [
        (0, 120.0),
        (10_000, 150.0),
    ]


@pytest.mark.parametrize(
    "writer",
    [
        lambda: notes_to_osu_mania(
            [NoteEvent(500, 0)],
            key_mode=4,
            bpm=120.0,
            offset_ms=0,
            audio_filename="game.flac",
            title="fixture",
        ),
        lambda: timing_to_osu_mania(
            (OsuBpmEvent(0, 120.0),),
            audio_filename="game.flac",
            title="fixture",
        ),
    ],
)
def test_writers_include_mapperatorinator_slider_required_fields(writer):
    text = writer()

    assert "Mode: 3" in text
    assert "Creator:ai-rhythm" in text
    assert "CircleSize:4" in text
    assert "HPDrainRate:5" in text
    assert "OverallDifficulty:8" in text
