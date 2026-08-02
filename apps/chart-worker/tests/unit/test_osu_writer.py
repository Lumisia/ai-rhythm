import pytest

from chart_worker.generation.osu_parser import parse_osu_mania
from chart_worker.generation.osu_writer import notes_to_osu_mania
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
