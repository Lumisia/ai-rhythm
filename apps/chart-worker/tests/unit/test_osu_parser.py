from pathlib import Path

import pytest

from chart_worker.generation.osu_parser import parse_osu_file, parse_osu_mania

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "mini-4k.osu"


def test_mania_parser_preserves_raw_uninherited_timing_as_an_immutable_tuple():
    text = (
        "osu file format v14\n\n[General]\nMode: 3\n\n[Difficulty]\nCircleSize:4\n"
        "\n[TimingPoints]\n"
        "-120,500,4,2,0,60,1,0\n"
        "500,-100,4,2,0,60,0,0\n"
        "16000,600,4,2,0,60,1,0\n"
        "\n[HitObjects]\n64,192,1000,1,0,0:0:0:0:\n"
    )

    beatmap = parse_osu_mania(text)

    assert isinstance(beatmap.bpm_events, tuple)
    assert [(event.time_ms, round(event.bpm, 3)) for event in beatmap.bpm_events] == [
        (-120, 120.0),
        (16000, 100.0),
    ]


def test_parses_mania_fixture():
    beatmap = parse_osu_file(FIXTURE)
    assert beatmap.key_mode == 4
    assert [note.lane for note in beatmap.notes] == [0, 1, 2, 3]
    assert [note.time_ms for note in beatmap.notes] == [1000, 1200, 1400, 1600]
    hold = next(note for note in beatmap.notes if note.kind == "HOLD")
    assert hold.duration_ms == 500
    assert [(event.time_ms, round(event.bpm, 3)) for event in beatmap.bpm_events] == [
        (0, 150.0),
        (12000, 120.0),
    ]


def test_rejects_non_mania_mode():
    with pytest.raises(ValueError, match="mania"):
        parse_osu_mania("osu file format v14\n\n[General]\nMode: 0\n")


def test_rejects_unsupported_key_mode():
    text = "osu file format v14\n\n[General]\nMode: 3\n\n[Difficulty]\nCircleSize:5\n"
    with pytest.raises(ValueError, match="unsupported key_mode"):
        parse_osu_mania(text)


def test_rejects_missing_key_mode_instead_of_assuming_4k():
    text = "osu file format v14\n\n[General]\nMode: 3\n\n[Difficulty]\n"
    with pytest.raises(ValueError, match="CircleSize"):
        parse_osu_mania(text)


def test_rejects_fractional_key_mode_instead_of_truncating_it():
    text = "osu file format v14\n\n[General]\nMode: 3\n\n[Difficulty]\nCircleSize:4.5\n"
    with pytest.raises(ValueError, match="unsupported key_mode"):
        parse_osu_mania(text)


def test_rejects_hold_without_an_end_time_instead_of_parsing_a_tap():
    text = (
        "osu file format v14\n\n[General]\nMode: 3\n\n[Difficulty]\nCircleSize:4\n"
        "\n[HitObjects]\n64,192,1000,128,0\n"
    )
    with pytest.raises(ValueError, match="hold"):
        parse_osu_mania(text)


def test_rejects_malformed_hit_object_instead_of_dropping_it():
    text = (
        "osu file format v14\n\n[General]\nMode: 3\n\n[Difficulty]\nCircleSize:4\n"
        "\n[HitObjects]\n64,192,1000\n"
    )
    with pytest.raises(ValueError, match="malformed HitObject"):
        parse_osu_mania(text)


def test_rejects_unsupported_hit_object_instead_of_parsing_a_tap():
    text = (
        "osu file format v14\n\n[General]\nMode: 3\n\n[Difficulty]\nCircleSize:4\n"
        "\n[HitObjects]\n64,192,1000,8,0,0:0:0:0:\n"
    )
    with pytest.raises(ValueError, match="unsupported HitObject type"):
        parse_osu_mania(text)


def test_reports_malformed_numeric_hit_object_with_context():
    text = (
        "osu file format v14\n\n[General]\nMode: 3\n\n[Difficulty]\nCircleSize:4\n"
        "\n[HitObjects]\nnot-a-number,192,1000,1,0,0:0:0:0:\n"
    )
    with pytest.raises(ValueError, match="malformed HitObject"):
        parse_osu_mania(text)


@pytest.mark.parametrize("object_type", [9, 136, 257, -127])
def test_rejects_mixed_or_unknown_hit_object_bits(object_type):
    text = (
        "osu file format v14\n\n[General]\nMode: 3\n\n[Difficulty]\nCircleSize:4\n"
        f"\n[HitObjects]\n64,192,1000,{object_type},0,1200:0:0:0:0:\n"
    )
    with pytest.raises(ValueError, match="unsupported HitObject type"):
        parse_osu_mania(text)


@pytest.mark.parametrize("end_ms", [1000, 900])
def test_degenerate_hold_is_demoted_to_tap(end_ms):
    """길이가 0 이하인 롱노트는 예외 대신 일반 노트가 된다."""
    text = (
        "osu file format v14\n\n[General]\nMode: 3\n\n[Difficulty]\nCircleSize:4\n"
        f"\n[HitObjects]\n64,192,1000,128,0,{end_ms}:0:0:0:0:\n"
    )
    beatmap = parse_osu_mania(text)
    assert [(note.kind, note.duration_ms) for note in beatmap.notes] == [("TAP", None)]
    assert beatmap.notes[0].time_ms == 1000


def test_allows_new_combo_and_combo_offset_auxiliary_bits():
    text = (
        "osu file format v14\n\n[General]\nMode: 3\n\n[Difficulty]\nCircleSize:4\n"
        "\n[HitObjects]\n64,192,1000,21,0,0:0:0:0:\n"
        "192,192,1200,164,0,1500:0:0:0:0:\n"
    )
    beatmap = parse_osu_mania(text)
    assert [note.kind for note in beatmap.notes] == ["TAP", "HOLD"]
