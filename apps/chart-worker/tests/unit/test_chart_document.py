import json
from uuid import UUID

import pytest
from pydantic import ValidationError

from chart_worker.schema.chart import (
    SCHEMA_VERSION,
    BpmEvent,
    ChartDocument,
    ChartMetrics,
    ChartNote,
    GeneratorInfo,
    chart_json_schema,
    notes_to_chart_notes,
)
from chart_worker.schema.note import NoteEvent
from chart_worker.schema.types import lane_semantics

UUID_A = UUID("11111111-1111-4111-8111-111111111111")
SHA = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"


def _metrics(**overrides) -> ChartMetrics:
    base = {
        "note_count": 2,
        "hold_count": 1,
        "avg_nps": 1.0,
        "p95_nps": 2.0,
        "peak_nps": 2.0,
        "chord_ratio": 0.0,
        "max_jack": 1,
        "project_rating": 1.2,
        "project_tier": "EASY",
        "pattern_entropy": 0.0,
        "drum_coverage": 1.0,
        "drum_precision": 1.0,
        "mean_abs_err_ms": 8.2,
        "side_note_ratio": 0.0,
        "side_hold_ratio": 0.0,
        "moved_note_ratio": 0.0,
    }
    return ChartMetrics(**(base | overrides))


def _document(**overrides) -> ChartDocument:
    base = {
        "chart_id": UUID_A,
        "song_version_id": UUID_A,
        "game_audio_asset_id": UUID_A,
        "audio_sha256": SHA,
        "key_mode": 4,
        "difficulty": "EASY",
        "lane_semantics": lane_semantics(4),
        "offset_ms": 0,
        "duration_ms": 2000,
        "bpm_events": [BpmEvent(time_ms=0, bpm=150.0)],
        "bpm_source": "BEAT_THIS",
        "notes": notes_to_chart_notes(
            [
                NoteEvent(time_ms=0, lane=0),
                NoteEvent(time_ms=500, lane=1, kind="HOLD", duration_ms=200),
            ]
        ),
        "auto_play_onsets": [900, 1200],
        "metrics": _metrics(),
        "generator": GeneratorInfo(
            name="mapperatorinator-v32",
            version="a1b2c3d",
            analysis_version="beatthis-0.1+librosa-0.10",
            postprocess_version="pp-v1",
            seed=918273,
        ),
    }
    return ChartDocument(**(base | overrides))


def test_notes_to_chart_notes_sorts_and_numbers_from_one():
    notes = notes_to_chart_notes(
        [
            NoteEvent(time_ms=500, lane=3),
            NoteEvent(time_ms=0, lane=2),
            NoteEvent(time_ms=0, lane=1),
        ]
    )
    assert [(note.id, note.time_ms, note.lane) for note in notes] == [
        (1, 0, 1),
        (2, 0, 2),
        (3, 500, 3),
    ]


def test_notes_to_chart_notes_keeps_hold_duration():
    [note] = notes_to_chart_notes([NoteEvent(time_ms=0, lane=0, kind="HOLD", duration_ms=250)])
    assert (note.kind, note.duration_ms, note.end_ms) == ("HOLD", 250, 250)


def test_json_uses_camel_case_and_type_key():
    payload = json.loads(_document().to_json())
    assert payload["schemaVersion"] == SCHEMA_VERSION
    assert payload["laneSemantics"] == ["MAIN_1", "MAIN_2", "MAIN_3", "MAIN_4"]
    assert payload["notes"][0] == {
        "id": 1,
        "lane": 0,
        "timeMs": 0,
        "type": "TAP",
        "durationMs": None,
    }
    assert payload["notes"][1]["type"] == "HOLD"
    assert payload["metrics"]["p95Nps"] == 2.0
    assert payload["metrics"]["meanAbsErrMs"] == 8.2
    assert payload["generator"]["analysisVersion"] == "beatthis-0.1+librosa-0.10"
    assert "time_ms" not in json.dumps(payload)


def test_round_trip_through_json():
    document = _document()
    assert ChartDocument.model_validate_json(document.to_json()) == document


def test_json_schema_is_generated_with_aliases():
    schema = chart_json_schema()
    assert "timeMs" in schema["$defs"]["ChartNote"]["properties"]
    assert "type" in schema["$defs"]["ChartNote"]["properties"]


def test_document_is_frozen():
    with pytest.raises(ValidationError):
        _document().offset_ms = 10


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"lane_semantics": lane_semantics(6)}, "laneSemantics"),
        ({"key_mode": 5}, "key_mode"),
        ({"duration_ms": 500}, "timeMs must be less than durationMs"),
        ({"duration_ms": 600}, "HOLD end time"),
        ({"auto_play_onsets": [1200, 900]}, "autoPlayOnsets"),
        ({"auto_play_onsets": [900, 900]}, "autoPlayOnsets"),
        ({"auto_play_onsets": [2000]}, "autoPlayOnsets"),
        ({"bpm_events": [BpmEvent(time_ms=10, bpm=150.0)]}, "bpmEvents must start"),
        (
            {
                "bpm_events": [
                    BpmEvent(time_ms=0, bpm=150.0),
                    BpmEvent(time_ms=0, bpm=120.0),
                ]
            },
            "bpmEvents must be sorted",
        ),
        ({"bpm_source": "GUESS"}, "bpm_source"),
        ({"difficulty": "NOMAL"}, "difficulty"),
    ],
)
def test_rejects_inconsistent_document(overrides, message):
    with pytest.raises(ValidationError, match=message):
        _document(**overrides)


def test_rejects_metrics_note_count_mismatch():
    with pytest.raises(ValidationError, match="noteCount"):
        _document(metrics=_metrics(note_count=99))


def test_rejects_metrics_hold_count_mismatch():
    with pytest.raises(ValidationError, match="holdCount"):
        _document(metrics=_metrics(hold_count=0))


def test_rejects_note_outside_key_mode():
    with pytest.raises(ValidationError, match="outside 4K"):
        _document(
            notes=notes_to_chart_notes([NoteEvent(time_ms=0, lane=9)]),
            metrics=_metrics(note_count=1, hold_count=0),
        )


def test_rejects_unsorted_notes():
    notes = notes_to_chart_notes([NoteEvent(time_ms=0, lane=0), NoteEvent(time_ms=500, lane=1)])
    with pytest.raises(ValidationError, match="sorted"):
        _document(
            notes=list(reversed(notes)),
            metrics=_metrics(note_count=2, hold_count=0),
        )


def test_rejects_duplicate_note_ids():
    note = ChartNote(id=1, lane=0, time_ms=0, type="TAP")
    with pytest.raises(ValidationError, match="duplicate note id"):
        _document(notes=[note, note], metrics=_metrics(note_count=2, hold_count=0))


def test_rejects_two_notes_in_the_same_lane_at_the_same_time():
    notes = notes_to_chart_notes([NoteEvent(time_ms=0, lane=0), NoteEvent(time_ms=0, lane=0)])
    with pytest.raises(ValidationError, match="duplicate note"):
        _document(notes=notes, metrics=_metrics(note_count=2, hold_count=0))


def test_rejects_overlapping_holds_in_one_lane():
    notes = notes_to_chart_notes(
        [
            NoteEvent(time_ms=0, lane=0, kind="HOLD", duration_ms=600),
            NoteEvent(time_ms=500, lane=0, kind="HOLD", duration_ms=200),
        ]
    )
    with pytest.raises(ValidationError, match="overlap"):
        _document(notes=notes, metrics=_metrics(note_count=2, hold_count=2))


def test_allows_touching_holds_in_one_lane():
    notes = notes_to_chart_notes(
        [
            NoteEvent(time_ms=0, lane=0, kind="HOLD", duration_ms=500),
            NoteEvent(time_ms=500, lane=0, kind="HOLD", duration_ms=200),
        ]
    )
    assert _document(notes=notes, metrics=_metrics(note_count=2, hold_count=2))


def test_rejects_unknown_key():
    with pytest.raises(ValidationError, match="[Ee]xtra"):
        ChartNote(id=1, lane=0, time_ms=0, type="TAP", velocity=3)


@pytest.mark.parametrize("sha", ["", "z" * 64, SHA.upper(), SHA[:-1]])
def test_rejects_malformed_audio_sha256(sha):
    with pytest.raises(ValidationError, match="audio_sha256"):
        _document(audio_sha256=sha)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"id": 1, "lane": 0, "time_ms": 0, "type": "HOLD"},
        {"id": 1, "lane": 0, "time_ms": 0, "type": "TAP", "duration_ms": 100},
    ],
)
def test_rejects_note_kind_and_duration_mismatch(kwargs):
    with pytest.raises(ValidationError, match="durationMs"):
        ChartNote(**kwargs)
