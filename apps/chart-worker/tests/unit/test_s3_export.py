from pathlib import Path

from chart_worker.audio.normalize import NormalizedAudio
from chart_worker.generation.mapperatorinator import GeneratedChart
from chart_worker.generation.osu_parser import OsuBpmEvent
from chart_worker.schema.note import NoteEvent
from chart_worker.stages.s3_export import run_export
from chart_worker.stages.types import GeneratedVariant, PreparedAudio
from tests.support import pass_acceptance

SHA = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"


def _prepared(tmp_path: Path) -> PreparedAudio:
    audio_path = tmp_path / "audio" / "game.flac"
    audio_path.parent.mkdir(parents=True)
    audio_path.write_bytes(b"audio")
    return PreparedAudio(
        normalized=NormalizedAudio(
            path=audio_path,
            profile_version="audio-profile-v2",
            sha256=SHA,
            duration_ms=4_000,
            sample_rate_hz=48_000,
            channels=2,
            source_duration_ms=4_000,
            trimmed_ms=0,
            gain_db=0.0,
            achieved_lufs=-14.0,
            achieved_true_peak_dbtp=-1.0,
            shortfall_lu=0.0,
            limited_by="LOUDNESS",
        )
    )


def test_export_preserves_mapperatorinator_notes_and_timing_exactly(tmp_path: Path):
    prepared = _prepared(tmp_path)
    notes = [
        NoteEvent(time_ms=500, lane=0),
        NoteEvent(time_ms=1_000, lane=3, kind="HOLD", duration_ms=750),
    ]
    timing = (
        OsuBpmEvent(time_ms=-120, bpm=120.0),
        OsuBpmEvent(time_ms=2_000, bpm=150.0),
    )
    raw_path = tmp_path / "raw" / "4k-normal.osu"
    raw_path.parent.mkdir(parents=True)
    raw_path.write_text("raw", encoding="utf-8")
    generated = GeneratedVariant(
        key_mode=4,
        difficulty="NORMAL",
        requested_star=1.5,
        raw_osu_path=raw_path,
        acceptance=pass_acceptance(),
        generated=GeneratedChart(
            notes=notes,
            key_mode=4,
            osu_text="raw",
            generator_name="mapperatorinator-v32",
            seed=7,
            bpm_events=timing,
        ),
    )
    before = tuple((note.time_ms, note.lane, note.kind, note.duration_ms) for note in notes)

    exported = run_export(
        prepared,
        (generated,),
        tmp_path,
        worker_version="test-build",
    )[0]

    document = exported.document
    after = tuple(
        (note.time_ms, note.lane, note.kind, note.duration_ms)
        for note in generated.generated.notes
    )
    assert after == before
    assert [
        (note.time_ms, note.lane, note.kind, note.duration_ms)
        for note in document.notes
    ] == list(before)
    assert [(event.time_ms, event.bpm) for event in document.bpm_events] == [
        (-120, 120.0),
        (2_000, 150.0),
    ]
    assert document.offset_ms == 0
    assert document.bpm_source == "MAPPERATORINATOR"
    assert document.metrics.moved_note_ratio == 0.0
    assert document.auto_play_onsets == []
    assert exported.path == tmp_path / "charts" / "4k-normal.chart.json"
    assert exported.path.is_file()


def test_export_rejects_a_generated_map_without_timing(tmp_path: Path):
    prepared = _prepared(tmp_path)
    raw_path = tmp_path / "raw" / "4k-easy.osu"
    raw_path.parent.mkdir(parents=True)
    raw_path.write_text("raw", encoding="utf-8")
    generated = GeneratedVariant(
        key_mode=4,
        difficulty="EASY",
        requested_star=1.0,
        raw_osu_path=raw_path,
        acceptance=pass_acceptance(),
        generated=GeneratedChart(
            notes=[NoteEvent(time_ms=500, lane=0)],
            key_mode=4,
            osu_text="raw",
            generator_name="mapperatorinator-v32",
            seed=3,
        ),
    )

    try:
        run_export(prepared, (generated,), tmp_path, worker_version="test")
    except ValueError as error:
        assert "timing" in str(error).lower()
    else:
        raise AssertionError("missing Mapperatorinator timing was accepted")
