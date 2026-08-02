import json
from pathlib import Path

import numpy as np

from chart_worker.analysis.audio_io import AudioSignal
from chart_worker.analysis.beat import BeatGrid
from chart_worker.analysis.onset import OnsetAnalysis
from chart_worker.audio.normalize import NormalizedAudio
from chart_worker.generation.mapperatorinator import GeneratedChart
from chart_worker.hashing import sha256_file
from chart_worker.schema.chart import ChartDocument
from chart_worker.schema.note import NoteEvent
from chart_worker.schema.playtest_run import AudioFileRef
from chart_worker.stages.s4_postprocess import run_postprocess
from chart_worker.stages.types import AnalysisStageResult, GeneratedVariant, StemStageResult

SHA = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"


def _inputs(tmp_path: Path):
    audio_path = tmp_path / "audio" / "game.flac"
    audio_path.parent.mkdir(parents=True)
    audio_path.write_bytes(b"game")
    analysis = AnalysisStageResult(
        normalized=NormalizedAudio(
            audio_path,
            "audio-profile-v1",
            SHA,
            2_000,
            48_000,
            2,
            2_000,
            0,
            0.0,
            -14.0,
            -1.0,
            0.0,
            "LOUDNESS",
        ),
        signal=AudioSignal(np.zeros((96_000, 2)), 48_000),
        beat_grid=BeatGrid((0, 500, 1_000, 1_500), (0,), 120.0, 4, 0.0, 4, 0, 0.0, 0.0),
        onsets=OnsetAnalysis(
            48_000,
            512,
            np.array([0.0, 0.8, 0.0]),
            np.array([[0.0, 0.2, 0.0], [0.0, 0.8, 0.0], [0.0, 0.1, 0.0]]),
            (500,),
        ),
        timing_osu_path=tmp_path / "analysis" / "timing.osu",
    )
    raw_path = tmp_path / "raw" / "4k-normal.osu"
    raw_path.parent.mkdir(parents=True)
    raw_path.write_text("raw", encoding="utf-8")
    original_note = NoteEvent(500, 0)
    generated = GeneratedVariant(
        key_mode=4,
        difficulty="NORMAL",
        requested_star=3.0,
        raw_osu_path=raw_path,
        generated=GeneratedChart([original_note], 4, "raw", "fake", 7),
    )
    stems = StemStageResult(
        game_ref=AudioFileRef(path="audio/game.flac", sha256=SHA),
        no_drums_ref=None,
        keys_ref=None,
        drum_onsets=(),
        keysound_manifest=None,
        keysound_manifest_path=None,
    )
    return analysis, generated, stems, original_note


def test_postprocess_keeps_times_and_writes_a_valid_camel_case_chart(tmp_path: Path):
    analysis, generated, stems, original_note = _inputs(tmp_path)
    result = run_postprocess(
        analysis,
        (generated,),
        stems,
        tmp_path,
        worker_version="test-build",
    )[0]

    payload = json.loads(result.path.read_text(encoding="utf-8"))
    document = ChartDocument.model_validate(payload)
    assert {note.time_ms for note in document.notes} <= {original_note.time_ms}
    assert payload["schemaVersion"] == 1
    assert payload["keyMode"] == 4
    assert payload["difficulty"] == "NORMAL"
    assert "schema_version" not in payload
    assert payload["autoPlayOnsets"] == []
    assert payload["metrics"]["noteCount"] == 1
    assert result.sha256 == sha256_file(result.path)


def test_postprocess_ids_are_deterministic_for_the_same_build(tmp_path: Path):
    analysis, generated, stems, _ = _inputs(tmp_path)
    first = run_postprocess(
        analysis, (generated,), stems, tmp_path, worker_version="same-build"
    )[0]
    first_id = first.document.chart_id
    second = run_postprocess(
        analysis, (generated,), stems, tmp_path, worker_version="same-build"
    )[0]
    assert second.document.chart_id == first_id


def test_postprocess_uses_only_unmatched_drum_onsets_for_autoplay(tmp_path: Path):
    analysis, generated, stems, _ = _inputs(tmp_path)
    stems = StemStageResult(
        game_ref=stems.game_ref,
        no_drums_ref=AudioFileRef(path="audio/no_drums.flac", sha256=SHA),
        keys_ref=AudioFileRef(path="audio/drums.flac", sha256=SHA),
        drum_onsets=(505, 900),
        keysound_manifest=None,
        keysound_manifest_path=tmp_path / "keysound-manifest.json",
    )
    result = run_postprocess(
        analysis, (generated,), stems, tmp_path, worker_version="test-build"
    )[0]
    assert result.document.auto_play_onsets == [900]
    assert result.document.metrics.drum_coverage == 0.5
