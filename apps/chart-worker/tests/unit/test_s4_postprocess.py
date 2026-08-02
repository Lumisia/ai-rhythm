import dataclasses
import json
from pathlib import Path

import numpy as np

from chart_worker.analysis.audio_io import AudioSignal
from chart_worker.analysis.beat import BeatGrid
from chart_worker.analysis.onset import OnsetAnalysis
from chart_worker.analysis.timing import TimingPoint, TimingSource
from chart_worker.audio.normalize import NormalizedAudio
from chart_worker.generation.mapperatorinator import GeneratedChart
from chart_worker.hashing import sha256_file
from chart_worker.schema.chart import ChartDocument
from chart_worker.schema.note import NoteEvent
from chart_worker.schema.playtest_run import AudioFileRef
from chart_worker.stages.s4_postprocess import (
    candidate_quality_of,
    run_postprocess,
    run_postprocess_variant,
)
from chart_worker.stages.types import AnalysisStageResult, GeneratedVariant, StemStageResult
from tests.support import timing_candidate

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
        timing_candidate=timing_candidate(),
        timing_osu_path=tmp_path / "analysis" / "timing.osu",
        timing_quality_report_path=tmp_path / "analysis" / "timing-quality-v1.json",
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


def test_postprocess_exports_every_selected_bpm_event(tmp_path: Path):
    analysis, generated, stems, _ = _inputs(tmp_path)
    analysis = dataclasses.replace(
        analysis,
        timing_candidate=timing_candidate(
            points=(
                TimingPoint(120, 120.0, 4, 0),
                TimingPoint(16_000, 100.0, 4, 32),
            ),
            duration_ms=20_000,
        ),
    )

    result = run_postprocess(
        analysis, (generated,), stems, tmp_path, worker_version="test"
    )[0]

    assert [event.time_ms for event in result.document.bpm_events] == [0, 16_000]
    assert result.document.bpm_source == "BEAT_THIS"
    assert result.document.offset_ms == 120


def test_postprocess_labels_super_timing_as_mapperatorinator(tmp_path: Path):
    analysis, generated, stems, _ = _inputs(tmp_path)
    analysis = dataclasses.replace(
        analysis,
        timing_candidate=timing_candidate(source=TimingSource.MAPPERATORINATOR_SUPER),
    )

    result = run_postprocess(
        analysis, (generated,), stems, tmp_path, worker_version="test"
    )[0]

    assert result.document.bpm_source == "MAPPERATORINATOR"


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
    first = run_postprocess(analysis, (generated,), stems, tmp_path, worker_version="same-build")[0]
    first_id = first.document.chart_id
    second = run_postprocess(analysis, (generated,), stems, tmp_path, worker_version="same-build")[
        0
    ]
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
    result = run_postprocess(analysis, (generated,), stems, tmp_path, worker_version="test-build")[
        0
    ]
    assert result.document.auto_play_onsets == [900]
    assert result.document.metrics.drum_coverage == 0.5


def test_candidate_evaluation_writes_only_attempt_artifact_until_selected(tmp_path: Path):
    analysis, generated, stems, _ = _inputs(tmp_path)
    canonical = tmp_path / "charts" / "4k-normal.chart.json"
    canonical.parent.mkdir(parents=True)
    canonical.write_text("winner", encoding="utf-8")

    evaluated = run_postprocess_variant(
        analysis,
        generated,
        stems,
        tmp_path,
        worker_version="test",
        write_output=False,
    )

    assert canonical.read_text(encoding="utf-8") == "winner"
    assert evaluated.path == generated.raw_osu_path.parent / "chart.json"
    assert evaluated.path.is_file()


def test_selected_candidate_is_the_only_variant_written_to_canonical_chart(tmp_path: Path):
    analysis, generated, stems, _ = _inputs(tmp_path)
    evaluated = run_postprocess_variant(
        analysis,
        generated,
        stems,
        tmp_path,
        worker_version="test",
        write_output=False,
    )
    selected = run_postprocess_variant(
        analysis,
        generated,
        stems,
        tmp_path,
        worker_version="test",
        write_output=True,
    )

    assert evaluated.path != selected.path
    assert selected.path == tmp_path / "charts" / "4k-normal.chart.json"
    assert selected.path.is_file()


def test_candidate_quality_extracts_reference_and_unavailable_drums(tmp_path: Path):
    analysis, generated, stems, _ = _inputs(tmp_path)
    result = run_postprocess_variant(
        analysis,
        generated,
        stems,
        tmp_path,
        worker_version="test",
        write_output=False,
    )

    quality = candidate_quality_of(
        analysis,
        generated,
        result,
        stems,
        reference_pass=True,
    )

    assert quality.removed_ratio == 0.0
    assert quality.rating_error == (
        result.document.metrics.project_rating - result.reports.difficulty.target_rating
    )
    assert quality.drum_precision is None
    assert quality.playability_passes == result.reports.playability.passes
    assert quality.hold_ratio_error == 0.15
    assert quality.reference_pass is True
    assert quality.requested_star == generated.requested_star
    assert quality.cfg_scale == generated.cfg_scale


def test_zero_raw_notes_are_always_a_failed_candidate(tmp_path: Path):
    analysis, generated, stems, _ = _inputs(tmp_path)
    generated = dataclasses.replace(
        generated,
        generated=dataclasses.replace(generated.generated, notes=[]),
    )
    result = run_postprocess_variant(
        analysis,
        generated,
        stems,
        tmp_path,
        worker_version="test",
        write_output=False,
    )

    quality = candidate_quality_of(
        analysis,
        generated,
        result,
        stems,
        reference_pass=None,
    )

    assert quality.removed_ratio == 1.0


def test_removed_ratio_sums_every_deletion_stage_over_raw_count(tmp_path: Path):
    analysis, generated, stems, original_note = _inputs(tmp_path)
    generated = dataclasses.replace(
        generated,
        generated=dataclasses.replace(generated.generated, notes=[original_note] * 10),
    )
    result = run_postprocess_variant(
        analysis,
        dataclasses.replace(
            generated,
            generated=dataclasses.replace(generated.generated, notes=[original_note]),
        ),
        stems,
        tmp_path,
        worker_version="test",
        write_output=False,
    )
    reports = dataclasses.replace(
        result.reports,
        conversion=dataclasses.replace(result.reports.conversion, deleted_count=1),
        difficulty=dataclasses.replace(result.reports.difficulty, removed_count=2),
        playability=dataclasses.replace(result.reports.playability, deleted_count=3),
    )

    quality = candidate_quality_of(
        analysis,
        generated,
        dataclasses.replace(result, reports=reports),
        stems,
        reference_pass=None,
    )

    assert quality.removed_ratio == 0.6
