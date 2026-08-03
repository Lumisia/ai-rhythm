import json
from dataclasses import replace
from pathlib import Path

import pytest

from chart_worker.errors import ErrorCode, WorkerError
from chart_worker.pipeline import PipelineOptions, run_pipeline
from chart_worker.schema.chart import ChartDocument
from chart_worker.schema.playtest_run import PlaytestRunManifest
from tests.support import fake_dependencies


def test_direct_pipeline_writes_twelve_unmodified_charts(tmp_path: Path):
    source = tmp_path / "fixture.wav"
    source.write_bytes(b"source")
    output_dir = tmp_path / "run"

    result = run_pipeline(
        PipelineOptions(
            source=source,
            output_dir=output_dir,
            title="fixture",
            generator="fake",
            seed=7,
        ),
        dependencies=fake_dependencies(),
    )

    manifest = PlaytestRunManifest.model_validate_json(
        result.manifest_path.read_text(encoding="utf-8")
    )
    assert len(manifest.charts) == 12
    assert len(result.raw_osu_paths) == 12
    assert set(result.elapsed_ms_by_stage) == {
        "prepare",
        "analysis",
        "generation",
        "export",
    }
    assert manifest.audio.no_drums is None
    assert manifest.audio.keys is None
    assert manifest.keysound_manifest_path is None
    assert not (output_dir / "analysis").exists()

    report = json.loads((output_dir / "generation-report.json").read_text())
    assert report["strategy"] == "MAPPERATORINATOR_DIRECT"
    assert report["timingAuthority"] == "FAKE"
    assert report["noteMutationEnabled"] is False
    assert report["attemptsPerChartMax"] == 3
    assert report["canonicalAudioSha256"] == manifest.audio.game.sha256
    assert report["timingReviewRequired"] is False
    assert report["elapsedMsByStage"] == result.elapsed_ms_by_stage
    assert len(report["charts"]) == 12

    for chart_report, chart_ref, raw_path in zip(
        report["charts"], manifest.charts, result.raw_osu_paths, strict=True
    ):
        document = ChartDocument.model_validate_json(
            (output_dir / chart_ref.path).read_text(encoding="utf-8")
        )
        assert chart_report["rawNoteCount"] == len(document.notes)
        assert chart_report["finalNoteCount"] == len(document.notes)
        assert chart_report["attemptCount"] == 1
        assert chart_report["attemptErrors"] == []
        assert chart_report["timingDiagnostics"]["status"] == "PASS"
        assert chart_report["cfgScale"] == 1.0
        assert chart_report["chartPath"] == chart_ref.path
        assert chart_report["rawOsuPath"] == raw_path.relative_to(output_dir).as_posix()
        assert "candidates" not in chart_report
        assert raw_path.parent == output_dir / "raw"


def test_generation_report_records_the_selected_retry_attempt(tmp_path: Path):
    source = tmp_path / "fixture.wav"
    source.write_bytes(b"source")
    dependencies = fake_dependencies()

    def generation(prepared, run_dir, generator, seed):
        variants = dependencies.generation(prepared, run_dir, generator, seed)
        retried = replace(
            variants[0],
            generated=replace(variants[0].generated, seed=19),
            attempt=2,
            attempt_errors=("lane 4 is outside requested 4K",),
        )
        return (retried, *variants[1:])

    run_pipeline(
        PipelineOptions(
            source=source,
            output_dir=tmp_path / "run",
            title="fixture",
            generator="fake",
            seed=7,
        ),
        dependencies=replace(dependencies, generation=generation),
    )

    report = json.loads((tmp_path / "run" / "generation-report.json").read_text())
    first = report["charts"][0]
    assert first["attemptCount"] == 2
    assert first["seed"] == 19
    assert first["attemptErrors"] == ["lane 4 is outside requested 4K"]


def test_pipeline_analyzes_only_the_canonical_game_audio_once(tmp_path: Path):
    source = tmp_path / "fixture.wav"
    source.write_bytes(b"source")
    dependencies = fake_dependencies()
    calls = []

    def analyze(path):
        calls.append(path)
        return dependencies.analyze(path)

    result = run_pipeline(
        PipelineOptions(
            source=source,
            output_dir=tmp_path / "run",
            title="fixture",
            generator="fake",
        ),
        dependencies=replace(dependencies, analyze=analyze),
    )

    assert calls == [result.output_dir / "audio" / "game.flac"]


def test_pipeline_rejects_canonical_audio_changed_after_prepare(tmp_path: Path):
    source = tmp_path / "fixture.wav"
    source.write_bytes(b"source")
    dependencies = fake_dependencies()
    analyze_calls = []

    def prepare(source_path, run_dir, config):
        prepared = dependencies.prepare(source_path, run_dir, config)
        prepared.normalized.path.write_bytes(b"tampered after hashing")
        return prepared

    def analyze(path):
        analyze_calls.append(path)
        return dependencies.analyze(path)

    with pytest.raises(WorkerError) as captured:
        run_pipeline(
            PipelineOptions(
                source=source,
                output_dir=tmp_path / "run",
                title="fixture",
                generator="fake",
            ),
            dependencies=replace(
                dependencies,
                prepare=prepare,
                analyze=analyze,
            ),
        )

    assert captured.value.code is ErrorCode.ASSET_HASH_MISMATCH
    assert analyze_calls == []


def test_pipeline_rejects_canonical_audio_changed_during_generation(tmp_path: Path):
    source = tmp_path / "fixture.wav"
    source.write_bytes(b"source")
    dependencies = fake_dependencies()
    export_calls = []

    def generation(prepared, run_dir, generator, seed):
        variants = dependencies.generation(prepared, run_dir, generator, seed)
        prepared.normalized.path.write_bytes(b"tampered during generation")
        return variants

    def export(prepared, generated, run_dir, worker_version):
        export_calls.append(run_dir)
        return dependencies.export(prepared, generated, run_dir, worker_version)

    with pytest.raises(WorkerError) as captured:
        run_pipeline(
            PipelineOptions(
                source=source,
                output_dir=tmp_path / "run",
                title="fixture",
                generator="fake",
            ),
            dependencies=replace(
                dependencies,
                generation=generation,
                export=export,
            ),
        )

    assert captured.value.code is ErrorCode.ASSET_HASH_MISMATCH
    assert export_calls == []


def test_pipeline_defaults_to_mapperatorinator():
    options = PipelineOptions(
        source=Path("song.wav"),
        output_dir=Path("run"),
        title="song",
    )
    assert options.generator == "mapperatorinator"


def test_pipeline_rejects_an_empty_title():
    with pytest.raises(ValueError, match="title"):
        PipelineOptions(
            source=Path("song.wav"),
            output_dir=Path("run"),
            title="  ",
        )


def test_pipeline_rejects_a_nonempty_output_without_deleting_existing_files(tmp_path: Path):
    source = tmp_path / "fixture.wav"
    source.write_bytes(b"source")
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    (output_dir / "keep.txt").write_text("keep", encoding="utf-8")

    with pytest.raises(ValueError, match="not empty"):
        run_pipeline(
            PipelineOptions(source=source, output_dir=output_dir, title="fixture"),
            dependencies=fake_dependencies(),
        )

    assert (output_dir / "keep.txt").read_text(encoding="utf-8") == "keep"
