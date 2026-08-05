import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from chart_worker import pipeline
from chart_worker.analysis.activity import AudioActivity
from chart_worker.errors import ErrorCode, WorkerError
from chart_worker.generation.fake import FakeGenerator
from chart_worker.hashing import sha256_file
from chart_worker.pipeline import PipelineOptions, run_pipeline
from chart_worker.schema.chart import ChartDocument
from chart_worker.schema.playtest_run import PlaytestRunManifest
from chart_worker.validation.quality_gate import GateAction
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
        "timing",
        "generation",
        "export",
    }
    assert manifest.audio.no_drums is None
    assert manifest.audio.keys is None
    assert manifest.keysound_manifest_path is None
    assert not (output_dir / "analysis").exists()

    report = json.loads((output_dir / "generation-report.json").read_text())
    assert report["strategy"] == "MAPPERATORINATOR_SHARED_TIMING"
    assert report["timingAuthority"] == "audio/timing-reference.osu"
    assert report["timingAuthoritySha256"] == sha256_file(
        output_dir / "audio" / "timing-reference.osu"
    )
    assert report["timingGenerationMode"] == "STANDARD"
    assert report["timingAttemptCount"] == 1
    assert report["noteMutationEnabled"] is False
    assert report["mapperatorinatorConstraintPatch"] is None
    assert report["attemptsPerChartMax"] == 3
    assert report["canonicalAudioSha256"] == manifest.audio.game.sha256
    assert report["qualityGateVersion"] == "quality-gate-v1"
    assert report["publishable"] is True
    assert report["status"] == "PASS"
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
        assert chart_report["acceptanceStatus"] == "PASS"
        assert chart_report["acceptanceReasons"] == []
        assert set(chart_report["acceptanceDecisions"]) == {
            "STRUCTURE",
            "TIMING_IDENTITY",
            "TIMING_ALIGNMENT",
            "COVERAGE",
        }
        assert "activeOnsetCount" in chart_report["timingDiagnostics"]
        assert "quietCoverageGaps" in chart_report["timingDiagnostics"]
        assert chart_report["cfgScale"] == 1.0
        assert chart_report["chartPath"] == chart_ref.path
        assert chart_report["rawOsuPath"] == raw_path.relative_to(output_dir).as_posix()
        assert "candidates" not in chart_report
        assert raw_path.parent == output_dir / "raw"


@pytest.mark.parametrize(
    ("code", "status"),
    [
        (ErrorCode.CHART_TIMING_REVIEW_REQUIRED, "REVIEW"),
        (ErrorCode.CHART_CANDIDATES_EXHAUSTED, "EXHAUSTED"),
    ],
)
def test_withheld_generation_writes_failure_report_without_publishable_artifacts(
    code: ErrorCode, status: str, tmp_path: Path
):
    source = tmp_path / "fixture.wav"
    source.write_bytes(b"source")
    output_dir = tmp_path / "run"
    dependencies = fake_dependencies()
    export_calls = []
    failure = WorkerError(
        code,
        "withheld fixture",
        context={"seed": 7, "attempts": [{"seed": 7, "gateReport": {"action": status}}]},
    )

    def generation(prepared, authority, analysis, run_dir, generator, seed):
        del prepared, authority, analysis, run_dir, generator, seed
        raise failure

    def export(prepared, generated, run_dir, worker_version):
        export_calls.append(run_dir)
        return dependencies.export(prepared, generated, run_dir, worker_version)

    with pytest.raises(WorkerError) as captured:
        run_pipeline(
            PipelineOptions(
                source=source,
                output_dir=output_dir,
                title="fixture",
                generator="fake",
                seed=7,
            ),
            dependencies=replace(
                dependencies,
                generation=generation,
                export=export,
            ),
        )

    assert captured.value is failure
    report = json.loads((output_dir / "generation-report.json").read_text())
    assert report["version"] == 1
    assert report["qualityGateVersion"] == "quality-gate-v1"
    assert report["runId"] == "00000000-0000-0000-0000-000000000007"
    assert report["publishable"] is False
    assert report["status"] == status
    assert report["error"] == {
        "code": code.value,
        "context": failure.context,
    }
    assert report["canonicalAudioSha256"] == sha256_file(
        output_dir / "audio" / "game.flac"
    )
    assert report["timingAuthority"] == "audio/timing-reference.osu"
    assert report["timingAuthoritySha256"] == sha256_file(
        output_dir / "audio" / "timing-reference.osu"
    )
    assert export_calls == []
    assert not (output_dir / "charts").exists()
    assert not (output_dir / "playtest-run-v1.json").exists()


@pytest.mark.parametrize(
    ("action", "code", "status"),
    [
        (
            GateAction.REVIEW,
            ErrorCode.CHART_TIMING_REVIEW_REQUIRED,
            "REVIEW",
        ),
        (
            GateAction.RETRY_MAP,
            ErrorCode.CHART_CANDIDATES_EXHAUSTED,
            "EXHAUSTED",
        ),
    ],
)
def test_pipeline_rejects_non_pass_variants_returned_by_generation_stage(
    action: GateAction, code: ErrorCode, status: str, tmp_path: Path
):
    source = tmp_path / "fixture.wav"
    source.write_bytes(b"source")
    output_dir = tmp_path / "run"
    dependencies = fake_dependencies()
    export_calls = []
    rejected_acceptance = None

    def generation(prepared, authority, analysis, run_dir, generator, seed):
        nonlocal rejected_acceptance
        variants = dependencies.generation(
            prepared,
            authority,
            analysis,
            run_dir,
            generator,
            seed,
        )
        first, *remaining = variants[0].acceptance.decisions
        rejected_acceptance = replace(
            variants[0].acceptance,
            action=action,
            decisions=(
                replace(first, action=action, reasons=(f"FIXTURE_{action.value}",)),
                *remaining,
            ),
        )
        return (
            replace(variants[0], acceptance=rejected_acceptance),
            *variants[1:],
        )

    def export(prepared, generated, run_dir, worker_version):
        export_calls.append(run_dir)
        return dependencies.export(prepared, generated, run_dir, worker_version)

    with pytest.raises(WorkerError) as captured:
        run_pipeline(
            PipelineOptions(
                source=source,
                output_dir=output_dir,
                title="fixture",
                generator="fake",
            ),
            dependencies=replace(
                dependencies,
                generation=generation,
                export=export,
            ),
        )

    assert rejected_acceptance is not None
    assert captured.value.code is code
    assert captured.value.context == {
        "variants": [
            {
                "key_mode": 4,
                "difficulty": "EASY",
                "gate_report": rejected_acceptance.to_report(),
            }
        ]
    }
    report = json.loads((output_dir / "generation-report.json").read_text())
    assert report["publishable"] is False
    assert report["status"] == status
    assert report["error"] == {
        "code": code.value,
        "context": captured.value.context,
    }
    assert export_calls == []
    assert not (output_dir / "charts").exists()
    assert not (output_dir / "playtest-run-v1.json").exists()


def test_generation_report_uses_recorded_acceptance_timing(tmp_path: Path):
    source = tmp_path / "fixture.wav"
    source.write_bytes(b"source")

    assert not hasattr(pipeline, "diagnose_chart_timing")

    run_pipeline(
        PipelineOptions(
            source=source,
            output_dir=tmp_path / "run",
            title="fixture",
            generator="fake",
        ),
        dependencies=fake_dependencies(),
    )


def test_pipeline_runs_shared_timing_before_twelve_maps_with_one_generator(
    tmp_path: Path,
):
    source = tmp_path / "fixture.wav"
    source.write_bytes(b"source")
    dependencies = fake_dependencies()
    calls = []
    select_calls = []

    class RecordingGenerator:
        def __init__(self):
            self.delegate = FakeGenerator()
            self.timing_calls = 0
            self.map_calls = 0

        def generate_timing(self, request, workdir):
            self.timing_calls += 1
            return self.delegate.generate_timing(request, workdir)

        def generate_map(self, request, workdir):
            self.map_calls += 1
            return self.delegate.generate_map(request, workdir)

    generator = RecordingGenerator()

    def prepare(source_path, run_dir, config):
        calls.append("prepare")
        return dependencies.prepare(source_path, run_dir, config)

    def analyze(path):
        calls.append("analyze")
        return dependencies.analyze(path)

    def select_generator(name, config):
        select_calls.append((name, config))
        return generator

    def timing(prepared, analysis, run_dir, selected_generator, seed):
        calls.append("timing")
        assert selected_generator is generator
        return dependencies.timing(prepared, analysis, run_dir, selected_generator, seed)

    def generation(prepared, authority, analysis, run_dir, selected_generator, seed):
        calls.append("generation")
        assert selected_generator is generator
        return dependencies.generation(
            prepared,
            authority,
            analysis,
            run_dir,
            selected_generator,
            seed,
        )

    def export(prepared, generated, run_dir, worker_version):
        calls.append("export")
        return dependencies.export(prepared, generated, run_dir, worker_version)

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
            select_generator=select_generator,
            timing=timing,
            generation=generation,
            export=export,
        ),
    )

    assert calls == ["prepare", "analyze", "timing", "generation", "export"]
    assert len(select_calls) == 1
    assert generator.timing_calls == 1
    assert generator.map_calls == 12


def test_generation_report_records_the_selected_retry_attempt(tmp_path: Path):
    source = tmp_path / "fixture.wav"
    source.write_bytes(b"source")
    dependencies = fake_dependencies()

    def generation(prepared, authority, analysis, run_dir, generator, seed):
        variants = dependencies.generation(
            prepared,
            authority,
            analysis,
            run_dir,
            generator,
            seed,
        )
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


def test_pipeline_passes_one_shared_onset_analysis_to_timing_and_map_generation(
    tmp_path: Path,
):
    source = tmp_path / "fixture.wav"
    source.write_bytes(b"source")
    dependencies = fake_dependencies()
    analyses = []
    timing_analyses = []
    generation_analyses = []

    def analyze(path):
        analysis = dependencies.analyze(path)
        analyses.append(analysis)
        return analysis

    def timing(prepared, analysis, run_dir, generator, seed):
        timing_analyses.append(analysis)
        return dependencies.timing(prepared, analysis, run_dir, generator, seed)

    def generation(prepared, authority, analysis, run_dir, generator, seed):
        generation_analyses.append(analysis)
        return dependencies.generation(
            prepared,
            authority,
            analysis,
            run_dir,
            generator,
            seed,
        )

    run_pipeline(
        PipelineOptions(
            source=source,
            output_dir=tmp_path / "run",
            title="fixture",
            generator="fake",
        ),
        dependencies=replace(
            dependencies,
            analyze=analyze,
            timing=timing,
            generation=generation,
        ),
    )

    assert timing_analyses == analyses
    assert generation_analyses == analyses


def test_pipeline_passes_shared_activity_to_every_chart_diagnostic(tmp_path: Path):
    source = tmp_path / "fixture.wav"
    source.write_bytes(b"source")
    dependencies = fake_dependencies()

    def analyze(path):
        analysis = dependencies.analyze(path)
        return replace(
            analysis,
            activity=AudioActivity(
                frame_ms=10.0,
                rms_db=np.full(200, -80.0),
                floor_db=-60.0,
                active_onset_ms=(),
            ),
        )

    with pytest.raises(WorkerError) as captured:
        run_pipeline(
            PipelineOptions(
                source=source,
                output_dir=tmp_path / "run",
                title="fixture",
                generator="fake",
            ),
            dependencies=replace(dependencies, analyze=analyze),
        )

    report = json.loads((tmp_path / "run" / "generation-report.json").read_text())
    assert captured.value.code is ErrorCode.CHART_TIMING_REVIEW_REQUIRED
    assert report["publishable"] is False
    assert report["status"] == "REVIEW"
    gate_report = report["error"]["context"]["gate_report"]
    assert gate_report["timing"]["activeOnsetCount"] == 0
    assert "LOW_ACTIVE_ONSET_SUPPORT" in gate_report["decisions"][
        "TIMING_ALIGNMENT"
    ]["reasons"]
    assert not (tmp_path / "run" / "charts").exists()


def test_generation_report_records_mapperatorinator_constraint_patch(tmp_path: Path):
    source = tmp_path / "fixture.wav"
    source.write_bytes(b"source")

    run_pipeline(
        PipelineOptions(
            source=source,
            output_dir=tmp_path / "run",
            title="fixture",
            generator="mapperatorinator",
        ),
        dependencies=fake_dependencies(),
    )

    report = json.loads((tmp_path / "run" / "generation-report.json").read_text())
    assert report["mapperatorinatorConstraintPatch"] == "mania-keycount-v1"


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

    def generation(prepared, authority, analysis, run_dir, generator, seed):
        variants = dependencies.generation(
            prepared,
            authority,
            analysis,
            run_dir,
            generator,
            seed,
        )
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


def test_pipeline_rejects_canonical_audio_changed_during_timing(tmp_path: Path):
    source = tmp_path / "fixture.wav"
    source.write_bytes(b"source")
    dependencies = fake_dependencies()
    generation_calls = []

    def timing(prepared, analysis, run_dir, generator, seed):
        authority = dependencies.timing(prepared, analysis, run_dir, generator, seed)
        prepared.normalized.path.write_bytes(b"tampered during timing")
        return authority

    def generation(prepared, authority, analysis, run_dir, generator, seed):
        generation_calls.append(run_dir)
        return dependencies.generation(
            prepared,
            authority,
            analysis,
            run_dir,
            generator,
            seed,
        )

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
                timing=timing,
                generation=generation,
            ),
        )

    assert captured.value.code is ErrorCode.ASSET_HASH_MISMATCH
    assert generation_calls == []


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
