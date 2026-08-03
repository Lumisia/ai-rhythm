import json
from dataclasses import replace
from pathlib import Path

import pytest

from chart_worker.bench import BenchmarkReport, run_benchmark
from chart_worker.pipeline import PipelineOptions
from tests.support import fake_dependencies


def test_benchmark_writes_direct_generation_report_for_all_charts(tmp_path: Path):
    source = tmp_path / "fixture.wav"
    source.write_bytes(b"source")
    output_dir = tmp_path / "run"

    result = run_benchmark(
        PipelineOptions(
            source=source,
            output_dir=output_dir,
            title="fixture",
            generator="fake",
            seed=7,
        ),
        dependencies=fake_dependencies(),
    )

    report = BenchmarkReport.model_validate_json(
        result.report_path.read_text(encoding="utf-8")
    )
    assert report.source_name == "fixture.wav"
    assert report.status == "PASS"
    assert len(report.charts) == 12
    assert set(report.elapsed_ms_by_stage) == {
        "prepare",
        "analysis",
        "generation",
        "export",
    }
    generation = json.loads((output_dir / "generation-report.json").read_text())
    assert generation["strategy"] == "MAPPERATORINATOR_DIRECT"
    assert all(chart["attemptCount"] == 1 for chart in generation["charts"])


def test_benchmark_rejects_a_generation_report_without_direct_strategy(
    tmp_path: Path, monkeypatch
):
    source = tmp_path / "fixture.wav"
    source.write_bytes(b"source")
    output_dir = tmp_path / "run"

    def corrupt(options, *, dependencies=None):
        result = __import__("chart_worker.pipeline", fromlist=["run_pipeline"]).run_pipeline(
            options, dependencies=dependencies
        )
        path = output_dir / "generation-report.json"
        payload = json.loads(path.read_text())
        payload["strategy"] = "HYBRID"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return result

    monkeypatch.setattr("chart_worker.bench.run_pipeline", corrupt)
    with pytest.raises(ValueError, match="direct strategy"):
        run_benchmark(
            PipelineOptions(
                source=source,
                output_dir=output_dir,
                title="fixture",
                generator="fake",
            ),
            dependencies=fake_dependencies(),
        )


def test_benchmark_requires_review_when_timing_diagnostics_do(tmp_path: Path):
    source = tmp_path / "fixture.wav"
    source.write_bytes(b"source")
    dependencies = fake_dependencies()
    aligned = dependencies.analyze(Path("unused"))
    review_analysis = replace(aligned, onset_ms=(1_937,))

    result = run_benchmark(
        PipelineOptions(
            source=source,
            output_dir=tmp_path / "run",
            title="fixture",
            generator="fake",
        ),
        dependencies=replace(
            dependencies,
            analyze=lambda _path: review_analysis,
        ),
    )

    assert result.report.status == "REVIEW"


def test_benchmark_requires_review_when_timing_diagnostics_are_insufficient(
    tmp_path: Path,
):
    source = tmp_path / "fixture.wav"
    source.write_bytes(b"source")
    dependencies = fake_dependencies()
    analyzed = dependencies.analyze(Path("unused"))

    result = run_benchmark(
        PipelineOptions(
            source=source,
            output_dir=tmp_path / "run",
            title="fixture",
            generator="fake",
        ),
        dependencies=replace(
            dependencies,
            analyze=lambda _path: replace(analyzed, onset_ms=()),
        ),
    )

    assert result.report.status == "REVIEW"
