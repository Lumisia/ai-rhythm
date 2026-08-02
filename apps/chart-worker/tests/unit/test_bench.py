import dataclasses
import json
from pathlib import Path

import pytest

import chart_worker.bench as bench_module
from chart_worker.bench import BenchmarkReport, run_benchmark
from chart_worker.errors import ErrorCode, WorkerError
from chart_worker.pipeline import PipelineOptions
from tests.support import fake_dependencies


def test_benchmark_writes_report_for_all_charts(tmp_path: Path):
    source = tmp_path / "fixture.wav"
    source.write_bytes(b"source")
    output_dir = tmp_path / "run"

    result = run_benchmark(
        PipelineOptions(source=source, output_dir=output_dir, title="fixture", seed=7),
        dependencies=fake_dependencies(),
    )

    report = BenchmarkReport.model_validate_json(result.report_path.read_text(encoding="utf-8"))
    assert report.source_name == "fixture.wav"
    assert report.status == "PASS"
    assert len(report.charts) == 12
    assert set(report.elapsed_ms_by_stage) == {
        "analysis",
        "timing",
        "generation",
        "stems",
        "postprocess",
    }
    assert (
        "reference accuracy UNAVAILABLE: one or more charts have no human reference onsets"
        in report.warnings
    )
    generation_report = json.loads(
        (output_dir / "generation-report.json").read_text(encoding="utf-8")
    )
    assert all(
        chart["referenceAccuracy"] == {"status": "UNAVAILABLE"}
        for chart in generation_report["charts"]
    )


def test_benchmark_preserves_pipeline_exhaustion_report_and_propagates_error(
    tmp_path: Path,
):
    source = tmp_path / "fixture.wav"
    source.write_bytes(b"source")
    reference = tmp_path / "labels.json"
    reference.write_text(
        json.dumps(
            {
                "version": 1,
                "charts": [
                    {
                        "keyMode": 4,
                        "difficulty": "NORMAL",
                        "sections": [{"id": "song", "onsetMs": [100, 300]}],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "run"

    with pytest.raises(WorkerError) as caught:
        run_benchmark(
            PipelineOptions(
                source=source,
                output_dir=output_dir,
                title="fixture",
                reference_onsets_path=reference,
            ),
            dependencies=fake_dependencies(),
        )

    assert caught.value.code is ErrorCode.CHART_CANDIDATES_EXHAUSTED
    generation_report = json.loads(
        (output_dir / "generation-report.json").read_text(encoding="utf-8")
    )
    assert generation_report["warnings"] == [
        "4K NORMAL: all chart candidates failed quality gates"
    ]
    assert len(generation_report["failedCombination"]["candidates"]) == 3
    benchmark = BenchmarkReport.model_validate_json(
        (output_dir / "benchmark-report.json").read_text(encoding="utf-8")
    )
    assert benchmark.status == "FAIL"
    assert benchmark.charts == []
    assert benchmark.elapsed_ms_by_stage == generation_report["elapsedMsByStage"]
    assert benchmark.warnings == generation_report["warnings"]
    assert not (output_dir / "playtest-run-v1.json").exists()


def test_benchmark_does_not_synthesize_report_for_other_worker_errors(tmp_path: Path):
    source = tmp_path / "fixture.wav"
    source.write_bytes(b"source")
    output_dir = tmp_path / "run"
    dependencies = fake_dependencies()

    def timing_failure(analysis, run_dir, config, enable_super_timing):
        del analysis, run_dir, config, enable_super_timing
        raise WorkerError(
            ErrorCode.CHART_TIMING_REVIEW_REQUIRED,
            "timing candidates disagree",
        )

    with pytest.raises(WorkerError) as caught:
        run_benchmark(
            PipelineOptions(source=source, output_dir=output_dir, title="fixture"),
            dependencies=dataclasses.replace(dependencies, timing=timing_failure),
        )

    assert caught.value.code is ErrorCode.CHART_TIMING_REVIEW_REQUIRED
    assert not (output_dir / "benchmark-report.json").exists()


def test_benchmark_preserves_original_exhaustion_when_failure_report_is_corrupt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    source = tmp_path / "fixture.wav"
    source.write_bytes(b"source")
    output_dir = tmp_path / "run"
    failure = WorkerError(
        ErrorCode.CHART_CANDIDATES_EXHAUSTED,
        "all chart candidates failed quality gates",
        context={"key_mode": 4, "difficulty": "NORMAL"},
    )

    def corrupt_exhaustion(options, *, dependencies=None):
        del dependencies
        options.output_dir.mkdir(parents=True)
        (options.output_dir / "generation-report.json").write_text(
            "not-json",
            encoding="utf-8",
        )
        raise failure

    monkeypatch.setattr(bench_module, "run_pipeline", corrupt_exhaustion)

    with pytest.raises(WorkerError) as caught:
        run_benchmark(
            PipelineOptions(source=source, output_dir=output_dir, title="fixture")
        )

    assert caught.value is failure
    assert not (output_dir / "benchmark-report.json").exists()
