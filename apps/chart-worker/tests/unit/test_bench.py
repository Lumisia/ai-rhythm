import json
from dataclasses import replace
from pathlib import Path

import pytest

from chart_worker.bench import BenchmarkReport, _model_inference_calls, run_benchmark
from chart_worker.errors import ErrorCode, WorkerError
from chart_worker.hashing import sha256_file
from chart_worker.pipeline import PipelineOptions
from tests.support import fake_dependencies


def test_benchmark_writes_shared_timing_generation_report_for_all_charts(
    tmp_path: Path,
):
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

    report = BenchmarkReport.model_validate_json(result.report_path.read_text(encoding="utf-8"))
    assert report.source_name == "fixture.wav"
    assert report.status == "PASS"
    assert len(report.charts) == 12
    assert set(report.elapsed_ms_by_stage) == {
        "prepare",
        "analysis",
        "timing",
        "generation",
        "export",
    }
    assert report.model_inference_calls == 13
    assert report.analysis_elapsed_ms == report.elapsed_ms_by_stage["analysis"]
    assert report.difficulty_selector_mode == "SHADOW_V2"
    assert report.intro_contract is not None
    generation = json.loads((output_dir / "generation-report.json").read_text())
    assert generation["strategy"] == "MAPPERATORINATOR_SHARED_TIMING"
    assert all(chart["attemptCount"] == 1 for chart in generation["charts"])


def test_model_call_count_prefers_actual_hydra_invocation_directories(tmp_path: Path):
    (tmp_path / "timing" / "attempt-1" / ".hydra-run").mkdir(parents=True)
    (tmp_path / "raw" / "attempt-1" / ".hydra-run").mkdir(parents=True)
    generation = {
        "timingAttemptCount": 4,
        "charts": [{"generationAttemptCount": 9}],
    }

    assert _model_inference_calls(generation, tmp_path) == 2


def test_benchmark_rejects_a_generation_report_without_shared_timing_strategy(
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
        manifest_path = output_dir / "playtest-run-v2.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["generationReport"]["sha256"] = sha256_file(path)
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        return result

    monkeypatch.setattr("chart_worker.bench.run_pipeline", corrupt)
    with pytest.raises(ValueError, match="shared-timing strategy"):
        run_benchmark(
            PipelineOptions(
                source=source,
                output_dir=output_dir,
                title="fixture",
                generator="fake",
            ),
            dependencies=fake_dependencies(),
        )


def test_benchmark_rejects_generation_report_hash_mismatch(tmp_path: Path, monkeypatch):
    source = tmp_path / "fixture.wav"
    source.write_bytes(b"source")
    output_dir = tmp_path / "run"

    def corrupt(options, *, dependencies=None):
        result = __import__("chart_worker.pipeline", fromlist=["run_pipeline"]).run_pipeline(
            options, dependencies=dependencies
        )
        path = output_dir / "generation-report.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["warnings"] = ["tampered-after-manifest"]
        path.write_text(json.dumps(payload), encoding="utf-8")
        return result

    monkeypatch.setattr("chart_worker.bench.run_pipeline", corrupt)
    with pytest.raises(ValueError, match="generation report hash"):
        run_benchmark(
            PipelineOptions(
                source=source,
                output_dir=output_dir,
                title="fixture",
                generator="fake",
            ),
            dependencies=fake_dependencies(),
        )


def test_benchmark_rejects_raw_model_output_when_quality_gates_reject(
    tmp_path: Path,
):
    """품질 축이 거절한 raw 출력은 benchmark에서도 배포하지 않는다."""
    source = tmp_path / "fixture.wav"
    source.write_bytes(b"source")
    dependencies = fake_dependencies()
    aligned = dependencies.analyze(Path("unused"))
    review_analysis = replace(aligned, onset_ms=(1_937,))

    output_dir = tmp_path / "run"
    with pytest.raises(WorkerError) as captured:
        run_benchmark(
            PipelineOptions(
                source=source,
                output_dir=output_dir,
                title="fixture",
                generator="fake",
            ),
            dependencies=replace(
                dependencies,
                analyze=lambda _path: review_analysis,
            ),
        )

    generation = json.loads((output_dir / "generation-report.json").read_text())
    assert captured.value.code is ErrorCode.CHART_CANDIDATES_EXHAUSTED
    assert generation["publishable"] is False
    assert generation["status"] == "EXHAUSTED"
    assert generation["charts"] == []


def test_insufficient_onset_evidence_is_a_diagnostic_not_a_status_downgrade(
    tmp_path: Path,
):
    """onset 근거가 약한 것만으로 곡 상태를 낮추지 않는다.

    24곡 배치에서 사용자가 정상으로 판정한 채보의 구간 precision50 이
    0.00 까지 내려갔다. librosa onset 은 음악적 정답이 아니므로
    timingReviewRequired 는 진단 플래그로만 남긴다.
    """
    source = tmp_path / "fixture.wav"
    source.write_bytes(b"source")
    dependencies = fake_dependencies()
    analyzed = dependencies.analyze(Path("unused"))

    output_dir = tmp_path / "run"
    result = run_benchmark(
        PipelineOptions(
            source=source,
            output_dir=output_dir,
            title="fixture",
            generator="fake",
        ),
        dependencies=replace(
            dependencies,
            analyze=lambda _path: replace(analyzed, onset_ms=()),
        ),
    )

    generation = json.loads((output_dir / "generation-report.json").read_text())
    assert generation["publishable"] is False
    assert generation["outcomeStatusV2"]["quality"] == "REVIEW"
    assert generation["publicationDecision"]["decision"] == "PLAYTEST_ONLY"
    assert generation["status"] == "PASS"
    assert generation["timingReviewRequired"] is True
    assert all(chart["acceptanceStatus"] == "REVIEW" for chart in generation["charts"])
    assert result.report.status == "PASS"
    assert result.report.missing_charts == []
    assert result.report_path == output_dir / "benchmark-report.json"
    assert result.report_path.is_file()
