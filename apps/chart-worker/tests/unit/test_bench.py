from pathlib import Path

from chart_worker.bench import BenchmarkReport, run_benchmark
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
    assert len(report.charts) == 12
    assert set(report.elapsed_ms_by_stage) == {"analysis", "generation", "stems", "postprocess"}
