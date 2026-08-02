import dataclasses
import json
from pathlib import Path

import pytest

from chart_worker.errors import ErrorCode, WorkerError
from chart_worker.pipeline import PipelineOptions, run_pipeline
from chart_worker.schema.playtest_run import PlaytestRunManifest
from tests.support import fake_dependencies


def test_fake_pipeline_writes_playtest_manifest(tmp_path: Path):
    source = tmp_path / "fixture.wav"
    source.write_bytes(b"source")
    output_dir = tmp_path / "run"

    result = run_pipeline(
        PipelineOptions(
            source=source,
            output_dir=output_dir,
            title="fixture",
            generator="fake",
            keysounds=False,
            seed=7,
        ),
        dependencies=fake_dependencies(),
    )

    manifest = PlaytestRunManifest.model_validate_json(
        result.manifest_path.read_text(encoding="utf-8")
    )
    assert len(manifest.charts) == 12
    assert len({(chart.key_mode, chart.difficulty) for chart in manifest.charts}) == 12
    assert all((output_dir / chart.path).is_file() for chart in manifest.charts)
    assert len(result.raw_osu_paths) == 12
    assert set(result.elapsed_ms_by_stage) == {
        "analysis",
        "timing",
        "generation",
        "stems",
        "postprocess",
    }
    assert (output_dir / manifest.generation_report_path).is_file()
    metadata = json.loads((output_dir / "analysis" / "analysis-v1.json").read_text())
    assert metadata["timingSelection"]["source"] == "BEAT_THIS_PIECEWISE"
    assert metadata["timingSelection"]["qualityReportPath"] == (
        "analysis/timing-quality-v1.json"
    )
    assert (output_dir / metadata["timingSelection"]["qualityReportPath"]).is_file()
    report = json.loads((output_dir / "generation-report.json").read_text())
    assert all(
        chart["referenceAccuracy"] == {"status": "UNAVAILABLE"}
        for chart in report["charts"]
    )


def test_pipeline_copies_reference_and_reports_measured_accuracy_per_chart(tmp_path: Path):
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
                        "sections": [
                            {
                                "id": "song",
                                "onsetMs": [0, 250, 500, 750, 1000, 1250, 1500, 1750],
                            }
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "run"

    run_pipeline(
        PipelineOptions(
            source=source,
            output_dir=output_dir,
            title="fixture",
            reference_onsets_path=reference,
        ),
        dependencies=fake_dependencies(),
    )

    copied = output_dir / "analysis" / "reference-onsets-v1.json"
    assert json.loads(copied.read_text()) == json.loads(reference.read_text())
    charts = json.loads((output_dir / "generation-report.json").read_text())["charts"]
    normal_4k = next(
        chart
        for chart in charts
        if chart["keyMode"] == 4 and chart["difficulty"] == "NORMAL"
    )
    assert normal_4k["referenceAccuracy"] == {
        "status": "PASS",
        "macroF1At20Ms": 1.0,
        "phaseAbsMs": 0.0,
        "p95AbsMs": 0.0,
    }
    assert all(
        chart["referenceAccuracy"] == {"status": "UNAVAILABLE"}
        for chart in charts
        if chart is not normal_4k
    )


def test_pipeline_reuses_reference_already_at_canonical_output_path(tmp_path: Path):
    source = tmp_path / "fixture.wav"
    source.write_bytes(b"source")
    output_dir = tmp_path / "run"
    reference = output_dir / "analysis" / "reference-onsets-v1.json"
    reference.parent.mkdir(parents=True)
    reference.write_text(json.dumps({"version": 1, "charts": []}), encoding="utf-8")

    run_pipeline(
        PipelineOptions(
            source=source,
            output_dir=output_dir,
            title="fixture",
            reference_onsets_path=reference,
            overwrite=True,
        ),
        dependencies=fake_dependencies(),
    )

    assert json.loads(reference.read_text()) == {"version": 1, "charts": []}


def test_reference_failure_writes_report_then_exhausts_the_first_candidate(tmp_path: Path):
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
        run_pipeline(
            PipelineOptions(
                source=source,
                output_dir=output_dir,
                title="fixture",
                reference_onsets_path=reference,
            ),
            dependencies=fake_dependencies(),
        )

    assert caught.value.code is ErrorCode.CHART_CANDIDATES_EXHAUSTED
    assert caught.value.context == {
        "candidates": [
            {
                "key_mode": 4,
                "difficulty": "NORMAL",
                "seed": 1,
                "timing_source": "BEAT_THIS_PIECEWISE",
                "failure_metrics": {
                    "status": "FAIL",
                    "macroF1At20Ms": 0.0,
                    "phaseAbsMs": 0.0,
                    "p95AbsMs": 0.0,
                },
                "raw_osu_path": str(output_dir / "raw" / "4k-normal.osu"),
                "chart_path": str(output_dir / "charts" / "4k-normal.chart.json"),
            }
        ]
    }
    charts = json.loads((output_dir / "generation-report.json").read_text())["charts"]
    normal_4k = next(
        chart
        for chart in charts
        if chart["keyMode"] == 4 and chart["difficulty"] == "NORMAL"
    )
    assert normal_4k["referenceAccuracy"]["status"] == "FAIL"


def test_fake_pipeline_never_enables_mapperatorinator_super_timing(tmp_path: Path):
    source = tmp_path / "fixture.wav"
    source.write_bytes(b"source")
    dependencies = fake_dependencies()

    def timing_stage(analysis, run_dir, config, enable_super_timing):
        del run_dir, config
        assert enable_super_timing is False
        analysis.timing_quality_report_path.write_text("{}\n", encoding="utf-8")
        return analysis

    run_pipeline(
        PipelineOptions(source=source, output_dir=tmp_path / "run", title="fixture"),
        dependencies=dataclasses.replace(dependencies, timing=timing_stage),
    )


def test_pipeline_refuses_a_nonempty_output_without_removing_it(tmp_path: Path):
    source = tmp_path / "fixture.wav"
    source.write_bytes(b"source")
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    marker = output_dir / "keep.txt"
    marker.write_text("keep", encoding="utf-8")

    with pytest.raises(ValueError, match="not empty"):
        run_pipeline(
            PipelineOptions(source=source, output_dir=output_dir, title="fixture"),
            dependencies=fake_dependencies(),
        )

    assert marker.read_text(encoding="utf-8") == "keep"
