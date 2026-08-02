import dataclasses
import json
import math
from pathlib import Path

import pytest

import chart_worker.pipeline as pipeline_module
from chart_worker.analysis.timing import project_beats
from chart_worker.errors import ErrorCode, WorkerError
from chart_worker.generation.fake import FakeGenerator
from chart_worker.generation.mapperatorinator import GeneratedChart
from chart_worker.pipeline import PipelineOptions, run_pipeline
from chart_worker.schema.chart import ChartDocument
from chart_worker.schema.playtest_run import PlaytestRunManifest
from tests.support import fake_dependencies


class RecordingFakeGenerator:
    def __init__(self):
        self.calls = []

    def __call__(self, request, workdir):
        self.calls.append(request)
        return FakeGenerator()(request, workdir)


def timing_candidate_payload(candidate):
    return {
        "source": candidate.source.value,
        "points": [
            {
                "timeMs": point.time_ms,
                "bpm": point.bpm,
                "meter": point.meter,
                "startBeatIndex": point.start_beat_index,
            }
            for point in candidate.points
        ],
        "projectedBeatMs": list(candidate.projected_beat_ms),
        "f1At20Ms": candidate.f1_20ms,
        "f1At50Ms": candidate.f1_50ms,
        "p95AbsMs": candidate.p95_abs_ms,
        "status": candidate.status.value,
        "reasons": list(candidate.reasons),
    }


def write_timing_quality_report(analysis) -> None:
    selected = timing_candidate_payload(analysis.timing_candidate)
    analysis.timing_quality_report_path.write_text(
        json.dumps(
            {
                "version": 1,
                "selected": selected,
                "warnings": [],
                "candidates": [selected],
            }
        )
        + "\n",
        encoding="utf-8",
    )


def mapperatorinator_dependencies(generator: RecordingFakeGenerator):
    dependencies = fake_dependencies()

    def timing_stage(analysis, run_dir, config, enable_super_timing):
        del run_dir, config
        assert enable_super_timing is True
        write_timing_quality_report(analysis)
        return analysis

    return dataclasses.replace(
        dependencies,
        timing=timing_stage,
        select_generator=lambda name, config: generator,
    )


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
    assert all(path.parent == output_dir / "raw" for path in result.raw_osu_paths)
    assert all(path.is_file() for path in result.raw_osu_paths)
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
    assert report["timing"]["selectedSource"] == "BEAT_THIS_PIECEWISE"
    assert report["timing"]["selected"] == report["timing"]["candidates"][0]
    assert report["timing"]["qualityReportPath"] == (
        "analysis/timing-quality-v1.json"
    )
    assert report["timing"]["warnings"] == []
    assert all(
        chart["referenceAccuracy"] == {"status": "UNAVAILABLE"}
        for chart in report["charts"]
    )

    for chart in report["charts"]:
        assert chart["candidateCount"] == 1
        assert chart["selectedAttempt"] == 1
        assert chart["selectedSeed"] == chart["candidates"][0]["seed"]
        assert chart["selectedParameters"] == chart["candidates"][0][
            "generationParameters"
        ]
        assert chart["rawNoteCount"] >= chart["finalNoteCount"]
        deleted = sum(chart["removals"].values())
        expected_ratio = deleted / chart["rawNoteCount"] if chart["rawNoteCount"] else 1.0
        assert chart["totalRemovalRatio"] == pytest.approx(expected_ratio)
        assert chart["signedRatingError"] == pytest.approx(
            chart["actualRating"] - chart["targetRating"]
        )
        assert chart["absoluteRatingError"] == pytest.approx(
            abs(chart["signedRatingError"])
        )
        assert chart["drumOnsetPrecision"] == {"status": "UNAVAILABLE"}
        assert chart["candidates"][0]["drumOnsetPrecision"] == {
            "status": "UNAVAILABLE"
        }
        assert chart["holdRatio"]["absoluteError"] == pytest.approx(
            abs(chart["holdRatio"]["actual"] - chart["holdRatio"]["target"])
        )
        assert chart["candidates"][0]["selection"]["status"] == "SELECTED"
        for field in ("chartPath", "rawOsuPath"):
            relative = Path(chart[field])
            assert not relative.is_absolute()
            assert (output_dir / relative).is_file()
        selected_candidate = chart["candidates"][0]
        assert selected_candidate["canonicalChartPath"] == chart["chartPath"]
        assert selected_candidate["canonicalRawOsuPath"] == chart["rawOsuPath"]
        for field in ("diagnosticRawOsuPath", "diagnosticChartPath"):
            relative = Path(selected_candidate[field])
            assert not relative.is_absolute()
            assert (output_dir / relative).is_file()


def _selected_timing_analysis(tmp_path: Path):
    source = tmp_path / "fixture.wav"
    source.write_bytes(b"source")
    run_dir = tmp_path / "run"
    dependencies = fake_dependencies()
    analysis = dependencies.analysis(source, run_dir, dependencies.config)
    return dependencies.timing(analysis, run_dir, dependencies.config, False), run_dir


def test_timing_report_accepts_structured_warning_context(tmp_path: Path):
    analysis, run_dir = _selected_timing_analysis(tmp_path)
    report = json.loads(analysis.timing_quality_report_path.read_text(encoding="utf-8"))
    report["warnings"] = [
        {
            "code": "CHART_GENERATION_FAILED",
            "message": "super timing unavailable",
            "context": {"stderr": "model missing"},
        }
    ]
    analysis.timing_quality_report_path.write_text(json.dumps(report), encoding="utf-8")

    payload = pipeline_module._timing_payload(analysis, run_dir)

    assert payload["warnings"] == report["warnings"]


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda report: report["candidates"][0].pop("projectedBeatMs"),
            "candidate fields",
        ),
        (
            lambda report: report["candidates"][0].__setitem__("p95AbsMs", math.nan),
            "finite number",
        ),
        (
            lambda report: report["candidates"][0].__setitem__("f1At20Ms", True),
            "finite number",
        ),
        (
            lambda report: report.__setitem__(
                "warnings", [{"code": "BROKEN", "message": "bad", "context": []}]
            ),
            "warning context",
        ),
    ],
)
def test_timing_report_rejects_malformed_candidate_or_warning(
    tmp_path: Path,
    mutation,
    message: str,
):
    analysis, run_dir = _selected_timing_analysis(tmp_path)
    report = json.loads(analysis.timing_quality_report_path.read_text(encoding="utf-8"))
    mutation(report)
    analysis.timing_quality_report_path.write_text(json.dumps(report), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        pipeline_module._timing_payload(analysis, run_dir)


def test_timing_report_rejects_selected_metric_mismatch(tmp_path: Path):
    analysis, run_dir = _selected_timing_analysis(tmp_path)
    report = json.loads(analysis.timing_quality_report_path.read_text(encoding="utf-8"))
    report["selected"]["f1At20Ms"] = 0.5
    analysis.timing_quality_report_path.write_text(json.dumps(report), encoding="utf-8")

    with pytest.raises(ValueError, match="selected candidate does not match analysis"):
        pipeline_module._timing_payload(analysis, run_dir)


def test_timing_report_accepts_negative_lead_in_point_and_projected_beat(tmp_path: Path):
    analysis, run_dir = _selected_timing_analysis(tmp_path)
    negative_point = dataclasses.replace(analysis.timing_candidate.points[0], time_ms=-120)
    negative_candidate = dataclasses.replace(
        analysis.timing_candidate,
        points=(negative_point,),
        projected_beat_ms=project_beats(
            (negative_point,),
            end_ms=analysis.normalized.duration_ms,
        ),
    )
    analysis = dataclasses.replace(analysis, timing_candidate=negative_candidate)
    write_timing_quality_report(analysis)

    payload = pipeline_module._timing_payload(analysis, run_dir)

    assert payload["selected"]["points"][0]["timeMs"] == -120
    assert payload["selected"]["projectedBeatMs"][0] == -120


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
    assert caught.value.context["key_mode"] == 4
    assert caught.value.context["difficulty"] == "NORMAL"
    candidates = caught.value.context["candidates"]
    assert [candidate["attempt"] for candidate in candidates] == [1, 2, 3]
    assert [candidate["seed"] for candidate in candidates] == [1, 10_001, 20_001]
    assert [candidate["parameters"] for candidate in candidates] == [
        {"requested_star": 3.0, "cfg_scale": 1.25},
        {"requested_star": 3.0, "cfg_scale": 1.0},
        {"requested_star": 3.0, "cfg_scale": 1.0},
    ]
    assert all(candidate["timing_source"] == "BEAT_THIS_PIECEWISE" for candidate in candidates)
    assert all(candidate["failure_metrics"]["reference_accuracy"]["status"] == "FAIL" for candidate in candidates)
    assert all(
        (output_dir / candidate["raw_osu_path"]).is_file()
        for candidate in candidates
    )
    assert all(
        (output_dir / candidate["chart_path"]).is_file()
        for candidate in candidates
    )
    assert len({candidate["raw_osu_path"] for candidate in candidates}) == 3
    assert len({candidate["chart_path"] for candidate in candidates}) == 3
    retry_dirs = sorted((output_dir / "raw" / "candidates").glob("*/attempt-2"))
    assert retry_dirs == [
        output_dir / "raw" / "candidates" / "4k-normal" / "attempt-2"
    ]

    report = json.loads((output_dir / "generation-report.json").read_text())
    assert report["failedCombination"]["keyMode"] == 4
    assert report["failedCombination"]["difficulty"] == "NORMAL"
    assert len(report["failedCombination"]["candidates"]) == 3
    assert not (output_dir / "playtest-run-v1.json").exists()


def test_pipeline_retry_parameters_follow_each_too_hard_candidate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
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
    real_quality_of = pipeline_module.candidate_quality_of

    def always_too_hard(*args, **kwargs):
        quality = real_quality_of(*args, **kwargs)
        return dataclasses.replace(quality, rating_error=0.3501)

    monkeypatch.setattr(pipeline_module, "candidate_quality_of", always_too_hard)

    with pytest.raises(WorkerError) as caught:
        run_pipeline(
            PipelineOptions(
                source=source,
                output_dir=tmp_path / "run",
                title="fixture",
                reference_onsets_path=reference,
            ),
            dependencies=fake_dependencies(),
        )

    candidates = caught.value.context["candidates"]
    assert [candidate["parameters"] for candidate in candidates] == [
        {"requested_star": 3.0, "cfg_scale": 1.25},
        {"requested_star": 3.0, "cfg_scale": 1.0},
        {"requested_star": 2.5, "cfg_scale": 1.0},
    ]


def test_mapperatorinator_compares_unguided_at_same_star_even_when_guided_passes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    source = tmp_path / "fixture.wav"
    source.write_bytes(b"source")
    generator = RecordingFakeGenerator()
    real_quality_of = pipeline_module.candidate_quality_of

    def passing_quality(*args, **kwargs):
        quality = real_quality_of(*args, **kwargs)
        return dataclasses.replace(
            quality,
            long_gap_bars=0.0,
            rating_error=0.0,
            removed_ratio=0.0,
            drum_precision=None,
            playability_passes=1,
            reference_pass=None,
        )

    monkeypatch.setattr(pipeline_module, "candidate_quality_of", passing_quality)

    result = run_pipeline(
        PipelineOptions(
            source=source,
            output_dir=tmp_path / "run",
            title="fixture",
            generator="mapperatorinator",
        ),
        dependencies=mapperatorinator_dependencies(generator),
    )

    normal_calls = [
        request
        for request in generator.calls
        if request.key_mode == 4 and request.difficulty == "NORMAL"
    ]
    assert [
        (request.seed, request.requested_star, request.cfg_scale)
        for request in normal_calls
    ] == [(1, 3.0, 1.25), (10_001, 3.0, 1.0)]
    assert len(result.raw_osu_paths) == 12
    assert all(path.is_file() for path in result.raw_osu_paths)
    assert result.manifest_path.is_file()
    report = json.loads((result.output_dir / "generation-report.json").read_text())
    normal_report = next(
        chart
        for chart in report["charts"]
        if chart["keyMode"] == 4 and chart["difficulty"] == "NORMAL"
    )
    assert normal_report["candidateCount"] == 2
    assert [candidate["attempt"] for candidate in normal_report["candidates"]] == [1, 2]
    assert sum(
        candidate["selection"]["status"] == "SELECTED"
        for candidate in normal_report["candidates"]
    ) == 1
    selected_candidate = next(
        candidate
        for candidate in normal_report["candidates"]
        if candidate["selection"]["status"] == "SELECTED"
    )
    rejected_candidate = next(
        candidate
        for candidate in normal_report["candidates"]
        if candidate["selection"]["status"] == "REJECTED"
    )
    assert selected_candidate["canonicalChartPath"] == normal_report["chartPath"]
    assert selected_candidate["canonicalRawOsuPath"] == normal_report["rawOsuPath"]
    assert "canonicalChartPath" not in rejected_candidate
    assert "canonicalRawOsuPath" not in rejected_candidate
    assert sum(
        candidate["selection"]["status"] == "REJECTED"
        for candidate in normal_report["candidates"]
    ) == 1


def test_mapperatorinator_failed_guided_candidate_adds_one_bounded_star_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    source = tmp_path / "fixture.wav"
    source.write_bytes(b"source")
    generator = RecordingFakeGenerator()
    real_quality_of = pipeline_module.candidate_quality_of

    def normal_is_too_hard(*args, **kwargs):
        quality = real_quality_of(*args, **kwargs)
        variant = args[1]
        return dataclasses.replace(
            quality,
            long_gap_bars=0.0,
            rating_error=0.3501 if variant.difficulty == "NORMAL" else 0.0,
            removed_ratio=0.0,
            drum_precision=None,
            playability_passes=1,
            reference_pass=None,
        )

    monkeypatch.setattr(pipeline_module, "candidate_quality_of", normal_is_too_hard)

    with pytest.raises(WorkerError) as caught:
        run_pipeline(
            PipelineOptions(
                source=source,
                output_dir=tmp_path / "run",
                title="fixture",
                generator="mapperatorinator",
            ),
            dependencies=mapperatorinator_dependencies(generator),
        )

    assert caught.value.code is ErrorCode.CHART_CANDIDATES_EXHAUSTED
    assert caught.value.context["difficulty"] == "NORMAL"
    candidates = caught.value.context["candidates"]
    assert [(candidate["seed"], candidate["parameters"]) for candidate in candidates] == [
        (1, {"requested_star": 3.0, "cfg_scale": 1.25}),
        (10_001, {"requested_star": 3.0, "cfg_scale": 1.0}),
        (20_001, {"requested_star": 2.5, "cfg_scale": 1.0}),
    ]


def test_reference_failure_retries_only_that_combination_and_selects_a_passing_seed(
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

    class MisalignedStructurallyBetterCandidates:
        def __call__(self, request, workdir):
            generated = FakeGenerator()(request, workdir)
            if request.key_mode == 4 and request.difficulty == "NORMAL" and request.seed == 1:
                return GeneratedChart(
                    notes=[
                        dataclasses.replace(note, time_ms=note.time_ms + 100)
                        for note in generated.notes
                    ],
                    key_mode=generated.key_mode,
                    osu_text="",
                    generator_name=generated.generator_name,
                    seed=generated.seed,
                )
            if (
                request.key_mode == 4
                and request.difficulty == "NORMAL"
                and request.seed == 20_001
            ):
                return GeneratedChart(
                    notes=[],
                    key_mode=generated.key_mode,
                    osu_text="",
                    generator_name=generated.generator_name,
                    seed=generated.seed,
                )
            return generated

    dependencies = dataclasses.replace(
        fake_dependencies(),
        select_generator=lambda name, config: MisalignedStructurallyBetterCandidates(),
    )
    output_dir = tmp_path / "run"
    result = run_pipeline(
        PipelineOptions(
            source=source,
            output_dir=output_dir,
            title="fixture",
            reference_onsets_path=reference,
        ),
        dependencies=dependencies,
    )

    normal_chart = ChartDocument.model_validate_json(
        (output_dir / "charts" / "4k-normal.chart.json").read_text(encoding="utf-8")
    )
    assert normal_chart.generator.seed == 10_001
    assert (output_dir / "raw" / "4k-normal.osu").is_file()
    assert sorted((output_dir / "raw" / "candidates").glob("*/attempt-2")) == [
        output_dir / "raw" / "candidates" / "4k-normal" / "attempt-2"
    ]
    manifest = PlaytestRunManifest.model_validate_json(
        result.manifest_path.read_text(encoding="utf-8")
    )
    assert all(Path(chart.path).parts[0] == "charts" for chart in manifest.charts)
    report_charts = json.loads((output_dir / "generation-report.json").read_text())["charts"]
    normal_report = next(
        chart
        for chart in report_charts
        if chart["keyMode"] == 4 and chart["difficulty"] == "NORMAL"
    )
    assert normal_report["referenceAccuracy"]["status"] == "PASS"


def test_fake_pipeline_never_enables_mapperatorinator_super_timing(tmp_path: Path):
    source = tmp_path / "fixture.wav"
    source.write_bytes(b"source")
    dependencies = fake_dependencies()

    def timing_stage(analysis, run_dir, config, enable_super_timing):
        del run_dir, config
        assert enable_super_timing is False
        write_timing_quality_report(analysis)
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
