"""원샷 실행의 시간과 채보 경고를 고정하는 benchmark 리포트."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from chart_worker.errors import ErrorCode, WorkerError
from chart_worker.hashing import sha256_file
from chart_worker.pipeline import (
    PipelineDependencies,
    PipelineOptions,
    PipelineResult,
    run_pipeline,
)
from chart_worker.postprocess.difficulty_solver import RATING_TOLERANCE
from chart_worker.rating.project_rating import TARGET_RATING
from chart_worker.schema.chart import CamelModel, ChartDocument, Sha256
from chart_worker.schema.playtest_run import PlaytestRunManifest, RunChartRef
from chart_worker.schema.types import DIFFICULTIES, KEY_MODES


class BenchmarkReport(CamelModel):
    status: Literal["PASS", "FAIL"]
    source_name: str
    source_sha256: Sha256
    generator: str
    keysounds: bool
    elapsed_ms_by_stage: dict[str, int]
    charts: list[RunChartRef]
    warnings: list[str]


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    pipeline: PipelineResult
    report_path: Path
    report: BenchmarkReport


def _warnings(
    manifest: PlaytestRunManifest,
    output_dir: Path,
    generation_report: dict[str, object],
) -> list[str]:
    documents = {
        (reference.key_mode, reference.difficulty): ChartDocument.model_validate_json(
            (output_dir / reference.path).read_text(encoding="utf-8")
        )
        for reference in manifest.charts
    }
    warnings = []
    charts_value = generation_report.get("charts")
    if not isinstance(charts_value, list):
        raise ValueError(  # noqa: TRY004 - persisted contract validation
            "generation report charts must be an array"
        )
    if any(
        isinstance(chart, dict)
        and chart.get("referenceAccuracy") == {"status": "UNAVAILABLE"}
        for chart in charts_value
    ):
        warnings.append(
            "reference accuracy UNAVAILABLE: one or more charts have no human reference onsets"
        )
    for (key_mode, difficulty), document in documents.items():
        rating = document.metrics.project_rating
        target = TARGET_RATING[difficulty]
        if rating > target + RATING_TOLERANCE:
            warnings.append(
                f"{key_mode}K {difficulty}: rating {rating:.3f} exceeds target {target:.3f}"
            )
        if document.metrics.project_tier != difficulty:
            warnings.append(
                f"{key_mode}K {difficulty}: measured tier is {document.metrics.project_tier}"
            )
    for key_mode in KEY_MODES:
        ratings = [
            documents[(key_mode, difficulty)].metrics.project_rating for difficulty in DIFFICULTIES
        ]
        for easier, harder, easier_rating, harder_rating in zip(
            DIFFICULTIES[:-1],
            DIFFICULTIES[1:],
            ratings[:-1],
            ratings[1:],
            strict=True,
        ):
            if harder_rating < easier_rating:
                warnings.append(
                    f"{key_mode}K rating inversion: {easier} {easier_rating:.3f} > "
                    f"{harder} {harder_rating:.3f}"
                )
    return warnings


def _load_generation_report(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(  # noqa: TRY004 - persisted contract validation
            "generation report must be an object"
        )
    elapsed = value.get("elapsedMsByStage")
    if not isinstance(elapsed, dict) or not all(
        isinstance(stage, str)
        and not isinstance(duration, bool)
        and isinstance(duration, int)
        and duration >= 0
        for stage, duration in elapsed.items()
    ):
        raise ValueError("generation report elapsedMsByStage is invalid")
    charts = value.get("charts")
    if not isinstance(charts, list):
        raise ValueError("generation report charts must be an array")  # noqa: TRY004
    warnings = value.get("warnings")
    if not isinstance(warnings, list) or not all(
        isinstance(warning, str) for warning in warnings
    ):
        raise ValueError("generation report warnings must be an array of strings")
    return value


def _write_failed_benchmark(
    options: PipelineOptions,
    generation_report: dict[str, object],
) -> Path:
    elapsed = generation_report["elapsedMsByStage"]
    warnings = generation_report["warnings"]
    if not any(
        "all chart candidates failed quality gates" in warning for warning in warnings
    ):
        raise ValueError("exhausted generation report is missing its failure warning")
    report = BenchmarkReport(
        status="FAIL",
        source_name=options.source.name,
        source_sha256=sha256_file(options.source),
        generator=options.generator,
        keysounds=options.keysounds,
        elapsed_ms_by_stage=elapsed,
        charts=[],
        warnings=warnings,
    )
    report_path = options.output_dir.resolve() / "benchmark-report.json"
    report_path.write_text(
        report.model_dump_json(by_alias=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return report_path


def run_benchmark(
    options: PipelineOptions,
    *,
    dependencies: PipelineDependencies | None = None,
) -> BenchmarkResult:
    try:
        pipeline = run_pipeline(options, dependencies=dependencies)
    except WorkerError as error:
        if error.code is not ErrorCode.CHART_CANDIDATES_EXHAUSTED:
            raise
        try:
            generation_report = _load_generation_report(
                options.output_dir.resolve() / "generation-report.json"
            )
            _write_failed_benchmark(options, generation_report)
        except (OSError, ValueError):
            # Failure reporting is best-effort. A corrupt/missing report or a
            # failed write must never hide the causal candidate-exhaustion error.
            pass
        raise
    manifest = PlaytestRunManifest.model_validate_json(
        pipeline.manifest_path.read_text(encoding="utf-8")
    )
    generation_report_path = pipeline.output_dir / manifest.generation_report_path
    generation_report = _load_generation_report(generation_report_path)
    report = BenchmarkReport(
        status="PASS",
        source_name=options.source.name,
        source_sha256=sha256_file(options.source),
        generator=options.generator,
        keysounds=options.keysounds,
        elapsed_ms_by_stage=pipeline.elapsed_ms_by_stage,
        charts=manifest.charts,
        warnings=_warnings(manifest, pipeline.output_dir, generation_report),
    )
    report_path = pipeline.output_dir / "benchmark-report.json"
    report_path.write_text(
        report.model_dump_json(by_alias=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return BenchmarkResult(pipeline=pipeline, report_path=report_path, report=report)
