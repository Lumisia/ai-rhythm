"""Direct generation timing and structural diagnostics."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from chart_worker.hashing import sha256_file
from chart_worker.pipeline import (
    PipelineDependencies,
    PipelineOptions,
    PipelineResult,
    run_pipeline,
)
from chart_worker.schema.chart import CamelModel, ChartDocument, Sha256
from chart_worker.schema.playtest_run import PlaytestRunManifest, RunChartRef
from chart_worker.schema.types import DIFFICULTIES, KEY_MODES


class BenchmarkReport(CamelModel):
    status: Literal["PASS", "REVIEW"]
    source_name: str
    source_sha256: Sha256
    generator: str
    elapsed_ms_by_stage: dict[str, int]
    charts: list[RunChartRef]
    warnings: list[str]


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    pipeline: PipelineResult
    report_path: Path
    report: BenchmarkReport


def _load_generation_report(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("generation report must be an object")  # noqa: TRY004
    if value.get("strategy") != "MAPPERATORINATOR_SHARED_TIMING":
        raise ValueError("generation report must use the shared-timing strategy")
    elapsed = value.get("elapsedMsByStage")
    if not isinstance(elapsed, dict) or set(elapsed) != {
        "prepare",
        "analysis",
        "timing",
        "generation",
        "export",
    }:
        raise ValueError("generation report elapsedMsByStage is invalid")
    charts = value.get("charts")
    if not isinstance(charts, list) or len(charts) != len(KEY_MODES) * len(DIFFICULTIES):
        raise ValueError("generation report must contain exactly 12 charts")
    return value


def _warnings(manifest: PlaytestRunManifest, output_dir: Path) -> list[str]:
    documents = {
        (reference.key_mode, reference.difficulty): ChartDocument.model_validate_json(
            (output_dir / reference.path).read_text(encoding="utf-8")
        )
        for reference in manifest.charts
    }
    warnings = []
    for key_mode in KEY_MODES:
        ratings = [
            documents[(key_mode, difficulty)].metrics.project_rating
            for difficulty in DIFFICULTIES
        ]
        for index in range(len(ratings) - 1):
            if ratings[index + 1] < ratings[index]:
                warnings.append(
                    f"{key_mode}K rating inversion: {DIFFICULTIES[index]} "
                    f"{ratings[index]:.3f} > {DIFFICULTIES[index + 1]} "
                    f"{ratings[index + 1]:.3f}"
                )
    return warnings


def run_benchmark(
    options: PipelineOptions,
    *,
    dependencies: PipelineDependencies | None = None,
) -> BenchmarkResult:
    pipeline = run_pipeline(options, dependencies=dependencies)
    manifest = PlaytestRunManifest.model_validate_json(
        pipeline.manifest_path.read_text(encoding="utf-8")
    )
    generation = _load_generation_report(
        pipeline.output_dir / manifest.generation_report_path
    )
    report = BenchmarkReport(
        status="REVIEW" if generation.get("timingReviewRequired") is True else "PASS",
        source_name=options.source.name,
        source_sha256=sha256_file(options.source),
        generator=options.generator,
        elapsed_ms_by_stage=pipeline.elapsed_ms_by_stage,
        charts=manifest.charts,
        warnings=_warnings(manifest, pipeline.output_dir),
    )
    report_path = pipeline.output_dir / "benchmark-report.json"
    report_path.write_text(
        report.model_dump_json(by_alias=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return BenchmarkResult(pipeline=pipeline, report_path=report_path, report=report)
