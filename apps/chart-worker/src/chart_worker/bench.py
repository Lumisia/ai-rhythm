"""원샷 실행의 시간과 채보 경고를 고정하는 benchmark 리포트."""

from dataclasses import dataclass
from pathlib import Path

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


def _warnings(manifest: PlaytestRunManifest, output_dir: Path) -> list[str]:
    documents = {
        (reference.key_mode, reference.difficulty): ChartDocument.model_validate_json(
            (output_dir / reference.path).read_text(encoding="utf-8")
        )
        for reference in manifest.charts
    }
    warnings = []
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


def run_benchmark(
    options: PipelineOptions,
    *,
    dependencies: PipelineDependencies | None = None,
) -> BenchmarkResult:
    pipeline = run_pipeline(options, dependencies=dependencies)
    manifest = PlaytestRunManifest.model_validate_json(
        pipeline.manifest_path.read_text(encoding="utf-8")
    )
    report = BenchmarkReport(
        source_name=options.source.name,
        source_sha256=sha256_file(options.source),
        generator=options.generator,
        keysounds=options.keysounds,
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
