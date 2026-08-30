"""Direct generation timing and structural diagnostics."""

import json
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Literal

from pydantic import Field

from chart_worker.hashing import sha256_file
from chart_worker.pipeline import (
    PipelineDependencies,
    PipelineOptions,
    PipelineResult,
    run_pipeline,
)
from chart_worker.schema.chart import CamelModel, ChartDocument, Sha256
from chart_worker.schema.playtest_run import (
    MissingChartRef,
    PlaytestRunManifestV3,
    RunChartRef,
)
from chart_worker.schema.types import DIFFICULTIES, KEY_MODES


class BenchmarkReport(CamelModel):
    status: Literal["PASS", "REVIEW", "PARTIAL"]
    source_name: str
    source_sha256: Sha256
    generator: str
    elapsed_ms_by_stage: dict[str, int]
    model_inference_calls: int
    analysis_elapsed_ms: int
    difficulty_selector_mode: Literal["CURRENT", "SHADOW_V2", "V2"]
    intro_contract: dict[str, object] | None = None
    charts: list[RunChartRef]
    missing_charts: list[MissingChartRef] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


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
    missing = value.get("missingCharts") or []
    expected = len(KEY_MODES) * len(DIFFICULTIES)
    if not isinstance(charts, list) or not charts:
        raise ValueError("generation report must contain at least one chart")
    if not isinstance(missing, list) or len(charts) + len(missing) != expected:
        raise ValueError(
            "generation report charts and missingCharts must cover all 12 combinations"
        )
    return value


def _warnings(manifest: PlaytestRunManifestV3, output_dir: Path) -> list[str]:
    documents = {
        (reference.key_mode, reference.difficulty): ChartDocument.model_validate_json(
            (output_dir / reference.path).read_text(encoding="utf-8")
        )
        for reference in manifest.charts
    }
    warnings = []
    for key_mode in KEY_MODES:
        # 발행되지 않은 난이도는 사다리 검사에서 빠진다. 없는 조합을
        # 기준점으로 삼으면 멀쩡한 조합이 역전으로 신고된다.
        present = [
            (difficulty, documents[(key_mode, difficulty)].metrics.project_rating)
            for difficulty in DIFFICULTIES
            if (key_mode, difficulty) in documents
        ]
        for (easier, easier_rating), (harder, harder_rating) in pairwise(present):
            if harder_rating < easier_rating:
                warnings.append(
                    f"{key_mode}K rating inversion: {easier} "
                    f"{easier_rating:.3f} > {harder} {harder_rating:.3f}"
                )
    return warnings


def _model_inference_calls(
    generation: dict[str, object],
    output_dir: Path,
) -> int:
    hydra_invocations = sum(path.is_dir() for path in output_dir.rglob(".hydra-run"))
    if hydra_invocations:
        return hydra_invocations
    timing_attempts = int(generation.get("timingAttemptCount") or 0)
    charts = generation.get("charts")
    if not isinstance(charts, list):
        return timing_attempts
    map_attempts = sum(
        int(chart.get("generationAttemptCount") or 0) for chart in charts if isinstance(chart, dict)
    )
    return timing_attempts + map_attempts


def _difficulty_selector_mode(
    generation: dict[str, object],
) -> Literal["CURRENT", "SHADOW_V2", "V2"]:
    comparisons = generation.get("difficultySelectionShadow")
    if not isinstance(comparisons, list) or not comparisons:
        return "CURRENT"
    modes = {comparison.get("mode") for comparison in comparisons if isinstance(comparison, dict)}
    if len(modes) != 1:
        raise ValueError("generation report has inconsistent selector modes")
    mode = modes.pop()
    if mode not in {"SHADOW_V2", "V2"}:
        raise ValueError("generation report has an invalid selector mode")
    return mode


def run_benchmark(
    options: PipelineOptions,
    *,
    dependencies: PipelineDependencies | None = None,
) -> BenchmarkResult:
    pipeline = run_pipeline(options, dependencies=dependencies)
    manifest = PlaytestRunManifestV3.model_validate_json(
        pipeline.manifest_path.read_text(encoding="utf-8")
    )
    generation_report_path = pipeline.output_dir / manifest.generation_report.path
    if sha256_file(generation_report_path) != manifest.generation_report.sha256:
        raise ValueError("generation report hash does not match playtest manifest")
    generation = _load_generation_report(generation_report_path)
    # 곡 상태는 생성 리포트가 정한다. timingReviewRequired 는 진단
    # 플래그라 사람 판정과 자주 어긋난다 (24곡 배치 실측).
    status = generation.get("status")
    if status not in {"PASS", "REVIEW", "PARTIAL"}:
        status = "REVIEW" if generation.get("timingReviewRequired") is True else "PASS"
    report = BenchmarkReport(
        status=status,
        source_name=options.source.name,
        source_sha256=sha256_file(options.source),
        generator=options.generator,
        elapsed_ms_by_stage=pipeline.elapsed_ms_by_stage,
        model_inference_calls=_model_inference_calls(
            generation,
            pipeline.output_dir,
        ),
        analysis_elapsed_ms=pipeline.elapsed_ms_by_stage["analysis"],
        difficulty_selector_mode=_difficulty_selector_mode(generation),
        intro_contract=(
            generation.get("introStartContract")
            if isinstance(generation.get("introStartContract"), dict)
            else None
        ),
        charts=manifest.charts,
        missing_charts=manifest.missing_charts,
        warnings=_warnings(manifest, pipeline.output_dir),
    )
    report_path = pipeline.output_dir / "benchmark-report.json"
    report_path.write_text(
        report.model_dump_json(by_alias=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return BenchmarkResult(pipeline=pipeline, report_path=report_path, report=report)
