"""로컬 한 번 실행용 chart-worker 오케스트레이션."""

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter_ns
from typing import Literal
from uuid import UUID, uuid4

from chart_worker.config import WorkerConfig, load_config
from chart_worker.generation.fake import FakeGenerator
from chart_worker.generation.mapperatorinator import ChartGenerator, MapperatorinatorGenerator
from chart_worker.schema.playtest_run import (
    PlaytestRunManifest,
    RunAudioRefs,
    RunChartRef,
)
from chart_worker.stages.s1_analyze import run_analysis
from chart_worker.stages.s2_generate import run_generation
from chart_worker.stages.s3_stems import run_stems
from chart_worker.stages.s4_postprocess import run_postprocess
from chart_worker.stages.types import (
    AnalysisStageResult,
    GeneratedVariant,
    PostprocessedVariant,
    StemStageResult,
)

GeneratorName = Literal["fake", "mapperatorinator"]
AnalysisStage = Callable[[Path, Path, WorkerConfig], AnalysisStageResult]
GenerationStage = Callable[
    [AnalysisStageResult, Path, ChartGenerator, int], tuple[GeneratedVariant, ...]
]
StemStage = Callable[[AnalysisStageResult, Path, bool], StemStageResult]
PostprocessStage = Callable[
    [AnalysisStageResult, tuple[GeneratedVariant, ...], StemStageResult, Path, str],
    tuple[PostprocessedVariant, ...],
]


def _analysis_stage(source: Path, run_dir: Path, config: WorkerConfig) -> AnalysisStageResult:
    return run_analysis(source, run_dir, config=config)


def _generation_stage(
    analysis: AnalysisStageResult,
    run_dir: Path,
    generator: ChartGenerator,
    seed: int,
) -> tuple[GeneratedVariant, ...]:
    return run_generation(analysis, run_dir, generator=generator, seed=seed)


def _stem_stage(analysis: AnalysisStageResult, run_dir: Path, enabled: bool) -> StemStageResult:
    return run_stems(analysis, run_dir, enabled=enabled)


def _postprocess_stage(
    analysis: AnalysisStageResult,
    generated: tuple[GeneratedVariant, ...],
    stems: StemStageResult,
    run_dir: Path,
    worker_version: str,
) -> tuple[PostprocessedVariant, ...]:
    return run_postprocess(
        analysis,
        generated,
        stems,
        run_dir,
        worker_version=worker_version,
    )


def _utc_now() -> datetime:
    return datetime.now(UTC)


@dataclass(frozen=True, slots=True)
class PipelineDependencies:
    config: WorkerConfig = field(default_factory=load_config)
    analysis: AnalysisStage = _analysis_stage
    generation: GenerationStage = _generation_stage
    stems: StemStage = _stem_stage
    postprocess: PostprocessStage = _postprocess_stage
    now: Callable[[], datetime] = _utc_now
    new_run_id: Callable[[], UUID] = uuid4


@dataclass(frozen=True, slots=True)
class PipelineOptions:
    source: Path
    output_dir: Path
    title: str
    generator: GeneratorName = "fake"
    keysounds: bool = False
    seed: int = 0
    worker_version: str = "local"
    overwrite: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "source", Path(self.source))
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        if not self.title.strip():
            raise ValueError("title must not be empty")
        if self.generator not in ("fake", "mapperatorinator"):
            raise ValueError(f"unsupported generator: {self.generator}")


@dataclass(frozen=True, slots=True)
class PipelineResult:
    run_id: UUID
    output_dir: Path
    manifest_path: Path
    chart_paths: tuple[Path, ...]
    raw_osu_paths: tuple[Path, ...]
    elapsed_ms_by_stage: dict[str, int]


def _prepare_run_dir(options: PipelineOptions) -> Path:
    if not options.source.is_file():
        raise ValueError(f"source does not exist or is not a file: {options.source}")
    run_dir = options.output_dir.resolve()
    if run_dir.exists() and not run_dir.is_dir():
        raise ValueError(f"output path is not a directory: {run_dir}")
    if run_dir.exists() and any(run_dir.iterdir()) and not options.overwrite:
        raise ValueError(f"output directory is not empty: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _relative(path: Path, run_dir: Path) -> str:
    try:
        return path.resolve().relative_to(run_dir).as_posix()
    except ValueError:
        raise ValueError(f"pipeline asset is outside the output directory: {path}") from None


def _select_generator(name: GeneratorName, config: WorkerConfig) -> ChartGenerator:
    if name == "fake":
        return FakeGenerator()
    return MapperatorinatorGenerator(config)


def _elapsed_ms(start_ns: int) -> int:
    return max(0, round((perf_counter_ns() - start_ns) / 1_000_000))


def _generation_report(
    options: PipelineOptions,
    run_id: UUID,
    elapsed: dict[str, int],
    charts: tuple[PostprocessedVariant, ...],
) -> dict[str, object]:
    return {
        "runId": str(run_id),
        "sourceName": options.source.name,
        "generator": options.generator,
        "keysounds": options.keysounds,
        "elapsedMsByStage": elapsed,
        "charts": [
            {
                "keyMode": result.document.key_mode,
                "difficulty": result.document.difficulty,
                "targetRating": result.reports.difficulty.target_rating,
                "actualRating": result.document.metrics.project_rating,
                "reachedTarget": result.reports.difficulty.reached_target,
                "removedCount": result.reports.difficulty.removed_count,
                "playabilityRecoveredCount": result.reports.playability.recovered_count,
                "playabilityDeletedCount": result.reports.playability.deleted_count,
            }
            for result in charts
        ],
    }


def run_pipeline(
    options: PipelineOptions,
    *,
    dependencies: PipelineDependencies | None = None,
) -> PipelineResult:
    dependencies = dependencies or PipelineDependencies()
    run_dir = _prepare_run_dir(options)
    run_id = dependencies.new_run_id()
    elapsed: dict[str, int] = {}

    started = perf_counter_ns()
    analysis = dependencies.analysis(options.source.resolve(), run_dir, dependencies.config)
    elapsed["analysis"] = _elapsed_ms(started)

    started = perf_counter_ns()
    generated = dependencies.generation(
        analysis,
        run_dir,
        _select_generator(options.generator, dependencies.config),
        options.seed,
    )
    elapsed["generation"] = _elapsed_ms(started)

    started = perf_counter_ns()
    stems = dependencies.stems(analysis, run_dir, options.keysounds)
    elapsed["stems"] = _elapsed_ms(started)

    started = perf_counter_ns()
    charts = dependencies.postprocess(
        analysis,
        generated,
        stems,
        run_dir,
        options.worker_version,
    )
    elapsed["postprocess"] = _elapsed_ms(started)

    report_path = run_dir / "generation-report.json"
    report_path.write_text(
        json.dumps(
            _generation_report(options, run_id, elapsed, charts),
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    manifest = PlaytestRunManifest(
        run_id=run_id,
        title=options.title,
        generated_at=dependencies.now(),
        worker_version=options.worker_version,
        audio=RunAudioRefs(
            game=stems.game_ref,
            no_drums=stems.no_drums_ref,
            keys=stems.keys_ref,
        ),
        charts=[
            RunChartRef(
                path=_relative(chart.path, run_dir),
                sha256=chart.sha256,
                key_mode=chart.document.key_mode,
                difficulty=chart.document.difficulty,
            )
            for chart in charts
        ],
        keysound_manifest_path=(
            _relative(stems.keysound_manifest_path, run_dir)
            if stems.keysound_manifest_path is not None
            else None
        ),
        generation_report_path=_relative(report_path, run_dir),
    )
    manifest_path = run_dir / "playtest-run-v1.json"
    manifest_path.write_text(
        manifest.model_dump_json(by_alias=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return PipelineResult(
        run_id=run_id,
        output_dir=run_dir,
        manifest_path=manifest_path,
        chart_paths=tuple(chart.path for chart in charts),
        raw_osu_paths=tuple(variant.raw_osu_path for variant in generated),
        elapsed_ms_by_stage=elapsed,
    )
