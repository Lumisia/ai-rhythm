"""Direct Mapperatorinator chart generation pipeline."""

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path
from time import perf_counter_ns
from typing import Literal
from uuid import UUID, uuid4

from chart_worker.config import WorkerConfig, load_config
from chart_worker.generation.fake import FakeGenerator
from chart_worker.generation.mapperatorinator import ChartGenerator, MapperatorinatorGenerator
from chart_worker.generation.params import DESCRIPTORS
from chart_worker.schema.playtest_run import (
    AudioFileRef,
    PlaytestRunManifest,
    RunAudioRefs,
    RunChartRef,
)
from chart_worker.stages.s1_prepare import run_prepare
from chart_worker.stages.s2_generate import run_generation
from chart_worker.stages.s3_export import run_export
from chart_worker.stages.types import ExportedVariant, GeneratedVariant, PreparedAudio

GeneratorName = Literal["fake", "mapperatorinator"]
PrepareStage = Callable[[Path, Path, WorkerConfig], PreparedAudio]
GenerationStage = Callable[
    [PreparedAudio, Path, ChartGenerator, int], tuple[GeneratedVariant, ...]
]
ExportStage = Callable[
    [PreparedAudio, tuple[GeneratedVariant, ...], Path, str],
    tuple[ExportedVariant, ...],
]


def _prepare_stage(source: Path, run_dir: Path, config: WorkerConfig) -> PreparedAudio:
    return run_prepare(source, run_dir, config=config)


def _generation_stage(
    prepared: PreparedAudio,
    run_dir: Path,
    generator: ChartGenerator,
    seed: int,
) -> tuple[GeneratedVariant, ...]:
    return run_generation(prepared, run_dir, generator=generator, seed=seed)


def _export_stage(
    prepared: PreparedAudio,
    generated: tuple[GeneratedVariant, ...],
    run_dir: Path,
    worker_version: str,
) -> tuple[ExportedVariant, ...]:
    return run_export(
        prepared,
        generated,
        run_dir,
        worker_version=worker_version,
    )


def _select_generator(name: GeneratorName, config: WorkerConfig) -> ChartGenerator:
    if name == "fake":
        return FakeGenerator()
    return MapperatorinatorGenerator(config)


def _utc_now() -> datetime:
    return datetime.now(UTC)


@dataclass(frozen=True, slots=True)
class PipelineDependencies:
    config: WorkerConfig = field(default_factory=load_config)
    prepare: PrepareStage = _prepare_stage
    select_generator: Callable[[GeneratorName, WorkerConfig], ChartGenerator] = _select_generator
    generation: GenerationStage = _generation_stage
    export: ExportStage = _export_stage
    now: Callable[[], datetime] = _utc_now
    new_run_id: Callable[[], UUID] = uuid4


@dataclass(frozen=True, slots=True)
class PipelineOptions:
    source: Path
    output_dir: Path
    title: str
    generator: GeneratorName = "mapperatorinator"
    seed: int = 0
    worker_version: str = "local"

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
    if run_dir.exists() and any(run_dir.iterdir()):
        raise ValueError(f"output directory is not empty: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _relative(path: Path, run_dir: Path) -> str:
    try:
        return path.resolve().relative_to(run_dir).as_posix()
    except ValueError:
        raise ValueError(f"pipeline asset is outside the output directory: {path}") from None


def _elapsed_ms(started_ns: int) -> int:
    return max(0, (perf_counter_ns() - started_ns) // 1_000_000)


def _max_gap_ms(variant: GeneratedVariant) -> int:
    times = sorted({note.time_ms for note in variant.generated.notes})
    return max((right - left for left, right in pairwise(times)), default=0)


def _generation_report(
    options: PipelineOptions,
    run_id: UUID,
    elapsed: dict[str, int],
    generated: tuple[GeneratedVariant, ...],
    exported: tuple[ExportedVariant, ...],
    run_dir: Path,
    config: WorkerConfig,
) -> dict[str, object]:
    charts = []
    for variant, result in zip(generated, exported, strict=True):
        notes = variant.generated.notes
        charts.append(
            {
                "keyMode": variant.key_mode,
                "difficulty": variant.difficulty,
                "descriptor": DESCRIPTORS[variant.difficulty][0],
                "precision": config.mapperatorinator_precision,
                "seed": variant.generated.seed,
                "requestedStar": variant.requested_star,
                "cfgScale": variant.cfg_scale,
                "attemptCount": 1,
                "rawNoteCount": len(notes),
                "finalNoteCount": len(result.document.notes),
                "holdCount": sum(note.kind == "HOLD" for note in notes),
                "firstNoteTimeMs": notes[0].time_ms if notes else None,
                "maxGapMs": _max_gap_ms(variant),
                "rawOsuPath": _relative(variant.raw_osu_path, run_dir),
                "chartPath": _relative(result.path, run_dir),
            }
        )
    return {
        "version": 1,
        "runId": str(run_id),
        "sourceName": options.source.name,
        "generator": options.generator,
        "strategy": "MAPPERATORINATOR_DIRECT",
        "timingAuthority": (
            "MAPPERATORINATOR" if options.generator == "mapperatorinator" else "FAKE"
        ),
        "noteMutationEnabled": False,
        "attemptsPerChart": 1,
        "elapsedMsByStage": elapsed,
        "warnings": [],
        "charts": charts,
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
    prepared = dependencies.prepare(
        options.source.resolve(),
        run_dir,
        dependencies.config,
    )
    elapsed["prepare"] = _elapsed_ms(started)

    started = perf_counter_ns()
    generator = dependencies.select_generator(options.generator, dependencies.config)
    generated = dependencies.generation(prepared, run_dir, generator, options.seed)
    elapsed["generation"] = _elapsed_ms(started)

    started = perf_counter_ns()
    exported = dependencies.export(
        prepared,
        generated,
        run_dir,
        options.worker_version,
    )
    elapsed["export"] = _elapsed_ms(started)

    report_path = run_dir / "generation-report.json"
    report_path.write_text(
        json.dumps(
            _generation_report(
                options,
                run_id,
                elapsed,
                generated,
                exported,
                run_dir,
                dependencies.config,
            ),
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    normalized = prepared.normalized
    manifest = PlaytestRunManifest(
        run_id=run_id,
        title=options.title,
        generated_at=dependencies.now(),
        worker_version=options.worker_version,
        audio=RunAudioRefs(
            game=AudioFileRef(
                path=_relative(normalized.path, run_dir),
                sha256=normalized.sha256,
            )
        ),
        charts=[
            RunChartRef(
                path=_relative(result.path, run_dir),
                sha256=result.sha256,
                key_mode=result.document.key_mode,
                difficulty=result.document.difficulty,
            )
            for result in exported
        ],
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
        chart_paths=tuple(result.path for result in exported),
        raw_osu_paths=tuple(variant.raw_osu_path for variant in generated),
        elapsed_ms_by_stage=elapsed,
    )
