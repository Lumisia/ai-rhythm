"""로컬 한 번 실행용 chart-worker 오케스트레이션."""

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from shutil import copyfile
from time import perf_counter_ns
from typing import Literal
from uuid import UUID, uuid4

from chart_worker.analysis.snapshot import save_analysis_snapshot
from chart_worker.analysis.timing import (
    ReferenceChart,
    ReferenceQuality,
    evaluate_reference,
    load_reference_onsets,
)
from chart_worker.config import WorkerConfig, load_config
from chart_worker.errors import ErrorCode, WorkerError
from chart_worker.generation.candidate_selection import (
    RETRY_SEED_STEP,
    CandidateParameters,
    CandidateQuality,
    needs_retry,
    select_candidate_index,
)
from chart_worker.generation.fake import FakeGenerator
from chart_worker.generation.mapperatorinator import ChartGenerator, MapperatorinatorGenerator
from chart_worker.schema.playtest_run import (
    PlaytestRunManifest,
    RunAudioRefs,
    RunChartRef,
)
from chart_worker.stages.s1_analyze import run_analysis
from chart_worker.stages.s2_generate import run_generation, run_generation_variant
from chart_worker.stages.s3_stems import run_stems
from chart_worker.stages.s4_postprocess import (
    candidate_quality_of,
    run_postprocess,
    run_postprocess_variant,
)
from chart_worker.stages.s15_select_timing import run_timing_selection
from chart_worker.stages.types import (
    AnalysisStageResult,
    GeneratedVariant,
    PostprocessedVariant,
    StemStageResult,
)

GeneratorName = Literal["fake", "mapperatorinator"]
AnalysisStage = Callable[[Path, Path, WorkerConfig], AnalysisStageResult]
TimingStage = Callable[[AnalysisStageResult, Path, WorkerConfig, bool], AnalysisStageResult]
GenerationStage = Callable[
    [AnalysisStageResult, Path, ChartGenerator, int], tuple[GeneratedVariant, ...]
]
StemStage = Callable[[AnalysisStageResult, Path, bool], StemStageResult]
PostprocessStage = Callable[
    [AnalysisStageResult, tuple[GeneratedVariant, ...], StemStageResult, Path, str],
    tuple[PostprocessedVariant, ...],
]
GenerationVariantStage = Callable[
    [AnalysisStageResult, Path, ChartGenerator, int, str, int, CandidateParameters],
    GeneratedVariant,
]
PostprocessVariantStage = Callable[
    [AnalysisStageResult, GeneratedVariant, StemStageResult, Path, str, bool],
    PostprocessedVariant,
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


def _generation_variant_stage(
    analysis: AnalysisStageResult,
    run_dir: Path,
    generator: ChartGenerator,
    key_mode: int,
    difficulty: str,
    attempt: int,
    parameters: CandidateParameters,
) -> GeneratedVariant:
    return run_generation_variant(
        analysis,
        run_dir,
        generator=generator,
        key_mode=key_mode,
        difficulty=difficulty,
        attempt=attempt,
        parameters=parameters,
    )


def _timing_stage(
    analysis: AnalysisStageResult,
    run_dir: Path,
    config: WorkerConfig,
    enable_super_timing: bool,
) -> AnalysisStageResult:
    return run_timing_selection(analysis, run_dir, config, enable_super_timing)


def _stem_stage(analysis: AnalysisStageResult, run_dir: Path, enabled: bool) -> StemStageResult:
    return run_stems(analysis, run_dir, enabled=enabled)


def _select_generator(name: GeneratorName, config: WorkerConfig) -> ChartGenerator:
    if name == "fake":
        return FakeGenerator()
    return MapperatorinatorGenerator(config)


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


def _postprocess_variant_stage(
    analysis: AnalysisStageResult,
    generated: GeneratedVariant,
    stems: StemStageResult,
    run_dir: Path,
    worker_version: str,
    write_output: bool,
) -> PostprocessedVariant:
    return run_postprocess_variant(
        analysis,
        generated,
        stems,
        run_dir,
        worker_version=worker_version,
        write_output=write_output,
    )


def _utc_now() -> datetime:
    return datetime.now(UTC)


@dataclass(frozen=True, slots=True)
class PipelineDependencies:
    config: WorkerConfig = field(default_factory=load_config)
    analysis: AnalysisStage = _analysis_stage
    timing: TimingStage = _timing_stage
    select_generator: Callable[[GeneratorName, WorkerConfig], ChartGenerator] = _select_generator
    generation: GenerationStage = _generation_stage
    generation_variant: GenerationVariantStage = _generation_variant_stage
    stems: StemStage = _stem_stage
    postprocess: PostprocessStage = _postprocess_stage
    postprocess_variant: PostprocessVariantStage = _postprocess_variant_stage
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
    reference_onsets_path: Path | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "source", Path(self.source))
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        if self.reference_onsets_path is not None:
            object.__setattr__(self, "reference_onsets_path", Path(self.reference_onsets_path))
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


@dataclass(frozen=True, slots=True)
class _CandidateEvaluation:
    variant: GeneratedVariant
    result: PostprocessedVariant
    quality: CandidateQuality
    reference_quality: ReferenceQuality | None


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


def _elapsed_ms(start_ns: int) -> int:
    return max(0, round((perf_counter_ns() - start_ns) / 1_000_000))


def _load_and_copy_references(
    source: Path | None,
    run_dir: Path,
) -> dict[tuple[int, str], ReferenceChart]:
    if source is None:
        return {}
    resolved_source = source.resolve()
    if not resolved_source.is_file():
        raise ValueError(f"reference onset file does not exist: {source}")
    references = load_reference_onsets(resolved_source)
    destination = (run_dir / "analysis" / "reference-onsets-v1.json").resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if resolved_source != destination:
        copyfile(resolved_source, destination)
    return references


def _reference_payload(quality: ReferenceQuality | None) -> dict[str, object]:
    if quality is None:
        return {"status": "UNAVAILABLE"}
    return {
        "status": "PASS" if quality.passes else "FAIL",
        "macroF1At20Ms": quality.macro_f1_20ms,
        "phaseAbsMs": quality.phase_abs_ms,
        "p95AbsMs": quality.p95_abs_ms,
    }


def _quality_payload(
    evaluation: _CandidateEvaluation,
    *,
    analysis: AnalysisStageResult,
) -> dict[str, object]:
    variant = evaluation.variant
    quality = evaluation.quality
    return {
        "attempt": variant.attempt,
        "seed": variant.generated.seed,
        "parameters": {
            "requested_star": variant.requested_star,
            "cfg_scale": variant.cfg_scale,
        },
        "timing_source": analysis.timing_candidate.source.value,
        "failure_metrics": {
            "long_gap_bars": quality.long_gap_bars,
            "rating_error": quality.rating_error,
            "removed_ratio": quality.removed_ratio,
            "drum_precision": quality.drum_precision,
            "playability_passes": quality.playability_passes,
            "hold_ratio_error": quality.hold_ratio_error,
            "reference_pass": quality.reference_pass,
            "reference_accuracy": _reference_payload(evaluation.reference_quality),
        },
        "raw_osu_path": str(variant.raw_osu_path),
        "chart_path": str(evaluation.result.path),
    }


def _write_exhausted_report(
    *,
    options: PipelineOptions,
    run_id: UUID,
    elapsed: dict[str, int],
    charts: tuple[PostprocessedVariant, ...],
    reference_quality: dict[tuple[int, str], ReferenceQuality | None],
    key_mode: int,
    difficulty: str,
    candidates: list[dict[str, object]],
    run_dir: Path,
) -> Path:
    payload = _generation_report(options, run_id, elapsed, charts, reference_quality)
    payload["failedCombination"] = {
        "keyMode": key_mode,
        "difficulty": difficulty,
        "candidates": candidates,
    }
    report_path = run_dir / "generation-report.json"
    report_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report_path


def _canonical_raw_path(variant: GeneratedVariant, run_dir: Path) -> Path:
    destination = run_dir / "raw" / (
        f"{variant.key_mode}k-{variant.difficulty.lower()}.osu"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    copyfile(variant.raw_osu_path, destination)
    return destination


def _requires_retry(
    quality: CandidateQuality,
    *,
    difficulty: str,
    generator: GeneratorName,
) -> bool:
    # FakeGenerator validates orchestration and serialization, not musical
    # quality. Its two-second fixture cannot meaningfully satisfy rating or
    # deletion gates; human labels remain a real observable contract.
    if generator == "fake":
        return quality.reference_pass is False
    return needs_retry(quality, difficulty=difficulty)


def _select_pipeline_candidate(
    evaluations: list[_CandidateEvaluation],
    *,
    difficulty: str,
    generator: GeneratorName,
) -> int:
    eligible = list(range(len(evaluations)))
    if generator == "fake":
        eligible = [
            index
            for index in eligible
            if evaluations[index].quality.reference_pass is not False
        ]
    selected_within_eligible = select_candidate_index(
        tuple(evaluations[index].quality for index in eligible),
        difficulty=difficulty,
    )
    return eligible[selected_within_eligible]


def _generation_report(
    options: PipelineOptions,
    run_id: UUID,
    elapsed: dict[str, int],
    charts: tuple[PostprocessedVariant, ...],
    reference_quality: dict[tuple[int, str], ReferenceQuality | None],
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
                "referenceAccuracy": _reference_payload(
                    reference_quality[
                        (result.document.key_mode, result.document.difficulty)
                    ]
                ),
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
    references = _load_and_copy_references(options.reference_onsets_path, run_dir)
    run_id = dependencies.new_run_id()
    elapsed: dict[str, int] = {}

    started = perf_counter_ns()
    analysis = dependencies.analysis(options.source.resolve(), run_dir, dependencies.config)
    elapsed["analysis"] = _elapsed_ms(started)

    started = perf_counter_ns()
    analysis = dependencies.timing(
        analysis,
        run_dir,
        dependencies.config,
        options.generator == "mapperatorinator",
    )
    save_analysis_snapshot(analysis, run_dir)
    elapsed["timing"] = _elapsed_ms(started)

    started = perf_counter_ns()
    generator = dependencies.select_generator(options.generator, dependencies.config)
    generated = dependencies.generation(
        analysis,
        run_dir,
        generator,
        options.seed,
    )
    elapsed["generation"] = _elapsed_ms(started)

    started = perf_counter_ns()
    stems = dependencies.stems(analysis, run_dir, options.keysounds)
    elapsed["stems"] = _elapsed_ms(started)

    started = perf_counter_ns()
    selected_charts: list[PostprocessedVariant] = []
    selected_reference_quality: dict[tuple[int, str], ReferenceQuality | None] = {}
    canonical_raw_paths: list[Path] = []
    for combination_index, initial_variant in enumerate(generated):
        key_mode = initial_variant.key_mode
        difficulty = initial_variant.difficulty
        reference = references.get((key_mode, difficulty))

        def evaluate(
            variant: GeneratedVariant,
            reference_chart: ReferenceChart | None,
        ) -> _CandidateEvaluation:
            result = dependencies.postprocess_variant(
                analysis,
                variant,
                stems,
                run_dir,
                options.worker_version,
                False,
            )
            reference_result = evaluate_reference(
                reference_chart,
                tuple(sorted({note.time_ms for note in result.document.notes})),
            )
            quality = candidate_quality_of(
                analysis,
                variant,
                result,
                stems,
                reference_pass=(
                    reference_result.passes if reference_result is not None else None
                ),
            )
            return _CandidateEvaluation(variant, result, quality, reference_result)

        evaluations = [evaluate(initial_variant, reference)]
        if _requires_retry(
            evaluations[0].quality,
            difficulty=difficulty,
            generator=options.generator,
        ):
            for attempt in (2, 3):
                retry_variant = dependencies.generation_variant(
                    analysis,
                    run_dir,
                    generator,
                    key_mode,
                    difficulty,
                    attempt,
                    CandidateParameters(
                        seed=(
                            options.seed
                            + combination_index
                            + (attempt - 1) * RETRY_SEED_STEP
                        ),
                        requested_star=initial_variant.requested_star,
                        cfg_scale=initial_variant.cfg_scale,
                    ),
                )
                evaluations.append(evaluate(retry_variant, reference))

        if all(
            _requires_retry(
                evaluation.quality,
                difficulty=difficulty,
                generator=options.generator,
            )
            for evaluation in evaluations
        ):
            candidate_payloads = [
                _quality_payload(evaluation, analysis=analysis)
                for evaluation in evaluations
            ]
            elapsed["postprocess"] = _elapsed_ms(started)
            _write_exhausted_report(
                options=options,
                run_id=run_id,
                elapsed=elapsed,
                charts=tuple(selected_charts),
                reference_quality=selected_reference_quality,
                key_mode=key_mode,
                difficulty=difficulty,
                candidates=candidate_payloads,
                run_dir=run_dir,
            )
            raise WorkerError(
                ErrorCode.CHART_CANDIDATES_EXHAUSTED,
                "all three chart candidates failed quality gates",
                context={
                    "key_mode": key_mode,
                    "difficulty": difficulty,
                    "candidates": candidate_payloads,
                },
            )

        selected_index = _select_pipeline_candidate(
            evaluations,
            difficulty=difficulty,
            generator=options.generator,
        )
        selected = evaluations[selected_index]
        canonical_chart = dependencies.postprocess_variant(
            analysis,
            selected.variant,
            stems,
            run_dir,
            options.worker_version,
            True,
        )
        selected_charts.append(canonical_chart)
        selected_reference_quality[(key_mode, difficulty)] = selected.reference_quality
        canonical_raw_paths.append(_canonical_raw_path(selected.variant, run_dir))

    charts = tuple(selected_charts)
    reference_quality = selected_reference_quality
    elapsed["postprocess"] = _elapsed_ms(started)

    report_path = run_dir / "generation-report.json"
    report_path.write_text(
        json.dumps(
            _generation_report(options, run_id, elapsed, charts, reference_quality),
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
        raw_osu_paths=tuple(canonical_raw_paths),
        elapsed_ms_by_stage=elapsed,
    )
