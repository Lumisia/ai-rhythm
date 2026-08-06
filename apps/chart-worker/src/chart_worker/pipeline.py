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

from chart_worker.analysis.onset import OnsetAnalysis, analyze_canonical_audio
from chart_worker.config import WorkerConfig, load_config
from chart_worker.errors import ErrorCode, WorkerError
from chart_worker.generation.fake import FakeGenerator
from chart_worker.generation.mapperatorinator import ChartGenerator, MapperatorinatorGenerator
from chart_worker.generation.mapperatorinator_patch import CONSTRAINT_PATCH_ID
from chart_worker.generation.params import DESCRIPTORS
from chart_worker.hashing import sha256_file
from chart_worker.schema.playtest_run import (
    AudioFileRef,
    PlaytestRunManifest,
    RunAudioRefs,
    RunChartRef,
)
from chart_worker.schema.types import DIFFICULTIES, KEY_MODES
from chart_worker.stages.s1_prepare import run_prepare
from chart_worker.stages.s2_generate import MAX_VARIANT_ATTEMPTS, run_generation
from chart_worker.stages.s2_timing import run_timing_generation
from chart_worker.stages.s3_export import run_export
from chart_worker.stages.types import (
    ExportedVariant,
    GeneratedVariant,
    PreparedAudio,
    SongTimingAuthority,
)
from chart_worker.validation.quality_gate import GateAction

GeneratorName = Literal["fake", "mapperatorinator"]
PrepareStage = Callable[[Path, Path, WorkerConfig], PreparedAudio]
AnalysisStage = Callable[[Path], OnsetAnalysis]
TimingStage = Callable[
    [PreparedAudio, OnsetAnalysis, Path, ChartGenerator, int], SongTimingAuthority
]
GenerationStage = Callable[
    [PreparedAudio, SongTimingAuthority, OnsetAnalysis, Path, ChartGenerator, int],
    tuple[GeneratedVariant, ...],
]
ExportStage = Callable[
    [PreparedAudio, tuple[GeneratedVariant, ...], Path, str],
    tuple[ExportedVariant, ...],
]


def _prepare_stage(source: Path, run_dir: Path, config: WorkerConfig) -> PreparedAudio:
    return run_prepare(source, run_dir, config=config)


def _generation_stage(
    prepared: PreparedAudio,
    authority: SongTimingAuthority,
    analysis: OnsetAnalysis,
    run_dir: Path,
    generator: ChartGenerator,
    seed: int,
) -> tuple[GeneratedVariant, ...]:
    return run_generation(
        prepared,
        authority,
        analysis,
        run_dir,
        generator=generator,
        seed=seed,
    )


def _timing_stage(
    prepared: PreparedAudio,
    analysis: OnsetAnalysis,
    run_dir: Path,
    generator: ChartGenerator,
    seed: int,
) -> SongTimingAuthority:
    return run_timing_generation(
        prepared,
        analysis,
        run_dir,
        generator=generator,
        seed=seed,
    )


def _analysis_stage(path: Path) -> OnsetAnalysis:
    return analyze_canonical_audio(path)


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
    analyze: AnalysisStage = _analysis_stage
    select_generator: Callable[[GeneratorName, WorkerConfig], ChartGenerator] = _select_generator
    timing: TimingStage = _timing_stage
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


def _require_canonical_audio_hash(prepared: PreparedAudio) -> None:
    normalized = prepared.normalized
    actual_audio_sha = sha256_file(normalized.path)
    if actual_audio_sha != normalized.sha256:
        raise WorkerError(
            ErrorCode.ASSET_HASH_MISMATCH,
            "canonical game audio changed after prepare",
            context={
                "path": str(normalized.path),
                "expected": normalized.sha256,
                "actual": actual_audio_sha,
            },
        )


def _max_gap_ms(variant: GeneratedVariant) -> int:
    times = sorted({note.time_ms for note in variant.generated.notes})
    return max((right - left for left, right in pairwise(times)), default=0)


def _require_difficulty_order_reports(
    generated: tuple[GeneratedVariant, ...],
) -> dict[str, dict[str, object]]:
    expected = {(key_mode, difficulty) for key_mode in KEY_MODES for difficulty in DIFFICULTIES}
    observed = [(variant.key_mode, variant.difficulty) for variant in generated]
    context: dict[str, object] = {}

    missing_variants = sorted(expected.difference(observed))
    if missing_variants:
        context["missingVariants"] = [
            {"keyMode": key_mode, "difficulty": difficulty}
            for key_mode, difficulty in missing_variants
        ]
    duplicate_variants = sorted(
        pair for pair in set(observed) if observed.count(pair) > 1
    )
    if duplicate_variants:
        context["duplicateVariants"] = [
            {"keyMode": key_mode, "difficulty": difficulty}
            for key_mode, difficulty in duplicate_variants
        ]
    missing_order = [
        {"keyMode": variant.key_mode, "difficulty": variant.difficulty}
        for variant in generated
        if variant.difficulty_order is None
    ]
    if missing_order:
        context["missingDifficultyOrder"] = missing_order

    reports: dict[str, dict[str, object]] = {}
    for key_mode in KEY_MODES:
        reviews = [
            variant.difficulty_order
            for variant in generated
            if variant.key_mode == key_mode and variant.difficulty_order is not None
        ]
        if not reviews:
            continue
        first = reviews[0]
        if any(review != first for review in reviews[1:]):
            context.setdefault("inconsistentDifficultyOrder", []).append(key_mode)
            continue
        reports[f"{key_mode}K"] = first.to_report()

    if context:
        raise WorkerError(
            ErrorCode.CHART_VALIDATION_FAILED,
            "generation stage returned incomplete difficulty-order evidence",
            context=context,
        )
    return reports


def _generation_report(
    options: PipelineOptions,
    run_id: UUID,
    elapsed: dict[str, int],
    generated: tuple[GeneratedVariant, ...],
    exported: tuple[ExportedVariant, ...],
    run_dir: Path,
    config: WorkerConfig,
    prepared: PreparedAudio,
    authority: SongTimingAuthority,
    difficulty_order_reports: dict[str, dict[str, object]],
) -> dict[str, object]:
    charts = []
    for variant, result in zip(generated, exported, strict=True):
        notes = variant.generated.notes
        acceptance = variant.acceptance.to_report()
        quality_profile = acceptance["qualityProfile"]
        charts.append(
            {
                "keyMode": variant.key_mode,
                "difficulty": variant.difficulty,
                "descriptor": DESCRIPTORS[variant.difficulty][0],
                "precision": config.mapperatorinator_precision,
                "seed": variant.generated.seed,
                "selectedSeed": variant.selected_seed,
                "requestedStar": variant.requested_star,
                "cfgScale": variant.cfg_scale,
                "attemptCount": variant.attempt,
                "candidateCount": variant.candidate_count,
                "generationAttemptCount": variant.generation_attempt_count,
                "attemptErrors": list(variant.attempt_errors),
                "attemptEvidence": list(variant.attempt_evidence),
                "acceptanceStatus": acceptance["action"],
                "acceptanceReasons": [
                    reason
                    for decision in variant.acceptance.decisions
                    for reason in decision.reasons
                ],
                "acceptanceDecisions": acceptance["decisions"],
                "timingDiagnostics": acceptance["timing"],
                "noteGrid": acceptance["noteGrid"],
                "difficultyProfile": (
                    quality_profile["difficultyProfile"]
                    if quality_profile is not None
                    else None
                ),
                "holdProfile": (
                    quality_profile["holdProfile"]
                    if quality_profile is not None
                    else None
                ),
                "patternProfile": (
                    quality_profile["patternProfile"]
                    if quality_profile is not None
                    else None
                ),
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
        "qualityGateVersion": "quality-gate-v1",
        "publishable": True,
        "status": "PASS",
        "runId": str(run_id),
        "sourceName": options.source.name,
        "generator": options.generator,
        "strategy": "MAPPERATORINATOR_SHARED_TIMING",
        "timingAuthority": _relative(authority.reference_path, run_dir),
        "timingAuthoritySha256": authority.sha256,
        "timingGenerationMode": authority.mode,
        "timingAttemptCount": authority.attempt_count,
        "timingAuthorityTempoMetrics": (
            authority.tempo_metrics.to_report()
            if authority.tempo_metrics is not None
            else None
        ),
        "timingAuthorityReview": (
            authority.review.to_report() if authority.review is not None else None
        ),
        "noteMutationEnabled": False,
        "mapperatorinatorConstraintPatch": (
            CONSTRAINT_PATCH_ID if options.generator == "mapperatorinator" else None
        ),
        "attemptsPerChartMax": MAX_VARIANT_ATTEMPTS,
        "canonicalAudioSha256": prepared.normalized.sha256,
        "timingReviewRequired": any(
            chart["acceptanceStatus"] != "PASS" for chart in charts
        ),
        "elapsedMsByStage": elapsed,
        "warnings": [],
        "difficultyOrder": difficulty_order_reports,
        "charts": charts,
    }


def _failure_generation_report(
    options: PipelineOptions,
    run_id: UUID,
    elapsed: dict[str, int],
    run_dir: Path,
    prepared: PreparedAudio,
    authority: SongTimingAuthority | None,
    error: WorkerError,
    *,
    failure_stage: str | None = None,
) -> dict[str, object]:
    status = {
        ErrorCode.CHART_TIMING_REVIEW_REQUIRED: "REVIEW",
        ErrorCode.CHART_CANDIDATES_EXHAUSTED: "EXHAUSTED",
        ErrorCode.CHART_VALIDATION_FAILED: "FAILED",
    }[error.code]
    report: dict[str, object] = {
        "version": 1,
        "qualityGateVersion": "quality-gate-v1",
        "runId": str(run_id),
        "sourceName": options.source.name,
        "generator": options.generator,
        "strategy": "MAPPERATORINATOR_SHARED_TIMING",
        "publishable": False,
        "status": status,
        "error": {
            "code": error.code.value,
            "context": error.context,
        },
        "timingAuthority": (
            _relative(authority.reference_path, run_dir) if authority is not None else None
        ),
        "timingAuthoritySha256": authority.sha256 if authority is not None else None,
        "timingGenerationMode": authority.mode if authority is not None else None,
        "timingAttemptCount": authority.attempt_count if authority is not None else None,
        "canonicalAudioSha256": prepared.normalized.sha256,
        "elapsedMsByStage": elapsed,
        "charts": [],
    }
    if failure_stage is not None:
        report["failureStage"] = failure_stage
    return report


def _write_generation_report(path: Path, report: dict[str, object]) -> None:
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _require_publishable_acceptance(generated: tuple[GeneratedVariant, ...]) -> None:
    rejected = [
        {
            "key_mode": variant.key_mode,
            "difficulty": variant.difficulty,
            "gate_report": variant.acceptance.to_report(),
        }
        for variant in generated
        if variant.acceptance.action is GateAction.RETRY_MAP
    ]
    if not rejected:
        return
    raise WorkerError(
        ErrorCode.CHART_CANDIDATES_EXHAUSTED,
        "generation stage returned non-publishable chart candidates",
        context={"variants": rejected},
    )


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
    normalized = prepared.normalized
    _require_canonical_audio_hash(prepared)
    onsets = dependencies.analyze(normalized.path)
    elapsed["analysis"] = _elapsed_ms(started)

    generator = dependencies.select_generator(options.generator, dependencies.config)

    started = perf_counter_ns()
    try:
        authority = dependencies.timing(prepared, onsets, run_dir, generator, options.seed)
    except WorkerError as error:
        if error.code is not ErrorCode.CHART_TIMING_REVIEW_REQUIRED:
            raise
        elapsed["timing"] = _elapsed_ms(started)
        _write_generation_report(
            run_dir / "generation-report.json",
            _failure_generation_report(
                options,
                run_id,
                elapsed,
                run_dir,
                prepared,
                None,
                error,
                failure_stage="TIMING",
            ),
        )
        raise
    _require_canonical_audio_hash(prepared)
    elapsed["timing"] = _elapsed_ms(started)

    started = perf_counter_ns()
    try:
        generated = dependencies.generation(
            prepared,
            authority,
            onsets,
            run_dir,
            generator,
            options.seed,
        )
        _require_publishable_acceptance(generated)
        difficulty_order_reports = _require_difficulty_order_reports(generated)
    except WorkerError as error:
        if error.code not in {
            ErrorCode.CHART_TIMING_REVIEW_REQUIRED,
            ErrorCode.CHART_CANDIDATES_EXHAUSTED,
            ErrorCode.CHART_VALIDATION_FAILED,
        }:
            raise
        elapsed["generation"] = _elapsed_ms(started)
        _write_generation_report(
            run_dir / "generation-report.json",
            _failure_generation_report(
                options,
                run_id,
                elapsed,
                run_dir,
                prepared,
                authority,
                error,
            ),
        )
        raise
    _require_canonical_audio_hash(prepared)
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
    _write_generation_report(
        report_path,
        _generation_report(
            options,
            run_id,
            elapsed,
            generated,
            exported,
            run_dir,
            dependencies.config,
            prepared,
            authority,
            difficulty_order_reports,
        ),
    )
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
