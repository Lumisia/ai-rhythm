"""로컬 한 번 실행용 chart-worker 오케스트레이션."""

import json
import math
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
    TimingCandidate,
    TimingSource,
    TimingStatus,
    evaluate_reference,
    load_reference_onsets,
)
from chart_worker.config import WorkerConfig, load_config
from chart_worker.errors import ErrorCode, WorkerError
from chart_worker.generation.candidate_selection import (
    MAX_LONG_GAP_BARS,
    MAX_PLAYABILITY_PASSES,
    MAX_RATING_ERROR,
    MAX_REMOVED_RATIO,
    MIN_DRUM_PRECISION,
    CandidateParameters,
    CandidateQuality,
    candidate_parameters,
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
from chart_worker.schema.types import TARGET_HOLD_RATIO
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


@dataclass(frozen=True, slots=True)
class _ChartSelection:
    result: PostprocessedVariant
    evaluations: tuple[_CandidateEvaluation, ...]
    selected_index: int
    canonical_raw_path: Path


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


def _report_asset_path(path: Path, run_dir: Path) -> str:
    relative = _relative(path, run_dir)
    if not path.resolve().is_file():
        raise ValueError(f"pipeline report asset does not exist: {path}")
    return relative


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


_TIMING_CANDIDATE_FIELDS = {
    "source",
    "points",
    "projectedBeatMs",
    "f1At20Ms",
    "f1At50Ms",
    "p95AbsMs",
    "status",
    "reasons",
}
_TIMING_POINT_FIELDS = {"timeMs", "bpm", "meter", "startBeatIndex"}


def _timing_candidate_payload(candidate: TimingCandidate) -> dict[str, object]:
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


def _require_int(value: object, *, name: str, minimum: int | None = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")  # noqa: TRY004
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def _require_finite_number(
    value: object,
    *,
    name: str,
    minimum: float = 0.0,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{name} must be a finite number")  # noqa: TRY004
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"{name} must be a finite number")
    if converted < minimum or (maximum is not None and converted > maximum):
        raise ValueError(f"{name} is outside the supported range")
    return converted


def _validate_timing_candidate(value: object, *, name: str) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{name} must be an object")
    if set(value) != _TIMING_CANDIDATE_FIELDS:
        raise ValueError(f"{name} candidate fields do not match the version-1 contract")
    if not isinstance(value["source"], str) or value["source"] not in {
        source.value for source in TimingSource
    }:
        raise ValueError(f"{name} source is unsupported")
    if not isinstance(value["status"], str) or value["status"] not in {
        status.value for status in TimingStatus
    }:
        raise ValueError(f"{name} status is unsupported")

    points = value["points"]
    if not isinstance(points, list):
        raise ValueError(f"{name} points must be an array")  # noqa: TRY004
    for index, point in enumerate(points):
        point_name = f"{name}.points[{index}]"
        if not isinstance(point, dict) or set(point) != _TIMING_POINT_FIELDS:
            raise ValueError(f"{point_name} fields do not match the version-1 contract")
        _require_int(point["timeMs"], name=f"{point_name}.timeMs", minimum=None)
        _require_finite_number(
            point["bpm"],
            name=f"{point_name}.bpm",
            minimum=float.fromhex("0x0.0000000000001p-1022"),
        )
        _require_int(point["meter"], name=f"{point_name}.meter", minimum=1)
        start_beat_index = point["startBeatIndex"]
        if start_beat_index is not None:
            _require_int(start_beat_index, name=f"{point_name}.startBeatIndex")

    projected = value["projectedBeatMs"]
    if not isinstance(projected, list):
        raise ValueError(f"{name}.projectedBeatMs must be an array")  # noqa: TRY004
    projected_values = [
        _require_int(
            item,
            name=f"{name}.projectedBeatMs[{index}]",
            minimum=None,
        )
        for index, item in enumerate(projected)
    ]
    if projected_values != sorted(set(projected_values)):
        raise ValueError(f"{name}.projectedBeatMs must be sorted without duplicates")

    _require_finite_number(
        value["f1At20Ms"], name=f"{name}.f1At20Ms", maximum=1.0
    )
    _require_finite_number(
        value["f1At50Ms"], name=f"{name}.f1At50Ms", maximum=1.0
    )
    _require_finite_number(value["p95AbsMs"], name=f"{name}.p95AbsMs")
    reasons = value["reasons"]
    if not isinstance(reasons, list) or not all(isinstance(reason, str) for reason in reasons):
        raise ValueError(f"{name}.reasons must be an array of strings")
    return value


def _validate_timing_warning(value: object, *, index: int) -> dict[str, object]:
    name = f"warnings[{index}]"
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")  # noqa: TRY004
    if not isinstance(value.get("code"), str) or not value["code"]:
        raise ValueError(f"{name} code must be a non-empty string")
    if not isinstance(value.get("message"), str):
        raise ValueError(f"{name} message must be a string")  # noqa: TRY004
    context = value.get("context")
    if not isinstance(context, dict) or not all(isinstance(key, str) for key in context):
        raise ValueError(f"{name} warning context must be an object")
    return value


def _timing_payload(analysis: AnalysisStageResult, run_dir: Path) -> dict[str, object]:
    quality_report_path = _report_asset_path(analysis.timing_quality_report_path, run_dir)
    value = json.loads(analysis.timing_quality_report_path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(  # noqa: TRY004 - persisted contract validation
            "timing quality report must be an object"
        )
    if set(value) != {"version", "selected", "warnings", "candidates"}:
        raise ValueError("timing quality report fields do not match the version-1 contract")
    if _require_int(value["version"], name="timing quality report version", minimum=1) != 1:
        raise ValueError("unsupported timing quality report version")
    selected = _validate_timing_candidate(value["selected"], name="selected")
    candidates = value["candidates"]
    warnings = value["warnings"]
    if not isinstance(candidates, list):
        raise ValueError(  # noqa: TRY004 - persisted contract validation
            "timing quality report candidates must be an array"
        )
    if not candidates:
        raise ValueError("timing quality report candidates must not be empty")
    if not isinstance(warnings, list):
        raise ValueError(  # noqa: TRY004 - persisted contract validation
            "timing quality report warnings must be an array"
        )
    validated_candidates = [
        _validate_timing_candidate(candidate, name=f"candidates[{index}]")
        for index, candidate in enumerate(candidates)
    ]
    validated_warnings = [
        _validate_timing_warning(warning, index=index)
        for index, warning in enumerate(warnings)
    ]
    expected_selected = _timing_candidate_payload(analysis.timing_candidate)
    if selected != expected_selected:
        raise ValueError("timing quality report selected candidate does not match analysis")
    if selected not in validated_candidates:
        raise ValueError("timing quality report selected candidate is absent from candidates")
    return {
        "qualityReportPath": quality_report_path,
        "selectedSource": selected["source"],
        "selected": selected,
        "candidates": validated_candidates,
        "warnings": validated_warnings,
    }


def _drum_payload(precision: float | None) -> dict[str, object]:
    if precision is None:
        return {"status": "UNAVAILABLE"}
    return {"status": "AVAILABLE", "precision": precision}


def _candidate_failure_reasons(
    quality: CandidateQuality,
    *,
    difficulty: str,
    generator: GeneratorName,
) -> list[str]:
    reasons = []
    if generator == "fake":
        if quality.reference_pass is False:
            reasons.append("human reference accuracy gate failed")
        return reasons
    if quality.long_gap_bars > MAX_LONG_GAP_BARS:
        reasons.append("long gap exceeds two bars")
    if abs(quality.rating_error) >= MAX_RATING_ERROR:
        reasons.append("rating error is at least 0.35")
    if quality.removed_ratio > MAX_REMOVED_RATIO:
        reasons.append("total removal ratio exceeds 0.45")
    if quality.playability_passes >= MAX_PLAYABILITY_PASSES:
        reasons.append("playability recovery reached eight passes")
    if quality.reference_pass is False:
        reasons.append("human reference accuracy gate failed")
    if (
        difficulty in ("HARD", "EXPERT")
        and quality.drum_precision is not None
        and quality.drum_precision < MIN_DRUM_PRECISION
    ):
        reasons.append("drum onset precision is below 0.70")
    return reasons


def _hold_ratio(result: PostprocessedVariant) -> float:
    notes = result.document.notes
    if not notes:
        return 0.0
    return sum(note.kind == "HOLD" for note in notes) / len(notes)


def _quality_payload(
    evaluation: _CandidateEvaluation,
    *,
    analysis: AnalysisStageResult,
    run_dir: Path,
    selection_status: str,
    selection_reasons: list[str],
) -> dict[str, object]:
    variant = evaluation.variant
    quality = evaluation.quality
    reports = evaluation.result.reports
    raw_note_count = len(variant.generated.notes)
    final_note_count = len(evaluation.result.document.notes)
    lane_deleted = reports.conversion.deleted_count
    difficulty_deleted = reports.difficulty.removed_count
    playability_deleted = reports.playability.deleted_count
    total_deleted = lane_deleted + difficulty_deleted + playability_deleted
    total_removal_ratio = total_deleted / raw_note_count if raw_note_count else 1.0
    target_rating = reports.difficulty.target_rating
    actual_rating = evaluation.result.document.metrics.project_rating
    signed_rating_error = actual_rating - target_rating
    actual_hold_ratio = _hold_ratio(evaluation.result)
    target_hold_ratio = TARGET_HOLD_RATIO[variant.difficulty]
    raw_osu_path = _report_asset_path(variant.raw_osu_path, run_dir)
    chart_path = _report_asset_path(evaluation.result.path, run_dir)
    return {
        "attempt": variant.attempt,
        "seed": variant.generated.seed,
        "parameters": {
            "requested_star": variant.requested_star,
            "cfg_scale": variant.cfg_scale,
        },
        "generationParameters": {
            "requestedStar": variant.requested_star,
            "cfgScale": variant.cfg_scale,
        },
        "selection": {
            "status": selection_status,
            "reasons": selection_reasons,
        },
        "rawNoteCount": raw_note_count,
        "finalNoteCount": final_note_count,
        "removals": {
            "laneConversion": lane_deleted,
            "difficultySolver": difficulty_deleted,
            "playability": playability_deleted,
        },
        "totalRemovalRatio": total_removal_ratio,
        "playabilityPasses": reports.playability.passes,
        "targetRating": target_rating,
        "actualRating": actual_rating,
        "signedRatingError": signed_rating_error,
        "absoluteRatingError": abs(signed_rating_error),
        "drumOnsetPrecision": _drum_payload(quality.drum_precision),
        "longGapBars": quality.long_gap_bars,
        "holdRatio": {
            "actual": actual_hold_ratio,
            "target": target_hold_ratio,
            "absoluteError": abs(actual_hold_ratio - target_hold_ratio),
        },
        "referenceAccuracy": _reference_payload(evaluation.reference_quality),
        "diagnosticRawOsuPath": raw_osu_path,
        "diagnosticChartPath": chart_path,
        "timing_source": analysis.timing_candidate.source.value,
        "failure_metrics": {
            "long_gap_bars": quality.long_gap_bars,
            "rating_error": quality.rating_error,
            "removed_ratio": quality.removed_ratio,
            "drum_precision": _drum_payload(quality.drum_precision),
            "playability_passes": quality.playability_passes,
            "hold_ratio_error": quality.hold_ratio_error,
            "reference_pass": quality.reference_pass,
            "reference_accuracy": _reference_payload(evaluation.reference_quality),
        },
        "raw_osu_path": raw_osu_path,
        "chart_path": chart_path,
    }


def _selection_payloads(
    selection: _ChartSelection,
    *,
    analysis: AnalysisStageResult,
    run_dir: Path,
    generator: GeneratorName,
) -> list[dict[str, object]]:
    selected_attempt = selection.evaluations[selection.selected_index].variant.attempt
    payloads = []
    for index, evaluation in enumerate(selection.evaluations):
        failure_reasons = _candidate_failure_reasons(
            evaluation.quality,
            difficulty=evaluation.variant.difficulty,
            generator=generator,
        )
        if index == selection.selected_index:
            reasons = [
                *failure_reasons,
                "selected by the approved lexicographic quality ranking",
            ]
            status = "SELECTED"
        else:
            reasons = [
                *failure_reasons,
                f"ranked below selected attempt {selected_attempt}",
            ]
            status = "REJECTED"
        payloads.append(
            _quality_payload(
                evaluation,
                analysis=analysis,
                run_dir=run_dir,
                selection_status=status,
                selection_reasons=reasons,
            )
        )
        if index == selection.selected_index:
            payloads[-1]["canonicalChartPath"] = _report_asset_path(
                selection.result.path,
                run_dir,
            )
            payloads[-1]["canonicalRawOsuPath"] = _report_asset_path(
                selection.canonical_raw_path,
                run_dir,
            )
    return payloads


def _write_exhausted_report(
    *,
    options: PipelineOptions,
    run_id: UUID,
    elapsed: dict[str, int],
    selections: tuple[_ChartSelection, ...],
    analysis: AnalysisStageResult,
    key_mode: int,
    difficulty: str,
    candidates: list[dict[str, object]],
    run_dir: Path,
) -> Path:
    payload = _generation_report(
        options,
        run_id,
        elapsed,
        selections,
        analysis=analysis,
        run_dir=run_dir,
    )
    payload["warnings"] = [
        f"{key_mode}K {difficulty}: all chart candidates failed quality gates"
    ]
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
    selections: tuple[_ChartSelection, ...],
    *,
    analysis: AnalysisStageResult,
    run_dir: Path,
) -> dict[str, object]:
    charts = []
    for selection in selections:
        result = selection.result
        selected = selection.evaluations[selection.selected_index]
        candidates = _selection_payloads(
            selection,
            analysis=analysis,
            run_dir=run_dir,
            generator=options.generator,
        )
        selected_payload = candidates[selection.selected_index]
        charts.append(
            {
                "keyMode": result.document.key_mode,
                "difficulty": result.document.difficulty,
                "candidateCount": len(candidates),
                "selectedAttempt": selected.variant.attempt,
                "selectedSeed": selected.variant.generated.seed,
                "selectedParameters": selected_payload["generationParameters"],
                "rawNoteCount": selected_payload["rawNoteCount"],
                "finalNoteCount": selected_payload["finalNoteCount"],
                "removals": selected_payload["removals"],
                "totalRemovalRatio": selected_payload["totalRemovalRatio"],
                "playabilityPasses": selected_payload["playabilityPasses"],
                "targetRating": selected_payload["targetRating"],
                "actualRating": selected_payload["actualRating"],
                "signedRatingError": selected_payload["signedRatingError"],
                "absoluteRatingError": selected_payload["absoluteRatingError"],
                "reachedTarget": result.reports.difficulty.reached_target,
                "removedCount": result.reports.difficulty.removed_count,
                "playabilityRecoveredCount": result.reports.playability.recovered_count,
                "playabilityDeletedCount": result.reports.playability.deleted_count,
                "drumOnsetPrecision": selected_payload["drumOnsetPrecision"],
                "longGapBars": selected_payload["longGapBars"],
                "holdRatio": selected_payload["holdRatio"],
                "referenceAccuracy": selected_payload["referenceAccuracy"],
                "chartPath": _report_asset_path(result.path, run_dir),
                "rawOsuPath": _report_asset_path(selection.canonical_raw_path, run_dir),
                "candidates": candidates,
            }
        )
    return {
        "runId": str(run_id),
        "sourceName": options.source.name,
        "generator": options.generator,
        "keysounds": options.keysounds,
        "elapsedMsByStage": elapsed,
        "timing": _timing_payload(analysis, run_dir),
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
    selections: list[_ChartSelection] = []
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
        initial_requires_retry = _requires_retry(
            evaluations[0].quality,
            difficulty=difficulty,
            generator=options.generator,
        )
        retry_attempts: tuple[int, ...] = ()
        if options.generator == "mapperatorinator":
            retry_attempts = (2, 3) if initial_requires_retry else (2,)
        elif initial_requires_retry:
            retry_attempts = (2, 3)

        for attempt in retry_attempts:
            retry_variant = dependencies.generation_variant(
                analysis,
                run_dir,
                generator,
                key_mode,
                difficulty,
                attempt,
                candidate_parameters(
                    options.seed,
                    combination_index,
                    attempt,
                    evaluations[-1].quality,
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
                _quality_payload(
                    evaluation,
                    analysis=analysis,
                    run_dir=run_dir,
                    selection_status="REJECTED",
                    selection_reasons=[
                        *_candidate_failure_reasons(
                            evaluation.quality,
                            difficulty=evaluation.variant.difficulty,
                            generator=options.generator,
                        ),
                        "all candidates failed quality gates",
                    ],
                )
                for evaluation in evaluations
            ]
            elapsed["postprocess"] = _elapsed_ms(started)
            _write_exhausted_report(
                options=options,
                run_id=run_id,
                elapsed=elapsed,
                selections=tuple(selections),
                analysis=analysis,
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
        canonical_raw_path = _canonical_raw_path(selected.variant, run_dir)
        canonical_raw_paths.append(canonical_raw_path)
        selections.append(
            _ChartSelection(
                result=canonical_chart,
                evaluations=tuple(evaluations),
                selected_index=selected_index,
                canonical_raw_path=canonical_raw_path,
            )
        )

    charts = tuple(selected_charts)
    elapsed["postprocess"] = _elapsed_ms(started)

    report_path = run_dir / "generation-report.json"
    report_path.write_text(
        json.dumps(
            _generation_report(
                options,
                run_id,
                elapsed,
                tuple(selections),
                analysis=analysis,
                run_dir=run_dir,
            ),
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
