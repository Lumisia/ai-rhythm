"""Direct Mapperatorinator chart generation pipeline."""

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path
from time import perf_counter_ns
from typing import Literal
from uuid import UUID, uuid4

from chart_worker.analysis.activity import (
    evaluate_boundary_policy,
    observe_outro,
)
from chart_worker.analysis.hold_lane_state import analyze_hold_lane_state
from chart_worker.analysis.onset import OnsetAnalysis, analyze_canonical_audio
from chart_worker.analysis.outro_evidence import build_outro_evidence_profile
from chart_worker.analysis.runtime_fingerprint import build_runtime_fingerprint
from chart_worker.config import WorkerConfig, load_config
from chart_worker.errors import ErrorCode, WorkerError
from chart_worker.generation.attempt_journal import build_attempt_journal_projection
from chart_worker.generation.diagnostic_fallback import (
    DIAGNOSTIC_FALLBACK_VERSION,
    DiagnosticFallbackExport,
    DiagnosticFallbackIdentity,
    export_diagnostic_fallback,
)
from chart_worker.generation.fake import FakeGenerator
from chart_worker.generation.generation_control import (
    MAX_CRASH_ATTEMPTS,
    MAX_TOTAL_ATTEMPTS,
    MAX_VARIANT_ATTEMPTS,
)
from chart_worker.generation.inference_session import (
    InferenceSession,
    ResidentProcessSession,
    SessionState,
    SongIdentity,
    SubprocessResidentTransport,
)
from chart_worker.generation.mapperatorinator import (
    ChartGenerator,
    GeneratedChart,
    MapperatorinatorGenerator,
    inference_env,
)
from chart_worker.generation.mapperatorinator_patch import (
    CONSTRAINT_PATCH_ID,
    EXPECTED_MAPPERATORINATOR_HEAD,
)
from chart_worker.generation.osu_parser import parse_osu_mania
from chart_worker.generation.params import DESCRIPTORS
from chart_worker.hashing import sha256_file
from chart_worker.schema.playtest_run import (
    AudioFileRef,
    CoverageSummary,
    MissingChartRef,
    OutcomeStatusSnapshot,
    PlaytestRunManifestV2,
    PublicationDecisionSnapshot,
    ReportFileRef,
    RunAudioRefs,
    RunChartRefV2,
)
from chart_worker.schema.types import DIFFICULTIES, KEY_MODES
from chart_worker.stages.authority_epoch import AuthorityEpochRecord
from chart_worker.stages.s1_prepare import run_prepare
from chart_worker.stages.s2_generate import (
    run_generation,
)
from chart_worker.stages.s2_timing import run_timing_generation
from chart_worker.stages.s3_export import run_export
from chart_worker.stages.timing_feedback import RetryTimingSignal
from chart_worker.stages.types import (
    ExportedVariant,
    GeneratedVariant,
    GenerationOutcome,
    MissingVariant,
    PreparedAudio,
    SongTimingAuthority,
)
from chart_worker.validation.difficulty_selector import DifficultySelectionComparison
from chart_worker.validation.final_difficulty_family import (
    DifficultyFamilyEntry,
    observe_final_difficulty_family,
)
from chart_worker.validation.generated_chart import validate_generated_chart
from chart_worker.validation.intro_phrase_family import IntroPhraseFamilyReview
from chart_worker.validation.intro_start_contract import (
    IntroContractReview,
    IntroStartContract,
)
from chart_worker.validation.outcome_status import (
    failure_outcome_status,
    success_outcome_status,
)
from chart_worker.validation.outro_family_review import OutroFamilyReview
from chart_worker.validation.publication_policy import (
    BoundaryPublicationAssessment,
    assess_boundary_publication,
    decide_publication,
)
from chart_worker.validation.quality_gate import QUALITY_GATE_VERSION, GateAction, GateAxis
from chart_worker.validation.song_family_selector import SongSelectionComparison
from chart_worker.validation.timing_authority import validate_timing_identity
from chart_worker.validation.timing_family_review import TimingFamilyReview
from chart_worker.validation.timing_review import TimingAuthorityAction

GeneratorName = Literal["fake", "mapperatorinator"]
_PLAYTEST_ONLY_PROVENANCES = frozenset(
    {"COVERAGE_REPAIR", "RAW_UNVERIFIED", "SAFE_FALLBACK"}
)
PrepareStage = Callable[[Path, Path, WorkerConfig], PreparedAudio]
AnalysisStage = Callable[[Path], OnsetAnalysis]
TimingStage = Callable[
    [PreparedAudio, OnsetAnalysis, Path, ChartGenerator, int], SongTimingAuthority
]
SuperTimingStage = Callable[
    [PreparedAudio, OnsetAnalysis, Path, ChartGenerator, int, int],
    SongTimingAuthority,
]
GenerationStage = Callable[
    [PreparedAudio, SongTimingAuthority, OnsetAnalysis, Path, ChartGenerator, int, int],
    GenerationOutcome,
]
ExportStage = Callable[
    [PreparedAudio, tuple[GeneratedVariant, ...], Path, str],
    tuple[ExportedVariant, ...],
]
OpenInferenceSession = Callable[[WorkerConfig, Path], InferenceSession]
BindInferenceSession = Callable[[ChartGenerator, InferenceSession], ChartGenerator]
_REPORTABLE_INFERENCE_FAILURES = frozenset(
    {
        ErrorCode.INFERENCE_PROTOCOL_FAILED,
        ErrorCode.INFERENCE_START_FAILED,
        ErrorCode.INFERENCE_COMPLETION_UNKNOWN,
        ErrorCode.INFERENCE_INVOCATION_CONFLICT,
    }
)


def _prepare_stage(source: Path, run_dir: Path, config: WorkerConfig) -> PreparedAudio:
    prepared = run_prepare(source, run_dir, config=config)
    return replace(
        prepared,
        difficulty_selector_mode=config.difficulty_selector_mode,
        boundary_policy_mode=config.boundary_policy_mode,
        beat_this_enabled=config.beat_this_enabled,
        beat_this_checkpoint=config.beat_this_checkpoint,
        beat_this_device=config.beat_this_device,
        beat_this_float16=config.beat_this_float16,
    )


def _generation_stage(
    prepared: PreparedAudio,
    authority: SongTimingAuthority,
    analysis: OnsetAnalysis,
    run_dir: Path,
    generator: ChartGenerator,
    seed: int,
    authority_epoch: int,
) -> GenerationOutcome:
    return run_generation(
        prepared,
        authority,
        analysis,
        run_dir,
        generator=generator,
        seed=seed,
        authority_epoch=authority_epoch,
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
        authority_epoch=1,
    )


def _super_timing_stage(
    prepared: PreparedAudio,
    analysis: OnsetAnalysis,
    run_dir: Path,
    generator: ChartGenerator,
    seed: int,
    authority_epoch: int,
) -> SongTimingAuthority:
    return run_timing_generation(
        prepared,
        analysis,
        run_dir,
        generator=generator,
        seed=seed,
        force_super=True,
        authority_epoch=authority_epoch,
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


def _open_inference_session(
    config: WorkerConfig,
    run_dir: Path,
    *,
    stderr_path: Path | None = None,
) -> InferenceSession:
    if config.mapperatorinator_backend != "song_session":
        raise ValueError("resident session factory requires song_session backend")
    if (
        config.mapperatorinator_home is None
        or config.mapperatorinator_python is None
        or config.mapperatorinator_model_root is None
        or config.mapperatorinator_model_revision is None
    ):
        raise ValueError("song_session configuration is incomplete")
    home = config.mapperatorinator_home.resolve(strict=True)
    python = config.mapperatorinator_python.resolve(strict=True)
    worker_source = home / "osuT5" / "osuT5" / "inference" / "resident_worker.py"
    if not worker_source.is_file():
        raise ValueError("patched Mapperatorinator resident_worker.py is missing")
    command = [
        str(python),
        "-u",
        "-m",
        "osuT5.osuT5.inference.resident_worker",
        "--home",
        str(home),
        "--job-root",
        str(run_dir),
        "--model-root",
        str(config.mapperatorinator_model_root),
        "--model-revision",
        config.mapperatorinator_model_revision,
        "--upstream-commit",
        EXPECTED_MAPPERATORINATOR_HEAD,
        "--patch-set-id",
        CONSTRAINT_PATCH_ID,
    ]
    transport = SubprocessResidentTransport(
        command=command,
        cwd=home,
        job_root=run_dir,
        stderr_path=stderr_path or run_dir / "resident-stderr.log",
        env=inference_env(),
        startup_timeout_sec=config.mapperatorinator_resident_startup_timeout_sec,
        invocation_timeout_sec=config.mapperatorinator_resident_invocation_timeout_sec,
        shutdown_timeout_sec=config.mapperatorinator_resident_close_timeout_sec,
        max_line_bytes=config.mapperatorinator_protocol_max_line_bytes,
    )
    return ResidentProcessSession(
        transport=transport,
        close_timeout_sec=config.mapperatorinator_resident_close_timeout_sec,
    )


def _bind_inference_session(
    generator: ChartGenerator,
    session: InferenceSession,
) -> ChartGenerator:
    if not isinstance(generator, MapperatorinatorGenerator):
        raise TypeError("song_session requires MapperatorinatorGenerator")
    return replace(generator, session=session)


def _song_config_digest(config: WorkerConfig) -> str:
    payload = {
        "backend": config.mapperatorinator_backend,
        "checkpointIntervalWindows": config.mapperatorinator_checkpoint_interval_windows,
        "holdStateMode": config.mapperatorinator_hold_state_mode,
        "modelRevision": config.mapperatorinator_model_revision,
        "modelRoot": str(config.mapperatorinator_model_root),
        "patchSetId": CONSTRAINT_PATCH_ID,
        "precision": config.mapperatorinator_precision,
        "tailRepairs": config.mapperatorinator_tail_repairs,
        "upstreamCommit": EXPECTED_MAPPERATORINATOR_HEAD,
    }
    canonical = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _utc_now() -> datetime:
    return datetime.now(UTC)


@dataclass(frozen=True, slots=True)
class PipelineDependencies:
    config: WorkerConfig = field(default_factory=load_config)
    prepare: PrepareStage = _prepare_stage
    analyze: AnalysisStage = _analysis_stage
    select_generator: Callable[[GeneratorName, WorkerConfig], ChartGenerator] = _select_generator
    open_inference_session: OpenInferenceSession = _open_inference_session
    bind_inference_session: BindInferenceSession = _bind_inference_session
    attached_inference_session: InferenceSession | None = None
    timing: TimingStage = _timing_stage
    super_timing: SuperTimingStage = _super_timing_stage
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


def _timing_authority_report(
    authority: SongTimingAuthority | None,
    run_dir: Path,
) -> dict[str, object]:
    if authority is None:
        return {
            "timingAuthority": None,
            "timingAuthoritySha256": None,
            "timingGenerationMode": None,
            "timingAttemptCount": None,
            "timingAuthorityTempoMetrics": None,
            "timingAuthorityReview": None,
            "timingAuthorityLeadingCoverage": None,
            "timingAuthorityLocalReview": None,
            "timingAuthorityRecoveryPreflight": None,
            "timingAuthoritySelection": None,
            "timingAuthorityIntegrity": None,
        }
    return {
        "timingAuthority": _relative(authority.reference_path, run_dir),
        "timingAuthoritySha256": authority.sha256,
        "timingGenerationMode": authority.mode,
        "timingAttemptCount": authority.attempt_count,
        "timingAuthorityTempoMetrics": (
            authority.tempo_metrics.to_report() if authority.tempo_metrics is not None else None
        ),
        "timingAuthorityReview": (
            authority.review.to_report() if authority.review is not None else None
        ),
        "timingAuthorityLeadingCoverage": (
            authority.leading_coverage.to_report()
            if authority.leading_coverage is not None
            else None
        ),
        "timingAuthorityLocalReview": (
            authority.local_review.to_report() if authority.local_review is not None else None
        ),
        "timingAuthorityRecoveryPreflight": (
            authority.recovery_preflight.to_report()
            if authority.recovery_preflight is not None
            else None
        ),
        "timingAuthoritySelection": (
            authority.candidate_selection.to_report()
            if authority.candidate_selection is not None
            else None
        ),
        "timingAuthorityIntegrity": (
            authority.timing_integrity.to_report()
            if authority.timing_integrity is not None
            else None
        ),
    }


def _authority_epoch_report(
    records: list[AuthorityEpochRecord],
    escalations: list[dict[str, object]],
    selected_epoch: int | None,
) -> dict[str, object]:
    return {
        "selectedAuthorityEpoch": selected_epoch,
        "timingCandidates": [record.to_report() for record in records],
        "mapTimingEscalations": escalations,
    }


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


def _coverage_summary(variant: GeneratedVariant) -> CoverageSummary:
    attack_gaps = tuple(
        gap
        for gap in variant.acceptance.timing.coverage_gaps
        if gap.opportunity is None
        or gap.opportunity.kind.value == "ATTACK_REQUIRED"
    )
    return CoverageSummary(
        first_note_time_ms=variant.acceptance.timing.first_note_time_ms,
        max_gap_ms=variant.acceptance.timing.max_gap_ms,
        attack_required_gap_count=len(attack_gaps),
        attack_required_gap_total_ms=sum(
            gap.end_ms - gap.start_ms for gap in attack_gaps
        ),
        repaired_gap_count=variant.coverage_repair_gap_count,
    )


def _playability_tier(variant: GeneratedVariant) -> str:
    if variant.provenance not in _PLAYTEST_ONLY_PROVENANCES:
        return "MODEL_PLAYABLE"
    if variant.provenance == "RAW_UNVERIFIED":
        return "DIAGNOSTIC_ONLY"
    if variant.acceptance.action is GateAction.RETRY_MAP:
        return "DIAGNOSTIC_ONLY"
    return "RECOVERY_PLAYABLE"


def _require_difficulty_order_reports(
    generated: tuple[GeneratedVariant, ...],
    missing: tuple[MissingVariant, ...] = (),
) -> dict[str, dict[str, object]]:
    expected = {(key_mode, difficulty) for key_mode in KEY_MODES for difficulty in DIFFICULTIES}
    declared_missing = {(entry.key_mode, entry.difficulty) for entry in missing}
    observed = [(variant.key_mode, variant.difficulty) for variant in generated]
    context: dict[str, object] = {}

    overlap = sorted(declared_missing.intersection(observed))
    if overlap:
        context["missingButGenerated"] = [
            {"keyMode": key_mode, "difficulty": difficulty} for key_mode, difficulty in overlap
        ]
    missing_variants = sorted(expected.difference(observed).difference(declared_missing))
    if missing_variants:
        context["missingVariants"] = [
            {"keyMode": key_mode, "difficulty": difficulty}
            for key_mode, difficulty in missing_variants
        ]
    duplicate_variants = sorted(pair for pair in set(observed) if observed.count(pair) > 1)
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
        report = first.to_report()
        report["finalFamilyObservation"] = observe_final_difficulty_family(
            key_mode,
            tuple(
                DifficultyFamilyEntry(
                    difficulty=variant.difficulty,
                    provenance=variant.provenance,
                    project_rating=(
                        variant.acceptance.profile.difficulty.project_rating
                        if variant.acceptance.profile is not None
                        else None
                    ),
                    ordering_score=(
                        variant.acceptance.profile.difficulty_vector_v2.ordering_score
                        if variant.acceptance.profile is not None
                        else None
                    ),
                )
                for variant in generated
                if variant.key_mode == key_mode
            ),
        ).to_report()
        reports[f"{key_mode}K"] = report

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
    analysis: OnsetAnalysis,
    authority: SongTimingAuthority,
    difficulty_order_reports: dict[str, dict[str, object]],
    authority_epochs: list[AuthorityEpochRecord],
    map_timing_escalations: list[dict[str, object]],
    selected_authority_epoch: int,
    boundary_publication_assessment: BoundaryPublicationAssessment,
    difficulty_selection_shadows: tuple[DifficultySelectionComparison, ...] = (),
    song_selection_shadow: SongSelectionComparison | None = None,
    intro_start_contract: IntroStartContract | None = None,
    intro_contract_review: IntroContractReview | None = None,
    intro_phrase_family_reviews: tuple[IntroPhraseFamilyReview, ...] = (),
    timing_family_reviews: tuple[TimingFamilyReview, ...] = (),
    outro_family_review: OutroFamilyReview | None = None,
    additional_inference_calls: int = 0,
    additional_inference_work_ms: int = 0,
    additional_inference_work_limit_ms: int = 0,
    missing: tuple[MissingVariant, ...] = (),
    music_bounds: dict[str, object] | None = None,
    diagnostic_fallbacks: tuple[DiagnosticFallbackExport, ...] = (),
    diagnostic_fallback_failures: tuple[dict[str, object], ...] = (),
    diagnostic_fallback_manifest: Path | None = None,
) -> dict[str, object]:
    strict_blockers = boundary_publication_assessment.strict_blockers
    charts = []
    resnap_collisions = []
    for variant, result in zip(generated, exported, strict=True):
        notes = variant.generated.notes
        acceptance = variant.acceptance.to_report()
        playtest_only = variant.provenance in _PLAYTEST_ONLY_PROVENANCES
        coverage_summary = _coverage_summary(variant)
        quality_profile = acceptance["qualityProfile"]
        resnap_diagnostics = variant.generated.resnap_diagnostics.to_report()
        hold_lane_state_trace = analyze_hold_lane_state(
            notes,
            variant.generated.resnap_diagnostics,
        ).to_report()
        resnap_collisions.extend(
            {
                "keyMode": variant.key_mode,
                "difficulty": variant.difficulty,
                **collision,
            }
            for collision in resnap_diagnostics["collisions"]
        )
        charts.append(
            {
                "keyMode": variant.key_mode,
                "difficulty": variant.difficulty,
                "descriptor": DESCRIPTORS[variant.difficulty][0],
                "descriptors": list(DESCRIPTORS[variant.difficulty]),
                "precision": config.mapperatorinator_precision,
                "seed": variant.generated.seed,
                "selectedSeed": variant.selected_seed,
                "requestedStar": variant.requested_star,
                "cfgScale": variant.cfg_scale,
                "attemptCount": variant.attempt,
                "candidateCount": variant.candidate_count,
                "generationAttemptCount": variant.generation_attempt_count,
                "provenance": variant.provenance,
                "recoveryReason": variant.recovery_reason,
                "productionEligible": not playtest_only,
                "distributionTier": (
                    "PLAYTEST_ONLY" if playtest_only else "PRODUCTION_CANDIDATE"
                ),
                "playabilityTier": _playability_tier(variant),
                "coverageSummary": coverage_summary.model_dump(by_alias=True),
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
                    quality_profile["difficultyProfile"] if quality_profile is not None else None
                ),
                "difficultyVectorV2": (
                    quality_profile["difficultyVectorV2"] if quality_profile is not None else None
                ),
                "holdProfile": (
                    quality_profile["holdProfile"] if quality_profile is not None else None
                ),
                "patternProfile": (
                    quality_profile["patternProfile"] if quality_profile is not None else None
                ),
                "rawNoteCount": len(notes),
                "finalNoteCount": len(result.document.notes),
                "holdCount": sum(note.kind == "HOLD" for note in notes),
                "firstNoteTimeMs": notes[0].time_ms if notes else None,
                "maxGapMs": _max_gap_ms(variant),
                "rawOsuPath": _relative(variant.raw_osu_path, run_dir),
                "chartPath": _relative(result.path, run_dir),
                "resnapDiagnostics": resnap_diagnostics,
                "holdLaneStateTrace": hold_lane_state_trace,
            }
        )
    timing_review_required = (
        (authority.review is not None and authority.review.action is TimingAuthorityAction.REVIEW)
        or any(chart["acceptanceStatus"] != "PASS" for chart in charts)
        or (intro_contract_review is not None and intro_contract_review.status == "REVIEW")
        or any(review.should_block_publication for review in intro_phrase_family_reviews)
        or any(review.status == "OUTLIER" for review in timing_family_reviews)
        or (outro_family_review is not None and outro_family_review.status == "REVIEW")
    )
    has_playtest_fallback = any(
        variant.provenance in _PLAYTEST_ONLY_PROVENANCES
        for variant in generated
    )
    has_order_retry = any(
        variant.difficulty_order is not None
        and variant.difficulty_order.status == "RETRY"
        for variant in generated
    )
    # timingReviewRequired 는 진단 플래그로 남긴다 (배치 실측상 잦고,
    # 사람 판정과 자주 어긋난다). 곡 상태를 낮추는 것은 실제 계약 위반
    # 두 가지뿐이다: 조합 누락(PARTIAL), 품질 축 우회 발행(REVIEW).
    if missing:
        status = "PARTIAL"
    elif (
        has_playtest_fallback
        or has_order_retry
        or (intro_contract_review is not None and intro_contract_review.status == "REVIEW")
        or any(review.should_block_publication for review in intro_phrase_family_reviews)
        or any(review.status == "OUTLIER" for review in timing_family_reviews)
        or (outro_family_review is not None and outro_family_review.status == "REVIEW")
    ):
        status = "REVIEW"
    else:
        status = "PASS"
    outcome_status_v2 = success_outcome_status(
        expected_slots=len(KEY_MODES) * len(DIFFICULTIES),
        generated_slots=len(generated),
        requires_review=(
            timing_review_required
            or has_playtest_fallback
            or has_order_retry
            or bool(missing)
        ),
    )
    publication_decision = decide_publication(
        outcome=outcome_status_v2,
        published_slots=len(generated),
        expected_slots=len(KEY_MODES) * len(DIFFICULTIES),
        strict_blockers=strict_blockers,
    )
    return {
        "version": 1,
        "qualityGateVersion": QUALITY_GATE_VERSION,
        "publishable": publication_decision.decision == "ALLOW_PRODUCTION",
        "status": status,
        "outcomeStatusV2": outcome_status_v2.to_report(),
        "strictBlockers": list(strict_blockers),
        "publicationDecision": publication_decision.to_report(),
        "boundaryPublicationAssessment": boundary_publication_assessment.to_report(),
        "runId": str(run_id),
        "sourceName": options.source.name,
        "generator": options.generator,
        "strategy": "MAPPERATORINATOR_SHARED_TIMING",
        "attemptJournal": build_attempt_journal_projection(
            run_dir / "attempt-journal.jsonl",
            relative_to=run_dir,
        ),
        **_timing_authority_report(authority, run_dir),
        **_authority_epoch_report(
            authority_epochs,
            map_timing_escalations,
            selected_authority_epoch,
        ),
        "resnapCollisions": resnap_collisions,
        "noteMutationEnabled": (
            intro_contract_review is not None and intro_contract_review.corrected_count > 0
        ),
        "mapperatorinatorConstraintPatch": (
            CONSTRAINT_PATCH_ID if options.generator == "mapperatorinator" else None
        ),
        "mapperatorinatorHoldStateMode": (
            config.mapperatorinator_hold_state_mode
            if options.generator == "mapperatorinator"
            else None
        ),
        "attemptsPerChartMax": MAX_TOTAL_ATTEMPTS,
        "qualityAttemptsPerChartMax": MAX_VARIANT_ATTEMPTS,
        "crashAttemptsPerChartMax": MAX_CRASH_ATTEMPTS,
        "canonicalAudioSha256": prepared.normalized.sha256,
        "runtimeFingerprint": build_runtime_fingerprint(
            config=config,
            prepared=prepared,
            analysis=analysis,
            authority=authority,
            generator=options.generator,
            worker_version=options.worker_version,
        ),
        "timingReviewRequired": timing_review_required,
        "elapsedMsByStage": elapsed,
        "warnings": [],
        "difficultyOrder": difficulty_order_reports,
        "difficultySelectionShadow": [
            comparison.to_report() for comparison in difficulty_selection_shadows
        ],
        "songSelectionShadow": (
            song_selection_shadow.to_report() if song_selection_shadow is not None else None
        ),
        "introStartContract": (
            intro_start_contract.to_report() if intro_start_contract is not None else None
        ),
        "introContractReview": (
            intro_contract_review.to_report() if intro_contract_review is not None else None
        ),
        "introPhraseFamilyReviews": [
            review.to_report() for review in intro_phrase_family_reviews
        ],
        "timingFamilyReviews": [review.to_report() for review in timing_family_reviews],
        "outroFamilyReview": (
            outro_family_review.to_report() if outro_family_review is not None else None
        ),
        "additionalInferenceCalls": additional_inference_calls,
        "additionalInferenceWorkMs": additional_inference_work_ms,
        "additionalInferenceWorkLimitMs": additional_inference_work_limit_ms,
        "musicBounds": music_bounds,
        "availableCharts": len(charts),
        "missingCharts": [entry.to_report() for entry in missing],
        "diagnosticRawFallbacks": [
            {
                **entry.to_report(relative_to=run_dir),
                "decision": "PLAYTEST_ONLY",
                "productionEligible": False,
            }
            for entry in diagnostic_fallbacks
        ],
        "diagnosticRawFallbackFailures": list(diagnostic_fallback_failures),
        "diagnosticFallbackManifest": (
            {
                "path": _relative(diagnostic_fallback_manifest, run_dir),
                "sha256": sha256_file(diagnostic_fallback_manifest),
            }
            if diagnostic_fallback_manifest is not None
            else None
        ),
        "charts": charts,
    }


def _failure_generation_report(
    options: PipelineOptions,
    config: WorkerConfig,
    run_id: UUID,
    elapsed: dict[str, int],
    run_dir: Path,
    prepared: PreparedAudio,
    authority: SongTimingAuthority | None,
    error: WorkerError,
    boundary_publication_assessment: BoundaryPublicationAssessment,
    *,
    failure_stage: str | None = None,
    authority_epochs: list[AuthorityEpochRecord] | None = None,
    map_timing_escalations: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    strict_blockers = boundary_publication_assessment.strict_blockers
    status = {
        ErrorCode.CHART_TIMING_REVIEW_REQUIRED: "REVIEW",
        ErrorCode.CHART_TIMING_CANDIDATE_FAILED: "EXHAUSTED",
        ErrorCode.CHART_CANDIDATES_EXHAUSTED: "EXHAUSTED",
        ErrorCode.CHART_VALIDATION_FAILED: "FAILED",
        ErrorCode.INFERENCE_PROTOCOL_FAILED: "FAILED",
        ErrorCode.INFERENCE_START_FAILED: "FAILED",
        ErrorCode.INFERENCE_COMPLETION_UNKNOWN: "FAILED",
        ErrorCode.INFERENCE_INVOCATION_CONFLICT: "FAILED",
    }[error.code]
    failure_category = {
        ErrorCode.CHART_TIMING_REVIEW_REQUIRED: "POLICY",
        ErrorCode.CHART_TIMING_CANDIDATE_FAILED: "GENERATION",
        ErrorCode.CHART_CANDIDATES_EXHAUSTED: "GENERATION",
        ErrorCode.CHART_VALIDATION_FAILED: "VALIDATION",
        ErrorCode.INFERENCE_PROTOCOL_FAILED: "INFRA",
        ErrorCode.INFERENCE_START_FAILED: "INFRA",
        ErrorCode.INFERENCE_COMPLETION_UNKNOWN: "INFRA",
        ErrorCode.INFERENCE_INVOCATION_CONFLICT: "INFRA",
    }[error.code]
    outcome_status_v2 = failure_outcome_status(category=failure_category)
    publication_decision = decide_publication(
        outcome=outcome_status_v2,
        published_slots=0,
        expected_slots=len(KEY_MODES) * len(DIFFICULTIES),
        strict_blockers=strict_blockers,
    )
    report: dict[str, object] = {
        "version": 1,
        "qualityGateVersion": QUALITY_GATE_VERSION,
        "runId": str(run_id),
        "sourceName": options.source.name,
        "generator": options.generator,
        "strategy": "MAPPERATORINATOR_SHARED_TIMING",
        "mapperatorinatorHoldStateMode": (
            config.mapperatorinator_hold_state_mode
            if options.generator == "mapperatorinator"
            else None
        ),
        "attemptJournal": build_attempt_journal_projection(
            run_dir / "attempt-journal.jsonl",
            relative_to=run_dir,
        ),
        "publishable": False,
        "status": status,
        "outcomeStatusV2": outcome_status_v2.to_report(),
        "strictBlockers": list(strict_blockers),
        "publicationDecision": publication_decision.to_report(),
        "boundaryPublicationAssessment": boundary_publication_assessment.to_report(),
        "error": {
            "code": error.code.value,
            "context": error.context,
        },
        **_timing_authority_report(authority, run_dir),
        **_authority_epoch_report(
            authority_epochs or [],
            map_timing_escalations or [],
            None,
        ),
        "resnapCollisions": [],
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


def _export_diagnostic_raw_fallbacks(
    outcome: GenerationOutcome,
    *,
    options: PipelineOptions,
    config: WorkerConfig,
    prepared: PreparedAudio,
    authority: SongTimingAuthority,
    run_dir: Path,
) -> tuple[
    tuple[DiagnosticFallbackExport, ...],
    tuple[dict[str, object], ...],
    Path | None,
]:
    candidates = outcome.diagnostic_raw_candidates
    if not candidates:
        return (), (), None
    failures: list[dict[str, object]] = []
    exports: list[DiagnosticFallbackExport] = []
    if options.generator != "mapperatorinator":
        failures.extend(
            {
                "keyMode": candidate.key_mode,
                "difficulty": candidate.difficulty,
                "reason": "DIAGNOSTIC_FALLBACK_REQUIRES_MAPPERATORINATOR",
            }
            for candidate in candidates
        )
    elif (
        config.mapperatorinator_hold_state_mode != "incremental"
        or config.mapperatorinator_model_root is None
        or config.mapperatorinator_model_revision is None
    ):
        failures.extend(
            {
                "keyMode": candidate.key_mode,
                "difficulty": candidate.difficulty,
                "reason": "DIAGNOSTIC_FALLBACK_IDENTITY_NOT_PINNED",
            }
            for candidate in candidates
        )
    else:
        identity = DiagnosticFallbackIdentity(
            audio_sha256=prepared.normalized.sha256,
            timing_sha256=authority.sha256,
            model_identity=(
                f"{config.mapperatorinator_model_root.resolve(strict=False)}"
                f"@{config.mapperatorinator_model_revision}"
            ),
            patch_set_id=CONSTRAINT_PATCH_ID,
            hold_state_mode=config.mapperatorinator_hold_state_mode,
        )
        for candidate in candidates:
            def validate_osu(text: str, *, _candidate=candidate) -> None:
                beatmap = parse_osu_mania(text)
                generated = GeneratedChart(
                    notes=beatmap.notes,
                    key_mode=beatmap.key_mode,
                    osu_text=text,
                    generator_name="diagnostic-raw-fallback",
                    seed=_candidate.seed,
                    bpm_events=beatmap.bpm_events,
                )
                validate_generated_chart(
                    generated,
                    key_mode=_candidate.key_mode,
                    duration_ms=prepared.normalized.duration_ms,
                )
                validate_timing_identity(beatmap.bpm_events, authority.bpm_events)

            try:
                exports.append(
                    export_diagnostic_fallback(
                        candidate,
                        run_dir=run_dir,
                        identity=identity,
                        validate_osu=validate_osu,
                    )
                )
            except (OSError, TypeError, ValueError, WorkerError) as error:
                failures.append(
                    {
                        "keyMode": candidate.key_mode,
                        "difficulty": candidate.difficulty,
                        "reason": "DIAGNOSTIC_FALLBACK_EXPORT_FAILED",
                        "errorType": type(error).__name__,
                        "message": str(error),
                    }
                )
    root = run_dir / "diagnostic-raw-fallback"
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "manifest-v1.json"
    _write_generation_report(
        manifest_path,
        {
            "version": DIAGNOSTIC_FALLBACK_VERSION,
            "decision": "PLAYTEST_ONLY",
            "productionEligible": False,
            "entries": [entry.to_report(relative_to=run_dir) for entry in exports],
            "failures": failures,
        },
    )
    return tuple(exports), tuple(failures), manifest_path


def _write_playtest_manifest(path: Path, manifest: PlaytestRunManifestV2) -> None:
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(
        manifest.model_dump_json(by_alias=True, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _require_publishable_acceptance(generated: tuple[GeneratedVariant, ...]) -> None:
    """Allow soft-gate playtest fallbacks, but never waive hard safety axes."""
    rejected = []
    for variant in generated:
        if variant.acceptance.action is not GateAction.RETRY_MAP:
            continue
        hard_safe = all(
            variant.acceptance.decision(axis).action is GateAction.PASS
            for axis in (
                GateAxis.STRUCTURE,
                GateAxis.TIMING_IDENTITY,
                GateAxis.SONG_BOUNDS,
            )
        )
        if (
            variant.provenance in _PLAYTEST_ONLY_PROVENANCES
            and hard_safe
        ):
            continue
        rejected.append(
            {
                "key_mode": variant.key_mode,
                "difficulty": variant.difficulty,
                "provenance": variant.provenance,
                "gate_report": variant.acceptance.to_report(),
            }
        )
    if not rejected:
        return
    raise WorkerError(
        ErrorCode.CHART_CANDIDATES_EXHAUSTED,
        "generation stage returned non-publishable chart candidates",
        context={"variants": rejected},
    )


def _run_pipeline(
    options: PipelineOptions,
    *,
    dependencies: PipelineDependencies,
    run_dir: Path,
    inference_session: InferenceSession | None,
) -> PipelineResult:
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
    boundary_evaluation = (
        evaluate_boundary_policy(
            onsets.activity,
            prepared.normalized.duration_ms,
            enforcement_mode=prepared.boundary_policy_mode,
            terminal_silence=onsets.terminal_silence,
        )
        if onsets.activity is not None
        else None
    )
    boundary_publication_assessment = assess_boundary_publication(
        policy_state=(
            boundary_evaluation.policy_state
            if boundary_evaluation is not None
            else None
        ),
        confidence=(
            boundary_evaluation.confidence
            if boundary_evaluation is not None
            else None
        ),
    )

    generator = dependencies.select_generator(options.generator, dependencies.config)
    if inference_session is not None:
        inference_session.begin_song(
            SongIdentity(
                song_id=str(run_id),
                audio_sha256=prepared.normalized.sha256,
                config_digest=_song_config_digest(dependencies.config),
            )
        )
        generator = dependencies.bind_inference_session(generator, inference_session)

    started = perf_counter_ns()
    try:
        authority = dependencies.timing(prepared, onsets, run_dir, generator, options.seed)
    except WorkerError as error:
        if error.code not in {
            ErrorCode.CHART_TIMING_REVIEW_REQUIRED,
            ErrorCode.CHART_TIMING_CANDIDATE_FAILED,
        } and error.code not in _REPORTABLE_INFERENCE_FAILURES:
            raise
        elapsed["timing"] = _elapsed_ms(started)
        _write_generation_report(
            run_dir / "generation-report.json",
            _failure_generation_report(
                options,
                dependencies.config,
                run_id,
                elapsed,
                run_dir,
                prepared,
                None,
                error,
                boundary_publication_assessment,
                failure_stage="TIMING",
            ),
        )
        raise
    _require_canonical_audio_hash(prepared)
    elapsed["timing"] = _elapsed_ms(started)

    authority_epochs: list[AuthorityEpochRecord] = []
    map_timing_escalations: list[dict[str, object]] = []
    generation_elapsed_ms = 0
    selected_authority_epoch: int | None = None

    outcome: GenerationOutcome | None = None
    for epoch in (1, 2):
        started = perf_counter_ns()
        try:
            outcome = dependencies.generation(
                prepared,
                authority,
                onsets,
                run_dir,
                generator,
                options.seed,
                epoch,
            )
            generated = outcome.variants
            _require_publishable_acceptance(generated)
            difficulty_order_reports = _require_difficulty_order_reports(generated, outcome.missing)
        except RetryTimingSignal as signal:
            generation_elapsed_ms += _elapsed_ms(started)
            escalation = {"epoch": epoch, **signal.to_context()}
            map_timing_escalations.append(escalation)
            if epoch == 2 or authority.mode != "STANDARD":
                authority_epochs.append(
                    AuthorityEpochRecord(
                        epoch=epoch,
                        authority_sha256=authority.sha256,
                        mode=authority.mode,
                        status="FAILED",
                        escalation=escalation,
                    )
                )
                error = WorkerError(
                    ErrorCode.CHART_CANDIDATES_EXHAUSTED,
                    "Final timing authority failed corroborated MAP timing checks",
                    context={
                        "reason": "FINAL_TIMING_AUTHORITY_MAP_FEEDBACK_EXHAUSTED",
                        "epochs": [record.to_report() for record in authority_epochs],
                        "mapTimingEscalations": map_timing_escalations,
                    },
                )
                elapsed["generation"] = generation_elapsed_ms
                _write_generation_report(
                    run_dir / "generation-report.json",
                    _failure_generation_report(
                        options,
                        dependencies.config,
                        run_id,
                        elapsed,
                        run_dir,
                        prepared,
                        authority,
                        error,
                        boundary_publication_assessment,
                        failure_stage="GENERATION",
                        authority_epochs=authority_epochs,
                        map_timing_escalations=map_timing_escalations,
                    ),
                )
                raise error from signal

            authority_epochs.append(
                AuthorityEpochRecord(
                    epoch=epoch,
                    authority_sha256=authority.sha256,
                    mode=authority.mode,
                    status="REJECTED_MAP_TIMING_FEEDBACK",
                    escalation=escalation,
                )
            )
            started = perf_counter_ns()
            try:
                authority = dependencies.super_timing(
                    prepared,
                    onsets,
                    run_dir,
                    generator,
                    options.seed,
                    epoch + 1,
                )
            except WorkerError as error:
                if error.code not in {
                    ErrorCode.CHART_TIMING_REVIEW_REQUIRED,
                    ErrorCode.CHART_TIMING_CANDIDATE_FAILED,
                } and error.code not in _REPORTABLE_INFERENCE_FAILURES:
                    raise
                elapsed["timing"] += _elapsed_ms(started)
                elapsed["generation"] = generation_elapsed_ms
                _write_generation_report(
                    run_dir / "generation-report.json",
                    _failure_generation_report(
                        options,
                        dependencies.config,
                        run_id,
                        elapsed,
                        run_dir,
                        prepared,
                        None,
                        error,
                        boundary_publication_assessment,
                        failure_stage="TIMING",
                        authority_epochs=authority_epochs,
                        map_timing_escalations=map_timing_escalations,
                    ),
                )
                raise
            elapsed["timing"] += _elapsed_ms(started)
            _require_canonical_audio_hash(prepared)
            continue
        except WorkerError as error:
            if error.code not in {
                ErrorCode.CHART_TIMING_REVIEW_REQUIRED,
                ErrorCode.CHART_CANDIDATES_EXHAUSTED,
                ErrorCode.CHART_VALIDATION_FAILED,
            } and error.code not in _REPORTABLE_INFERENCE_FAILURES:
                raise
            generation_elapsed_ms += _elapsed_ms(started)
            authority_epochs.append(
                AuthorityEpochRecord(
                    epoch=epoch,
                    authority_sha256=authority.sha256,
                    mode=authority.mode,
                    status="FAILED",
                )
            )
            elapsed["generation"] = generation_elapsed_ms
            _write_generation_report(
                run_dir / "generation-report.json",
                _failure_generation_report(
                    options,
                    dependencies.config,
                    run_id,
                    elapsed,
                    run_dir,
                    prepared,
                    authority,
                    error,
                    boundary_publication_assessment,
                    failure_stage="GENERATION",
                    authority_epochs=authority_epochs,
                    map_timing_escalations=map_timing_escalations,
                ),
            )
            raise
        else:
            generation_elapsed_ms += _elapsed_ms(started)
            authority_epochs.append(
                AuthorityEpochRecord(
                    epoch=epoch,
                    authority_sha256=authority.sha256,
                    mode=authority.mode,
                    status="SELECTED",
                )
            )
            selected_authority_epoch = epoch
            break
    else:
        raise AssertionError("authority epoch attempts were unexpectedly exhausted")

    _require_canonical_audio_hash(prepared)
    elapsed["generation"] = generation_elapsed_ms
    assert selected_authority_epoch is not None

    started = perf_counter_ns()
    exported = dependencies.export(
        prepared,
        generated,
        run_dir,
        options.worker_version,
    )
    elapsed["export"] = _elapsed_ms(started)

    if outcome.diagnostic_raw_candidates:
        started = perf_counter_ns()
        diagnostic_fallbacks, diagnostic_fallback_failures, diagnostic_manifest = (
            _export_diagnostic_raw_fallbacks(
                outcome,
                options=options,
                config=dependencies.config,
                prepared=prepared,
                authority=authority,
                run_dir=run_dir,
            )
        )
        elapsed["diagnosticFallback"] = _elapsed_ms(started)
    else:
        diagnostic_fallbacks = ()
        diagnostic_fallback_failures = ()
        diagnostic_manifest = None

    report_path = run_dir / "generation-report.json"
    assert outcome is not None
    outro_observation = (
        observe_outro(onsets.activity, prepared.normalized.duration_ms)
        if onsets.activity is not None
        else None
    )
    outro_evidence_profile = (
        build_outro_evidence_profile(
            activity=onsets.activity,
            onset_ms=onsets.onset_ms,
            duration_ms=prepared.normalized.duration_ms,
        )
        if onsets.activity is not None
        else None
    )
    generation_report = _generation_report(
        options,
        run_id,
        elapsed,
        generated,
        exported,
        run_dir,
        dependencies.config,
        prepared,
        onsets,
        authority,
        difficulty_order_reports,
        authority_epochs,
        map_timing_escalations,
        selected_authority_epoch,
        boundary_publication_assessment,
        difficulty_selection_shadows=outcome.difficulty_selection_shadows,
        song_selection_shadow=outcome.song_selection_shadow,
        intro_start_contract=outcome.intro_start_contract,
        intro_contract_review=outcome.intro_contract_review,
        intro_phrase_family_reviews=outcome.intro_phrase_family_reviews,
        timing_family_reviews=outcome.timing_family_reviews,
        outro_family_review=outcome.outro_family_review,
        additional_inference_calls=outcome.additional_inference_calls,
        additional_inference_work_ms=outcome.additional_inference_work_ms,
        additional_inference_work_limit_ms=(
            outcome.additional_inference_work_limit_ms
        ),
        missing=outcome.missing,
        music_bounds={
            "audioDurationMs": prepared.normalized.duration_ms,
            "musicEndMs": (
                (
                    prepared.normalized.duration_ms
                    if boundary_evaluation.effective_source
                    == "FULL_DURATION_BASELINE"
                    else boundary_evaluation.provisional_decision.selected_music_end_ms
                )
                if boundary_evaluation is not None
                else None
            ),
            "outroObservation": (
                outro_observation.to_report() if outro_observation is not None else None
            ),
            "outroEvidenceProfile": (
                outro_evidence_profile.to_report()
                if outro_evidence_profile is not None
                else None
            ),
            "terminalSilenceObservation": (
                onsets.terminal_silence.to_report()
                if onsets.terminal_silence is not None
                else None
            ),
            "outroPolicyDecision": (
                boundary_evaluation.provisional_decision.to_report()
                if boundary_evaluation is not None
                else None
            ),
            "boundaryPolicyEvaluation": (
                boundary_evaluation.to_report()
                if boundary_evaluation is not None
                else None
            ),
            "songBoundaryContract": (
                boundary_evaluation.effective_contract.to_report()
                if boundary_evaluation is not None
                else None
            ),
        },
        diagnostic_fallbacks=diagnostic_fallbacks,
        diagnostic_fallback_failures=diagnostic_fallback_failures,
        diagnostic_fallback_manifest=diagnostic_manifest,
    )
    _write_generation_report(report_path, generation_report)
    manifest = PlaytestRunManifestV2(
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
            RunChartRefV2(
                path=_relative(result.path, run_dir),
                sha256=result.sha256,
                key_mode=result.document.key_mode,
                difficulty=result.document.difficulty,
                provenance=variant.provenance,
                production_eligible=(
                    variant.provenance
                    not in _PLAYTEST_ONLY_PROVENANCES
                ),
                distribution_tier=(
                    "PLAYTEST_ONLY"
                    if variant.provenance in _PLAYTEST_ONLY_PROVENANCES
                    else "PRODUCTION_CANDIDATE"
                ),
                playability_tier=_playability_tier(variant),
                coverage_summary=_coverage_summary(variant),
            )
            for variant, result in zip(generated, exported, strict=True)
        ],
        missing_charts=[
            MissingChartRef(
                key_mode=entry.key_mode,
                difficulty=entry.difficulty,
                reason=entry.reason,
            )
            for entry in outcome.missing
        ],
        generation_report=ReportFileRef(
            path=_relative(report_path, run_dir),
            sha256=sha256_file(report_path),
        ),
        outcome=OutcomeStatusSnapshot.model_validate(generation_report["outcomeStatusV2"]),
        strict_blockers=generation_report["strictBlockers"],
        publication=PublicationDecisionSnapshot.model_validate(
            generation_report["publicationDecision"]
        ),
    )
    manifest_path = run_dir / "playtest-run-v2.json"
    _write_playtest_manifest(manifest_path, manifest)
    return PipelineResult(
        run_id=run_id,
        output_dir=run_dir,
        manifest_path=manifest_path,
        chart_paths=tuple(result.path for result in exported),
        raw_osu_paths=tuple(variant.raw_osu_path for variant in generated),
        elapsed_ms_by_stage=elapsed,
    )


def run_pipeline(
    options: PipelineOptions,
    *,
    dependencies: PipelineDependencies | None = None,
) -> PipelineResult:
    dependencies = dependencies or PipelineDependencies()
    run_dir = _prepare_run_dir(options)
    resident_requested = (
        options.generator == "mapperatorinator"
        and dependencies.config.mapperatorinator_backend == "song_session"
    )
    if dependencies.attached_inference_session is not None and not resident_requested:
        raise ValueError("an attached inference session requires mapperatorinator song_session")

    session = dependencies.attached_inference_session
    owns_session = False
    if resident_requested and session is None:
        session = dependencies.open_inference_session(dependencies.config, run_dir)
        owns_session = True
    try:
        return _run_pipeline(
            options,
            dependencies=dependencies,
            run_dir=run_dir,
            inference_session=session,
        )
    finally:
        if session is not None:
            try:
                if session.state is SessionState.SONG_ACTIVE:
                    session.end_song()
            finally:
                if owns_session:
                    session.close()
