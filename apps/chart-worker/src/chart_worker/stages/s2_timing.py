"""S2 timing: promote one validated timing reference for the canonical song audio."""

from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from time import perf_counter_ns

from chart_worker.analysis.beat import BeatGrid, bpm_events_of
from chart_worker.analysis.beat_corroboration import measure_beat_corroboration
from chart_worker.analysis.beat_this_backend import BeatThisFileAnalyzer
from chart_worker.analysis.grid_alignment import (
    TempoCandidateMetrics,
    measure_tempo_candidates,
)
from chart_worker.analysis.local_timing import measure_local_timing
from chart_worker.analysis.onset import OnsetAnalysis
from chart_worker.errors import ErrorCode, WorkerError
from chart_worker.generation.mapperatorinator import ChartGenerator, GeneratedTiming
from chart_worker.generation.osu_parser import OsuBpmEvent, parse_osu_file
from chart_worker.generation.osu_writer import timing_to_osu_mania
from chart_worker.generation.params import TimingGenerationRequest
from chart_worker.hashing import sha256_file
from chart_worker.stages.types import PreparedAudio, SongTimingAuthority
from chart_worker.validation.leading_timing_coverage import (
    LeadingTimingCoverage,
    review_leading_timing_coverage,
)
from chart_worker.validation.local_timing_review import (
    LocalTimingAuthorityReview,
    review_local_timing_authority,
)
from chart_worker.validation.recovery_preflight import (
    RecoveryPreflight,
    review_recovery_preflight,
)
from chart_worker.validation.timing_authority import (
    TimingAuthorityValidationError,
    validate_timing_events,
    validate_timing_identity,
)
from chart_worker.validation.timing_candidate_selector import (
    TimingCandidateEvidence,
    TimingCandidateSelection,
    build_timing_candidate_evidence,
    select_timing_candidate,
    timing_candidates_need_external_corroboration,
)
from chart_worker.validation.timing_integrity import (
    TimingIntegrityAssessment,
    TimingIntegrityStatus,
    assess_timing_integrity,
)
from chart_worker.validation.timing_review import (
    TimingAuthorityAction,
    TimingAuthorityReview,
    review_timing_authority,
)

_SUPER_TIMING_REVIEW_REASONS = frozenset(
    {"TEMPO_EVIDENCE_DISAGREES", "WEAK_BASE_TEMPO_SUPPORT"}
)

BeatAnalyzer = Callable[[Path], BeatGrid]


def _should_try_super_timing(review: TimingAuthorityReview) -> bool:
    return review.action is TimingAuthorityAction.RETRY_TIMING or bool(
        _SUPER_TIMING_REVIEW_REASONS.intersection(review.reasons)
    )


def _has_healthy_candidate(
    candidates: list["_EvaluatedTimingCandidate"],
) -> bool:
    return any(
        candidate.integrity.status is TimingIntegrityStatus.HEALTHY
        for candidate in candidates
    )


def _merge_reviews(
    tempo: TimingAuthorityReview,
    leading: LeadingTimingCoverage,
) -> TimingAuthorityReview:
    priority = {
        TimingAuthorityAction.PASS: 0,
        TimingAuthorityAction.REVIEW: 1,
        TimingAuthorityAction.RETRY_TIMING: 2,
    }
    action = max((tempo.action, leading.action), key=priority.__getitem__)
    leading_reasons = (
        () if leading.action is TimingAuthorityAction.PASS else leading.reasons
    )
    reasons = tuple(dict.fromkeys((*tempo.reasons, *leading_reasons)))
    return TimingAuthorityReview(action, reasons)


def _validate_candidate(generated: GeneratedTiming, prepared: PreparedAudio) -> None:
    validate_timing_events(generated.bpm_events, prepared.normalized.duration_ms)


def _candidate_authority(
    generated: GeneratedTiming,
    prepared: PreparedAudio,
    reference_path: Path,
) -> SongTimingAuthority:
    return SongTimingAuthority(
        reference_path=reference_path,
        sha256="",
        audio_sha256=prepared.normalized.sha256,
        bpm_events=generated.bpm_events,
        generator_name=generated.generator_name,
        seed=generated.seed,
        mode=generated.mode,
        attempt_count=0,
    )


def _promote(generated: GeneratedTiming, prepared: PreparedAudio, reference_path: Path) -> None:
    reference_path.parent.mkdir(parents=True, exist_ok=True)
    candidate_path = reference_path.with_name(f"{reference_path.stem}-candidate.osu")
    try:
        candidate_path.write_text(
            timing_to_osu_mania(
                generated.bpm_events,
                audio_filename=prepared.normalized.path.name,
                title=prepared.normalized.path.stem,
            ),
            encoding="utf-8",
        )
        validate_timing_identity(parse_osu_file(candidate_path).bpm_events, generated.bpm_events)
        candidate_path.replace(reference_path)
    except TimingAuthorityValidationError:
        candidate_path.unlink(missing_ok=True)
        reference_path.unlink(missing_ok=True)
        raise


@dataclass(frozen=True, slots=True)
class _EvaluatedTimingCandidate:
    generated: GeneratedTiming
    attempt_count: int
    metrics: TempoCandidateMetrics
    review: TimingAuthorityReview
    leading: LeadingTimingCoverage
    local_review: LocalTimingAuthorityReview
    recovery_preflight: RecoveryPreflight
    integrity: TimingIntegrityAssessment
    evidence: TimingCandidateEvidence


def _evaluate_timing_candidate(
    generated: GeneratedTiming,
    *,
    attempt_count: int,
    prepared: PreparedAudio,
    analysis: OnsetAnalysis,
    reference_path: Path,
    authority_epoch: int,
    workdir: Path,
    run_dir: Path,
) -> tuple[_EvaluatedTimingCandidate, dict[str, object], bool]:
    """Apply the same evidence and hard gates to every timing backend."""
    _validate_candidate(generated, prepared)
    leading = review_leading_timing_coverage(
        generated.bpm_events,
        analysis,
        duration_ms=prepared.normalized.duration_ms,
    )
    metrics = measure_tempo_candidates(generated.bpm_events, analysis)
    review = _merge_reviews(review_timing_authority(metrics), leading)
    recovery_preflight = review_recovery_preflight(
        _candidate_authority(generated, prepared, reference_path),
        analysis,
        duration_ms=prepared.normalized.duration_ms,
        boundary_policy_mode=prepared.boundary_policy_mode,
    )
    local_metrics = measure_local_timing(
        generated.bpm_events,
        analysis,
        duration_ms=prepared.normalized.duration_ms,
    )
    local_review = review_local_timing_authority(
        local_metrics,
        recovery_preflight,
    )
    integrity = assess_timing_integrity(local_review, recovery_preflight)
    evidence = build_timing_candidate_evidence(
        epoch=authority_epoch,
        mode=generated.mode,
        structurally_valid=True,
        local_review=local_review,
        tempo_metrics=metrics,
        integrity=integrity,
    )
    evaluated = _EvaluatedTimingCandidate(
        generated=generated,
        attempt_count=attempt_count,
        metrics=metrics,
        review=review,
        leading=leading,
        local_review=local_review,
        recovery_preflight=recovery_preflight,
        integrity=integrity,
        evidence=evidence,
    )
    hard_rejected = (
        review.action is TimingAuthorityAction.RETRY_TIMING
        or leading.action is TimingAuthorityAction.RETRY_TIMING
        or local_review.action is TimingAuthorityAction.RETRY_TIMING
        or integrity.status is TimingIntegrityStatus.DAMAGED
    )
    report = {
        "attempt": attempt_count,
        "seed": generated.seed,
        "mode": generated.mode,
        "workdir": workdir.relative_to(run_dir).as_posix(),
        "review": review.to_report(),
        "leadingCoverage": leading.to_report(),
        "tempoMetrics": metrics.to_report(),
        "localTimingReview": local_review.to_report(),
        "recoveryPreflight": recovery_preflight.to_report(),
        "timingIntegrity": integrity.to_report(),
        "candidateEvidence": evidence.to_report(),
    }
    return evaluated, report, hard_rejected


def _generated_from_beat_grid(
    grid: BeatGrid,
    *,
    checkpoint: str,
) -> GeneratedTiming:
    fitted = bpm_events_of(grid)
    return GeneratedTiming(
        osu_text="",
        bpm_events=tuple(OsuBpmEvent(event.time_ms, event.bpm) for event in fitted),
        generator_name=f"beat-this:{checkpoint}",
        seed=None,
        mode="BEAT_THIS_FALLBACK",
    )


def _evaluate_independent_fallback(
    *,
    prepared: PreparedAudio,
    analysis: OnsetAnalysis,
    run_dir: Path,
    reference_path: Path,
    authority_epoch: int,
    attempt_count: int,
    beat_analyzer: BeatAnalyzer | None,
) -> tuple[_EvaluatedTimingCandidate | None, dict[str, object]]:
    """Build one independent timing candidate and pass it through normal gates."""
    analyzer = beat_analyzer or BeatThisFileAnalyzer(
        checkpoint=prepared.beat_this_checkpoint,
        device=prepared.beat_this_device,
        float16=prepared.beat_this_float16,
    )
    workdir = (
        run_dir
        / "timing"
        / "work"
        / f"epoch-{authority_epoch}"
        / "beat-this-fallback"
        / f"attempt-{attempt_count}"
    )
    started_ns = perf_counter_ns()
    try:
        grid = analyzer(prepared.normalized.path)
        generated = _generated_from_beat_grid(
            grid,
            checkpoint=prepared.beat_this_checkpoint,
        )
        evaluated, report, hard_rejected = _evaluate_timing_candidate(
            generated,
            attempt_count=attempt_count,
            prepared=prepared,
            analysis=analysis,
            reference_path=reference_path,
            authority_epoch=authority_epoch,
            workdir=workdir,
            run_dir=run_dir,
        )
        report["independentFallbackStatus"] = (
            "REJECTED" if hard_rejected else "VIABLE"
        )
        report["independentFallbackElapsedMs"] = max(
            0,
            (perf_counter_ns() - started_ns) // 1_000_000,
        )
        return (None if hard_rejected else evaluated), report
    except Exception as error:  # noqa: BLE001 - optional backend degrade boundary
        return None, {
            "attempt": attempt_count,
            "seed": None,
            "mode": "BEAT_THIS_FALLBACK",
            "workdir": workdir.relative_to(run_dir).as_posix(),
            "independentFallbackStatus": (
                "UNAVAILABLE" if isinstance(error, ModuleNotFoundError) else "FAILED"
            ),
            "independentFallbackError": f"{type(error).__name__}: {error}",
            "independentFallbackElapsedMs": max(
                0,
                (perf_counter_ns() - started_ns) // 1_000_000,
            ),
        }


def _select_and_promote(
    candidates: list[_EvaluatedTimingCandidate],
    *,
    prepared: PreparedAudio,
    reference_path: Path,
    beat_analyzer: BeatAnalyzer | None,
) -> tuple[_EvaluatedTimingCandidate, TimingCandidateSelection]:
    """Promote the best structurally valid candidate, removing identity failures."""
    last_error: TimingAuthorityValidationError | None = None
    while candidates:
        external_status = "NOT_REQUESTED"
        external_error = None
        external_elapsed_ms = None
        evidence = tuple(candidate.evidence for candidate in candidates)
        if (
            prepared.beat_this_enabled
            and timing_candidates_need_external_corroboration(evidence)
        ):
            analyzer = beat_analyzer or BeatThisFileAnalyzer(
                checkpoint=prepared.beat_this_checkpoint,
                device=prepared.beat_this_device,
                float16=prepared.beat_this_float16,
            )
            started_ns = perf_counter_ns()
            try:
                beat_grid = analyzer(prepared.normalized.path)
                candidates[:] = [
                    replace(
                        candidate,
                        evidence=replace(
                            candidate.evidence,
                            external_beat_f1_by_level=(
                                corroboration.f1_by_level
                            ),
                            best_external_beat_f1=corroboration.best_f1,
                        ),
                    )
                    for candidate in candidates
                    for corroboration in (
                        measure_beat_corroboration(
                            candidate.generated.bpm_events,
                            beat_grid,
                            duration_ms=prepared.normalized.duration_ms,
                        ),
                    )
                ]
                external_status = "AVAILABLE"
            except ModuleNotFoundError as error:
                external_status = "UNAVAILABLE"
                external_error = str(error)
            # Optional corroboration must never turn a viable internal timing
            # candidate into a failed song.  Third-party model and audio
            # loader failures are therefore an explicit degrade boundary.
            except Exception as error:  # noqa: BLE001
                external_status = "FAILED"
                external_error = f"{type(error).__name__}: {error}"
            external_elapsed_ms = max(
                0,
                (perf_counter_ns() - started_ns) // 1_000_000,
            )
        selection = select_timing_candidate(
            tuple(candidate.evidence for candidate in candidates)
        )
        selection = replace(
            selection,
            external_beat_status=external_status,
            external_beat_error=external_error,
            external_beat_elapsed_ms=external_elapsed_ms,
        )
        selected = candidates[selection.selected_index]
        try:
            _promote(selected.generated, prepared, reference_path)
        except TimingAuthorityValidationError as error:
            last_error = error
            candidates.pop(selection.selected_index)
            continue
        return selected, selection
    assert last_error is not None
    raise last_error


def _to_authority(
    selected: _EvaluatedTimingCandidate,
    selection: TimingCandidateSelection,
    *,
    prepared: PreparedAudio,
    reference_path: Path,
) -> SongTimingAuthority:
    generated = selected.generated
    return SongTimingAuthority(
        reference_path=reference_path,
        sha256=sha256_file(reference_path),
        audio_sha256=prepared.normalized.sha256,
        bpm_events=generated.bpm_events,
        generator_name=generated.generator_name,
        seed=generated.seed,
        mode=generated.mode,
        attempt_count=selected.attempt_count,
        tempo_metrics=selected.metrics,
        review=selected.review,
        leading_coverage=selected.leading,
        local_review=selected.local_review,
        recovery_preflight=selected.recovery_preflight,
        candidate_selection=selection,
        timing_integrity=selected.integrity,
    )


def run_timing_generation(
    prepared: PreparedAudio,
    analysis: OnsetAnalysis,
    run_dir: Path,
    *,
    generator: ChartGenerator,
    seed: int,
    force_super: bool = False,
    authority_epoch: int = 1,
    beat_analyzer: BeatAnalyzer | None = None,
) -> SongTimingAuthority:
    """Generate one authority, optionally using only the bounded Super attempt."""
    reference_path = run_dir / "audio" / "timing-reference.osu"
    errors: list[str] = []
    seeds: list[int] = []
    attempt_reviews: list[dict[str, object]] = []
    viable_candidates: list[_EvaluatedTimingCandidate] = []

    attempt_modes = (True,) if force_super else (False, True)
    for attempt_count, super_timing in enumerate(attempt_modes, start=1):
        request = TimingGenerationRequest(
            audio_path=prepared.normalized.path,
            duration_ms=prepared.normalized.duration_ms,
            seed=seed,
            super_timing=super_timing,
        )
        seeds.append(seed)
        # epoch 를 경로에 넣어야 재선택(epoch 2)이 epoch 1 의 산출물과
        # 충돌하지 않는다. force_super=True 면 attempt_count 가 항상 1이라
        # epoch 없는 경로는 이전 실행의 .osu 를 재사용하게 되고,
        # _require_clean_output_dir 가 곡 전체를 실패시킨다 (24곡 배치
        # song 24 FAILED 의 원인).
        workdir = (
            run_dir
            / "timing"
            / "work"
            / f"epoch-{authority_epoch}"
            / ("super" if super_timing else "standard")
            / f"attempt-{attempt_count}"
        )
        try:
            generated = generator.generate_timing(request, workdir)
            evaluated, attempt_report, hard_rejected = _evaluate_timing_candidate(
                generated,
                attempt_count=attempt_count,
                prepared=prepared,
                analysis=analysis,
                reference_path=reference_path,
                authority_epoch=authority_epoch,
                workdir=workdir,
                run_dir=run_dir,
            )
            attempt_reviews.append(attempt_report)
            review = evaluated.review
            local_review = evaluated.local_review
            if not hard_rejected:
                viable_candidates.append(evaluated)
            terminal_hard_reject = (
                (prepared.beat_this_enabled and hard_rejected)
                or evaluated.leading.action is TimingAuthorityAction.RETRY_TIMING
                or local_review.action is TimingAuthorityAction.RETRY_TIMING
                or evaluated.integrity.status is TimingIntegrityStatus.DAMAGED
            )
            if super_timing and terminal_hard_reject:
                if viable_candidates and (
                    _has_healthy_candidate(viable_candidates)
                    or not prepared.beat_this_enabled
                ):
                    selected, selection = _select_and_promote(
                        viable_candidates,
                        prepared=prepared,
                        reference_path=reference_path,
                        beat_analyzer=beat_analyzer,
                    )
                    return _to_authority(
                        selected,
                        selection,
                        prepared=prepared,
                        reference_path=reference_path,
                    )
                reference_path.unlink(missing_ok=True)
                break
            retry_with_super = not super_timing and (
                _should_try_super_timing(review)
                or local_review.action is TimingAuthorityAction.RETRY_TIMING
                or evaluated.integrity.status
                is not TimingIntegrityStatus.HEALTHY
            )
            needs_independent_corroboration = (
                super_timing
                and prepared.beat_this_enabled
                and viable_candidates
                and not _has_healthy_candidate(viable_candidates)
            )
            if needs_independent_corroboration:
                reference_path.unlink(missing_ok=True)
                break
            if not retry_with_super and viable_candidates:
                selected, selection = _select_and_promote(
                    viable_candidates,
                    prepared=prepared,
                    reference_path=reference_path,
                    beat_analyzer=beat_analyzer,
                )
                return _to_authority(
                    selected,
                    selection,
                    prepared=prepared,
                    reference_path=reference_path,
                )
        except TimingAuthorityValidationError as error:
            errors.append(str(error))
            reference_path.unlink(missing_ok=True)
            if not super_timing:
                continue
            if viable_candidates and (
                _has_healthy_candidate(viable_candidates)
                or not prepared.beat_this_enabled
            ):
                selected, selection = _select_and_promote(
                    viable_candidates,
                    prepared=prepared,
                    reference_path=reference_path,
                    beat_analyzer=beat_analyzer,
                )
                return _to_authority(
                    selected,
                    selection,
                    prepared=prepared,
                    reference_path=reference_path,
                )
            break

        reference_path.unlink(missing_ok=True)
        if retry_with_super:
            continue
        raise WorkerError(
            ErrorCode.CHART_TIMING_REVIEW_REQUIRED,
            "timing candidate requires human review before MAP generation",
            context={
                "reasons": review.reasons,
                "attempt_count": attempt_count,
                "attempts": attempt_reviews,
            },
        )

    if prepared.beat_this_enabled:
        fallback, fallback_report = _evaluate_independent_fallback(
            prepared=prepared,
            analysis=analysis,
            run_dir=run_dir,
            reference_path=reference_path,
            authority_epoch=authority_epoch,
            attempt_count=len(seeds) + 1,
            beat_analyzer=beat_analyzer,
        )
        attempt_reviews.append(fallback_report)
        if fallback is not None:
            viable_candidates.append(fallback)

    if viable_candidates:
        selected, selection = _select_and_promote(
            viable_candidates,
            prepared=prepared,
            reference_path=reference_path,
            beat_analyzer=beat_analyzer,
        )
        return _to_authority(
            selected,
            selection,
            prepared=prepared,
            reference_path=reference_path,
        )

    raise WorkerError(
        ErrorCode.CHART_TIMING_CANDIDATE_FAILED,
        "all bounded timing candidates failed validation",
        context={"errors": errors, "seeds": seeds, "attempts": attempt_reviews},
    )
