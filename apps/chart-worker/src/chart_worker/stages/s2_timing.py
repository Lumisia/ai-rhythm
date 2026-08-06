"""S2 timing: promote one validated timing reference for the canonical song audio."""

from pathlib import Path

from chart_worker.analysis.grid_alignment import measure_tempo_candidates
from chart_worker.analysis.onset import OnsetAnalysis
from chart_worker.errors import ErrorCode, WorkerError
from chart_worker.generation.mapperatorinator import ChartGenerator, GeneratedTiming
from chart_worker.generation.osu_parser import parse_osu_file
from chart_worker.generation.osu_writer import timing_to_osu_mania
from chart_worker.generation.params import TimingGenerationRequest
from chart_worker.hashing import sha256_file
from chart_worker.stages.types import PreparedAudio, SongTimingAuthority
from chart_worker.validation.timing_authority import (
    TimingAuthorityValidationError,
    validate_timing_events,
    validate_timing_identity,
)
from chart_worker.validation.timing_review import (
    TimingAuthorityAction,
    TimingAuthorityReview,
    review_timing_authority,
)

_SUPER_TIMING_REVIEW_REASONS = frozenset(
    {"TEMPO_EVIDENCE_DISAGREES", "WEAK_BASE_TEMPO_SUPPORT"}
)


def _should_try_super_timing(review: TimingAuthorityReview) -> bool:
    return review.action is TimingAuthorityAction.RETRY_TIMING or bool(
        _SUPER_TIMING_REVIEW_REASONS.intersection(review.reasons)
    )


def _validate_candidate(generated: GeneratedTiming, prepared: PreparedAudio) -> None:
    validate_timing_events(generated.bpm_events, prepared.normalized.duration_ms)


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


def run_timing_generation(
    prepared: PreparedAudio,
    analysis: OnsetAnalysis,
    run_dir: Path,
    *,
    generator: ChartGenerator,
    seed: int,
) -> SongTimingAuthority:
    """Generate one publishable timing authority with at most one Super fallback."""
    reference_path = run_dir / "audio" / "timing-reference.osu"
    errors: list[str] = []
    seeds: list[int] = []
    attempt_reviews: list[dict[str, object]] = []

    for attempt_count, super_timing in enumerate((False, True), start=1):
        request = TimingGenerationRequest(
            audio_path=prepared.normalized.path,
            duration_ms=prepared.normalized.duration_ms,
            seed=seed,
            super_timing=super_timing,
        )
        seeds.append(seed)
        workdir = run_dir / "timing" / "work" / f"attempt-{attempt_count}"
        try:
            generated = generator.generate_timing(request, workdir)
            _validate_candidate(generated, prepared)
            metrics = measure_tempo_candidates(generated.bpm_events, analysis)
            review = review_timing_authority(metrics)
            attempt_reviews.append(
                {
                    "attempt": attempt_count,
                    "seed": generated.seed,
                    "mode": generated.mode,
                    "workdir": workdir.relative_to(run_dir).as_posix(),
                    "review": review.to_report(),
                    "tempoMetrics": metrics.to_report(),
                }
            )
            retry_with_super = not super_timing and _should_try_super_timing(review)
            publishable_review = review.action is TimingAuthorityAction.REVIEW
            if not retry_with_super and (
                review.action is TimingAuthorityAction.PASS or publishable_review
            ):
                _promote(generated, prepared, reference_path)
        except TimingAuthorityValidationError as error:
            errors.append(str(error))
            reference_path.unlink(missing_ok=True)
            if not super_timing:
                continue
            raise WorkerError(
                ErrorCode.CHART_TIMING_CANDIDATE_FAILED,
                "standard and Super Timing candidates failed structural validation",
                context={"errors": errors, "seeds": seeds},
            ) from error

        if review.action in {
            TimingAuthorityAction.PASS,
            TimingAuthorityAction.REVIEW,
        } and not retry_with_super:
            return SongTimingAuthority(
                reference_path=reference_path,
                sha256=sha256_file(reference_path),
                audio_sha256=prepared.normalized.sha256,
                bpm_events=generated.bpm_events,
                generator_name=generated.generator_name,
                seed=generated.seed,
                mode=generated.mode,
                attempt_count=attempt_count,
                tempo_metrics=metrics,
                review=review,
            )
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

    raise AssertionError("timing generation attempts were unexpectedly exhausted")
