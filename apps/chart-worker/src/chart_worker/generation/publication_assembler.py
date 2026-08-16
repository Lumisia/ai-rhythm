"""Assemble selected candidates into stable raw files and report entries."""

from dataclasses import dataclass
from pathlib import Path

from chart_worker.errors import ErrorCode, WorkerError
from chart_worker.generation.candidate_state import (
    Candidate,
    VariantState,
    candidate_evidence,
)
from chart_worker.generation.diagnostic_fallback import (
    DiagnosticRawCandidate,
    select_diagnostic_candidate,
)
from chart_worker.generation.generation_control import RecoveryKind
from chart_worker.schema.types import DIFFICULTIES
from chart_worker.stages.types import (
    GeneratedVariant,
    MissingVariant,
    PreparedAudio,
    SongTimingAuthority,
)
from chart_worker.validation.difficulty_order import DifficultyOrderReview
from chart_worker.validation.generated_chart import GeneratedChartValidationError
from chart_worker.validation.serialized_candidate import validate_serialized_candidate
from chart_worker.validation.timing_authority import TimingAuthorityValidationError

Selection = tuple[
    dict[str, VariantState],
    dict[str, Candidate | None],
    DifficultyOrderReview | None,
]


@dataclass(frozen=True, slots=True)
class PublicationAssembly:
    variants: tuple[GeneratedVariant, ...]
    missing: tuple[MissingVariant, ...]
    diagnostic_raw_candidates: tuple[DiagnosticRawCandidate, ...] = ()


def _diagnostic_projection(
    state: VariantState,
    candidate: Candidate,
) -> DiagnosticRawCandidate:
    return DiagnosticRawCandidate.create(
        key_mode=state.key_mode,
        difficulty=state.difficulty,
        seed=candidate.seed,
        attempt=candidate.attempt,
        osu_text=candidate.osu_text,
        source_workdir=candidate.workdir,
        gate_report=candidate.acceptance.to_report(),
        attempt_errors=tuple(state.attempt_errors),
        attempt_evidence=tuple(state.attempt_evidence),
    )


def _record_unselected_candidates(
    state: VariantState,
    selected: Candidate | None,
    review: DifficultyOrderReview | None,
    *,
    run_dir: Path,
) -> None:
    for candidate in (
        *state.candidates.admitted,
        *state.candidates.raw_rejected,
        *state.candidates.safe_fallbacks,
    ):
        if candidate is selected:
            continue
        evidence = candidate_evidence(
            candidate,
            reason=(
                state.publication_block_reason
                if state.publication_block_reason is not None
                else "NOT_SELECTED_BEST_MONOTONIC_FAMILY"
                if selected is not None
                else "DROPPED_FOR_MONOTONICITY"
            ),
            run_dir=run_dir,
        )
        if review is not None:
            evidence["selectedDifficultyOrder"] = review.to_report()
        state.attempt_evidence.append(evidence)


def _promote_key_mode(
    candidates: tuple[Candidate, ...],
    *,
    prepared: PreparedAudio,
    authority: SongTimingAuthority,
    run_dir: Path,
) -> tuple[Path, ...]:
    raw_paths = tuple(
        run_dir
        / "raw"
        / f"{candidate.request.key_mode}k-{candidate.request.difficulty.lower()}.osu"
        for candidate in candidates
    )
    raw_paths[0].parent.mkdir(parents=True, exist_ok=True)
    try:
        for candidate, raw_path in zip(candidates, raw_paths, strict=True):
            raw_path.write_text(candidate.osu_text, encoding="utf-8")
            validate_serialized_candidate(
                raw_path.read_text(encoding="utf-8-sig"),
                candidate.generated,
                authority,
                prepared,
                candidate.request.key_mode,
            )
    except (
        GeneratedChartValidationError,
        TimingAuthorityValidationError,
        WorkerError,
        OSError,
    ) as error:
        for raw_path in raw_paths:
            raw_path.unlink(missing_ok=True)
        cause_code = (
            error.code.value if isinstance(error, WorkerError) else type(error).__name__
        )
        raise WorkerError(
            ErrorCode.CHART_CANDIDATES_EXHAUSTED,
            f"{candidates[0].request.key_mode}K selected charts failed stable promotion",
            context={
                "key_mode": candidates[0].request.key_mode,
                "failure_stage": "PROMOTION",
                "paths": [
                    raw_path.relative_to(run_dir).as_posix() for raw_path in raw_paths
                ],
                "selected_seeds": [candidate.seed for candidate in candidates],
                "cause_code": cause_code,
                "cause": str(error),
            },
        ) from error
    return raw_paths


def assemble_publication(
    selections: list[Selection],
    *,
    prepared: PreparedAudio,
    authority: SongTimingAuthority,
    run_dir: Path,
) -> PublicationAssembly:
    """Promote one immutable selection snapshot and clean up atomically on failure."""

    variants: list[GeneratedVariant] = []
    missing: list[MissingVariant] = []
    diagnostic_raw_candidates: list[DiagnosticRawCandidate] = []
    promoted_paths: list[Path] = []
    try:
        for states, assignment, order_review in selections:
            chosen = tuple(
                assignment[difficulty]
                for difficulty in DIFFICULTIES
                if assignment[difficulty] is not None
            )
            raw_paths: tuple[Path, ...] = ()
            if chosen:
                raw_paths = _promote_key_mode(
                    chosen,
                    prepared=prepared,
                    authority=authority,
                    run_dir=run_dir,
                )
                promoted_paths.extend(raw_paths)
            for candidate, raw_path in zip(chosen, raw_paths, strict=True):
                state = states[candidate.request.difficulty]
                _record_unselected_candidates(
                    state,
                    candidate,
                    order_review,
                    run_dir=run_dir,
                )
                variants.append(
                    GeneratedVariant(
                        key_mode=candidate.request.key_mode,
                        difficulty=candidate.request.difficulty,
                        requested_star=candidate.request.requested_star,
                        raw_osu_path=raw_path,
                        generated=candidate.generated,
                        cfg_scale=candidate.request.cfg_scale,
                        attempt=candidate.attempt,
                        attempt_errors=tuple(state.attempt_errors),
                        attempt_evidence=tuple(state.attempt_evidence),
                        timing_authority_sha256=authority.sha256,
                        acceptance=candidate.acceptance,
                        candidate_count=(
                            len(state.candidates.admitted)
                            + len(state.candidates.raw_rejected)
                            + len(state.candidates.safe_fallbacks)
                        ),
                        generation_attempt_count=(
                            state.budget.next_attempt
                            - 1
                            + int(
                                state.recovery.was_attempted(
                                    RecoveryKind.PARTIAL_REMAP
                                )
                            )
                        ),
                        selected_seed=candidate.seed,
                        difficulty_order=order_review,
                        provenance=candidate.provenance,
                        recovery_reason=candidate.recovery_reason,
                    )
                )
            for difficulty in DIFFICULTIES:
                if assignment[difficulty] is not None:
                    continue
                state = states[difficulty]
                had_verified_candidates = bool(state.candidates.admitted)
                had_rejected_raw = bool(state.candidates.raw_rejected)
                if had_verified_candidates or had_rejected_raw:
                    _record_unselected_candidates(
                        state,
                        None,
                        order_review,
                        run_dir=run_dir,
                    )
                if had_rejected_raw:
                    diagnostic_raw_candidates.append(
                        select_diagnostic_candidate(
                            tuple(
                                _diagnostic_projection(state, candidate)
                                for candidate in state.candidates.raw_rejected
                            ),
                            key_mode=state.key_mode,
                            difficulty=difficulty,
                        )
                    )
                missing.append(
                    MissingVariant(
                        key_mode=state.key_mode,
                        difficulty=difficulty,
                        reason=(
                            state.publication_block_reason
                            if state.publication_block_reason is not None
                            else "DROPPED_FOR_MONOTONICITY"
                            if had_verified_candidates
                            else "QUALITY_GATE_REJECTED"
                            if had_rejected_raw
                            else "NO_PUBLISHABLE_CANDIDATE"
                        ),
                        attempt_errors=tuple(state.attempt_errors),
                        attempt_evidence=tuple(state.attempt_evidence),
                    )
                )
    except Exception:
        for raw_path in promoted_paths:
            raw_path.unlink(missing_ok=True)
        raise

    if not variants:
        first_exhausted = next(
            (
                state.exhausted_error
                for states, _, _ in selections
                for state in states.values()
                if state.exhausted_error is not None
            ),
            None,
        )
        if first_exhausted is not None:
            raise first_exhausted
        raise WorkerError(
            ErrorCode.CHART_CANDIDATES_EXHAUSTED,
            "no variant produced a publishable candidate",
            context={"missing": [entry.to_report() for entry in missing]},
        )

    return PublicationAssembly(
        variants=tuple(variants),
        missing=tuple(missing),
        diagnostic_raw_candidates=tuple(diagnostic_raw_candidates),
    )
