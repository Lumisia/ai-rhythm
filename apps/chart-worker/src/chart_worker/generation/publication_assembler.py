"""Assemble selected candidates into stable raw files and report entries."""

import hashlib
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
from chart_worker.validation.difficulty_order import (
    DifficultyOrderReview,
    review_difficulty_order,
)
from chart_worker.validation.generated_chart import GeneratedChartValidationError
from chart_worker.validation.serialized_candidate import validate_serialized_candidate
from chart_worker.validation.timing_authority import TimingAuthorityValidationError

Selection = tuple[
    dict[str, VariantState],
    dict[str, Candidate | None],
    DifficultyOrderReview | None,
]


def classify_family_assignment_kinds(
    assignment: dict[str, Candidate | None],
) -> dict[str, str]:
    """Classify target labels without confusing object identity with payload identity."""
    selected = {
        difficulty: candidate
        for difficulty, candidate in assignment.items()
        if candidate is not None
    }
    by_payload: dict[str, list[str]] = {}
    for difficulty in DIFFICULTIES:
        candidate = selected.get(difficulty)
        if candidate is None:
            continue
        payload = hashlib.sha256(candidate.osu_text.encode("utf-8")).hexdigest()
        by_payload.setdefault(payload, []).append(difficulty)

    kinds: dict[str, str] = {}
    for target_difficulties in by_payload.values():
        primary_target = next(
            (
                target
                for target in target_difficulties
                if selected[target].request.difficulty == target
            ),
            target_difficulties[0],
        )
        for target in target_difficulties:
            candidate = selected[target]
            if target != primary_target:
                kinds[target] = "EMERGENCY_DUPLICATE"
            elif candidate.request.difficulty == target:
                kinds[target] = "ORIGINAL"
            else:
                kinds[target] = "REASSIGNED"
    return kinds


def validate_unique_family_payloads(
    assignment: dict[str, Candidate | None],
    *,
    key_mode: int,
) -> None:
    """Reject an unresolved duplicate family before any stable file is written."""

    by_payload: dict[str, list[str]] = {}
    for difficulty in DIFFICULTIES:
        candidate = assignment.get(difficulty)
        if candidate is None:
            continue
        payload = hashlib.sha256(candidate.osu_text.encode("utf-8")).hexdigest()
        by_payload.setdefault(payload, []).append(difficulty)
    duplicate_groups = tuple(
        tuple(difficulties)
        for difficulties in by_payload.values()
        if len(difficulties) > 1
    )
    if duplicate_groups:
        raise WorkerError(
            ErrorCode.CHART_CANDIDATES_EXHAUSTED,
            f"{key_mode}K selected family contains duplicate payload",
            context={
                "key_mode": key_mode,
                "failure_stage": "UNIQUE_FAMILY_PUBLICATION_GUARD",
                "duplicate_difficulty_groups": [
                    list(group) for group in duplicate_groups
                ],
            },
        )


def validate_monotonic_family_difficulty(
    review: DifficultyOrderReview | None,
    *,
    key_mode: int,
) -> None:
    """Require strict and materially separated relative label order."""

    if review is None or review.status == "RETRY" or review.ambiguous_pairs:
        raise WorkerError(
            ErrorCode.CHART_CANDIDATES_EXHAUSTED,
            f"{key_mode}K selected family difficulty is not strictly increasing",
            context={
                "key_mode": key_mode,
                "failure_stage": "MONOTONIC_FAMILY_PUBLICATION_GUARD",
                "difficulty_order": review.to_report() if review is not None else None,
            },
        )
    if review.narrow_pairs:
        raise WorkerError(
            ErrorCode.CHART_CANDIDATES_EXHAUSTED,
            f"{key_mode}K selected family difficulty is not materially separated",
            context={
                "key_mode": key_mode,
                "failure_stage": "DIFFICULTY_SEPARATION_PUBLICATION_GUARD",
                "difficulty_order": review.to_report(),
            },
        )


@dataclass(frozen=True, slots=True)
class PublicationAssembly:
    variants: tuple[GeneratedVariant, ...]
    missing: tuple[MissingVariant, ...]
    diagnostic_raw_candidates: tuple[DiagnosticRawCandidate, ...] = ()


def _review_effective_assignment(
    assignment: dict[str, Candidate | None],
) -> DifficultyOrderReview | None:
    profiles = {}
    for difficulty in DIFFICULTIES:
        candidate = assignment[difficulty]
        profile = (
            getattr(candidate.acceptance, "profile", None)
            if candidate is not None
            else None
        )
        if candidate is None or profile is None:
            return None
        profiles[difficulty] = profile.difficulty
    return review_difficulty_order(profiles)


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
    assignments: tuple[tuple[str, Candidate], ...],
    *,
    prepared: PreparedAudio,
    authority: SongTimingAuthority,
    run_dir: Path,
) -> tuple[Path, ...]:
    raw_paths = tuple(
        run_dir
        / "raw"
        / f"{candidate.request.key_mode}k-{target_difficulty.lower()}.osu"
        for target_difficulty, candidate in assignments
    )
    raw_paths[0].parent.mkdir(parents=True, exist_ok=True)
    try:
        for (_target_difficulty, candidate), raw_path in zip(
            assignments, raw_paths, strict=True
        ):
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
            f"{assignments[0][1].request.key_mode}K selected charts failed stable promotion",
            context={
                "key_mode": assignments[0][1].request.key_mode,
                "failure_stage": "PROMOTION",
                "paths": [
                    raw_path.relative_to(run_dir).as_posix() for raw_path in raw_paths
                ],
                "selected_seeds": [candidate.seed for _target, candidate in assignments],
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
    """Promote immutable key families and isolate unresolved family contracts."""

    variants: list[GeneratedVariant] = []
    missing: list[MissingVariant] = []
    diagnostic_raw_candidates: list[DiagnosticRawCandidate] = []
    promoted_paths: list[Path] = []
    try:
        for states, assignment, order_review in selections:
            key_mode = next(iter(states.values())).key_mode
            effective_assignment = assignment
            family_resolution_state = "RESOLVED"
            family_resolution_reasons: tuple[str, ...] = ()
            family_production_eligible = True
            if prepared.difficulty_family_resolution_enabled:
                try:
                    validate_unique_family_payloads(assignment, key_mode=key_mode)
                except WorkerError as error:
                    failure_stage = str(
                        error.context.get(
                            "failure_stage",
                            "DIFFICULTY_FAMILY_PUBLICATION_GUARD",
                        )
                    )
                    family_resolution_state = "UNRESOLVED"
                    family_resolution_reasons = (failure_stage,)
                    family_production_eligible = False
                    # Duplicate payloads are never exported under multiple
                    # labels. Keep one deterministic owner so later per-slot
                    # fallback resolution can fill only the affected slots.
                    seen_payloads: set[str] = set()
                    effective_assignment = {}
                    for difficulty in DIFFICULTIES:
                        candidate = assignment[difficulty]
                        if candidate is None:
                            effective_assignment[difficulty] = None
                            continue
                        payload = hashlib.sha256(
                            candidate.osu_text.encode("utf-8")
                        ).hexdigest()
                        effective_assignment[difficulty] = (
                            None if payload in seen_payloads else candidate
                        )
                        seen_payloads.add(payload)
                    for state in states.values():
                        if state.publication_block_reason is None:
                            state.publication_block_reason = failure_stage
                        state.attempt_evidence.append(
                            {
                                "reason": "DIFFICULTY_FAMILY_PUBLICATION_BLOCKED",
                                "failureStage": failure_stage,
                                "errorCode": error.code.value,
                                "context": dict(error.context),
                                "mutatesPublishedCharts": True,
                            }
                        )
                else:
                    try:
                        validate_monotonic_family_difficulty(
                            order_review,
                            key_mode=key_mode,
                        )
                    except WorkerError as error:
                        failure_stage = str(
                            error.context.get(
                                "failure_stage",
                                "MONOTONIC_FAMILY_PUBLICATION_GUARD",
                            )
                        )
                        family_resolution_state = "UNRESOLVED"
                        family_resolution_reasons = (failure_stage,)
                        family_production_eligible = False
                        for state in states.values():
                            if state.publication_block_reason is None:
                                state.publication_block_reason = failure_stage
                            state.attempt_evidence.append(
                                {
                                    "reason": "DIFFICULTY_FAMILY_PRODUCTION_BLOCKED",
                                    "failureStage": failure_stage,
                                    "errorCode": error.code.value,
                                    "context": dict(error.context),
                                    "mutatesPlaytestCharts": False,
                                }
                            )
                    else:
                        narrow_pairs = tuple(
                            getattr(order_review, "narrow_pairs", ())
                            if order_review is not None
                            else ()
                        )
                        if narrow_pairs:
                            family_resolution_state = "NARROW_REVIEW"
                            family_resolution_reasons = (
                                "NARROW_ADJACENT_DIFFICULTY_GAP",
                            )
                            family_production_eligible = False
            effective_assignment = dict(effective_assignment)
            raw_unverified_targets: set[str] = set()
            unresolved_safe_targets: set[str] = set()
            selected_payloads = {
                hashlib.sha256(candidate.osu_text.encode("utf-8")).hexdigest()
                for candidate in effective_assignment.values()
                if candidate is not None
            }
            for difficulty in DIFFICULTIES:
                if effective_assignment[difficulty] is not None:
                    continue
                state = states[difficulty]
                safe_alternatives = tuple(
                    candidate
                    for candidate in (
                        *state.candidates.admitted,
                        *state.candidates.safe_fallbacks,
                    )
                    if hashlib.sha256(
                        candidate.osu_text.encode("utf-8")
                    ).hexdigest()
                    not in selected_payloads
                )
                if safe_alternatives:
                    selected_candidate = safe_alternatives[0]
                    effective_assignment[difficulty] = selected_candidate
                    selected_payloads.add(
                        hashlib.sha256(
                            selected_candidate.osu_text.encode("utf-8")
                        ).hexdigest()
                    )
                    unresolved_safe_targets.add(difficulty)
                    state.attempt_evidence.append(
                        {
                            "reason": "UNRESOLVED_FAMILY_HARD_SAFE_PLAYTEST_RETURN",
                            "selectedSeed": selected_candidate.seed,
                            "selectedAttempt": selected_candidate.attempt,
                        }
                    )
                    continue
                projected = tuple(
                    (candidate, _diagnostic_projection(state, candidate))
                    for candidate in state.candidates.raw_rejected
                    if hashlib.sha256(candidate.osu_text.encode("utf-8")).hexdigest()
                    not in selected_payloads
                )
                if not projected:
                    continue
                selected_projection = select_diagnostic_candidate(
                    tuple(projection for _candidate, projection in projected),
                    key_mode=state.key_mode,
                    difficulty=difficulty,
                )
                selected_candidate = next(
                    candidate
                    for candidate, projection in projected
                    if projection is selected_projection
                )
                effective_assignment[difficulty] = selected_candidate
                selected_payloads.add(
                    hashlib.sha256(
                        selected_candidate.osu_text.encode("utf-8")
                    ).hexdigest()
                )
                raw_unverified_targets.add(difficulty)
                diagnostic_raw_candidates.append(selected_projection)
                state.attempt_evidence.append(
                    {
                        "reason": "QUALITY_REJECTED_HARD_SAFE_PLAYTEST_RETURN",
                        "selectedSeed": selected_candidate.seed,
                        "selectedAttempt": selected_candidate.attempt,
                    }
                )
            if raw_unverified_targets or unresolved_safe_targets:
                family_resolution_state = "UNRESOLVED"
                family_resolution_reasons = tuple(
                    sorted(
                        {
                            *family_resolution_reasons,
                            *(
                                {"QUALITY_REJECTED_HARD_SAFE_PLAYTEST_RETURN"}
                                if raw_unverified_targets
                                else set()
                            ),
                            *(
                                {"UNRESOLVED_FAMILY_HARD_SAFE_PLAYTEST_RETURN"}
                                if unresolved_safe_targets
                                else set()
                            ),
                        }
                    )
                )
                family_production_eligible = False
            effective_order_review = order_review
            if (
                prepared.difficulty_family_resolution_enabled
                and all(
                    effective_assignment[difficulty] is not None
                    for difficulty in DIFFICULTIES
                )
            ):
                effective_order_review = _review_effective_assignment(
                    effective_assignment
                )
                try:
                    validate_unique_family_payloads(
                        effective_assignment,
                        key_mode=key_mode,
                    )
                    validate_monotonic_family_difficulty(
                        effective_order_review,
                        key_mode=key_mode,
                    )
                except WorkerError as error:
                    failure_stage = str(
                        error.context.get(
                            "failure_stage",
                            "POST_FILL_FAMILY_PUBLICATION_GUARD",
                        )
                    )
                    family_resolution_state = "UNRESOLVED"
                    family_resolution_reasons = tuple(
                        sorted({*family_resolution_reasons, failure_stage})
                    )
                    family_production_eligible = False
                    for state in states.values():
                        state.attempt_evidence.append(
                            {
                                "reason": "POST_FILL_FAMILY_PUBLICATION_BLOCKED",
                                "failureStage": failure_stage,
                                "errorCode": error.code.value,
                                "context": dict(error.context),
                                "mutatesPlaytestCharts": False,
                            }
                        )
            chosen = tuple(
                (difficulty, effective_assignment[difficulty])
                for difficulty in DIFFICULTIES
                if effective_assignment[difficulty] is not None
            )
            assignment_kinds = classify_family_assignment_kinds(effective_assignment)
            raw_paths: tuple[Path, ...] = ()
            if chosen:
                raw_paths = _promote_key_mode(
                    chosen,
                    prepared=prepared,
                    authority=authority,
                    run_dir=run_dir,
                )
                promoted_paths.extend(raw_paths)
            for (target_difficulty, candidate), raw_path in zip(
                chosen, raw_paths, strict=True
            ):
                state = states[target_difficulty]
                _record_unselected_candidates(
                    state,
                    candidate,
                    effective_order_review,
                    run_dir=run_dir,
                )
                variants.append(
                    # Hard-safe quality rejections are playable evidence, not
                    # production candidates. Their raw provenance remains
                    # explicit even though the immutable payload is promoted
                    # through the normal chart export path.
                    GeneratedVariant(
                        key_mode=candidate.request.key_mode,
                        difficulty=target_difficulty,
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
                        difficulty_order=effective_order_review,
                        provenance=(
                            "RAW_UNVERIFIED"
                            if target_difficulty in raw_unverified_targets
                            else candidate.provenance
                        ),
                        family_assignment_kind=assignment_kinds[target_difficulty],
                        source_difficulty=candidate.request.difficulty,
                        recovery_reason=(
                            "QUALITY_REJECTED_HARD_SAFE_PLAYTEST_RETURN"
                            if target_difficulty in raw_unverified_targets
                            else candidate.recovery_reason
                        ),
                        coverage_repair_gap_count=(
                            candidate.coverage_repair_gap_count
                        ),
                        production_eligible=(
                            family_production_eligible
                            and assignment_kinds[target_difficulty] == "ORIGINAL"
                            and candidate.provenance
                            not in {"COVERAGE_REPAIR", "RAW_UNVERIFIED", "SAFE_FALLBACK"}
                        ),
                        family_resolution_state=family_resolution_state,
                        family_resolution_reasons=family_resolution_reasons,
                    )
                )
            for difficulty in DIFFICULTIES:
                if effective_assignment[difficulty] is not None:
                    continue
                state = states[difficulty]
                had_verified_candidates = bool(state.candidates.admitted)
                had_rejected_raw = bool(state.candidates.raw_rejected)
                if had_verified_candidates or had_rejected_raw:
                    _record_unselected_candidates(
                        state,
                        None,
                        effective_order_review,
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
