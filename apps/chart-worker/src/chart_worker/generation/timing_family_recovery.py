"""Cross-key timing-family review, reselection, and one bounded retry."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

from chart_worker.analysis.onset import OnsetAnalysis
from chart_worker.errors import WorkerError
from chart_worker.generation.candidate_state import (
    Candidate,
    VariantState,
    candidate_replacement_allowed,
)
from chart_worker.generation.family_selection import review_candidates
from chart_worker.generation.generation_control import (
    AdditionalInferenceBudget,
    RecoveryKind,
)
from chart_worker.generation.inference_execution import (
    error_report_json,
    record_candidate_event,
    record_gate_event,
    run_inference_with_journal,
)
from chart_worker.generation.mapperatorinator import ChartGenerator, GeneratedChart
from chart_worker.schema.types import DIFFICULTIES, KEY_MODES
from chart_worker.stages.types import PreparedAudio, SongTimingAuthority
from chart_worker.validation.difficulty_order import DifficultyOrderReview
from chart_worker.validation.generated_chart import (
    GeneratedChartValidationError,
    validate_generated_chart,
)
from chart_worker.validation.quality_gate import ChartAcceptance, GateAction
from chart_worker.validation.timing_authority import (
    TimingAuthorityValidationError,
    validate_timing_identity,
)
from chart_worker.validation.timing_family_review import (
    TimingFamilyCandidate,
    TimingFamilyReview,
    review_timing_family,
)

_VARIANT_COUNT = len(KEY_MODES) * len(DIFFICULTIES)

Selection = tuple[
    dict[str, VariantState],
    dict[str, Candidate | None],
    DifficultyOrderReview | None,
]
EvaluateCandidate = Callable[..., ChartAcceptance]
SerializeCandidate = Callable[..., str]
IntroAnchorCovered = Callable[[GeneratedChart, SongTimingAuthority], bool | None]


def timing_family_reviews(
    selections: list[Selection],
) -> tuple[TimingFamilyReview, ...]:
    reviews = []
    for difficulty in DIFFICULTIES:
        candidates = tuple(
            TimingFamilyCandidate(
                key_mode=candidate.request.key_mode,
                difficulty=difficulty,
                diagnostics=candidate.acceptance.timing,
            )
            for _states, assignment, _review in selections
            if (candidate := assignment[difficulty]) is not None
        )
        reviews.append(review_timing_family(candidates))
    return tuple(reviews)


def _timing_review_for(
    reviews: tuple[TimingFamilyReview, ...],
    difficulty: str,
) -> TimingFamilyReview | None:
    return next((review for review in reviews if review.difficulty == difficulty), None)


def _family_review(
    assignment: dict[str, Candidate | None],
) -> DifficultyOrderReview | None:
    chosen = tuple(
        candidate
        for difficulty in DIFFICULTIES
        if (candidate := assignment[difficulty]) is not None
    )
    return review_candidates(chosen) if chosen else None


def _try_timing_family_retry(
    state: VariantState,
    source: Candidate,
    *,
    prepared: PreparedAudio,
    authority: SongTimingAuthority,
    onset_analysis: OnsetAnalysis,
    run_dir: Path,
    generator: ChartGenerator,
    base_seed: int,
    authority_epoch: int,
    inference_budget: AdditionalInferenceBudget,
    evaluate_candidate: EvaluateCandidate,
    serialize_candidate: SerializeCandidate,
    intro_anchor_covered: IntroAnchorCovered,
) -> Candidate | None:
    """Generate one alternate only for a corroborated cross-key outlier."""

    if state.recovery.was_attempted(RecoveryKind.TIMING_FAMILY):
        return None
    if not inference_budget.consume():
        state.attempt_evidence.append(
            {
                "reason": "TIMING_FAMILY_RETRY_BUDGET_EXHAUSTED",
                "budgetLimit": inference_budget.limit,
                "budgetUsed": inference_budget.used,
            }
        )
        return None
    if not state.recovery.claim(RecoveryKind.TIMING_FAMILY):
        return None
    attempt = state.budget.next_attempt
    retry_seed = base_seed + state.flat_index + (attempt - 1) * _VARIANT_COUNT
    state.budget.reserve_additional_attempt(seed=retry_seed)
    request = replace(
        source.request,
        timing_reference_path=authority.reference_path,
        seed=retry_seed,
        partial_start_ms=None,
        partial_end_ms=None,
        add_to_beatmap=False,
    )
    workdir = (
        run_dir
        / "raw"
        / "work"
        / f"epoch-{authority_epoch}"
        / f"{state.key_mode}k-{state.difficulty.lower()}"
        / "timing-family-retry"
    )
    inference_completed = False
    try:
        generated = run_inference_with_journal(
            state,
            generator=generator,
            request=request,
            workdir=workdir,
            run_dir=run_dir,
            prepared=prepared,
            authority=authority,
            authority_epoch=authority_epoch,
            attempt=attempt,
            seed=retry_seed,
            purpose="TIMING_FAMILY_RETRY",
        )
        inference_completed = True
        state.budget.record_quality_attempt()
        acceptance = evaluate_candidate(
            generated,
            authority,
            onset_analysis,
            requested_key_mode=state.key_mode,
            requested_difficulty=state.difficulty,
            duration_ms=prepared.normalized.duration_ms,
            boundary_policy_mode=prepared.boundary_policy_mode,
        )
        record_gate_event(
            state,
            authority_epoch=authority_epoch,
            attempt=attempt,
            seed=retry_seed,
            purpose="TIMING_FAMILY_RETRY",
            acceptance=acceptance,
        )
        evidence = {
            "seed": retry_seed,
            "workdir": workdir.relative_to(run_dir).as_posix(),
            "reason": (
                "TIMING_FAMILY_RETRY_CANDIDATE"
                if acceptance.action is not GateAction.RETRY_MAP
                else "TIMING_FAMILY_RETRY_GATE_REJECTED"
            ),
            "gateReport": acceptance.to_report(),
        }
        state.attempt_evidence.append(evidence)
        if acceptance.action is GateAction.RETRY_MAP:
            state.attempt_errors.append(
                json.dumps(
                    evidence,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
            )
            record_candidate_event(
                state,
                admitted=False,
                authority_epoch=authority_epoch,
                attempt=attempt,
                seed=retry_seed,
                purpose="TIMING_FAMILY_RETRY",
                reason="QUALITY_GATE_RETRY",
                acceptance=acceptance,
            )
            return None
        validate_timing_identity(generated.bpm_events, authority.bpm_events)
        validate_generated_chart(
            generated,
            key_mode=state.key_mode,
            duration_ms=prepared.normalized.duration_ms,
            max_note_start_ms=request.max_note_start_ms,
        )
        record_candidate_event(
            state,
            admitted=True,
            authority_epoch=authority_epoch,
            attempt=attempt,
            seed=retry_seed,
            purpose="TIMING_FAMILY_RETRY",
            reason="TIMING_FAMILY_OUTLIER",
            acceptance=acceptance,
        )
        return Candidate(
            request=request,
            generated=generated,
            acceptance=acceptance,
            osu_text=serialize_candidate(
                generated,
                authority=authority,
                prepared=prepared,
                key_mode=state.key_mode,
            ),
            workdir=workdir,
            attempt=attempt,
            seed=retry_seed,
            provenance="RETRY",
            recovery_reason="TIMING_FAMILY_OUTLIER",
            intro_anchor_covered=intro_anchor_covered(generated, authority),
        )
    except (
        GeneratedChartValidationError,
        TimingAuthorityValidationError,
        WorkerError,
        OSError,
        RuntimeError,
        ValueError,
    ) as error:
        if inference_completed:
            record_candidate_event(
                state,
                admitted=False,
                authority_epoch=authority_epoch,
                attempt=attempt,
                seed=retry_seed,
                purpose="TIMING_FAMILY_RETRY",
                reason="VALIDATION_ERROR",
            )
        state.attempt_errors.append(error_report_json(error))
        return None


def apply_timing_family_recovery(
    selections: list[Selection],
    *,
    prepared: PreparedAudio,
    authority: SongTimingAuthority,
    onset_analysis: OnsetAnalysis,
    run_dir: Path,
    generator: ChartGenerator,
    base_seed: int,
    authority_epoch: int,
    inference_budget: AdditionalInferenceBudget,
    evaluate_candidate: EvaluateCandidate,
    serialize_candidate: SerializeCandidate,
    intro_anchor_covered: IntroAnchorCovered,
    allow_model_retry: bool = True,
) -> tuple[list[Selection], tuple[TimingFamilyReview, ...]]:
    """Reselect for free, then spend at most the shared one-call budget."""

    reviews = timing_family_reviews(selections)
    outliers = sorted(
        (review for review in reviews if review.status == "OUTLIER"),
        key=lambda review: (
            review.overall_sibling_gap or 0.0,
            review.longest_local_outlier_run,
        ),
        reverse=True,
    )
    if not outliers:
        return selections, reviews

    target_review = outliers[0]
    assert target_review.target_key_mode is not None
    assert target_review.difficulty is not None
    family_index = next(
        index
        for index, (states, _assignment, _review) in enumerate(selections)
        if states[target_review.difficulty].key_mode == target_review.target_key_mode
    )
    states, original_assignment, _order_review = selections[family_index]
    state = states[target_review.difficulty]
    source = original_assignment[target_review.difficulty]
    if source is None:
        return selections, reviews

    options = sorted(
        (
            candidate
            for candidate in state.candidates.admitted
            if candidate is not source
        ),
        key=lambda candidate: (
            candidate.provenance == "RAW_UNVERIFIED",
            candidate.attempt,
            candidate.seed,
        ),
    )
    for option in options:
        assignment = dict(original_assignment)
        assignment[target_review.difficulty] = option
        order_review = _family_review(assignment)
        if order_review is not None and order_review.status == "RETRY":
            continue
        trial = list(selections)
        trial[family_index] = (states, assignment, order_review)
        trial_reviews = timing_family_reviews(trial)
        after = _timing_review_for(trial_reviews, target_review.difficulty)
        if after is not None and after.status != "OUTLIER":
            if not candidate_replacement_allowed(
                state,
                source,
                option,
                stage="TIMING_FAMILY_EXISTING_RESELECT",
                objective_improved=True,
            ):
                continue
            state.attempt_evidence.append(
                {
                    "reason": "TIMING_FAMILY_EXISTING_CANDIDATE_RESELECTED",
                    "sourceSeed": source.seed,
                    "selectedSeed": option.seed,
                    "before": target_review.to_report(),
                    "after": after.to_report(),
                }
            )
            return trial, trial_reviews

    if not allow_model_retry:
        return selections, reviews

    repaired = _try_timing_family_retry(
        state,
        source,
        prepared=prepared,
        authority=authority,
        onset_analysis=onset_analysis,
        run_dir=run_dir,
        generator=generator,
        base_seed=base_seed,
        authority_epoch=authority_epoch,
        inference_budget=inference_budget,
        evaluate_candidate=evaluate_candidate,
        serialize_candidate=serialize_candidate,
        intro_anchor_covered=intro_anchor_covered,
    )
    if repaired is None:
        return selections, reviews
    state.candidates.admit(repaired)
    assignment = dict(original_assignment)
    assignment[target_review.difficulty] = repaired
    order_review = _family_review(assignment)
    if order_review is not None and order_review.status == "RETRY":
        state.attempt_evidence.append(
            {
                "reason": "TIMING_FAMILY_RETRY_ORDER_REJECTED",
                "seed": repaired.seed,
                "difficultyOrder": order_review.to_report(),
            }
        )
        return selections, reviews
    trial = list(selections)
    trial[family_index] = (states, assignment, order_review)
    trial_reviews = timing_family_reviews(trial)
    after = _timing_review_for(trial_reviews, target_review.difficulty)
    if after is None or after.status == "OUTLIER":
        state.attempt_evidence.append(
            {
                "reason": "TIMING_FAMILY_RETRY_NOT_IMPROVED",
                "seed": repaired.seed,
                "before": target_review.to_report(),
                "after": after.to_report() if after is not None else None,
            }
        )
        return selections, reviews
    if not candidate_replacement_allowed(
        state,
        source,
        repaired,
        stage="TIMING_FAMILY_RETRY_RESELECT",
        objective_improved=True,
    ):
        return selections, reviews
    state.attempt_evidence.append(
        {
            "reason": "TIMING_FAMILY_RETRY_SELECTED",
            "seed": repaired.seed,
            "before": target_review.to_report(),
            "after": after.to_report(),
        }
    )
    return trial, trial_reviews
