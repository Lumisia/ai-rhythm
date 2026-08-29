"""Cross-difficulty intro-family evidence, recovery, and publication blocking."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

from chart_worker.analysis.intro_anchor import GRID_SUPPORT_WINDOW_MS
from chart_worker.analysis.onset import OnsetAnalysis
from chart_worker.analysis.song_context import SongAnalysisContext
from chart_worker.generation.candidate_state import (
    Candidate,
    VariantState,
    candidate_replacement_allowed,
)
from chart_worker.generation.family_selection import (
    candidate_stable_id,
    first_row_ms,
    review_candidates,
)
from chart_worker.generation.generation_control import AdditionalInferenceBudget
from chart_worker.generation.intro_recovery import (
    execute_intro_retry,
    intro_phrase_recovery_end_ms,
    intro_region_recovery_end_ms,
)
from chart_worker.generation.intro_required_gameplay import (
    plan_intro_required_gameplay_interval,
)
from chart_worker.generation.mapperatorinator import ChartGenerator, GeneratedChart
from chart_worker.generation.partial_remap import PartialRemapWindow
from chart_worker.generation.required_gameplay_interval import (
    RequiredGameplayIntervalMode,
    tempo_map_addresses,
)
from chart_worker.schema.types import DIFFICULTIES
from chart_worker.stages.types import PreparedAudio, SongTimingAuthority
from chart_worker.validation.difficulty_order import DifficultyOrderReview
from chart_worker.validation.intro_phrase_family import (
    IntroPhraseChartView,
    IntroPhraseFamilyReview,
    corroborate_intro_phrase_review,
    review_intro_phrase_pair,
)
from chart_worker.validation.intro_region_contract import (
    IntroRegionCandidateReview,
    IntroRegionContract,
    build_intro_region_contract,
    review_intro_region_candidate,
)
from chart_worker.validation.intro_start_contract import IntroCandidateView
from chart_worker.validation.quality_gate import ChartAcceptance

Selection = tuple[
    dict[str, VariantState],
    dict[str, Candidate | None],
    DifficultyOrderReview | None,
]
EvaluateCandidate = Callable[..., ChartAcceptance]
SerializeCandidate = Callable[..., str]
IntroAnchorCovered = Callable[[GeneratedChart, SongTimingAuthority], bool | None]


@dataclass(frozen=True, slots=True)
class IntroRecoveryTarget:
    key_mode: int
    difficulty: str
    kind: Literal["REGION", "PHRASE"]
    partial_end_ms: int
    severity_ms: int
    before_report: dict[str, object]


def _region_defects(
    selections: list[Selection],
    contract: IntroRegionContract,
) -> tuple[tuple[int, str, IntroRegionCandidateReview], ...]:
    defects: list[tuple[int, str, IntroRegionCandidateReview]] = []
    for states, assignment, _review in selections:
        for difficulty in DIFFICULTIES:
            candidate = assignment[difficulty]
            if candidate is None:
                continue
            review = review_intro_region_candidate(
                contract,
                first_row_ms=first_row_ms(candidate),
            )
            if review.status == "DEFECT":
                defects.append((states[difficulty].key_mode, difficulty, review))
    return tuple(defects)


def _record_unaddressed_intro_tempo(
    selections: list[Selection],
    contract: IntroRegionContract,
    authority: SongTimingAuthority,
) -> None:
    """Explain why a confirmed intro defect cannot use model recovery."""

    allowed = contract.allowed_first_row_ms
    if contract.status != "CONFIRMED" or allowed is None:
        return
    if tempo_map_addresses(allowed[1], authority.bpm_events):
        return

    reason = "INTRO_REGION_RECOVERY_SKIPPED_UNADDRESSED_TEMPO"
    for key_mode, difficulty, _review in _region_defects(selections, contract):
        state = next(
            states[difficulty]
            for states, _assignment, _order_review in selections
            if states[difficulty].key_mode == key_mode
        )
        if any(evidence.get("reason") == reason for evidence in state.attempt_evidence):
            continue
        state.attempt_evidence.append(
            {
                "reason": reason,
                "introRegionEndMs": allowed[1],
                "firstTimingEventMs": authority.bpm_events[0].time_ms,
                "modelRetryAttempted": False,
                "preservedSelectedCandidate": True,
            }
        )


def intro_recovery_targets(
    selections: list[Selection],
    *,
    song_context: SongAnalysisContext,
    authority: SongTimingAuthority,
    duration_ms: int,
    run_dir: Path,
) -> tuple[IntroRecoveryTarget, ...]:
    """Rank identity-free intro defects after zero-cost candidate selection."""

    contract = build_intro_region_contract(song_context)
    targets: list[IntroRecoveryTarget] = []
    allowed = contract.allowed_first_row_ms
    if contract.status == "CONFIRMED" and allowed is not None:
        region_end_ms = intro_region_recovery_end_ms(
            allowed[1],
            authority.bpm_events,
            duration_ms=duration_ms,
        )
        if region_end_ms is not None:
            for key_mode, difficulty, review in _region_defects(selections, contract):
                severity_ms = (
                    review.lateness_ms
                    if review.lateness_ms is not None
                    else duration_ms
                )
                targets.append(
                    IntroRecoveryTarget(
                        key_mode=key_mode,
                        difficulty=difficulty,
                        kind="REGION",
                        partial_end_ms=region_end_ms,
                        severity_ms=severity_ms,
                        before_report=review.to_report(),
                    )
                )

    # Preserve the older, deliberately narrow isolated-first-row detector as
    # an independent fallback.  It catches an early ghost row followed by a
    # long gap, which a first-row region check alone cannot see.
    for review in intro_phrase_family_reviews(
        selections,
        song_context=song_context,
        run_dir=run_dir,
        intro_region_contract=contract,
    ):
        if review.status != "DEFECT":
            continue
        partial_end_ms = intro_phrase_recovery_end_ms(
            review.expert.second_row_ms,
            authority.bpm_events,
            duration_ms=duration_ms,
        )
        if partial_end_ms is None:
            continue
        targets.append(
            IntroRecoveryTarget(
                key_mode=review.hard.key_mode,
                difficulty="EXPERT",
                kind="PHRASE",
                partial_end_ms=partial_end_ms,
                severity_ms=review.gap_delta_ms or 0,
                before_report=review.to_report(),
            )
        )

    return tuple(
        sorted(
            targets,
            key=lambda target: (
                0 if target.kind == "REGION" else 1,
                -target.severity_ms,
                target.key_mode,
                DIFFICULTIES.index(target.difficulty),
            ),
        )
    )


def intro_candidate_view(
    candidate: Candidate,
    song_context: SongAnalysisContext,
) -> IntroCandidateView:
    first = first_row_ms(candidate)
    audio_supported = first is not None and any(
        abs(onset_ms - first) <= GRID_SUPPORT_WINDOW_MS
        for onset_ms in song_context.onset_analysis.onset_ms
    )
    anchor = song_context.intro_anchor
    if (
        first is not None
        and anchor.status == "CONFIRMED"
        and anchor.anchor_ms is not None
        and abs(anchor.anchor_ms - first) <= GRID_SUPPORT_WINDOW_MS
    ):
        audio_supported = True
    return IntroCandidateView(
        key_mode=candidate.request.key_mode,
        difficulty=candidate.request.difficulty,
        first_row_ms=first,
        seed=candidate.seed,
        raw_supported=first is not None,
        audio_supported=audio_supported,
    )


def intro_phrase_view(
    candidate: Candidate | None,
    *,
    key_mode: int,
    difficulty: str,
    song_context: SongAnalysisContext,
    run_dir: Path | None,
) -> IntroPhraseChartView:
    if candidate is None:
        return IntroPhraseChartView(
            key_mode=key_mode,
            difficulty=difficulty,
            first_row_ms=None,
            second_row_ms=None,
            post_first_gap_beats=None,
        )
    rows = tuple(sorted({note.time_ms for note in candidate.generated.notes}))
    first = rows[0] if rows else None
    second = rows[1] if len(rows) >= 2 else None
    gap_beats = (
        round(song_context.tempo_map.beats_between(first, second), 6)
        if first is not None and second is not None
        else None
    )
    intro = intro_candidate_view(candidate, song_context)
    return IntroPhraseChartView(
        key_mode=key_mode,
        difficulty=difficulty,
        first_row_ms=first,
        second_row_ms=second,
        post_first_gap_beats=gap_beats,
        first_row_audio_supported=intro.audio_supported,
        candidate_id=candidate_stable_id(
            candidate,
            key_mode=key_mode,
            difficulty=difficulty,
            run_dir=run_dir,
        ),
        seed=candidate.seed,
        attempt=candidate.attempt,
    )


def intro_phrase_pair_review(
    states: dict[str, VariantState],
    assignment: dict[str, Candidate | None],
    *,
    song_context: SongAnalysisContext,
    run_dir: Path | None,
    intro_region_contract: IntroRegionContract | None = None,
) -> IntroPhraseFamilyReview:
    key_mode = states["HARD"].key_mode
    hard = intro_phrase_view(
        assignment["HARD"],
        key_mode=key_mode,
        difficulty="HARD",
        song_context=song_context,
        run_dir=run_dir,
    )
    expert = intro_phrase_view(
        assignment["EXPERT"],
        key_mode=key_mode,
        difficulty="EXPERT",
        song_context=song_context,
        run_dir=run_dir,
    )
    start_delta_beats = (
        round(
            song_context.tempo_map.beats_between(
                min(hard.first_row_ms, expert.first_row_ms),
                max(hard.first_row_ms, expert.first_row_ms),
            ),
            6,
        )
        if hard.first_row_ms is not None and expert.first_row_ms is not None
        else None
    )
    review = review_intro_phrase_pair(
        hard,
        expert,
        start_delta_beats=start_delta_beats,
    )
    contract = (
        intro_region_contract
        if intro_region_contract is not None
        else build_intro_region_contract(song_context)
    )
    return corroborate_intro_phrase_review(
        review,
        hard_region=review_intro_region_candidate(
            contract,
            first_row_ms=hard.first_row_ms,
        ),
        expert_region=review_intro_region_candidate(
            contract,
            first_row_ms=expert.first_row_ms,
        ),
    )


def intro_phrase_family_reviews(
    selections: list[Selection],
    *,
    song_context: SongAnalysisContext,
    run_dir: Path,
    intro_region_contract: IntroRegionContract | None = None,
) -> tuple[IntroPhraseFamilyReview, ...]:
    contract = (
        intro_region_contract
        if intro_region_contract is not None
        else build_intro_region_contract(song_context)
    )
    return tuple(
        intro_phrase_pair_review(
            states,
            assignment,
            song_context=song_context,
            run_dir=run_dir,
            intro_region_contract=contract,
        )
        for states, assignment, _review in selections
    )


def _family_review(
    assignment: dict[str, Candidate | None],
) -> DifficultyOrderReview | None:
    chosen = tuple(
        candidate
        for difficulty in DIFFICULTIES
        if (candidate := assignment[difficulty]) is not None
    )
    return review_candidates(chosen) if chosen else None


def _reselect_existing_intro_region_candidates(
    selections: list[Selection],
    contract: IntroRegionContract,
) -> list[Selection]:
    """Use an already-generated same-slot candidate before spending inference."""

    updated = list(selections)
    for family_index, (states, original_assignment, _review) in enumerate(updated):
        for difficulty in DIFFICULTIES:
            state = states[difficulty]
            source = original_assignment[difficulty]
            if source is None:
                continue
            before = review_intro_region_candidate(
                contract,
                first_row_ms=first_row_ms(source),
            )
            if before.status != "DEFECT":
                continue
            for option in sorted(
                (
                    candidate
                    for candidate in state.candidates.admitted
                    if candidate is not source
                ),
                key=lambda candidate: (candidate.attempt, candidate.seed),
            ):
                after = review_intro_region_candidate(
                    contract,
                    first_row_ms=first_row_ms(option),
                )
                if after.status != "PASS":
                    continue
                assignment = dict(original_assignment)
                assignment[difficulty] = option
                order_review = _family_review(assignment)
                if order_review is not None and order_review.status == "RETRY":
                    continue
                if not candidate_replacement_allowed(
                    state,
                    source,
                    option,
                    stage="INTRO_REGION_EXISTING_RESELECT",
                    objective_improved=True,
                ):
                    continue
                state.attempt_evidence.append(
                    {
                        "reason": "INTRO_REGION_EXISTING_CANDIDATE_RESELECTED",
                        "sourceSeed": source.seed,
                        "selectedSeed": option.seed,
                        "before": before.to_report(),
                        "after": after.to_report(),
                    }
                )
                original_assignment = assignment
                updated[family_index] = (states, assignment, order_review)
                break
    return updated


def apply_intro_phrase_family_recovery(
    selections: list[Selection],
    song_context: SongAnalysisContext,
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
    block_unresolved: bool = True,
    model_retry_target: tuple[int, str] | None = None,
) -> tuple[list[Selection], tuple[IntroPhraseFamilyReview, ...]]:
    """Repair audio-confirmed intro defects across every key and difficulty."""

    intro_region_contract = build_intro_region_contract(song_context)
    updated = _reselect_existing_intro_region_candidates(
        selections,
        intro_region_contract,
    )
    _record_unaddressed_intro_tempo(
        updated,
        intro_region_contract,
        authority,
    )
    initial_reviews = intro_phrase_family_reviews(
        updated,
        song_context=song_context,
        run_dir=run_dir,
        intro_region_contract=intro_region_contract,
    )

    for before in initial_reviews:
        if before.status != "DEFECT":
            continue
        key_mode = before.hard.key_mode
        family_index = next(
            index
            for index, (states, _assignment, _review) in enumerate(updated)
            if states["EXPERT"].key_mode == key_mode
        )
        states, original_assignment, _order_review = updated[family_index]
        state = states["EXPERT"]
        source = original_assignment["EXPERT"]
        if source is None:
            continue
        for option in sorted(
            (
                candidate
                for candidate in state.candidates.admitted
                if candidate is not source
            ),
            key=lambda candidate: (candidate.attempt, candidate.seed),
        ):
            assignment = dict(original_assignment)
            assignment["EXPERT"] = option
            order_review = _family_review(assignment)
            if order_review is not None and order_review.status == "RETRY":
                continue
            after = intro_phrase_pair_review(
                states,
                assignment,
                song_context=song_context,
                run_dir=run_dir,
                intro_region_contract=intro_region_contract,
            )
            if after.status in {"DEFECT", "INSUFFICIENT"}:
                continue
            if not candidate_replacement_allowed(
                state,
                source,
                option,
                stage="INTRO_PHRASE_EXISTING_RESELECT",
                objective_improved=True,
            ):
                continue
            state.attempt_evidence.append(
                {
                    "reason": "INTRO_PHRASE_EXISTING_CANDIDATE_RESELECTED",
                    "sourceSeed": source.seed,
                    "selectedSeed": option.seed,
                    "before": before.to_report(),
                    "after": after.to_report(),
                }
            )
            updated[family_index] = (states, assignment, order_review)
            break

    targets = intro_recovery_targets(
        updated,
        song_context=song_context,
        authority=authority,
        duration_ms=prepared.normalized.duration_ms,
        run_dir=run_dir,
    )
    target = next(
        (
            item
            for item in targets
            if model_retry_target is None
            or (item.key_mode, item.difficulty) == model_retry_target
        ),
        None,
    )
    if target is not None and allow_model_retry:
        family_index = next(
            index
            for index, (states, _assignment, _review) in enumerate(updated)
            if states[target.difficulty].key_mode == target.key_mode
        )
        states, original_assignment, _order_review = updated[family_index]
        state = states[target.difficulty]
        source = original_assignment[target.difficulty]
        if source is not None:
            partial_window = PartialRemapWindow(0, target.partial_end_ms)
            interval_decision = plan_intro_required_gameplay_interval(
                intro_region_contract,
                partial_window=partial_window,
                timing_authority_digest=authority.sha256,
                mode=RequiredGameplayIntervalMode.SHADOW_ENFORCE,
            )
            state.attempt_evidence.append(
                {
                    "reason": "INTRO_REQUIRED_GAMEPLAY_INTERVAL_PLANNED",
                    "decision": interval_decision.reason,
                    "introRegion": intro_region_contract.to_report(),
                    "partialWindow": {
                        "startMs": partial_window.start_ms,
                        "endMs": partial_window.end_ms,
                    },
                    "requiredGameplayInterval": (
                        {
                            "startMs": interval_decision.interval.start_ms,
                            "endMs": interval_decision.interval.end_ms,
                            "evidenceClass": (
                                interval_decision.interval.evidence_class.value
                            ),
                            "evidenceDigest": (
                                interval_decision.interval.evidence_digest
                            ),
                            "mode": interval_decision.interval.mode.value,
                        }
                        if interval_decision.interval is not None
                        else None
                    ),
                }
            )
            is_region = target.kind == "REGION"
            evidence_prefix = "INTRO_REGION" if is_region else "INTRO_PHRASE"
            recovery_reason = (
                "INTRO_REGION_DEFECT"
                if is_region
                else "INTRO_PHRASE_FAMILY_DEFECT"
            )
            repaired = (
                execute_intro_retry(
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
                    recovery_reason=recovery_reason,
                    evidence_prefix=evidence_prefix,
                    workdir_name=(
                        "intro-region-retry" if is_region else "intro-phrase-retry"
                    ),
                    partial_end_ms=target.partial_end_ms,
                    required_gameplay_interval=interval_decision.interval,
                )
                if interval_decision.interval is not None
                else None
            )
            if repaired is not None:
                state.candidates.admit(repaired)
                assignment = dict(original_assignment)
                assignment[target.difficulty] = repaired
                order_review = _family_review(assignment)
                if order_review is not None and order_review.status == "RETRY":
                    state.attempt_evidence.append(
                        {
                            "reason": f"{evidence_prefix}_RETRY_ORDER_REJECTED",
                            "seed": repaired.seed,
                            "difficultyOrder": order_review.to_report(),
                        }
                    )
                else:
                    if is_region:
                        after = review_intro_region_candidate(
                            intro_region_contract,
                            first_row_ms=first_row_ms(repaired),
                        )
                        improved = after.status == "PASS"
                    else:
                        after = intro_phrase_pair_review(
                            states,
                            assignment,
                            song_context=song_context,
                            run_dir=run_dir,
                            intro_region_contract=intro_region_contract,
                        )
                        improved = after.status not in {"DEFECT", "INSUFFICIENT"}
                    if improved:
                        if candidate_replacement_allowed(
                            state,
                            source,
                            repaired,
                            stage=f"{evidence_prefix}_RETRY_RESELECT",
                            objective_improved=True,
                        ):
                            state.attempt_evidence.append(
                                {
                                    "reason": f"{evidence_prefix}_RETRY_SELECTED",
                                    "sourceSeed": source.seed,
                                    "selectedSeed": repaired.seed,
                                    "before": target.before_report,
                                    "after": after.to_report(),
                                }
                            )
                            updated[family_index] = (states, assignment, order_review)
                    else:
                        state.attempt_evidence.append(
                            {
                                "reason": f"{evidence_prefix}_RETRY_NOT_IMPROVED",
                                "sourceSeed": source.seed,
                                "rejectedSeed": repaired.seed,
                                "before": target.before_report,
                                "after": after.to_report(),
                            }
                        )

    reviews = intro_phrase_family_reviews(
        updated,
        song_context=song_context,
        run_dir=run_dir,
        intro_region_contract=intro_region_contract,
    )
    blocked: dict[int, IntroPhraseFamilyReview] = {}
    if not block_unresolved:
        return updated, reviews

    for key_mode, difficulty, region_review in _region_defects(
        updated,
        intro_region_contract,
    ):
        family_index = next(
            index
            for index, (states, _assignment, _order_review) in enumerate(updated)
            if states[difficulty].key_mode == key_mode
        )
        states, assignment, _order_review = updated[family_index]
        state = states[difficulty]
        source = assignment[difficulty]
        state.publication_block_reason = "INTRO_REGION_DEFECT_UNRESOLVED"
        state.attempt_evidence.append(
            {
                "reason": "INTRO_REGION_DEFECT_PLAYTEST_ONLY",
                "selectedSeed": source.seed if source is not None else None,
                "selectedProvenance": source.provenance if source is not None else None,
                "review": region_review.to_report(),
            }
        )
        assignment = dict(assignment)
        if source is not None:
            if source.provenance in {"RAW_UNVERIFIED", "SAFE_FALLBACK"}:
                degraded = source
            else:
                degraded = replace(
                    source,
                    provenance="RAW_UNVERIFIED",
                    recovery_reason="INTRO_REGION_DEFECT_UNRESOLVED",
                )
                state.candidates.reject(degraded)
            assignment[difficulty] = degraded
        updated[family_index] = (states, assignment, _family_review(assignment))

    for review in reviews:
        if review.status != "DEFECT":
            continue
        key_mode = review.hard.key_mode
        family_index = next(
            index
            for index, (states, _assignment, _order_review) in enumerate(updated)
            if states["EXPERT"].key_mode == key_mode
        )
        states, assignment, _order_review = updated[family_index]
        state = states["EXPERT"]
        source = assignment["EXPERT"]
        if state.publication_block_reason == "INTRO_REGION_DEFECT_UNRESOLVED":
            blocked[key_mode] = review
            continue
        state.publication_block_reason = "INTRO_PHRASE_DEFECT_UNRESOLVED"
        state.attempt_evidence.append(
            {
                "reason": "INTRO_PHRASE_DEFECT_PLAYTEST_ONLY",
                "selectedSeed": source.seed if source is not None else None,
                "selectedProvenance": source.provenance if source is not None else None,
                "review": review.to_report(),
            }
        )
        blocked[key_mode] = review
        assignment = dict(assignment)
        if source is not None:
            if source.provenance in {"RAW_UNVERIFIED", "SAFE_FALLBACK"}:
                degraded = source
            else:
                degraded = replace(
                    source,
                    provenance="RAW_UNVERIFIED",
                    recovery_reason="INTRO_PHRASE_DEFECT_UNRESOLVED",
                )
                state.candidates.reject(degraded)
            assignment["EXPERT"] = degraded
        updated[family_index] = (states, assignment, _family_review(assignment))

    final_reviews = intro_phrase_family_reviews(
        updated,
        song_context=song_context,
        run_dir=run_dir,
        intro_region_contract=intro_region_contract,
    )
    return updated, tuple(
        blocked.get(review.hard.key_mode, review) for review in final_reviews
    )
