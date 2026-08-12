"""Cross-difficulty intro-family evidence, recovery, and publication blocking."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

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
from chart_worker.generation.intro_recovery import execute_intro_retry
from chart_worker.generation.mapperatorinator import ChartGenerator, GeneratedChart
from chart_worker.schema.types import DIFFICULTIES
from chart_worker.stages.types import PreparedAudio, SongTimingAuthority
from chart_worker.validation.difficulty_order import DifficultyOrderReview
from chart_worker.validation.intro_phrase_family import (
    IntroPhraseChartView,
    IntroPhraseFamilyReview,
    review_intro_phrase_pair,
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
    return review_intro_phrase_pair(
        hard,
        expert,
        start_delta_beats=start_delta_beats,
    )


def intro_phrase_family_reviews(
    selections: list[Selection],
    *,
    song_context: SongAnalysisContext,
    run_dir: Path,
) -> tuple[IntroPhraseFamilyReview, ...]:
    return tuple(
        intro_phrase_pair_review(
            states,
            assignment,
            song_context=song_context,
            run_dir=run_dir,
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
) -> tuple[list[Selection], tuple[IntroPhraseFamilyReview, ...]]:
    """Repair only the high-confidence isolated EXPERT first-row defect."""

    updated = list(selections)
    initial_reviews = intro_phrase_family_reviews(
        updated,
        song_context=song_context,
        run_dir=run_dir,
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

    reviews = intro_phrase_family_reviews(
        updated,
        song_context=song_context,
        run_dir=run_dir,
    )
    unresolved = sorted(
        (review for review in reviews if review.status == "DEFECT"),
        key=lambda review: (
            review.gap_delta_ms or 0,
            review.gap_ratio or 0.0,
            -review.hard.key_mode,
        ),
        reverse=True,
    )
    if unresolved and allow_model_retry:
        before = unresolved[0]
        key_mode = before.hard.key_mode
        family_index = next(
            index
            for index, (states, _assignment, _review) in enumerate(updated)
            if states["EXPERT"].key_mode == key_mode
        )
        states, original_assignment, _order_review = updated[family_index]
        state = states["EXPERT"]
        source = original_assignment["EXPERT"]
        if source is not None:
            repaired = execute_intro_retry(
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
                recovery_reason="INTRO_PHRASE_FAMILY_DEFECT",
                evidence_prefix="INTRO_PHRASE",
                workdir_name="intro-phrase-retry",
            )
            if repaired is not None:
                state.candidates.admit(repaired)
                assignment = dict(original_assignment)
                assignment["EXPERT"] = repaired
                order_review = _family_review(assignment)
                if order_review is not None and order_review.status == "RETRY":
                    state.attempt_evidence.append(
                        {
                            "reason": "INTRO_PHRASE_RETRY_ORDER_REJECTED",
                            "seed": repaired.seed,
                            "difficultyOrder": order_review.to_report(),
                        }
                    )
                else:
                    after = intro_phrase_pair_review(
                        states,
                        assignment,
                        song_context=song_context,
                        run_dir=run_dir,
                    )
                    if after.status not in {"DEFECT", "INSUFFICIENT"}:
                        if candidate_replacement_allowed(
                            state,
                            source,
                            repaired,
                            stage="INTRO_PHRASE_RETRY_RESELECT",
                            objective_improved=True,
                        ):
                            state.attempt_evidence.append(
                                {
                                    "reason": "INTRO_PHRASE_RETRY_SELECTED",
                                    "sourceSeed": source.seed,
                                    "selectedSeed": repaired.seed,
                                    "before": before.to_report(),
                                    "after": after.to_report(),
                                }
                            )
                            updated[family_index] = (states, assignment, order_review)
                    else:
                        state.attempt_evidence.append(
                            {
                                "reason": "INTRO_PHRASE_RETRY_NOT_IMPROVED",
                                "sourceSeed": source.seed,
                                "rejectedSeed": repaired.seed,
                                "before": before.to_report(),
                                "after": after.to_report(),
                            }
                        )

    reviews = intro_phrase_family_reviews(
        updated,
        song_context=song_context,
        run_dir=run_dir,
    )
    blocked: dict[int, IntroPhraseFamilyReview] = {}
    if not block_unresolved:
        return updated, reviews
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
        state.publication_block_reason = "INTRO_PHRASE_DEFECT_UNRESOLVED"
        state.attempt_evidence.append(
            {
                "reason": "INTRO_PHRASE_DEFECT_PUBLICATION_BLOCKED",
                "selectedSeed": source.seed if source is not None else None,
                "review": review.to_report(),
            }
        )
        blocked[key_mode] = review
        assignment = dict(assignment)
        assignment["EXPERT"] = None
        updated[family_index] = (states, assignment, _family_review(assignment))

    final_reviews = intro_phrase_family_reviews(
        updated,
        song_context=song_context,
        run_dir=run_dir,
    )
    return updated, tuple(
        blocked.get(review.hard.key_mode, review) for review in final_reviews
    )
