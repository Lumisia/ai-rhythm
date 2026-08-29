"""Legacy exact-row observation with free fallback when no region is known."""

from __future__ import annotations

from chart_worker.analysis.song_context import SongAnalysisContext
from chart_worker.generation.candidate_state import (
    Candidate,
    VariantState,
    candidate_replacement_allowed,
)
from chart_worker.generation.family_selection import first_row_ms, review_candidates
from chart_worker.generation.intro_family_recovery import (
    intro_candidate_view,
    intro_phrase_pair_review,
)
from chart_worker.schema.types import DIFFICULTIES
from chart_worker.validation.difficulty_order import DifficultyOrderReview
from chart_worker.validation.intro_start_contract import (
    IntroContractReview,
    IntroStartContract,
    build_intro_start_contract,
    validate_exact_first_row,
)

Selection = tuple[
    dict[str, VariantState],
    dict[str, Candidate | None],
    DifficultyOrderReview | None,
]


def _selected_candidates(selections: list[Selection]) -> tuple[Candidate, ...]:
    return tuple(
        candidate
        for _states, assignment, _review in selections
        for difficulty in DIFFICULTIES
        if (candidate := assignment[difficulty]) is not None
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


def try_exact_intro_candidate(
    state: VariantState,
    assignment: dict[str, Candidate | None],
    *,
    canonical_ms: int,
) -> Candidate | None:
    current = assignment[state.difficulty]
    if current is None:
        return None
    options = sorted(
        state.candidates.admitted,
        key=lambda candidate: (candidate.attempt, candidate.seed),
    )
    for candidate in options:
        if first_row_ms(candidate) != canonical_ms:
            continue
        trial = dict(assignment)
        trial[state.difficulty] = candidate
        review = _family_review(trial)
        if (
            (review is None or review.status != "RETRY")
            and candidate_replacement_allowed(
                state,
                current,
                candidate,
                stage="INTRO_EXACT_EXISTING_RESELECT",
                objective_improved=True,
            )
        ):
            return candidate
    return None


def apply_intro_start_contract(
    selections: list[Selection],
    song_context: SongAnalysisContext,
) -> tuple[list[Selection], IntroStartContract, IntroContractReview]:
    """Use exact matching only when adaptive region evidence is unavailable."""

    initial = _selected_candidates(selections)
    contract = build_intro_start_contract(
        song_context,
        tuple(intro_candidate_view(candidate, song_context) for candidate in initial),
    )
    canonical_ms = contract.canonical_first_row_ms
    correction_reasons: list[str] = []
    corrected_count = 0
    region_authoritative = (
        contract.intro_region is not None
        and contract.intro_region.status == "CONFIRMED"
    )
    correction_supported = not region_authoritative and (
        contract.audio_supported
        or (contract.raw_supported and contract.candidate_support_count >= 2)
    )
    updated = []
    for states, original_assignment, _original_review in selections:
        assignment = dict(original_assignment)
        if canonical_ms is not None and correction_supported:
            for difficulty in DIFFICULTIES:
                current = assignment[difficulty]
                if current is None or first_row_ms(current) == canonical_ms:
                    continue
                replacement = try_exact_intro_candidate(
                    states[difficulty],
                    assignment,
                    canonical_ms=canonical_ms,
                )
                if replacement is not None:
                    trial = dict(assignment)
                    trial[difficulty] = replacement
                    phrase_review = intro_phrase_pair_review(
                        states,
                        trial,
                        song_context=song_context,
                        run_dir=None,
                    )
                    if phrase_review.status == "DEFECT":
                        correction_reasons.append(
                            f"{states[difficulty].key_mode}K {difficulty}:"
                            "EXACT_CANDIDATE_REJECTED_INTRO_PHRASE_DEFECT"
                        )
                        continue
                    assignment = trial
                    corrected_count += 1
                    correction_reasons.append(
                        f"{states[difficulty].key_mode}K {difficulty}:"
                        "EXACT_CANDIDATE_RESELECTED"
                    )
                    continue
                correction_reasons.append(
                    f"{states[difficulty].key_mode}K {difficulty}:MUTATION_DISABLED"
                )
        updated.append((states, assignment, _family_review(assignment)))

    final_candidates = _selected_candidates(updated)
    review = validate_exact_first_row(
        contract,
        tuple(
            intro_candidate_view(candidate, song_context)
            for candidate in final_candidates
        ),
        corrected_count=corrected_count,
        correction_reasons=tuple(correction_reasons),
    )
    return updated, contract, review
