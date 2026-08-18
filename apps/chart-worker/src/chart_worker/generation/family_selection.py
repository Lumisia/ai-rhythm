"""Pure candidate-family selection and shadow-comparison policy.

Generation and recovery populate candidate repositories.  This module only
projects those candidates into deterministic family choices and diagnostics;
it performs no model calls and writes no artifacts.
"""

import hashlib
import json
from itertools import product
from pathlib import Path

from chart_worker.analysis.activity import SongBoundaryContract
from chart_worker.generation.candidate_state import Candidate, VariantState
from chart_worker.schema.types import DIFFICULTIES
from chart_worker.stages.types import PreparedAudio, SongTimingAuthority
from chart_worker.validation.difficulty_order import (
    DifficultyOrderReview,
    review_difficulty_order,
)
from chart_worker.validation.difficulty_selector import (
    DifficultyCandidateView,
    DifficultySelectionComparison,
    SelectionMode,
    compare_family_candidates,
)
from chart_worker.validation.intro_start_contract import IntroStartContract
from chart_worker.validation.quality_gate import QUALITY_GATE_VERSION, GateAction
from chart_worker.validation.song_family_selector import (
    CandidateSnapshot,
    ProtectedMetrics,
    SelectorMode,
    SongSelectionComparison,
    TimingSectionSnapshot,
    compare_song_families,
)

Selection = tuple[
    dict[str, VariantState],
    dict[str, Candidate | None],
    DifficultyOrderReview | None,
]


def review_candidates(candidates: tuple[Candidate, ...]) -> DifficultyOrderReview:
    profiles = {}
    for candidate in candidates:
        if candidate.acceptance.profile is None:
            raise ValueError("ladder candidate must carry a chart quality profile")
        profiles[candidate.request.difficulty] = candidate.acceptance.profile.difficulty
    return review_difficulty_order(profiles)


def family_score(
    assignment: tuple[Candidate | None, ...],
    review: DifficultyOrderReview | None,
) -> tuple[int, int, int, int, tuple[tuple[int, int], ...]]:
    """Return the deterministic lexicographic score; lower is preferred."""
    chosen = tuple(candidate for candidate in assignment if candidate is not None)
    missing = len(assignment) - len(chosen)
    raw = sum(1 for candidate in chosen if candidate.provenance == "RAW_UNVERIFIED")
    intro_misses = sum(
        1 for candidate in chosen if candidate.intro_anchor_covered is False
    )
    crowded = (
        len(review.narrow_pairs) + len(review.ambiguous_pairs)
        if review is not None
        else 0
    )
    order = tuple((candidate.attempt, candidate.seed) for candidate in chosen)
    return (missing, raw, intro_misses, crowded, order)


def has_complete_model_family(states: dict[str, VariantState]) -> bool:
    pools = tuple(
        tuple(states[difficulty].candidates.admitted) for difficulty in DIFFICULTIES
    )
    if any(not pool for pool in pools):
        return False
    return any(
        review_candidates(combo).status != "RETRY" for combo in product(*pools)
    )


def select_family(
    states: dict[str, VariantState],
) -> tuple[dict[str, Candidate | None], DifficultyOrderReview | None]:
    options_by_difficulty = []
    for difficulty in DIFFICULTIES:
        state = states[difficulty]
        pool = sorted(
            state.candidates.admitted,
            key=lambda candidate: (candidate.attempt, candidate.seed),
        )
        options_by_difficulty.append((*pool, None))

    best_assignment: dict[str, Candidate | None] | None = None
    best_review: DifficultyOrderReview | None = None
    best_score: tuple | None = None
    for combo in product(*options_by_difficulty):
        chosen = tuple(candidate for candidate in combo if candidate is not None)
        review: DifficultyOrderReview | None = None
        if chosen:
            review = review_candidates(chosen)
            if review.status == "RETRY":
                continue
        score = family_score(combo, review)
        if best_score is None or score < best_score:
            best_score = score
            best_assignment = dict(zip(DIFFICULTIES, combo, strict=True))
            best_review = review
    assert best_assignment is not None
    return best_assignment, best_review


def candidate_stable_id(
    candidate: Candidate,
    *,
    key_mode: int,
    difficulty: str,
    run_dir: Path | None = None,
) -> str:
    try:
        payload_ref = (
            candidate.workdir.relative_to(run_dir).as_posix()
            if run_dir is not None
            else candidate.workdir.as_posix()
        )
    except ValueError:
        payload_ref = candidate.workdir.as_posix()
    identity = {
        "keyMode": key_mode,
        "difficulty": difficulty,
        "attempt": candidate.attempt,
        "seed": candidate.seed,
        "provenance": candidate.provenance,
        "payloadRef": payload_ref,
    }
    digest = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:12]
    return (
        f"{key_mode}k:{difficulty}:a{candidate.attempt}:"
        f"s{candidate.seed}:{candidate.provenance}:{digest}"
    )


def compare_difficulty_selection(
    states: dict[str, VariantState],
    assignment: dict[str, Candidate | None],
    *,
    mode: SelectionMode,
) -> tuple[dict[str, Candidate | None], DifficultySelectionComparison]:
    pools: dict[str, tuple[DifficultyCandidateView, ...]] = {}
    current: dict[str, str | None] = {}
    candidate_ids: dict[int, str] = {}
    key_mode = next(iter(states.values())).key_mode
    for difficulty in DIFFICULTIES:
        candidates = tuple(states[difficulty].candidates.admitted)
        views = []
        for candidate in candidates:
            candidate_id = candidate_stable_id(
                candidate,
                key_mode=key_mode,
                difficulty=difficulty,
            )
            candidate_ids[id(candidate)] = candidate_id
            profile = candidate.acceptance.profile
            vector = profile.difficulty_vector_v2 if profile is not None else None
            views.append(
                DifficultyCandidateView(
                    candidate_id=candidate_id,
                    difficulty=difficulty,
                    seed=candidate.seed,
                    attempt=candidate.attempt,
                    provenance=candidate.provenance,
                    intro_anchor_covered=candidate.intro_anchor_covered,
                    current_rating=(
                        profile.difficulty.project_rating
                        if profile is not None
                        else float("inf")
                    ),
                    v2_ordering_score=(vector.ordering_score if vector is not None else None),
                    vector_v2=vector.to_report() if vector is not None else None,
                )
            )
        pools[difficulty] = tuple(views)
        selected = assignment[difficulty]
        current[difficulty] = (
            candidate_ids[id(selected)] if selected is not None else None
        )
    selected_ids, comparison = compare_family_candidates(
        pools,
        current,
        mode=mode,
    )
    if comparison is None:
        raise AssertionError("difficulty comparison is required outside CURRENT mode")
    by_candidate_id = {
        candidate_id: candidate
        for difficulty in DIFFICULTIES
        for candidate in states[difficulty].candidates.admitted
        if (candidate_id := candidate_ids[id(candidate)])
    }
    selected = {
        difficulty: (
            by_candidate_id[candidate_id]
            if (candidate_id := selected_ids[difficulty]) is not None
            else None
        )
        for difficulty in DIFFICULTIES
    }
    return selected, comparison


def song_selection_context_id(
    prepared: PreparedAudio,
    authority: SongTimingAuthority,
    boundary: SongBoundaryContract | None,
    intro_contract: IntroStartContract | None = None,
) -> str:
    payload = {
        "version": "song-family-selector-v2-shadow-2-policy-context",
        "audioSha256": authority.audio_sha256,
        "timingAuthoritySha256": authority.sha256,
        "durationMs": prepared.normalized.duration_ms,
        "difficultySelectorMode": prepared.difficulty_selector_mode,
        "qualityGateVersion": QUALITY_GATE_VERSION,
        "songBoundaryContractSha256": (
            boundary.stable_sha256() if boundary is not None else None
        ),
        "introStartContract": (
            intro_contract.to_report() if intro_contract is not None else None
        ),
    }
    serialized = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def first_row_ms(candidate: Candidate) -> int | None:
    return min((note.time_ms for note in candidate.generated.notes), default=None)


def compare_song_selection(
    selections: list[Selection],
    *,
    prepared: PreparedAudio,
    authority: SongTimingAuthority,
    run_dir: Path,
    intro_contract: IntroStartContract,
    boundary: SongBoundaryContract | None,
    mode: SelectorMode,
) -> tuple[list[Selection], SongSelectionComparison]:
    context_id = song_selection_context_id(
        prepared,
        authority,
        boundary,
        intro_contract,
    )
    pools: dict[tuple[int, str], tuple[CandidateSnapshot, ...]] = {}
    current: dict[tuple[int, str], str | None] = {}
    candidates_by_id: dict[str, Candidate] = {}
    for states, assignment, _review in selections:
        key_mode = next(iter(states.values())).key_mode
        for difficulty in DIFFICULTIES:
            state = states[difficulty]
            snapshots = []
            ids: dict[int, str] = {}
            for candidate in state.candidates.playtest_candidates:
                candidate_id = candidate_stable_id(
                    candidate,
                    key_mode=key_mode,
                    difficulty=difficulty,
                    run_dir=run_dir,
                )
                if candidate_id in ids.values():
                    raise ValueError(f"duplicate deterministic candidate id: {candidate_id}")
                ids[id(candidate)] = candidate_id
                candidates_by_id[candidate_id] = candidate
                acceptance = candidate.acceptance
                profile = acceptance.profile
                vector = profile.difficulty_vector_v2 if profile is not None else None
                snapshots.append(
                    CandidateSnapshot(
                        candidate_id=candidate_id,
                        context_id=context_id,
                        key_mode=key_mode,
                        difficulty=difficulty,
                        attempt=candidate.attempt,
                        seed=candidate.seed,
                        provenance=candidate.provenance,
                        hard_eligible=(
                            acceptance.action is not GateAction.RETRY_MAP
                            and candidate.provenance
                            not in {"RAW_UNVERIFIED", "SAFE_FALLBACK"}
                        ),
                        axis_actions=tuple(
                            (decision.axis.value, decision.action.value)
                            for decision in acceptance.decisions
                        ),
                        protected_metrics=ProtectedMetrics(
                            matched_precision_50=(
                                acceptance.timing.overall.matched_precision_50
                            ),
                            active_gap_count=len(acceptance.timing.coverage_gaps),
                            hold_integrity_violations=0,
                            review_rank=(
                                0
                                if acceptance.action is GateAction.PASS
                                else 1
                                if acceptance.action is GateAction.REVIEW
                                else 2
                            ),
                        ),
                        difficulty_ordering_score=(
                            vector.ordering_score if vector is not None else None
                        ),
                        first_row_ms=first_row_ms(candidate),
                        timing_sections=tuple(
                            TimingSectionSnapshot(
                                row_count=section.metrics.row_count,
                                matched_precision_50=(
                                    section.metrics.matched_precision_50
                                ),
                            )
                            for section in acceptance.timing.sections
                        ),
                        candidate_payload_ref=(
                            candidate.workdir.relative_to(run_dir).as_posix()
                        ),
                    )
                )
            pools[(key_mode, difficulty)] = tuple(snapshots)
            selected = assignment[difficulty]
            current[(key_mode, difficulty)] = (
                ids[id(selected)] if selected is not None else None
            )
    selected, comparison = compare_song_families(
        pools,
        current,
        canonical_first_row_ms=intro_contract.canonical_first_row_ms,
        mode=mode,
    )
    if mode == "SHADOW_V2":
        return selections, comparison

    updated: list[Selection] = []
    for states, _assignment, _review in selections:
        key_mode = next(iter(states.values())).key_mode
        assignment = {
            difficulty: (
                candidates_by_id[candidate_id]
                if (candidate_id := selected[(key_mode, difficulty)]) is not None
                else None
            )
            for difficulty in DIFFICULTIES
        }
        chosen = tuple(
            candidate
            for difficulty in DIFFICULTIES
            if (candidate := assignment[difficulty]) is not None
        )
        updated.append(
            (
                states,
                assignment,
                review_candidates(chosen) if chosen else None,
            )
        )
    return updated, comparison


def compare_song_selection_shadow(
    selections: list[Selection],
    *,
    prepared: PreparedAudio,
    authority: SongTimingAuthority,
    run_dir: Path,
    intro_contract: IntroStartContract,
    boundary: SongBoundaryContract | None,
) -> SongSelectionComparison:
    """Backward-compatible observation-only wrapper."""
    _unchanged, comparison = compare_song_selection(
        selections,
        prepared=prepared,
        authority=authority,
        run_dir=run_dir,
        intro_contract=intro_contract,
        boundary=boundary,
        mode="SHADOW_V2",
    )
    return comparison
