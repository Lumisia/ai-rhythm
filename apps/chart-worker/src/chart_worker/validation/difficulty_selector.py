"""Deterministic current-vs-v2 difficulty-family selection comparison."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from itertools import pairwise, product
from typing import Literal

from chart_worker.schema.types import DIFFICULTIES

SelectionMode = Literal["CURRENT", "SHADOW_V2", "V2"]
ShadowDifferenceReason = Literal[
    "SAME_SELECTION",
    "LOWER_V2_ORDER_COST",
    "FEWER_V2_INVERSIONS",
    "FEWER_V2_NARROW_GAPS",
    "CURRENT_CANDIDATE_MISSING_VECTOR",
]

MIN_V2_ADJACENT_GAP = 0.30


@dataclass(frozen=True, slots=True)
class DifficultyCandidateView:
    candidate_id: str
    difficulty: str
    seed: int
    attempt: int
    provenance: str
    intro_anchor_covered: bool | None
    current_rating: float
    v2_ordering_score: float | None
    vector_v2: dict[str, object] | None

    def to_report(self) -> dict[str, object]:
        return {
            "candidateId": self.candidate_id,
            "difficulty": self.difficulty,
            "seed": self.seed,
            "attempt": self.attempt,
            "provenance": self.provenance,
            "introAnchorCovered": self.intro_anchor_covered,
            "currentRating": self.current_rating,
            "vectorV2": self.vector_v2,
        }


@dataclass(frozen=True, slots=True)
class _AssignmentMetrics:
    inversions: tuple[tuple[str, str], ...]
    narrow_pairs: tuple[tuple[str, str], ...]
    order_cost: float


@dataclass(frozen=True, slots=True)
class DifficultySelectionComparison:
    mode: Literal["SHADOW_V2", "V2"]
    current_assignment: dict[str, str | None]
    shadow_assignment: dict[str, str | None]
    current_seeds: tuple[int | None, ...]
    shadow_seeds: tuple[int | None, ...]
    current_inversions: tuple[tuple[str, str], ...]
    shadow_inversions: tuple[tuple[str, str], ...]
    current_narrow_pairs: tuple[tuple[str, str], ...]
    shadow_narrow_pairs: tuple[tuple[str, str], ...]
    current_order_cost: float
    shadow_order_cost: float
    reason: ShadowDifferenceReason
    candidates: tuple[dict[str, object], ...]

    def to_report(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "currentAssignment": self.current_assignment,
            "shadowAssignment": self.shadow_assignment,
            "currentSeeds": list(self.current_seeds),
            "shadowSeeds": list(self.shadow_seeds),
            "currentInversions": [list(pair) for pair in self.current_inversions],
            "shadowInversions": [list(pair) for pair in self.shadow_inversions],
            "currentNarrowPairs": [list(pair) for pair in self.current_narrow_pairs],
            "shadowNarrowPairs": [list(pair) for pair in self.shadow_narrow_pairs],
            "currentOrderCost": self.current_order_cost,
            "shadowOrderCost": self.shadow_order_cost,
            "reason": self.reason,
            "candidates": list(self.candidates),
        }


def _candidate_map(
    pools: Mapping[str, tuple[DifficultyCandidateView, ...]],
) -> dict[str, DifficultyCandidateView]:
    result: dict[str, DifficultyCandidateView] = {}
    for difficulty in DIFFICULTIES:
        for candidate in pools.get(difficulty, ()):
            if candidate.difficulty != difficulty:
                raise ValueError("candidate difficulty does not match its pool")
            if candidate.candidate_id in result:
                raise ValueError(f"duplicate candidate_id: {candidate.candidate_id}")
            result[candidate.candidate_id] = candidate
    return result


def _metrics(
    assignment: Mapping[str, str | None],
    by_id: Mapping[str, DifficultyCandidateView],
) -> _AssignmentMetrics:
    ordered: list[tuple[str, float]] = []
    for difficulty in DIFFICULTIES:
        candidate_id = assignment.get(difficulty)
        if candidate_id is None:
            continue
        score = by_id[candidate_id].v2_ordering_score
        if score is None:
            continue
        ordered.append((difficulty, score))
    inversions: list[tuple[str, str]] = []
    narrow: list[tuple[str, str]] = []
    cost = 0.0
    for (easier, easier_score), (harder, harder_score) in pairwise(ordered):
        gap = harder_score - easier_score
        if gap < 0:
            inversions.append((easier, harder))
        if gap < MIN_V2_ADJACENT_GAP:
            narrow.append((easier, harder))
            cost += MIN_V2_ADJACENT_GAP - gap
    return _AssignmentMetrics(tuple(inversions), tuple(narrow), round(cost, 6))


def _selection_score(
    assignment: Mapping[str, str | None],
    by_id: Mapping[str, DifficultyCandidateView],
) -> tuple[object, ...]:
    chosen = [
        by_id[candidate_id]
        for difficulty in DIFFICULTIES
        if (candidate_id := assignment.get(difficulty)) is not None
    ]
    metrics = _metrics(assignment, by_id)
    missing = len(DIFFICULTIES) - len(chosen)
    missing_vectors = sum(item.v2_ordering_score is None for item in chosen)
    raw = sum(item.provenance == "RAW_UNVERIFIED" for item in chosen)
    intro_misses = sum(item.intro_anchor_covered is False for item in chosen)
    current_rating_inversions = sum(
        harder.current_rating < easier.current_rating
        for easier, harder in pairwise(chosen)
    )
    return (
        missing,
        missing_vectors,
        raw,
        intro_misses,
        current_rating_inversions,
        len(metrics.inversions),
        len(metrics.narrow_pairs),
        metrics.order_cost,
    )


def _shadow_assignment(
    pools: Mapping[str, tuple[DifficultyCandidateView, ...]],
    by_id: Mapping[str, DifficultyCandidateView],
    current: Mapping[str, str | None],
) -> dict[str, str | None]:
    options = tuple(
        tuple(candidate.candidate_id for candidate in pools.get(difficulty, ()))
        + (None,)
        for difficulty in DIFFICULTIES
    )
    # An equal score is not an improvement. Keeping the current assignment is
    # important after intro/HOLD recovery, where two candidates can have the
    # same difficulty vector but different recovery provenance.
    best: dict[str, str | None] = dict(current)
    best_score = _selection_score(best, by_id)
    for combination in product(*options):
        assignment = dict(zip(DIFFICULTIES, combination, strict=True))
        score = _selection_score(assignment, by_id)
        if score < best_score:
            best = assignment
            best_score = score
    return best


def _seeds(
    assignment: Mapping[str, str | None],
    by_id: Mapping[str, DifficultyCandidateView],
) -> tuple[int | None, ...]:
    return tuple(
        by_id[candidate_id].seed if (candidate_id := assignment.get(difficulty)) else None
        for difficulty in DIFFICULTIES
    )


def compare_family_candidates(
    pools: Mapping[str, tuple[DifficultyCandidateView, ...]],
    current_assignment: Mapping[str, str | None],
    *,
    mode: SelectionMode = "SHADOW_V2",
) -> tuple[dict[str, str | None], DifficultySelectionComparison | None]:
    """Return CURRENT unchanged in shadow mode and attach a v2 comparison."""
    current = {
        difficulty: current_assignment.get(difficulty)
        for difficulty in DIFFICULTIES
    }
    if mode == "CURRENT":
        return current, None
    by_id = _candidate_map(pools)
    unknown = {item for item in current.values() if item is not None} - set(by_id)
    if unknown:
        raise ValueError(f"current assignment contains unknown candidates: {sorted(unknown)}")

    current_missing_vector = any(
        by_id[candidate_id].v2_ordering_score is None
        for candidate_id in current.values()
        if candidate_id is not None
    )
    shadow = (
        current
        if current_missing_vector
        else _shadow_assignment(pools, by_id, current)
    )
    current_metrics = _metrics(current, by_id)
    shadow_metrics = _metrics(shadow, by_id)
    if current_missing_vector:
        reason: ShadowDifferenceReason = "CURRENT_CANDIDATE_MISSING_VECTOR"
    elif shadow == current:
        reason = "SAME_SELECTION"
    elif len(shadow_metrics.inversions) < len(current_metrics.inversions):
        reason = "FEWER_V2_INVERSIONS"
    elif len(shadow_metrics.narrow_pairs) < len(current_metrics.narrow_pairs):
        reason = "FEWER_V2_NARROW_GAPS"
    else:
        reason = "LOWER_V2_ORDER_COST"
    comparison = DifficultySelectionComparison(
        mode=mode,
        current_assignment=current,
        shadow_assignment=shadow,
        current_seeds=_seeds(current, by_id),
        shadow_seeds=_seeds(shadow, by_id),
        current_inversions=current_metrics.inversions,
        shadow_inversions=shadow_metrics.inversions,
        current_narrow_pairs=current_metrics.narrow_pairs,
        shadow_narrow_pairs=shadow_metrics.narrow_pairs,
        current_order_cost=current_metrics.order_cost,
        shadow_order_cost=shadow_metrics.order_cost,
        reason=reason,
        candidates=tuple(
            candidate.to_report()
            for difficulty in DIFFICULTIES
            for candidate in pools.get(difficulty, ())
        ),
    )
    selected = shadow if mode == "V2" else current
    return selected, comparison
