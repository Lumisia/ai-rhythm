"""Bounded, deterministic song-wide candidate selection.

The selector is intentionally independent from stage-private candidate objects.
It receives immutable snapshots, rejects mixed evaluation contexts, prunes only
within a slot, and compares 4K/6K/7K families after every recovery stage.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from itertools import pairwise, product
from typing import Literal

from chart_worker.schema.types import DIFFICULTIES, KEY_MODES
from chart_worker.validation.timing_family_review import (
    LOCAL_DENSITY_RATIO_MIN,
    LOCAL_SIBLING_GAP_MIN,
    MIN_CONSECUTIVE_LOCAL_OUTLIERS,
    OVERALL_SIBLING_GAP_MIN,
)

SelectorMode = Literal["SHADOW_V2", "V2"]
MATCHED_PRECISION_EPSILON = 0.005
MIN_DIFFICULTY_GAP = 0.30
MAX_KEY_FAMILIES = 24


@dataclass(frozen=True, slots=True)
class TimingSectionSnapshot:
    row_count: int
    matched_precision_50: float | None


@dataclass(frozen=True, slots=True)
class ProtectedMetrics:
    matched_precision_50: float | None
    active_gap_count: int
    hold_integrity_violations: int
    review_rank: int


@dataclass(frozen=True, slots=True)
class CandidateSnapshot:
    candidate_id: str
    context_id: str
    key_mode: int
    difficulty: str
    attempt: int
    seed: int
    provenance: str
    hard_eligible: bool
    axis_actions: tuple[tuple[str, str], ...]
    protected_metrics: ProtectedMetrics
    difficulty_ordering_score: float | None
    first_row_ms: int | None
    timing_sections: tuple[TimingSectionSnapshot, ...]
    candidate_payload_ref: str

    def to_report(self) -> dict[str, object]:
        return {
            "candidateId": self.candidate_id,
            "contextId": self.context_id,
            "keyMode": self.key_mode,
            "difficulty": self.difficulty,
            "attempt": self.attempt,
            "seed": self.seed,
            "provenance": self.provenance,
            "hardEligible": self.hard_eligible,
            "axisActions": {axis: action for axis, action in self.axis_actions},
            "protectedMetrics": {
                "matchedPrecision50": self.protected_metrics.matched_precision_50,
                "activeGapCount": self.protected_metrics.active_gap_count,
                "holdIntegrityViolations": (
                    self.protected_metrics.hold_integrity_violations
                ),
                "reviewRank": self.protected_metrics.review_rank,
            },
            "difficultyOrderingScore": self.difficulty_ordering_score,
            "firstRowMs": self.first_row_ms,
            "candidatePayloadRef": self.candidate_payload_ref,
        }


@dataclass(frozen=True, slots=True)
class SongFamilyScore:
    missing_slots: int
    hard_violations: int
    intro_violations: int
    cross_key_outliers: int
    difficulty_violations: int
    review_rank: int
    hold_integrity_violations: int
    active_gap_count: int
    worst_matched_precision_50: float | None
    total_attempts: int

    def objective(self) -> tuple[object, ...]:
        return (
            self.missing_slots,
            self.hard_violations,
            self.intro_violations,
            self.cross_key_outliers,
            self.difficulty_violations,
            self.review_rank,
            self.hold_integrity_violations,
            self.active_gap_count,
            -(
                self.worst_matched_precision_50
                if self.worst_matched_precision_50 is not None
                else -1.0
            ),
            self.total_attempts,
        )

    def to_report(self) -> dict[str, object]:
        return {
            "missingSlots": self.missing_slots,
            "hardViolations": self.hard_violations,
            "introViolations": self.intro_violations,
            "crossKeyOutliers": self.cross_key_outliers,
            "difficultyViolations": self.difficulty_violations,
            "reviewRank": self.review_rank,
            "holdIntegrityViolations": self.hold_integrity_violations,
            "activeGapCount": self.active_gap_count,
            "worstMatchedPrecision50": self.worst_matched_precision_50,
            "totalAttempts": self.total_attempts,
        }


@dataclass(frozen=True, slots=True)
class SongSelectionComparison:
    mode: SelectorMode
    context_id: str
    current_assignment: dict[str, str | None]
    shadow_assignment: dict[str, str | None]
    current_score: SongFamilyScore
    shadow_score: SongFamilyScore
    reason: str
    pruned_candidate_ids: tuple[str, ...]
    key_families_evaluated: dict[int, int]
    key_families_retained: dict[int, int]
    song_families_evaluated: int

    def to_report(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "contextId": self.context_id,
            "currentAssignment": self.current_assignment,
            "shadowAssignment": self.shadow_assignment,
            "currentScore": self.current_score.to_report(),
            "shadowScore": self.shadow_score.to_report(),
            "reason": self.reason,
            "prunedCandidateIds": list(self.pruned_candidate_ids),
            "keyFamiliesEvaluated": {
                str(key): count for key, count in self.key_families_evaluated.items()
            },
            "keyFamiliesRetained": {
                str(key): count for key, count in self.key_families_retained.items()
            },
            "songFamiliesEvaluated": self.song_families_evaluated,
        }


Slot = tuple[int, str]
Assignment = dict[Slot, str | None]


def _precision_value(value: float | None) -> float:
    return -1.0 if value is None else value


def _dominates(left: CandidateSnapshot, right: CandidateSnapshot) -> bool:
    if left.hard_eligible != right.hard_eligible:
        return left.hard_eligible
    # Intro alignment and difficulty ordering are family-level constraints.
    # A candidate that looks better in isolation must not erase the only row or
    # ordering score that can form a valid song family later.
    if left.first_row_ms != right.first_row_ms:
        return False
    if left.difficulty_ordering_score != right.difficulty_ordering_score:
        return False
    lp = left.protected_metrics
    rp = right.protected_metrics
    no_worse = (
        _precision_value(lp.matched_precision_50)
        >= _precision_value(rp.matched_precision_50) - MATCHED_PRECISION_EPSILON
        and lp.active_gap_count <= rp.active_gap_count
        and lp.hold_integrity_violations <= rp.hold_integrity_violations
        and lp.review_rank <= rp.review_rank
        and left.attempt <= right.attempt
    )
    strictly_better = (
        _precision_value(lp.matched_precision_50)
        > _precision_value(rp.matched_precision_50) + MATCHED_PRECISION_EPSILON
        or lp.active_gap_count < rp.active_gap_count
        or lp.hold_integrity_violations < rp.hold_integrity_violations
        or lp.review_rank < rp.review_rank
        or left.attempt < right.attempt
    )
    return no_worse and strictly_better


def _prune_slot(
    candidates: tuple[CandidateSnapshot, ...],
) -> tuple[tuple[CandidateSnapshot, ...], tuple[str, ...]]:
    eligible = tuple(item for item in candidates if item.hard_eligible)
    pruned = {item.candidate_id for item in candidates if not item.hard_eligible}
    retained = []
    for item in eligible:
        if any(
            other is not item and _dominates(other, item)
            for other in eligible
        ):
            pruned.add(item.candidate_id)
        else:
            retained.append(item)
    retained.sort(key=lambda item: item.candidate_id)
    return tuple(retained), tuple(sorted(pruned))


def _v2_replacement_allowed(
    current: CandidateSnapshot,
    challenger: CandidateSnapshot,
    *,
    canonical_first_row_ms: int | None,
) -> bool:
    """Require monotonic protected quality before enforcing a family change."""
    if challenger.candidate_id == current.candidate_id:
        return True
    if not challenger.hard_eligible:
        return False
    if not current.hard_eligible:
        return True
    current_metrics = current.protected_metrics
    challenger_metrics = challenger.protected_metrics
    if challenger_metrics.review_rank > current_metrics.review_rank:
        return False
    if (
        challenger_metrics.hold_integrity_violations
        > current_metrics.hold_integrity_violations
        or challenger_metrics.active_gap_count > current_metrics.active_gap_count
    ):
        return False
    if current_metrics.matched_precision_50 is not None and (
        challenger_metrics.matched_precision_50 is None
        or challenger_metrics.matched_precision_50
        < current_metrics.matched_precision_50 - MATCHED_PRECISION_EPSILON
    ):
        return False
    return not (
        canonical_first_row_ms is not None
        and current.first_row_ms == canonical_first_row_ms
        and challenger.first_row_ms != canonical_first_row_ms
    )


def _difficulty_violations(candidates: tuple[CandidateSnapshot, ...]) -> int:
    ordered = sorted(candidates, key=lambda item: DIFFICULTIES.index(item.difficulty))
    violations = 0
    for easier, harder in pairwise(ordered):
        if (
            easier.difficulty_ordering_score is None
            or harder.difficulty_ordering_score is None
        ):
            violations += 1
            continue
        if harder.difficulty_ordering_score - easier.difficulty_ordering_score < MIN_DIFFICULTY_GAP:
            violations += 1
    return violations


def _longest_true_run(flags: list[bool]) -> int:
    longest = current = 0
    for flag in flags:
        current = current + 1 if flag else 0
        longest = max(longest, current)
    return longest


def _is_cross_key_outlier(candidates: tuple[CandidateSnapshot, ...]) -> bool:
    if len(candidates) != 3:
        return False
    for target in candidates:
        siblings = tuple(item for item in candidates if item is not target)
        precision = target.protected_metrics.matched_precision_50
        sibling_precision = tuple(
            item.protected_metrics.matched_precision_50 for item in siblings
        )
        if precision is None or any(value is None for value in sibling_precision):
            continue
        sibling_median = sum(value for value in sibling_precision if value is not None) / 2
        overall_gap = sibling_median - precision
        section_count = min(
            len(target.timing_sections),
            *(len(item.timing_sections) for item in siblings),
        )
        flags = []
        for index in range(section_count):
            section = target.timing_sections[index]
            sibling_sections = tuple(item.timing_sections[index] for item in siblings)
            sibling_values = tuple(
                item.matched_precision_50 for item in sibling_sections
            )
            if section.matched_precision_50 is None or any(
                value is None for value in sibling_values
            ):
                flags.append(False)
                continue
            sibling_section_precision = sum(
                value for value in sibling_values if value is not None
            ) / 2
            sibling_rows = sum(item.row_count for item in sibling_sections) / 2
            flags.append(
                section.row_count >= 8
                and sibling_section_precision - section.matched_precision_50
                >= LOCAL_SIBLING_GAP_MIN
                and section.row_count / max(1.0, sibling_rows)
                >= LOCAL_DENSITY_RATIO_MIN
            )
        if (
            overall_gap >= OVERALL_SIBLING_GAP_MIN
            and _longest_true_run(flags) >= MIN_CONSECUTIVE_LOCAL_OUTLIERS
        ):
            return True
    return False


def _score(
    assignment: Mapping[Slot, str | None],
    by_id: Mapping[str, CandidateSnapshot],
    *,
    canonical_first_row_ms: int | None,
) -> SongFamilyScore:
    chosen = tuple(
        by_id[candidate_id]
        for key_mode in KEY_MODES
        for difficulty in DIFFICULTIES
        if (candidate_id := assignment.get((key_mode, difficulty))) is not None
    )
    precisions = tuple(
        item.protected_metrics.matched_precision_50
        for item in chosen
        if item.protected_metrics.matched_precision_50 is not None
    )
    intro_violations = (
        sum(item.first_row_ms != canonical_first_row_ms for item in chosen)
        if canonical_first_row_ms is not None
        else max(0, len({item.first_row_ms for item in chosen}) - 1)
    )
    cross_key_outliers = 0
    for difficulty in DIFFICULTIES:
        family = tuple(item for item in chosen if item.difficulty == difficulty)
        cross_key_outliers += int(_is_cross_key_outlier(family))
    return SongFamilyScore(
        missing_slots=len(KEY_MODES) * len(DIFFICULTIES) - len(chosen),
        hard_violations=sum(not item.hard_eligible for item in chosen),
        intro_violations=intro_violations,
        cross_key_outliers=cross_key_outliers,
        difficulty_violations=sum(
            _difficulty_violations(
                tuple(item for item in chosen if item.key_mode == key_mode)
            )
            for key_mode in KEY_MODES
        ),
        review_rank=sum(item.protected_metrics.review_rank for item in chosen),
        hold_integrity_violations=sum(
            item.protected_metrics.hold_integrity_violations for item in chosen
        ),
        active_gap_count=sum(
            item.protected_metrics.active_gap_count for item in chosen
        ),
        worst_matched_precision_50=(min(precisions) if precisions else None),
        total_attempts=sum(item.attempt for item in chosen),
    )


def _key_families(
    key_mode: int,
    pools: Mapping[Slot, tuple[CandidateSnapshot, ...]],
    by_id: Mapping[str, CandidateSnapshot],
    *,
    canonical_first_row_ms: int | None,
) -> tuple[list[Assignment], int]:
    options = tuple(
        tuple(item.candidate_id for item in pools[(key_mode, difficulty)]) + (None,)
        for difficulty in DIFFICULTIES
    )
    families = [
        {
            (key_mode, difficulty): candidate_id
            for difficulty, candidate_id in zip(DIFFICULTIES, combination, strict=True)
        }
        for combination in product(*options)
    ]
    families.sort(
        key=lambda family: (
            _score(
                family,
                by_id,
                canonical_first_row_ms=canonical_first_row_ms,
            ).objective(),
            tuple("~" if item is None else item for item in family.values()),
        )
    )
    return families[:MAX_KEY_FAMILIES], len(families)


def _report_assignment(assignment: Mapping[Slot, str | None]) -> dict[str, str | None]:
    return {
        f"{key_mode}K:{difficulty}": assignment.get((key_mode, difficulty))
        for key_mode in KEY_MODES
        for difficulty in DIFFICULTIES
    }


def _assignment_tie_key(assignment: Mapping[Slot, str | None]) -> tuple[str, ...]:
    reported = _report_assignment(assignment)
    return tuple(
        f"{slot}={'~' if candidate_id is None else candidate_id}"
        for slot, candidate_id in sorted(reported.items())
    )


def compare_song_families(
    pools: Mapping[Slot, tuple[CandidateSnapshot, ...]],
    current_assignment: Mapping[Slot, str | None],
    *,
    canonical_first_row_ms: int | None,
    mode: SelectorMode = "SHADOW_V2",
) -> tuple[Assignment, SongSelectionComparison]:
    expected_slots = {
        (key_mode, difficulty)
        for key_mode in KEY_MODES
        for difficulty in DIFFICULTIES
    }
    normalized_pools = {slot: tuple(pools.get(slot, ())) for slot in expected_slots}
    all_candidates = tuple(
        item for slot in expected_slots for item in normalized_pools[slot]
    )
    contexts = {item.context_id for item in all_candidates}
    if len(contexts) > 1:
        raise ValueError(f"candidates span {len(contexts)} evaluation contexts")
    context_id = next(iter(contexts), "EMPTY")
    by_id = {item.candidate_id: item for item in all_candidates}
    if len(by_id) != len(all_candidates):
        raise ValueError("candidate_id values must be globally unique")
    for slot, candidates in normalized_pools.items():
        if any((item.key_mode, item.difficulty) != slot for item in candidates):
            raise ValueError(f"candidate does not match slot {slot}")
    current: Assignment = {
        slot: current_assignment.get(slot) for slot in expected_slots
    }
    unknown = {item for item in current.values() if item is not None} - set(by_id)
    if unknown:
        raise ValueError(f"current assignment contains unknown candidates: {sorted(unknown)}")

    pruned_pools = {}
    pruned_ids = []
    for slot in expected_slots:
        candidates = normalized_pools[slot]
        if mode == "V2" and (current_id := current[slot]) is not None:
            current_candidate = by_id[current_id]
            allowed = tuple(
                item
                for item in candidates
                if _v2_replacement_allowed(
                    current_candidate,
                    item,
                    canonical_first_row_ms=canonical_first_row_ms,
                )
            )
            pruned_ids.extend(
                item.candidate_id for item in candidates if item not in allowed
            )
            candidates = allowed
        retained, removed = _prune_slot(candidates)
        pruned_pools[slot] = retained
        pruned_ids.extend(removed)

    key_families = {}
    evaluated = {}
    for key_mode in KEY_MODES:
        key_families[key_mode], evaluated[key_mode] = _key_families(
            key_mode,
            pruned_pools,
            by_id,
            canonical_first_row_ms=canonical_first_row_ms,
        )

    best = current
    best_score = _score(
        current, by_id, canonical_first_row_ms=canonical_first_row_ms
    )
    best_key = (best_score.objective(), _assignment_tie_key(best))
    song_family_count = 0
    for combination in product(*(key_families[key] for key in KEY_MODES)):
        song_family_count += 1
        assignment = {slot: item for family in combination for slot, item in family.items()}
        score = _score(
            assignment,
            by_id,
            canonical_first_row_ms=canonical_first_row_ms,
        )
        key = (score.objective(), _assignment_tie_key(assignment))
        if key < best_key:
            best = assignment
            best_score = score
            best_key = key

    reason = "SAME_SELECTION"
    if best != current:
        current_score = _score(
            current, by_id, canonical_first_row_ms=canonical_first_row_ms
        )
        names = (
            "MISSING_SLOTS",
            "HARD_ELIGIBILITY",
            "INTRO_CONTRACT",
            "CROSS_KEY_TIMING",
            "DIFFICULTY_ORDER",
            "QUALITY_STATUS",
            "HOLD_INTEGRITY",
            "ACTIVE_GAPS",
            "WORST_MATCHED_PRECISION",
            "ATTEMPT_COST",
        )
        reason = next(
            (
                name
                for name, before, after in zip(
                    names,
                    current_score.objective(),
                    best_score.objective(),
                    strict=True,
                )
                if before != after
            ),
            "DETERMINISTIC_TIE_BREAK",
        )
    comparison = SongSelectionComparison(
        mode=mode,
        context_id=context_id,
        current_assignment=_report_assignment(current),
        shadow_assignment=_report_assignment(best),
        current_score=_score(
            current, by_id, canonical_first_row_ms=canonical_first_row_ms
        ),
        shadow_score=best_score,
        reason=reason,
        pruned_candidate_ids=tuple(sorted(set(pruned_ids))),
        key_families_evaluated=evaluated,
        key_families_retained={
            key: len(families) for key, families in key_families.items()
        },
        song_families_evaluated=song_family_count,
    )
    return (best if mode == "V2" else current), comparison
