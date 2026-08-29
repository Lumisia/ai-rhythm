"""Bounded same-key assignment that puts gameplay safety before labels.

The selector is deliberately independent from generation objects.  It does not
modify notes, invoke a model, or interpret a provenance name as quality truth.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import pairwise, product
from typing import Literal

from chart_worker.schema.types import DIFFICULTIES, KEY_MODES
from chart_worker.validation.difficulty_order import MIN_ADJACENT_RATING_GAP

EvidenceState = Literal["CONFIRMED_SAFE", "VIOLATION", "UNKNOWN"]
TerminalEvidenceConfidence = Literal["CONFIRMED", "PROVISIONAL", "UNKNOWN"]
UniquePayloadStatus = Literal["SATISFIED", "UNAVAILABLE"]
FamilyFeasibilityStatus = Literal["SATISFIED", "UNAVAILABLE"]
PostResolutionOrderingStatus = Literal[
    "NOT_REQUESTED",
    "ORDERED",
    "ALREADY_ORDERED",
    "UNAVAILABLE",
]
Assignment = tuple[tuple[str, str], ...]
SAFE_FAMILY_ASSIGNMENT_VERSION = "safe-family-assignment-v3"
MAX_SAFE_SUBSTITUTES_PER_KEY = 20
REQUIRED_DIFFICULTY_METRICS = ("ORDERING_SCORE", "PROJECT_RATING")


def _exact_non_negative_int(value: object, *, name: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be an exact integer")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value


def _exact_string(value: object, *, name: str) -> str:
    if type(value) is not str or not value:
        raise TypeError(f"{name} must be a non-empty exact string")
    return value


def _sha256(value: object, *, name: str) -> str:
    digest = _exact_string(value, name=name)
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return digest


@dataclass(frozen=True, slots=True)
class SafeFamilyCandidate:
    candidate_id: str
    candidate_payload_sha256: str
    key_mode: int
    source_difficulty: str
    provenance: str
    hard_safe: bool
    intro_state: EvidenceState
    boundary_state: EvidenceState
    attack_gap_count: int
    attack_gap_total_ms: int
    active_gap_count: int
    max_active_gap_ms: int
    difficulty_scores: tuple[tuple[str, float], ...]
    review_rank: int
    publication_rank: int
    recovery_trust_rank: int
    matched_f1_50: float | None
    attempt: int
    intro_distance_ms: int | None = None
    tail_coverage_deficit_ms: int = 0
    tail_active_onset_count: int = 0
    terminal_overflow_ms: int = 0
    terminal_overflow_confidence: TerminalEvidenceConfidence = "UNKNOWN"

    def __post_init__(self) -> None:
        _exact_string(self.candidate_id, name="candidate_id")
        _sha256(self.candidate_payload_sha256, name="candidate_payload_sha256")
        if type(self.key_mode) is not int or self.key_mode not in KEY_MODES:
            raise ValueError("key_mode is unsupported")
        if self.source_difficulty not in DIFFICULTIES:
            raise ValueError("source_difficulty is unsupported")
        _exact_string(self.provenance, name="provenance")
        if type(self.hard_safe) is not bool:
            raise TypeError("hard_safe must be an exact boolean")
        if self.intro_state not in {"CONFIRMED_SAFE", "VIOLATION", "UNKNOWN"}:
            raise ValueError("intro_state is unsupported")
        if self.boundary_state not in {"CONFIRMED_SAFE", "VIOLATION", "UNKNOWN"}:
            raise ValueError("boundary_state is unsupported")
        for name in (
            "attack_gap_count",
            "attack_gap_total_ms",
            "active_gap_count",
            "max_active_gap_ms",
            "review_rank",
            "publication_rank",
            "recovery_trust_rank",
            "attempt",
            "tail_coverage_deficit_ms",
            "tail_active_onset_count",
            "terminal_overflow_ms",
        ):
            _exact_non_negative_int(getattr(self, name), name=name)
        if self.intro_distance_ms is not None:
            _exact_non_negative_int(self.intro_distance_ms, name="intro_distance_ms")
        if self.terminal_overflow_confidence not in {
            "CONFIRMED",
            "PROVISIONAL",
            "UNKNOWN",
        }:
            raise ValueError("terminal_overflow_confidence is unsupported")
        if self.active_gap_count < self.attack_gap_count:
            raise ValueError("active_gap_count cannot be smaller than attack_gap_count")
        if type(self.difficulty_scores) is not tuple:
            raise TypeError("difficulty_scores must be a tuple")
        names = []
        for name, value in self.difficulty_scores:
            _exact_string(name, name="difficulty score name")
            if type(value) is not float or not math.isfinite(value):
                raise ValueError("difficulty scores must be finite exact floats")
            names.append(name)
        if names != sorted(set(names)):
            raise ValueError("difficulty_scores must be sorted with unique names")
        if self.matched_f1_50 is not None and (
            type(self.matched_f1_50) is not float
            or not math.isfinite(self.matched_f1_50)
            or not 0.0 <= self.matched_f1_50 <= 1.0
        ):
            raise ValueError("matched_f1_50 must be a finite float in [0, 1]")

    @property
    def scores(self) -> dict[str, float]:
        return dict(self.difficulty_scores)


@dataclass(frozen=True, slots=True)
class SafeFamilyScore:
    hard_violations: int
    intro_violations: int
    boundary_unknown: int
    attack_gap_count: int
    attack_gap_total_ms: int
    active_gap_count: int
    max_active_gap_ms: int
    duplicate_payloads: int
    difficulty_violations: int
    difficulty_regression_magnitude: float
    difficulty_unscored_pairs: int
    minimum_relative_separation: float
    intro_unknown: int
    review_rank: int
    publication_rank: int
    recovery_trust_rank: int
    worst_matched_f1_50: float | None
    source_reassignments: int
    current_changes: int
    attempts: int
    intro_distance_ms: int
    tail_coverage_deficit_ms: int
    tail_active_onset_count: int
    confirmed_terminal_overflow_ms: int
    provisional_terminal_overflow_ms: int
    minimum_project_rating_gap: float | None
    required_difficulty_metrics_missing: tuple[str, ...]

    def objective(self) -> tuple[object, ...]:
        return (
            self.hard_violations,
            self.confirmed_terminal_overflow_ms,
            self.intro_violations,
            self.boundary_unknown,
            self.attack_gap_count,
            self.attack_gap_total_ms,
            self.tail_active_onset_count,
            self.tail_coverage_deficit_ms,
            self.provisional_terminal_overflow_ms,
            self.active_gap_count,
            self.max_active_gap_ms,
            self.difficulty_violations,
            self.difficulty_regression_magnitude,
            self.difficulty_unscored_pairs,
            len(self.required_difficulty_metrics_missing),
            -self.minimum_relative_separation,
            self.duplicate_payloads,
            self.intro_unknown,
            self.intro_distance_ms,
            self.review_rank,
            self.publication_rank,
            self.recovery_trust_rank,
            -(self.worst_matched_f1_50 if self.worst_matched_f1_50 is not None else -1.0),
            self.source_reassignments,
            self.current_changes,
            self.attempts,
        )

    def to_report(self) -> dict[str, object]:
        return {
            "hardViolations": self.hard_violations,
            "confirmedTerminalOverflowMs": self.confirmed_terminal_overflow_ms,
            "provisionalTerminalOverflowMs": self.provisional_terminal_overflow_ms,
            "introViolations": self.intro_violations,
            "introDistanceMs": self.intro_distance_ms,
            "boundaryUnknown": self.boundary_unknown,
            "attackGapCount": self.attack_gap_count,
            "attackGapTotalMs": self.attack_gap_total_ms,
            "activeGapCount": self.active_gap_count,
            "maxActiveGapMs": self.max_active_gap_ms,
            "tailCoverageDeficitMs": self.tail_coverage_deficit_ms,
            "tailActiveOnsetCount": self.tail_active_onset_count,
            "duplicatePayloads": self.duplicate_payloads,
            "difficultyViolations": self.difficulty_violations,
            "difficultyRegressionMagnitude": self.difficulty_regression_magnitude,
            "difficultyUnscoredPairs": self.difficulty_unscored_pairs,
            "minimumProjectRatingGap": self.minimum_project_rating_gap,
            "requiredDifficultyMetricsMissing": list(
                self.required_difficulty_metrics_missing
            ),
            "minimumRelativeSeparation": self.minimum_relative_separation,
            "introUnknown": self.intro_unknown,
            "reviewRank": self.review_rank,
            "publicationRank": self.publication_rank,
            "recoveryTrustRank": self.recovery_trust_rank,
            "worstMatchedF150": self.worst_matched_f1_50,
            "sourceReassignments": self.source_reassignments,
            "currentChanges": self.current_changes,
            "attempts": self.attempts,
        }


@dataclass(frozen=True, slots=True)
class SafeFamilyAssignmentDecision:
    key_mode: int
    current_assignment: Assignment
    selected_assignment: Assignment
    current_score: SafeFamilyScore
    selected_score: SafeFamilyScore
    source_difficulties: tuple[tuple[str, str], ...]
    reassigned_slots: tuple[str, ...]
    emergency_duplicate_slots: tuple[str, ...]
    unique_payload_status: UniquePayloadStatus
    family_feasibility_status: FamilyFeasibilityStatus
    family_feasibility_reasons: tuple[str, ...]
    reason: str
    candidates_evaluated: int
    assignments_evaluated: int
    post_resolution_ordering_status: PostResolutionOrderingStatus = "NOT_REQUESTED"
    version: Literal["safe-family-assignment-v3"] = SAFE_FAMILY_ASSIGNMENT_VERSION
    additional_model_calls: Literal[0] = 0

    @property
    def changed(self) -> bool:
        return self.selected_assignment != self.current_assignment

    def to_report(self) -> dict[str, object]:
        return {
            "version": self.version,
            "keyMode": self.key_mode,
            "currentAssignment": dict(self.current_assignment),
            "selectedAssignment": dict(self.selected_assignment),
            "sourceDifficulties": dict(self.source_difficulties),
            "reassignedSlots": list(self.reassigned_slots),
            "emergencyDuplicateSlots": list(self.emergency_duplicate_slots),
            "uniquePayloadStatus": self.unique_payload_status,
            "familyFeasibilityStatus": self.family_feasibility_status,
            "familyFeasibilityReasons": list(self.family_feasibility_reasons),
            "currentScore": self.current_score.to_report(),
            "selectedScore": self.selected_score.to_report(),
            "reason": self.reason,
            "changed": self.changed,
            "postResolutionOrderingStatus": self.post_resolution_ordering_status,
            "candidatesEvaluated": self.candidates_evaluated,
            "assignmentsEvaluated": self.assignments_evaluated,
            "additionalModelCalls": self.additional_model_calls,
        }


def _normalize_assignment(value: Assignment, *, candidates: set[str]) -> Assignment:
    if type(value) is not tuple:
        raise TypeError("current_assignment must be a tuple")
    normalized = []
    for entry in value:
        if type(entry) is not tuple or len(entry) != 2:
            raise TypeError("current_assignment entries must be exact pairs")
        difficulty, candidate_id = entry
        if difficulty not in DIFFICULTIES:
            raise ValueError("current_assignment difficulty is unsupported")
        _exact_string(candidate_id, name="current_assignment candidate")
        if candidate_id not in candidates:
            raise ValueError(f"current_assignment references unknown candidate: {candidate_id}")
        normalized.append((difficulty, candidate_id))
    by_difficulty = dict(normalized)
    if set(by_difficulty) != set(DIFFICULTIES) or len(normalized) != len(DIFFICULTIES):
        raise ValueError("current_assignment must contain exactly four unique difficulties")
    return tuple((difficulty, by_difficulty[difficulty]) for difficulty in DIFFICULTIES)


def _metric_ranges(
    candidates: tuple[SafeFamilyCandidate, ...],
) -> dict[str, tuple[float, float]]:
    names = sorted({name for candidate in candidates for name, _value in candidate.difficulty_scores})
    ranges = {}
    for name in names:
        values = [candidate.scores[name] for candidate in candidates if name in candidate.scores]
        ranges[name] = (min(values), max(values))
    return ranges


def _difficulty_evidence(
    assignment: tuple[SafeFamilyCandidate, ...],
    *,
    metric_ranges: dict[str, tuple[float, float]],
) -> tuple[int, float, int, float]:
    violations = 0
    regression_magnitude = 0.0
    unscored = 0
    separations = []
    for easier, harder in pairwise(assignment):
        for name, (minimum, maximum) in metric_ranges.items():
            easier_score = easier.scores.get(name)
            harder_score = harder.scores.get(name)
            if easier_score is None or harder_score is None:
                unscored += 1
                continue
            delta = harder_score - easier_score
            if delta <= 0.0:
                violations += 1
                span = maximum - minimum
                if delta < 0.0 and span > 0.0:
                    regression_magnitude += -delta / span
                separations.append(0.0)
                continue
            span = maximum - minimum
            separations.append(delta / span if span > 0.0 else 0.0)
    return (
        violations,
        round(regression_magnitude, 6),
        unscored,
        round(min(separations), 6) if separations else 0.0,
    )


def _minimum_adjacent_metric_gap(
    assignment: tuple[SafeFamilyCandidate, ...],
    *,
    metric_name: str,
) -> float | None:
    values = tuple(candidate.scores.get(metric_name) for candidate in assignment)
    if any(value is None for value in values):
        return None
    return round(
        min(
            float(harder) - float(easier)
            for easier, harder in pairwise(values)
        ),
        6,
    )


def _missing_required_difficulty_metrics(
    assignment: tuple[SafeFamilyCandidate, ...],
) -> tuple[str, ...]:
    return tuple(
        metric_name
        for metric_name in REQUIRED_DIFFICULTY_METRICS
        if any(metric_name not in candidate.scores for candidate in assignment)
    )


def _score(
    assignment: tuple[SafeFamilyCandidate, ...],
    *,
    current: Assignment,
    metric_ranges: dict[str, tuple[float, float]],
) -> SafeFamilyScore:
    current_by_difficulty = dict(current)
    difficulty_violations, regression_magnitude, unscored, separation = _difficulty_evidence(
        assignment,
        metric_ranges=metric_ranges,
    )
    f1_values = tuple(
        candidate.matched_f1_50
        for candidate in assignment
        if candidate.matched_f1_50 is not None
    )
    return SafeFamilyScore(
        hard_violations=sum(
            (not candidate.hard_safe) or candidate.boundary_state == "VIOLATION"
            for candidate in assignment
        ),
        intro_violations=sum(candidate.intro_state == "VIOLATION" for candidate in assignment),
        boundary_unknown=sum(candidate.boundary_state == "UNKNOWN" for candidate in assignment),
        attack_gap_count=sum(candidate.attack_gap_count for candidate in assignment),
        attack_gap_total_ms=sum(candidate.attack_gap_total_ms for candidate in assignment),
        active_gap_count=sum(candidate.active_gap_count for candidate in assignment),
        max_active_gap_ms=max(candidate.max_active_gap_ms for candidate in assignment),
        duplicate_payloads=len(assignment)
        - len({candidate.candidate_payload_sha256 for candidate in assignment}),
        difficulty_violations=difficulty_violations,
        difficulty_regression_magnitude=regression_magnitude,
        difficulty_unscored_pairs=unscored,
        minimum_relative_separation=separation,
        intro_unknown=sum(candidate.intro_state == "UNKNOWN" for candidate in assignment),
        review_rank=sum(candidate.review_rank for candidate in assignment),
        publication_rank=sum(candidate.publication_rank for candidate in assignment),
        recovery_trust_rank=sum(candidate.recovery_trust_rank for candidate in assignment),
        worst_matched_f1_50=min(f1_values) if f1_values else None,
        source_reassignments=sum(
            target != candidate.source_difficulty
            for target, candidate in zip(DIFFICULTIES, assignment, strict=True)
        ),
        current_changes=sum(
            current_by_difficulty[target] != candidate.candidate_id
            for target, candidate in zip(DIFFICULTIES, assignment, strict=True)
        ),
        attempts=sum(candidate.attempt for candidate in assignment),
        intro_distance_ms=sum(
            candidate.intro_distance_ms or 0 for candidate in assignment
        ),
        tail_coverage_deficit_ms=sum(
            candidate.tail_coverage_deficit_ms for candidate in assignment
        ),
        tail_active_onset_count=sum(
            candidate.tail_active_onset_count for candidate in assignment
        ),
        confirmed_terminal_overflow_ms=sum(
            candidate.terminal_overflow_ms
            for candidate in assignment
            if candidate.terminal_overflow_confidence == "CONFIRMED"
        ),
        provisional_terminal_overflow_ms=sum(
            candidate.terminal_overflow_ms
            for candidate in assignment
            if candidate.terminal_overflow_confidence == "PROVISIONAL"
        ),
        minimum_project_rating_gap=_minimum_adjacent_metric_gap(
            assignment,
            metric_name="PROJECT_RATING",
        ),
        required_difficulty_metrics_missing=_missing_required_difficulty_metrics(
            assignment
        ),
    )


def _family_feasibility_reasons(score: SafeFamilyScore) -> tuple[str, ...]:
    reasons = []
    if score.duplicate_payloads:
        reasons.append("PAYLOAD_UNIQUENESS")
    if score.hard_violations:
        reasons.append("HARD_SAFETY")
    if score.confirmed_terminal_overflow_ms:
        reasons.append("CONFIRMED_TERMINAL_BOUNDARY")
    if score.intro_violations:
        reasons.append("INTRO_COVERAGE")
    if score.attack_gap_count:
        reasons.append("ATTACK_REQUIRED_GAP")
    if score.tail_active_onset_count and score.tail_coverage_deficit_ms:
        reasons.append("ACTIVE_TAIL_COVERAGE")
    if score.required_difficulty_metrics_missing or score.difficulty_unscored_pairs:
        reasons.append("DIFFICULTY_EVIDENCE")
    if score.difficulty_violations:
        reasons.append("DIFFICULTY_ORDER")
    if (
        score.minimum_project_rating_gap is not None
        and score.minimum_project_rating_gap < MIN_ADJACENT_RATING_GAP
    ):
        reasons.append("PROJECT_RATING_SEPARATION")
    return tuple(reasons)


def _preserves_target_quality(
    assignment: tuple[SafeFamilyCandidate, ...],
    *,
    current: tuple[SafeFamilyCandidate, ...],
) -> bool:
    """Reject a relabel that introduces a new target-local quality defect.

    Family aggregates cannot see a defect moving from EXPERT to EASY because
    their sums remain unchanged.  Compare each proposal with the chart already
    occupying that target label before considering relative difficulty gains.
    """

    for baseline, proposed in zip(current, assignment, strict=True):
        baseline_confirmed_overflow = (
            baseline.terminal_overflow_ms
            if baseline.terminal_overflow_confidence == "CONFIRMED"
            else 0
        )
        proposed_confirmed_overflow = (
            proposed.terminal_overflow_ms
            if proposed.terminal_overflow_confidence == "CONFIRMED"
            else 0
        )
        if proposed_confirmed_overflow > baseline_confirmed_overflow:
            return False
        if proposed.attack_gap_count > baseline.attack_gap_count:
            return False
        if (
            proposed.tail_active_onset_count > baseline.tail_active_onset_count
            or proposed.tail_coverage_deficit_ms
            > baseline.tail_coverage_deficit_ms
        ) and (
            proposed.tail_active_onset_count > 0
            and proposed.tail_coverage_deficit_ms > 0
        ):
            return False
        if (
            baseline.intro_state == "CONFIRMED_SAFE"
            and proposed.intro_state == "VIOLATION"
        ):
            return False
    return True


def _post_resolution_order(
    assignment: tuple[SafeFamilyCandidate, ...],
    *,
    current: Assignment,
    metric_ranges: dict[str, tuple[float, float]],
) -> tuple[
    tuple[SafeFamilyCandidate, ...],
    SafeFamilyScore,
    PostResolutionOrderingStatus,
]:
    """Order one frozen four-payload family without changing its contents.

    Target-local quality vetoes apply while choosing which payloads survive.
    Once that set is frozen, a permutation cannot create or remove a payload
    defect.  Difficulty labels therefore follow measured project rating while
    every quality observation remains attached to the same immutable payload.
    """

    score = _score(
        assignment,
        current=current,
        metric_ranges=metric_ranges,
    )
    if score.duplicate_payloads or score.required_difficulty_metrics_missing:
        return assignment, score, "UNAVAILABLE"

    ordered = tuple(
        sorted(
            assignment,
            key=lambda candidate: (
                candidate.scores["PROJECT_RATING"],
                candidate.scores["ORDERING_SCORE"],
                candidate.candidate_id,
            ),
        )
    )
    ordered_score = _score(
        ordered,
        current=current,
        metric_ranges=metric_ranges,
    )
    project_ratings = tuple(
        candidate.scores["PROJECT_RATING"] for candidate in ordered
    )
    if any(
        harder <= easier
        for easier, harder in pairwise(project_ratings)
    ):
        return ordered, ordered_score, "UNAVAILABLE"
    return (
        ordered,
        ordered_score,
        "ALREADY_ORDERED" if ordered == assignment else "ORDERED",
    )


def _duplicate_slots(assignment: tuple[SafeFamilyCandidate, ...]) -> tuple[str, ...]:
    by_payload: dict[str, list[tuple[str, SafeFamilyCandidate]]] = {}
    for target, candidate in zip(DIFFICULTIES, assignment, strict=True):
        by_payload.setdefault(candidate.candidate_payload_sha256, []).append(
            (target, candidate)
        )
    duplicates: set[str] = set()
    for occurrences in by_payload.values():
        if len(occurrences) == 1:
            continue
        primary_target = next(
            (
                target
                for target, candidate in occurrences
                if target == candidate.source_difficulty
            ),
            occurrences[0][0],
        )
        duplicates.update(target for target, _candidate in occurrences if target != primary_target)
    return tuple(difficulty for difficulty in DIFFICULTIES if difficulty in duplicates)


def _has_material_gameplay_defect(candidate: SafeFamilyCandidate) -> bool:
    """Return true only for evidence strong enough to authorize substitution.

    Relative difficulty estimates are intentionally excluded.  They are noisy
    ranking evidence, not permission to replace an otherwise playable chart.
    UNKNOWN intro/boundary evidence is likewise not treated as a defect.
    """
    return (
        not candidate.hard_safe
        or candidate.intro_state == "VIOLATION"
        or candidate.boundary_state == "VIOLATION"
        or candidate.attack_gap_count > 0
    )


def _authorizes_existing_candidate_substitution(
    candidate: SafeFamilyCandidate,
) -> bool:
    """Allow replacing a fallback without making fallbacks unsafe candidates."""

    return (
        _has_material_gameplay_defect(candidate)
        or candidate.provenance == "SAFE_FALLBACK"
    )


def _substitute_priority(candidate: SafeFamilyCandidate) -> tuple[object, ...]:
    return (
        candidate.review_rank,
        candidate.publication_rank,
        candidate.recovery_trust_rank,
        -(candidate.matched_f1_50 if candidate.matched_f1_50 is not None else -1.0),
        candidate.attempt,
        candidate.candidate_id,
    )


def _bounded_safe_substitutes(
    candidates: tuple[SafeFamilyCandidate, ...],
) -> tuple[SafeFamilyCandidate, ...]:
    """Keep a deterministic quality/difficulty frontier for bounded search."""
    eligible = tuple(
        candidate
        for candidate in candidates
        if not _has_material_gameplay_defect(candidate)
    )
    if len(eligible) <= MAX_SAFE_SUBSTITUTES_PER_KEY:
        return tuple(sorted(eligible, key=lambda item: item.candidate_id))

    chosen: dict[str, SafeFamilyCandidate] = {}

    def retain(candidate: SafeFamilyCandidate) -> None:
        if len(chosen) < MAX_SAFE_SUBSTITUTES_PER_KEY:
            chosen.setdefault(candidate.candidate_id, candidate)

    for difficulty in DIFFICULTIES:
        source_candidates = tuple(
            candidate
            for candidate in eligible
            if candidate.source_difficulty == difficulty
        )
        if source_candidates:
            retain(min(source_candidates, key=_substitute_priority))

    metric_names = sorted(
        {name for candidate in eligible for name, _value in candidate.difficulty_scores}
    )
    for name in metric_names:
        scored = tuple(candidate for candidate in eligible if name in candidate.scores)
        if not scored:
            continue
        retain(min(scored, key=lambda item: (item.scores[name], item.candidate_id)))
        retain(max(scored, key=lambda item: (item.scores[name], item.candidate_id)))

    for candidate in sorted(eligible, key=_substitute_priority):
        retain(candidate)
        if len(chosen) == MAX_SAFE_SUBSTITUTES_PER_KEY:
            break
    return tuple(sorted(chosen.values(), key=lambda item: item.candidate_id))


def _reason(current: SafeFamilyScore, selected: SafeFamilyScore, *, changed: bool) -> str:
    if not changed:
        return "UNCHANGED_BEST_EVIDENCE"
    names = (
        "HARD_SAFETY",
        "CONFIRMED_TERMINAL_OVERFLOW",
        "INTRO_COVERAGE",
        "BOUNDARY_CONFIDENCE",
        "ATTACK_GAPS",
        "ATTACK_GAP_DURATION",
        "TAIL_ACTIVE_ONSETS",
        "TAIL_COVERAGE_DEFICIT",
        "PROVISIONAL_TERMINAL_OVERFLOW",
        "ACTIVE_GAPS",
        "MAX_ACTIVE_GAP",
        "RELATIVE_DIFFICULTY_ORDER",
        "RELATIVE_DIFFICULTY_REGRESSION_MAGNITUDE",
        "DIFFICULTY_EVIDENCE",
        "REQUIRED_DIFFICULTY_METRICS",
        "RELATIVE_DIFFICULTY_SEPARATION",
        "PAYLOAD_DUPLICATION",
        "INTRO_EVIDENCE",
        "INTRO_DISTANCE",
        "QUALITY_REVIEW",
        "PUBLICATION_TIER",
        "RECOVERY_TRUST",
        "TIMING_F1",
        "SOURCE_REASSIGNMENT",
        "CURRENT_ASSIGNMENT_CHANGES",
        "ATTEMPT_COST",
    )
    for name, before, after in zip(
        names,
        current.objective(),
        selected.objective(),
        strict=True,
    ):
        if before != after:
            return name
    return "DETERMINISTIC_TIE_BREAK"


def select_safe_family_assignment(
    candidates: tuple[SafeFamilyCandidate, ...],
    *,
    current_assignment: Assignment,
    post_resolution_ordering: bool = False,
) -> SafeFamilyAssignmentDecision:
    """Choose four target labels from one key without mutating any payload."""
    if type(candidates) is not tuple or not candidates:
        raise TypeError("candidates must be a non-empty tuple")
    if any(not isinstance(candidate, SafeFamilyCandidate) for candidate in candidates):
        raise TypeError("candidates must contain SafeFamilyCandidate values")
    if type(post_resolution_ordering) is not bool:
        raise TypeError("post_resolution_ordering must be an exact boolean")
    key_modes = {candidate.key_mode for candidate in candidates}
    if len(key_modes) != 1:
        raise ValueError("candidates must belong to one key mode")
    by_id = {candidate.candidate_id: candidate for candidate in candidates}
    if len(by_id) != len(candidates):
        raise ValueError("candidate identities must be unique")
    current = _normalize_assignment(current_assignment, candidates=set(by_id))
    current_candidates = tuple(by_id[candidate_id] for _difficulty, candidate_id in current)
    metric_ranges = _metric_ranges(candidates)
    current_score = _score(
        current_candidates,
        current=current,
        metric_ranges=metric_ranges,
    )

    best: tuple[SafeFamilyCandidate, ...] | None = None
    best_score: SafeFamilyScore | None = None
    best_key: tuple[object, ...] | None = None
    best_feasible: tuple[SafeFamilyCandidate, ...] | None = None
    best_feasible_score: SafeFamilyScore | None = None
    best_feasible_key: tuple[object, ...] | None = None
    assignments_evaluated = 0
    safe_substitutes = _bounded_safe_substitutes(candidates)
    family_requires_substitution = bool(
        _family_feasibility_reasons(current_score)
    )
    options_by_target = tuple(
        (current_candidate,)
        if post_resolution_ordering
        else tuple(
            dict.fromkeys(
                (
                    *current_candidates,
                    *(
                        safe_substitutes
                        if family_requires_substitution
                        or _authorizes_existing_candidate_substitution(current_candidate)
                        else ()
                    ),
                )
            )
        )
        for current_candidate in current_candidates
    )
    for assignment in product(*options_by_target):
        assignments_evaluated += 1
        score = _score(
            assignment,
            current=current,
            metric_ranges=metric_ranges,
        )
        # Payload uniqueness is a publication invariant, not a soft scoring
        # preference.  Keeping duplicate assignments in this comparison is
        # what previously allowed NORMAL bytes to be relabelled as EXPERT.
        if score.duplicate_payloads:
            continue
        if not _preserves_target_quality(assignment, current=current_candidates):
            continue
        key = (score.objective(), tuple(candidate.candidate_id for candidate in assignment))
        if best_key is None or key < best_key:
            best = assignment
            best_score = score
            best_key = key
        if not _family_feasibility_reasons(score) and (
            best_feasible_key is None or key < best_feasible_key
        ):
            best_feasible = assignment
            best_feasible_score = score
            best_feasible_key = key

    unique_payload_status: UniquePayloadStatus = "SATISFIED"
    if post_resolution_ordering:
        # The last pipeline phase has no candidate-substitution authority.
        # Candidate selection, compilation, recovery, and fallback supply have
        # already finished; only permute the four frozen current payloads.
        best = current_candidates
        best_score = current_score
        if current_score.duplicate_payloads:
            unique_payload_status = "UNAVAILABLE"
    elif best is None or best_score is None:
        # Preserve the immutable current snapshot only as explicit evidence for
        # the bounded compiler/retry stage.  Publication has a second guard and
        # cannot promote this unresolved duplicate family.
        best = current_candidates
        best_score = current_score
        unique_payload_status = "UNAVAILABLE"

    if (
        not post_resolution_ordering
        and best_feasible is not None
        and best_feasible_score is not None
    ):
        best = best_feasible
        best_score = best_feasible_score

    post_resolution_ordering_status: PostResolutionOrderingStatus = "NOT_REQUESTED"
    if post_resolution_ordering:
        (
            best,
            best_score,
            post_resolution_ordering_status,
        ) = _post_resolution_order(
            best,
            current=current,
            metric_ranges=metric_ranges,
        )

    feasibility_reasons = _family_feasibility_reasons(best_score)
    family_feasibility_status: FamilyFeasibilityStatus = (
        "UNAVAILABLE" if feasibility_reasons else "SATISFIED"
    )

    selected = tuple(
        (difficulty, candidate.candidate_id)
        for difficulty, candidate in zip(DIFFICULTIES, best, strict=True)
    )
    source_difficulties = tuple(
        (difficulty, candidate.source_difficulty)
        for difficulty, candidate in zip(DIFFICULTIES, best, strict=True)
    )
    reassigned = tuple(
        difficulty
        for difficulty, candidate in zip(DIFFICULTIES, best, strict=True)
        if difficulty != candidate.source_difficulty
    )
    duplicates = _duplicate_slots(best)
    changed = selected != current
    return SafeFamilyAssignmentDecision(
        key_mode=next(iter(key_modes)),
        current_assignment=current,
        selected_assignment=selected,
        current_score=current_score,
        selected_score=best_score,
        source_difficulties=source_difficulties,
        reassigned_slots=reassigned,
        emergency_duplicate_slots=duplicates,
        unique_payload_status=unique_payload_status,
        family_feasibility_status=family_feasibility_status,
        family_feasibility_reasons=feasibility_reasons,
        reason=(
            "POST_RESOLUTION_DIFFICULTY_ORDERING_UNAVAILABLE"
            if post_resolution_ordering_status == "UNAVAILABLE"
            else "POST_RESOLUTION_DIFFICULTY_ORDERED"
            if post_resolution_ordering_status == "ORDERED"
            else "POST_RESOLUTION_DIFFICULTY_ALREADY_ORDERED"
            if post_resolution_ordering_status == "ALREADY_ORDERED"
            else "UNIQUE_PAYLOAD_UNAVAILABLE"
            if unique_payload_status == "UNAVAILABLE"
            else "FAMILY_FEASIBILITY_UNAVAILABLE"
            if family_feasibility_status == "UNAVAILABLE"
            else _reason(current_score, best_score, changed=changed)
        ),
        candidates_evaluated=len(candidates),
        assignments_evaluated=assignments_evaluated,
        post_resolution_ordering_status=post_resolution_ordering_status,
    )
