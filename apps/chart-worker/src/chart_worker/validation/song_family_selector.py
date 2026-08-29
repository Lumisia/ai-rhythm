"""Bounded, deterministic song-wide candidate selection.

The selector is intentionally independent from stage-private candidate objects.
It receives immutable snapshots, rejects mixed evaluation contexts, prunes only
within a slot, and compares 4K/6K/7K families after every recovery stage.
"""

from __future__ import annotations

import hashlib
import json
import math
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
MATCHED_RECALL_EPSILON = 0.005
MATCHED_F1_EPSILON = 0.005
# A hard-ineligible fallback is allowed to yield to a playable challenger when
# their one-to-one timing F1 is broadly comparable.  This is a conservative
# policy margin, not a calibrated musical-quality guarantee; larger tradeoffs
# remain REVIEW until human-labelled calibration is available.
MAX_HARD_INELIGIBLE_MATCHED_F1_REGRESSION = 0.05
MIN_DIFFICULTY_GAP = 0.30
MAX_KEY_FAMILIES = 24
SONG_SELECTION_REPLAY_VERSION = "song-selection-replay-v2"


@dataclass(frozen=True, slots=True)
class TimingSectionSnapshot:
    row_count: int
    matched_precision_50: float | None


@dataclass(frozen=True, slots=True)
class ProtectedMetrics:
    row_count: int
    onset_count: int
    matched_count_50: int
    matched_precision_50: float | None
    matched_recall_50: float | None
    matched_f1_50: float | None
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
    candidate_payload_sha256: str

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
                "rowCount": self.protected_metrics.row_count,
                "onsetCount": self.protected_metrics.onset_count,
                "matchedCount50": self.protected_metrics.matched_count_50,
                "matchedPrecision50": self.protected_metrics.matched_precision_50,
                "matchedRecall50": self.protected_metrics.matched_recall_50,
                "matchedF150": self.protected_metrics.matched_f1_50,
                "activeGapCount": self.protected_metrics.active_gap_count,
                "holdIntegrityViolations": (
                    self.protected_metrics.hold_integrity_violations
                ),
                "reviewRank": self.protected_metrics.review_rank,
            },
            "difficultyOrderingScore": self.difficulty_ordering_score,
            "firstRowMs": self.first_row_ms,
            "timingSections": [
                {
                    "rowCount": section.row_count,
                    "matchedPrecision50": section.matched_precision_50,
                }
                for section in self.timing_sections
            ],
            "candidatePayloadRef": self.candidate_payload_ref,
            "candidatePayloadSha256": self.candidate_payload_sha256,
        }


@dataclass(frozen=True, slots=True)
class SongSelectionReplayInput:
    context_id: str
    canonical_first_row_ms: int | None
    candidates: tuple[CandidateSnapshot, ...]
    current_assignment: tuple[tuple[str, str | None], ...]

    def to_report(self) -> dict[str, object]:
        return {
            "version": SONG_SELECTION_REPLAY_VERSION,
            "contextId": self.context_id,
            "canonicalFirstRowMs": self.canonical_first_row_ms,
            "currentAssignment": dict(self.current_assignment),
            "candidates": [item.to_report() for item in self.candidates],
        }

    def stable_sha256(self) -> str:
        serialized = json.dumps(
            self.to_report(),
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(serialized).hexdigest()


@dataclass(frozen=True, slots=True)
class SongFamilyScore:
    missing_slots: int
    hard_violations: int
    intro_violations: int
    cross_key_outliers: int
    difficulty_violations: int
    difficulty_unscored_pairs: int
    difficulty_deficit: float
    review_rank: int
    hold_integrity_violations: int
    active_gap_count: int
    worst_matched_f1_50: float | None
    worst_matched_precision_50: float | None
    total_attempts: int

    def objective(self) -> tuple[object, ...]:
        return (
            self.missing_slots,
            self.hard_violations,
            self.intro_violations,
            self.cross_key_outliers,
            self.difficulty_violations,
            self.difficulty_unscored_pairs,
            self.difficulty_deficit,
            self.review_rank,
            self.hold_integrity_violations,
            self.active_gap_count,
            -(
                self.worst_matched_f1_50
                if self.worst_matched_f1_50 is not None
                else -1.0
            ),
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
            "difficultyUnscoredPairs": self.difficulty_unscored_pairs,
            "difficultyDeficit": self.difficulty_deficit,
            "reviewRank": self.review_rank,
            "holdIntegrityViolations": self.hold_integrity_violations,
            "activeGapCount": self.active_gap_count,
            "worstMatchedF150": self.worst_matched_f1_50,
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
    replay_input: SongSelectionReplayInput

    @property
    def replay_input_sha256(self) -> str:
        return self.replay_input.stable_sha256()

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
            "replayInput": self.replay_input.to_report(),
            "replayInputSha256": self.replay_input_sha256,
        }


Slot = tuple[int, str]
Assignment = dict[Slot, str | None]


def _require_mapping(value: object, *, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(type(key) is not str for key in value):
        raise ValueError(f"{name} must be an object with string keys")
    return value


def _require_keys(
    value: Mapping[str, object],
    expected: set[str],
    *,
    name: str,
) -> None:
    actual = set(value)
    if actual != expected:
        raise ValueError(
            f"{name} keys differ: missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )


def _require_string(value: object, *, name: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _require_sha256(value: object, *, name: str) -> str:
    digest = _require_string(value, name=name)
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return digest


def _require_int(
    value: object,
    *,
    name: str,
    minimum: int | None = None,
) -> int:
    if type(value) is not int:
        raise ValueError(f"{name} must be an integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def _require_optional_float(value: object, *, name: str) -> float | None:
    if value is None:
        return None
    if type(value) is not float or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite float or null")
    return value


def _require_optional_probability(value: object, *, name: str) -> float | None:
    number = _require_optional_float(value, name=name)
    if number is not None and not 0.0 <= number <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1")
    return number


def _validate_matched_metrics(
    *,
    name: str,
    row_count: int,
    onset_count: int,
    matched_count_50: int,
    matched_precision_50: float | None,
    matched_recall_50: float | None,
    matched_f1_50: float | None,
) -> None:
    if matched_count_50 > min(row_count, onset_count):
        raise ValueError(f"{name}.matchedCount50 exceeds row/onset counts")
    if row_count == 0 or onset_count == 0:
        if matched_count_50 != 0 or any(
            value is not None
            for value in (
                matched_precision_50,
                matched_recall_50,
                matched_f1_50,
            )
        ):
            raise ValueError(f"{name} empty evidence must have null matched metrics")
        return
    expected = {
        "matchedPrecision50": round(matched_count_50 / row_count, 6),
        "matchedRecall50": round(matched_count_50 / onset_count, 6),
        "matchedF150": round(2 * matched_count_50 / (row_count + onset_count), 6),
    }
    actual = {
        "matchedPrecision50": matched_precision_50,
        "matchedRecall50": matched_recall_50,
        "matchedF150": matched_f1_50,
    }
    for field, expected_value in expected.items():
        value = actual[field]
        if value is None or not math.isclose(
            value,
            expected_value,
            rel_tol=0.0,
            abs_tol=5e-7,
        ):
            raise ValueError(
                f"{name}.{field} is inconsistent with matchedCount50"
            )


def _candidate_from_report(value: object, *, index: int) -> CandidateSnapshot:
    name = f"replayInput.candidates[{index}]"
    report = _require_mapping(value, name=name)
    _require_keys(
        report,
        {
            "candidateId",
            "contextId",
            "keyMode",
            "difficulty",
            "attempt",
            "seed",
            "provenance",
            "hardEligible",
            "axisActions",
            "protectedMetrics",
            "difficultyOrderingScore",
            "firstRowMs",
            "timingSections",
            "candidatePayloadRef",
            "candidatePayloadSha256",
        },
        name=name,
    )
    key_mode = _require_int(report["keyMode"], name=f"{name}.keyMode")
    if key_mode not in KEY_MODES:
        raise ValueError(f"{name}.keyMode is unsupported")
    difficulty = _require_string(report["difficulty"], name=f"{name}.difficulty")
    if difficulty not in DIFFICULTIES:
        raise ValueError(f"{name}.difficulty is unsupported")
    hard_eligible = report["hardEligible"]
    if type(hard_eligible) is not bool:
        raise ValueError(f"{name}.hardEligible must be a boolean")
    axis_report = _require_mapping(report["axisActions"], name=f"{name}.axisActions")
    axis_actions = tuple(
        (
            _require_string(axis, name=f"{name}.axisActions key"),
            _require_string(action, name=f"{name}.axisActions[{axis}]"),
        )
        for axis, action in sorted(axis_report.items())
    )
    metrics = _require_mapping(
        report["protectedMetrics"], name=f"{name}.protectedMetrics"
    )
    _require_keys(
        metrics,
        {
            "rowCount",
            "onsetCount",
            "matchedCount50",
            "matchedPrecision50",
            "matchedRecall50",
            "matchedF150",
            "activeGapCount",
            "holdIntegrityViolations",
            "reviewRank",
        },
        name=f"{name}.protectedMetrics",
    )
    row_count = _require_int(
        metrics["rowCount"],
        name=f"{name}.protectedMetrics.rowCount",
        minimum=0,
    )
    onset_count = _require_int(
        metrics["onsetCount"],
        name=f"{name}.protectedMetrics.onsetCount",
        minimum=0,
    )
    matched_count_50 = _require_int(
        metrics["matchedCount50"],
        name=f"{name}.protectedMetrics.matchedCount50",
        minimum=0,
    )
    matched_precision_50 = _require_optional_probability(
        metrics["matchedPrecision50"],
        name=f"{name}.protectedMetrics.matchedPrecision50",
    )
    matched_recall_50 = _require_optional_probability(
        metrics["matchedRecall50"],
        name=f"{name}.protectedMetrics.matchedRecall50",
    )
    matched_f1_50 = _require_optional_probability(
        metrics["matchedF150"],
        name=f"{name}.protectedMetrics.matchedF150",
    )
    _validate_matched_metrics(
        name=f"{name}.protectedMetrics",
        row_count=row_count,
        onset_count=onset_count,
        matched_count_50=matched_count_50,
        matched_precision_50=matched_precision_50,
        matched_recall_50=matched_recall_50,
        matched_f1_50=matched_f1_50,
    )
    raw_sections = report["timingSections"]
    if type(raw_sections) is not list:
        raise ValueError(f"{name}.timingSections must be an array")
    sections = []
    for section_index, raw_section in enumerate(raw_sections):
        section_name = f"{name}.timingSections[{section_index}]"
        section = _require_mapping(raw_section, name=section_name)
        _require_keys(
            section,
            {"rowCount", "matchedPrecision50"},
            name=section_name,
        )
        sections.append(
            TimingSectionSnapshot(
                row_count=_require_int(
                    section["rowCount"],
                    name=f"{section_name}.rowCount",
                    minimum=0,
                ),
                matched_precision_50=_require_optional_float(
                    section["matchedPrecision50"],
                    name=f"{section_name}.matchedPrecision50",
                ),
            )
        )
    first_row = report["firstRowMs"]
    if first_row is not None:
        first_row = _require_int(first_row, name=f"{name}.firstRowMs", minimum=0)
    return CandidateSnapshot(
        candidate_id=_require_string(
            report["candidateId"], name=f"{name}.candidateId"
        ),
        context_id=_require_string(report["contextId"], name=f"{name}.contextId"),
        key_mode=key_mode,
        difficulty=difficulty,
        attempt=_require_int(report["attempt"], name=f"{name}.attempt", minimum=1),
        seed=_require_int(report["seed"], name=f"{name}.seed"),
        provenance=_require_string(
            report["provenance"], name=f"{name}.provenance"
        ),
        hard_eligible=hard_eligible,
        axis_actions=axis_actions,
        protected_metrics=ProtectedMetrics(
            row_count=row_count,
            onset_count=onset_count,
            matched_count_50=matched_count_50,
            matched_precision_50=matched_precision_50,
            matched_recall_50=matched_recall_50,
            matched_f1_50=matched_f1_50,
            active_gap_count=_require_int(
                metrics["activeGapCount"],
                name=f"{name}.protectedMetrics.activeGapCount",
                minimum=0,
            ),
            hold_integrity_violations=_require_int(
                metrics["holdIntegrityViolations"],
                name=f"{name}.protectedMetrics.holdIntegrityViolations",
                minimum=0,
            ),
            review_rank=_require_int(
                metrics["reviewRank"],
                name=f"{name}.protectedMetrics.reviewRank",
                minimum=0,
            ),
        ),
        difficulty_ordering_score=_require_optional_float(
            report["difficultyOrderingScore"],
            name=f"{name}.difficultyOrderingScore",
        ),
        first_row_ms=first_row,
        timing_sections=tuple(sections),
        candidate_payload_ref=_require_string(
            report["candidatePayloadRef"], name=f"{name}.candidatePayloadRef"
        ),
        candidate_payload_sha256=_require_sha256(
            report["candidatePayloadSha256"],
            name=f"{name}.candidatePayloadSha256",
        ),
    )


def _replay_input_from_report(value: object) -> SongSelectionReplayInput:
    report = _require_mapping(value, name="replayInput")
    _require_keys(
        report,
        {
            "version",
            "contextId",
            "canonicalFirstRowMs",
            "currentAssignment",
            "candidates",
        },
        name="replayInput",
    )
    if report["version"] != SONG_SELECTION_REPLAY_VERSION:
        raise ValueError("unsupported replay input version")
    context_id = _require_string(report["contextId"], name="replayInput.contextId")
    canonical_first_row = report["canonicalFirstRowMs"]
    if canonical_first_row is not None:
        canonical_first_row = _require_int(
            canonical_first_row,
            name="replayInput.canonicalFirstRowMs",
            minimum=0,
        )
    assignment = _require_mapping(
        report["currentAssignment"], name="replayInput.currentAssignment"
    )
    expected_assignment_keys = {
        f"{key_mode}K:{difficulty}"
        for key_mode in KEY_MODES
        for difficulty in DIFFICULTIES
    }
    _require_keys(
        assignment,
        expected_assignment_keys,
        name="replayInput.currentAssignment",
    )
    normalized_assignment = []
    for slot, candidate_id in sorted(assignment.items()):
        if candidate_id is not None:
            candidate_id = _require_string(
                candidate_id, name=f"replayInput.currentAssignment[{slot}]"
            )
        normalized_assignment.append((slot, candidate_id))
    raw_candidates = report["candidates"]
    if type(raw_candidates) is not list:
        raise ValueError("replayInput.candidates must be an array")
    candidates = tuple(
        sorted(
            (
                _candidate_from_report(item, index=index)
                for index, item in enumerate(raw_candidates)
            ),
            key=lambda item: item.candidate_id,
        )
    )
    return SongSelectionReplayInput(
        context_id=context_id,
        canonical_first_row_ms=canonical_first_row,
        candidates=candidates,
        current_assignment=tuple(normalized_assignment),
    )


def _metric_value(value: float | None) -> float:
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
        _metric_value(lp.matched_precision_50)
        >= _metric_value(rp.matched_precision_50) - MATCHED_PRECISION_EPSILON
        and _metric_value(lp.matched_recall_50)
        >= _metric_value(rp.matched_recall_50) - MATCHED_PRECISION_EPSILON
        and _metric_value(lp.matched_f1_50)
        >= _metric_value(rp.matched_f1_50) - MATCHED_F1_EPSILON
        and lp.active_gap_count <= rp.active_gap_count
        and lp.hold_integrity_violations <= rp.hold_integrity_violations
        and lp.review_rank <= rp.review_rank
        and left.attempt <= right.attempt
    )
    strictly_better = (
        _metric_value(lp.matched_precision_50)
        > _metric_value(rp.matched_precision_50) + MATCHED_PRECISION_EPSILON
        or _metric_value(lp.matched_recall_50)
        > _metric_value(rp.matched_recall_50) + MATCHED_PRECISION_EPSILON
        or _metric_value(lp.matched_f1_50)
        > _metric_value(rp.matched_f1_50) + MATCHED_F1_EPSILON
        or lp.active_gap_count < rp.active_gap_count
        or lp.hold_integrity_violations < rp.hold_integrity_violations
        or lp.review_rank < rp.review_rank
        or left.attempt < right.attempt
    )
    return no_worse and strictly_better


def _prune_slot(
    candidates: tuple[CandidateSnapshot, ...],
    *,
    protected_candidate_id: str | None,
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
    if protected_candidate_id is not None:
        protected = next(
            (
                item
                for item in candidates
                if item.candidate_id == protected_candidate_id
            ),
            None,
        )
        if protected is not None and protected not in retained:
            retained.append(protected)
            pruned.discard(protected_candidate_id)
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
    current_metrics = current.protected_metrics
    challenger_metrics = challenger.protected_metrics
    if (
        challenger_metrics.hold_integrity_violations
        > current_metrics.hold_integrity_violations
        or challenger_metrics.active_gap_count > current_metrics.active_gap_count
    ):
        return False
    if (
        canonical_first_row_ms is not None
        and current.first_row_ms == canonical_first_row_ms
        and challenger.first_row_ms != canonical_first_row_ms
    ):
        return False
    if current.hard_eligible:
        if challenger_metrics.review_rank > current_metrics.review_rank:
            return False
        for current_value, challenger_value, epsilon in (
            (
                current_metrics.matched_precision_50,
                challenger_metrics.matched_precision_50,
                MATCHED_PRECISION_EPSILON,
            ),
            (
                current_metrics.matched_recall_50,
                challenger_metrics.matched_recall_50,
                MATCHED_PRECISION_EPSILON,
            ),
            (
                current_metrics.matched_f1_50,
                challenger_metrics.matched_f1_50,
                MATCHED_F1_EPSILON,
            ),
        ):
            if current_value is not None and (
                challenger_value is None
                or challenger_value < current_value - epsilon
            ):
                return False
        return True
    current_f1 = current_metrics.matched_f1_50
    challenger_f1 = challenger_metrics.matched_f1_50
    return current_f1 is None or (
        challenger_f1 is not None
        and challenger_f1
        >= current_f1 - MAX_HARD_INELIGIBLE_MATCHED_F1_REGRESSION
        and challenger_metrics.matched_count_50
        >= current_metrics.matched_count_50
        and (
            current_metrics.matched_recall_50 is None
            or (
                challenger_metrics.matched_recall_50 is not None
                and challenger_metrics.matched_recall_50
                >= current_metrics.matched_recall_50 - MATCHED_RECALL_EPSILON
            )
        )
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


def _difficulty_deficit(candidates: tuple[CandidateSnapshot, ...]) -> float:
    ordered = sorted(candidates, key=lambda item: DIFFICULTIES.index(item.difficulty))
    deficit = 0.0
    for easier, harder in pairwise(ordered):
        if (
            easier.difficulty_ordering_score is None
            or harder.difficulty_ordering_score is None
        ):
            continue
        gap = harder.difficulty_ordering_score - easier.difficulty_ordering_score
        deficit += max(0.0, MIN_DIFFICULTY_GAP - gap)
    return round(deficit, 6)


def _difficulty_unscored_pairs(candidates: tuple[CandidateSnapshot, ...]) -> int:
    ordered = sorted(candidates, key=lambda item: DIFFICULTIES.index(item.difficulty))
    return sum(
        easier.difficulty_ordering_score is None
        or harder.difficulty_ordering_score is None
        for easier, harder in pairwise(ordered)
    )


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
    f1_values = tuple(
        item.protected_metrics.matched_f1_50
        for item in chosen
        if item.protected_metrics.matched_f1_50 is not None
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
        difficulty_unscored_pairs=sum(
            _difficulty_unscored_pairs(
                tuple(item for item in chosen if item.key_mode == key_mode)
            )
            for key_mode in KEY_MODES
        ),
        difficulty_deficit=round(
            sum(
                _difficulty_deficit(
                    tuple(item for item in chosen if item.key_mode == key_mode)
                )
                for key_mode in KEY_MODES
            ),
            6,
        ),
        review_rank=sum(item.protected_metrics.review_rank for item in chosen),
        hold_integrity_violations=sum(
            item.protected_metrics.hold_integrity_violations for item in chosen
        ),
        active_gap_count=sum(
            item.protected_metrics.active_gap_count for item in chosen
        ),
        worst_matched_f1_50=(min(f1_values) if f1_values else None),
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
    replay_input = SongSelectionReplayInput(
        context_id=context_id,
        canonical_first_row_ms=canonical_first_row_ms,
        candidates=tuple(sorted(all_candidates, key=lambda item: item.candidate_id)),
        current_assignment=tuple(sorted(_report_assignment(current).items())),
    )

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
        retained, removed = _prune_slot(
            candidates,
            protected_candidate_id=current[slot],
        )
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
            "DIFFICULTY_EVIDENCE",
            "DIFFICULTY_SEVERITY",
            "QUALITY_STATUS",
            "HOLD_INTEGRITY",
            "ACTIVE_GAPS",
            "WORST_MATCHED_F1",
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
        replay_input=replay_input,
    )
    return (best if mode == "V2" else current), comparison


def replay_song_selection_report(
    report: Mapping[str, object],
    *,
    mode: SelectorMode,
) -> tuple[Assignment, SongSelectionComparison]:
    """Replay a recorded selection without audio, model inference, or mutation."""
    replay_input = _replay_input_from_report(report.get("replayInput"))
    recorded_digest = report.get("replayInputSha256")
    if (
        type(recorded_digest) is not str
        or len(recorded_digest) != 64
        or any(character not in "0123456789abcdef" for character in recorded_digest)
        or recorded_digest != replay_input.stable_sha256()
    ):
        raise ValueError("replay input digest does not match candidate evidence")

    pools: dict[Slot, list[CandidateSnapshot]] = {
        (key_mode, difficulty): []
        for key_mode in KEY_MODES
        for difficulty in DIFFICULTIES
    }
    for candidate in replay_input.candidates:
        pools[(candidate.key_mode, candidate.difficulty)].append(candidate)
    reported_current = dict(replay_input.current_assignment)
    current: Assignment = {
        (key_mode, difficulty): reported_current[f"{key_mode}K:{difficulty}"]
        for key_mode in KEY_MODES
        for difficulty in DIFFICULTIES
    }
    selected, comparison = compare_song_families(
        {slot: tuple(candidates) for slot, candidates in pools.items()},
        current,
        canonical_first_row_ms=replay_input.canonical_first_row_ms,
        mode=mode,
    )
    if comparison.replay_input_sha256 != recorded_digest:
        raise ValueError("replayed selection changed the recorded evidence digest")
    return selected, comparison
