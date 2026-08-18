"""Final difficulty-family observations without calibrated-tier claims."""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import pairwise
from typing import Literal

from chart_worker.schema.types import DIFFICULTIES, KEY_MODES

FINAL_DIFFICULTY_FAMILY_OBSERVATION_VERSION = "final-difficulty-family-observation-v1"
_RECOVERY_PROVENANCES = frozenset(
    {
        "PARTIAL_REMAP",
        "INTRO_RECOVERY",
        "INTRO_ALIGNED",
        "COVERAGE_REPAIR",
        "RAW_UNVERIFIED",
        "SAFE_FALLBACK",
    }
)


@dataclass(frozen=True, slots=True)
class DifficultyFamilyEntry:
    difficulty: str
    provenance: str
    project_rating: float | None
    ordering_score: float | None

    def __post_init__(self) -> None:
        if type(self.difficulty) is not str or self.difficulty not in DIFFICULTIES:
            raise ValueError(f"unsupported difficulty: {self.difficulty!r}")
        if type(self.provenance) is not str or not self.provenance:
            raise ValueError("provenance must be a non-empty exact string")
        for field, value in (
            ("project_rating", self.project_rating),
            ("ordering_score", self.ordering_score),
        ):
            if value is None:
                continue
            if type(value) not in {int, float} or not math.isfinite(value):
                raise ValueError(f"{field} must be a finite exact number or None")

    def to_report(self) -> dict[str, object]:
        return {
            "difficulty": self.difficulty,
            "provenance": self.provenance,
            "projectRating": self.project_rating,
            "orderingScore": self.ordering_score,
        }


@dataclass(frozen=True, slots=True)
class FinalDifficultyFamilyObservation:
    version: Literal["final-difficulty-family-observation-v1"]
    key_mode: int
    calibration_state: Literal["UNAVAILABLE"]
    contract_status: Literal["UNCALIBRATED"]
    provisional_concern: Literal[
        "NONE",
        "INCOMPLETE_EVIDENCE",
        "METRIC_DISAGREEMENT",
        "CONSENSUS_INVERSION",
        "RECOVERY_INVERSION",
    ]
    entries: tuple[DifficultyFamilyEntry, ...]
    missing_difficulties: tuple[str, ...]
    missing_metric_difficulties: tuple[str, ...]
    project_rating_inversions: tuple[tuple[str, str], ...]
    ordering_score_inversions: tuple[tuple[str, str], ...]
    recovery_difficulties: tuple[str, ...]

    def to_report(self) -> dict[str, object]:
        return {
            "version": self.version,
            "keyMode": self.key_mode,
            "calibrationState": self.calibration_state,
            "contractStatus": self.contract_status,
            "provisionalConcern": self.provisional_concern,
            "entries": [entry.to_report() for entry in self.entries],
            "missingDifficulties": list(self.missing_difficulties),
            "missingMetricDifficulties": list(self.missing_metric_difficulties),
            "projectRatingInversions": [
                list(pair) for pair in self.project_rating_inversions
            ],
            "orderingScoreInversions": [
                list(pair) for pair in self.ordering_score_inversions
            ],
            "recoveryDifficulties": list(self.recovery_difficulties),
            "policyState": "OBSERVATION_ONLY",
            "mutatesSelection": False,
            "mutatesCharts": False,
        }


def _inversions(
    entries: tuple[DifficultyFamilyEntry, ...],
    field: Literal["project_rating", "ordering_score"],
) -> tuple[tuple[str, str], ...]:
    available = tuple(entry for entry in entries if getattr(entry, field) is not None)
    return tuple(
        (easier.difficulty, harder.difficulty)
        for easier, harder in pairwise(available)
        if getattr(harder, field) < getattr(easier, field)
    )


def observe_final_difficulty_family(
    key_mode: int,
    entries: tuple[DifficultyFamilyEntry, ...],
) -> FinalDifficultyFamilyObservation:
    """Compare within-key labels while keeping all scores explicitly uncalibrated."""
    if type(key_mode) is not int or key_mode not in KEY_MODES:
        raise ValueError(f"unsupported key mode: {key_mode!r}")
    if type(entries) is not tuple or not entries:
        raise TypeError("entries must be a non-empty tuple")
    if any(not isinstance(entry, DifficultyFamilyEntry) for entry in entries):
        raise TypeError("entries must contain DifficultyFamilyEntry values")
    difficulties = tuple(entry.difficulty for entry in entries)
    if len(set(difficulties)) != len(difficulties):
        raise ValueError("difficulty entries must be unique")
    ordered = tuple(sorted(entries, key=lambda entry: DIFFICULTIES.index(entry.difficulty)))
    missing = tuple(difficulty for difficulty in DIFFICULTIES if difficulty not in difficulties)
    missing_metrics = tuple(
        entry.difficulty
        for entry in ordered
        if entry.project_rating is None or entry.ordering_score is None
    )
    project_inversions = _inversions(ordered, "project_rating")
    ordering_inversions = _inversions(ordered, "ordering_score")
    recovery_difficulties = tuple(
        entry.difficulty for entry in ordered if entry.provenance in _RECOVERY_PROVENANCES
    )

    if missing or missing_metrics:
        concern = "INCOMPLETE_EVIDENCE"
    elif set(project_inversions) != set(ordering_inversions):
        concern = "METRIC_DISAGREEMENT"
    elif project_inversions:
        inverted_harder = {harder for _easier, harder in project_inversions}
        concern = (
            "RECOVERY_INVERSION"
            if inverted_harder.intersection(recovery_difficulties)
            else "CONSENSUS_INVERSION"
        )
    else:
        concern = "NONE"

    return FinalDifficultyFamilyObservation(
        version=FINAL_DIFFICULTY_FAMILY_OBSERVATION_VERSION,
        key_mode=key_mode,
        calibration_state="UNAVAILABLE",
        contract_status="UNCALIBRATED",
        provisional_concern=concern,
        entries=ordered,
        missing_difficulties=missing,
        missing_metric_difficulties=missing_metrics,
        project_rating_inversions=project_inversions,
        ordering_score_inversions=ordering_inversions,
        recovery_difficulties=recovery_difficulties,
    )
