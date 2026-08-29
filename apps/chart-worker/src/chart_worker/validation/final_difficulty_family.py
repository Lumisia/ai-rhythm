"""Final difficulty-family observations without calibrated-tier claims."""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import pairwise
from typing import Literal, TypeAlias

from chart_worker.schema.types import DIFFICULTIES, KEY_MODES

FINAL_DIFFICULTY_FAMILY_OBSERVATION_VERSION = "final-difficulty-family-observation-v2"
CalibrationState: TypeAlias = Literal[
    "UNAVAILABLE",
    "PILOT_ONLY",
    "REPORT_ONLY_VALIDATED",
    "ACTIVATION_ELIGIBLE",
    "ENFORCED",
]
CalibrationContractStatus: TypeAlias = Literal["UNCALIBRATED", "CALIBRATED"]
CALIBRATION_STATES: tuple[CalibrationState, ...] = (
    "UNAVAILABLE",
    "PILOT_ONLY",
    "REPORT_ONLY_VALIDATED",
    "ACTIVATION_ELIGIBLE",
    "ENFORCED",
)
# The frozen evidence is a single-rater enriched pilot.  It supports screening
# research, but it is neither an independent production holdout nor an active
# calibration model.  Callers must pass this state explicitly.
CURRENT_DIFFICULTY_FAMILY_CALIBRATION_STATE: Literal["PILOT_ONLY"] = "PILOT_ONLY"
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
    version: Literal["final-difficulty-family-observation-v2"]
    key_mode: int
    calibration_state: CalibrationState
    contract_status: CalibrationContractStatus
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

    @property
    def requires_review(self) -> bool:
        """Whether relative-order evidence is unresolved.

        This does not claim calibrated player tiers.  It only prevents an
        observed inversion, metric disagreement, or incomplete family from
        being reported as a clean quality PASS.
        """

        return self.provisional_concern != "NONE"

    @property
    def resolution_status(self) -> Literal["NO_OBSERVED_CONCERN", "UNRESOLVED"]:
        """Describe observation status without implying calibrated correctness."""

        return "UNRESOLVED" if self.requires_review else "NO_OBSERVED_CONCERN"

    @property
    def production_calibration_enforced(self) -> bool:
        """Whether an independently activated calibration contract is live."""

        return self.calibration_state == "ENFORCED"

    def to_report(self) -> dict[str, object]:
        return {
            "version": self.version,
            "keyMode": self.key_mode,
            "calibrationState": self.calibration_state,
            "contractStatus": self.contract_status,
            "productionCalibrationEnforced": self.production_calibration_enforced,
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
            "requiresReview": self.requires_review,
            "resolutionStatus": self.resolution_status,
            "policyState": "REPORTING_ENFORCED",
            "mutatesSelection": False,
            "mutatesCharts": False,
            "mutatesQualityStatus": True,
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
    *,
    calibration_state: CalibrationState,
) -> FinalDifficultyFamilyObservation:
    """Compare within-key labels under one caller-declared calibration state."""
    if type(key_mode) is not int or key_mode not in KEY_MODES:
        raise ValueError(f"unsupported key mode: {key_mode!r}")
    if type(entries) is not tuple or not entries:
        raise TypeError("entries must be a non-empty tuple")
    if any(not isinstance(entry, DifficultyFamilyEntry) for entry in entries):
        raise TypeError("entries must contain DifficultyFamilyEntry values")
    if type(calibration_state) is not str or calibration_state not in CALIBRATION_STATES:
        raise ValueError("calibration_state is unsupported")
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
        calibration_state=calibration_state,
        contract_status=(
            "CALIBRATED" if calibration_state == "ENFORCED" else "UNCALIBRATED"
        ),
        provisional_concern=concern,
        entries=ordered,
        missing_difficulties=missing,
        missing_metric_difficulties=missing_metrics,
        project_rating_inversions=project_inversions,
        ordering_score_inversions=ordering_inversions,
        recovery_difficulties=recovery_difficulties,
    )
