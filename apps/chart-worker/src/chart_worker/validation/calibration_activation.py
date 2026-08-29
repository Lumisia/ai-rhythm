"""Pre-registered report-only activation evidence for difficulty calibration."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Literal

import numpy as np

_ONE_SIDED_ALPHA = 0.05
_MAX_SEVERE_REGRESSION_RATE_UPPER = 0.05
_BOOTSTRAP_RESAMPLES = 20_000


def _sha256(value: object, *, name: str) -> str:
    if type(value) is not str or len(value) != 64:
        raise TypeError(f"{name} must be a lowercase SHA-256 digest")
    if any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _non_negative_int(value: object, *, name: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be an exact integer")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True, slots=True)
class HeldOutSongComparisonV3:
    audio_sha256: str
    current_disagreement: bool
    calibrated_disagreement: bool

    def __post_init__(self) -> None:
        _sha256(self.audio_sha256, name="audio_sha256")
        if (
            type(self.current_disagreement) is not bool
            or type(self.calibrated_disagreement) is not bool
        ):
            raise TypeError("disagreement flags must be exact booleans")

    @property
    def delta(self) -> int:
        return int(self.calibrated_disagreement) - int(self.current_disagreement)

    def to_report(self) -> dict[str, object]:
        return {
            "audioSha256": self.audio_sha256,
            "currentDisagreement": self.current_disagreement,
            "calibratedDisagreement": self.calibrated_disagreement,
            "delta": self.delta,
        }


def one_sided_zero_event_upper_bound(sample_count: int) -> float:
    """Exact binomial upper bound when zero severe events were observed."""
    count = _non_negative_int(sample_count, name="sample_count")
    if count == 0:
        return 1.0
    return 1.0 - _ONE_SIDED_ALPHA ** (1.0 / count)


def _paired_bootstrap_upper(
    deltas: tuple[int, ...],
    *,
    seed_material: str,
) -> tuple[float, float]:
    if not deltas:
        return 0.0, 1.0
    values = np.asarray(deltas, dtype=np.float64)
    point = float(values.mean())
    seed = int.from_bytes(hashlib.sha256(seed_material.encode()).digest()[:8], "big")
    rng = np.random.default_rng(seed)
    indices = rng.integers(
        0,
        len(values),
        size=(_BOOTSTRAP_RESAMPLES, len(values)),
    )
    means = values[indices].mean(axis=1)
    upper = float(np.quantile(means, 0.95, method="higher"))
    return point, upper


@dataclass(frozen=True, slots=True)
class CalibrationActivationEvidenceV3:
    model_sha256: str
    label_sha256: str
    feature_schema_sha256: str
    corpus_sha256: str
    fold_assignment_sha256: str
    held_out: tuple[HeldOutSongComparisonV3, ...]
    held_out_song_count: int
    difficulty_delta_point_estimate: float
    difficulty_delta_ci_upper: float
    severe_regression_rate_upper: float
    structure_or_hold_regressions: int
    completeness_regressions: int
    severe_gap_or_intro_regressions: int
    unknown_pair_count: int
    blockers: tuple[str, ...]
    activation_eligible: bool
    activation_state: Literal["REPORT_ONLY"] = "REPORT_ONLY"

    def to_report(self) -> dict[str, object]:
        return {
            "version": "calibration-activation-evidence-v3",
            "modelSha256": self.model_sha256,
            "labelSha256": self.label_sha256,
            "featureSchemaSha256": self.feature_schema_sha256,
            "corpusSha256": self.corpus_sha256,
            "foldAssignmentSha256": self.fold_assignment_sha256,
            "heldOut": [item.to_report() for item in self.held_out],
            "heldOutSongCount": self.held_out_song_count,
            "difficultyDeltaPointEstimate": self.difficulty_delta_point_estimate,
            "difficultyDeltaCiUpper": self.difficulty_delta_ci_upper,
            "severeRegressionRateUpper": self.severe_regression_rate_upper,
            "structureOrHoldRegressions": self.structure_or_hold_regressions,
            "completenessRegressions": self.completeness_regressions,
            "severeGapOrIntroRegressions": self.severe_gap_or_intro_regressions,
            "unknownPairCount": self.unknown_pair_count,
            "blockers": list(self.blockers),
            "activationEligible": self.activation_eligible,
            "activationState": self.activation_state,
            "automaticallyEnforced": False,
        }

    def stable_sha256(self) -> str:
        return _canonical_sha256(self.to_report())


def evaluate_calibration_activation_v3(
    *,
    model_sha256: str,
    label_sha256: str,
    feature_schema_sha256: str,
    corpus_sha256: str,
    fold_assignment_sha256: str,
    held_out: tuple[HeldOutSongComparisonV3, ...],
    structure_or_hold_regressions: int,
    completeness_regressions: int,
    severe_gap_or_intro_regressions: int,
    unknown_pair_count: int,
) -> CalibrationActivationEvidenceV3:
    hashes = tuple(
        _sha256(value, name=name)
        for name, value in (
            ("model_sha256", model_sha256),
            ("label_sha256", label_sha256),
            ("feature_schema_sha256", feature_schema_sha256),
            ("corpus_sha256", corpus_sha256),
            ("fold_assignment_sha256", fold_assignment_sha256),
        )
    )
    if type(held_out) is not tuple or any(
        not isinstance(item, HeldOutSongComparisonV3) for item in held_out
    ):
        raise TypeError("held_out must be a tuple of song comparisons")
    if tuple(sorted(held_out, key=lambda item: item.audio_sha256)) != held_out or len(
        {item.audio_sha256 for item in held_out}
    ) != len(held_out):
        raise ValueError("held_out songs must be sorted and unique")
    structure_count = _non_negative_int(
        structure_or_hold_regressions,
        name="structure_or_hold_regressions",
    )
    completeness_count = _non_negative_int(
        completeness_regressions,
        name="completeness_regressions",
    )
    severe_count = _non_negative_int(
        severe_gap_or_intro_regressions,
        name="severe_gap_or_intro_regressions",
    )
    unknown_count = _non_negative_int(unknown_pair_count, name="unknown_pair_count")
    point, upper = _paired_bootstrap_upper(
        tuple(item.delta for item in held_out),
        seed_material=":".join(hashes),
    )
    severe_upper = one_sided_zero_event_upper_bound(len(held_out)) if severe_count == 0 else 1.0
    blockers = []
    if not held_out:
        blockers.append("NO_HELD_OUT_SONGS")
    if structure_count:
        blockers.append("STRUCTURE_OR_HOLD_REGRESSION")
    if completeness_count:
        blockers.append("TWELVE_SLOT_COMPLETENESS_REGRESSION")
    if severe_count:
        blockers.append("SEVERE_GAP_OR_INTRO_REGRESSION")
    if severe_upper >= _MAX_SEVERE_REGRESSION_RATE_UPPER:
        blockers.append("SEVERE_REGRESSION_RATE_UNCERTAIN")
    if unknown_count:
        blockers.append("CALIBRATION_COVERAGE_UNKNOWN")
    if point >= 0.0 or upper > 0.0:
        blockers.append("NO_HELD_OUT_DIFFICULTY_IMPROVEMENT")
    unique_blockers = tuple(dict.fromkeys(blockers))
    return CalibrationActivationEvidenceV3(
        model_sha256=hashes[0],
        label_sha256=hashes[1],
        feature_schema_sha256=hashes[2],
        corpus_sha256=hashes[3],
        fold_assignment_sha256=hashes[4],
        held_out=held_out,
        held_out_song_count=len(held_out),
        difficulty_delta_point_estimate=point,
        difficulty_delta_ci_upper=upper,
        severe_regression_rate_upper=severe_upper,
        structure_or_hold_regressions=structure_count,
        completeness_regressions=completeness_count,
        severe_gap_or_intro_regressions=severe_count,
        unknown_pair_count=unknown_count,
        blockers=unique_blockers,
        activation_eligible=not unique_blockers,
    )


def _exact_mapping(
    value: object,
    *,
    name: str,
    keys: frozenset[str],
) -> dict[str, object]:
    if type(value) is not dict or any(type(key) is not str for key in value):
        raise TypeError(f"{name} must be an exact string-keyed object")
    actual = frozenset(value)
    if actual != keys:
        raise ValueError(
            f"{name} keys differ: missing={sorted(keys - actual)}, "
            f"extra={sorted(actual - keys)}"
        )
    return value


def _parse_held_out(value: object) -> HeldOutSongComparisonV3:
    report = _exact_mapping(
        value,
        name="held-out song comparison",
        keys=frozenset(
            {
                "audioSha256",
                "currentDisagreement",
                "calibratedDisagreement",
                "delta",
            }
        ),
    )
    parsed = HeldOutSongComparisonV3(
        audio_sha256=report["audioSha256"],
        current_disagreement=report["currentDisagreement"],
        calibrated_disagreement=report["calibratedDisagreement"],
    )
    if parsed.to_report() != report:
        raise ValueError("held-out song comparison contains an inconsistent delta")
    return parsed


def parse_calibration_activation_evidence_v3(
    value: object,
) -> CalibrationActivationEvidenceV3:
    """Strictly recompute activation aggregates from held-out song records."""
    report = _exact_mapping(
        value,
        name="calibration activation evidence",
        keys=frozenset(
            {
                "version",
                "modelSha256",
                "labelSha256",
                "featureSchemaSha256",
                "corpusSha256",
                "foldAssignmentSha256",
                "heldOut",
                "heldOutSongCount",
                "difficultyDeltaPointEstimate",
                "difficultyDeltaCiUpper",
                "severeRegressionRateUpper",
                "structureOrHoldRegressions",
                "completenessRegressions",
                "severeGapOrIntroRegressions",
                "unknownPairCount",
                "blockers",
                "activationEligible",
                "activationState",
                "automaticallyEnforced",
            }
        ),
    )
    if report["version"] != "calibration-activation-evidence-v3":
        raise ValueError("unsupported calibration activation evidence version")
    held_out_value = report["heldOut"]
    if type(held_out_value) is not list:
        raise TypeError("heldOut must be an exact list")
    recomputed = evaluate_calibration_activation_v3(
        model_sha256=report["modelSha256"],
        label_sha256=report["labelSha256"],
        feature_schema_sha256=report["featureSchemaSha256"],
        corpus_sha256=report["corpusSha256"],
        fold_assignment_sha256=report["foldAssignmentSha256"],
        held_out=tuple(_parse_held_out(item) for item in held_out_value),
        structure_or_hold_regressions=report["structureOrHoldRegressions"],
        completeness_regressions=report["completenessRegressions"],
        severe_gap_or_intro_regressions=report["severeGapOrIntroRegressions"],
        unknown_pair_count=report["unknownPairCount"],
    )
    canonical_report = json.dumps(
        report,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    recomputed_report = json.dumps(
        recomputed.to_report(),
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    if canonical_report != recomputed_report:
        raise ValueError("activation evidence differs from strict recalculation")
    for name in (
        "difficultyDeltaPointEstimate",
        "difficultyDeltaCiUpper",
        "severeRegressionRateUpper",
    ):
        if type(report[name]) is not float or not math.isfinite(report[name]):
            raise TypeError(f"{name} must be a finite exact float")
    return recomputed
