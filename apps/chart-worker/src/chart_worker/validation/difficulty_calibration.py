"""Small report-only pairwise calibration with song-group isolation."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Literal

import numpy as np

from chart_worker.analysis.chart_profile import ChartQualityProfile
from chart_worker.schema.types import KEY_MODES
from chart_worker.validation.family_evidence_v3 import CandidateFamilyEvidenceV3
from chart_worker.validation.mania_star_evidence import ManiaStarEvidenceV1
from chart_worker.validation.pairwise_labels import (
    PairwiseLabelV1,
    PairwiseTaskV1,
    canonical_answer,
)
from chart_worker.validation.song_family_selector_v3 import CalibrationPredictionV3

CALIBRATION_MODEL_VERSION = "difficulty-pairwise-calibration-v3"
CALIBRATION_FEATURE_VERSION = "difficulty-calibration-feature-v3"
CALIBRATION_FEATURE_NAMES_V3 = (
    "officialStarRating",
    "projectRating",
    "avgNps",
    "p95Nps",
    "peakNps",
    "chordRatio",
    "maxJack",
    "holdNoteRatio",
    "holdTimeOccupancyRatio",
    "holdMeanDurationMs",
    "holdP95DurationMs",
    "holdMaxDurationMs",
    "holdMaxConcurrent",
    "holdMaxHeldLaneRatio",
    "holdMaxReleaseCount250Ms",
    "densityStrain",
    "jackStrain",
    "chordLoad",
    "lnStrain",
    "coordination",
    "peakSkill",
    "boundedStamina",
    "orderingScore",
    "maxSectionPeak",
    "meanHoldBeats",
    "p95HoldBeats",
    "holdOccupancyRatio",
    "overlapInputLoad",
    "releaseLoad",
)


def _exact_string(value: object, *, name: str) -> str:
    if type(value) is not str or not value:
        raise TypeError(f"{name} must be a non-empty exact string")
    return value


def _sha256(value: object, *, name: str) -> str:
    digest = _exact_string(value, name=name)
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return digest


def _finite_float(value: object, *, name: str) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise TypeError(f"{name} must be a finite exact float")
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


def _exact_list(value: object, *, name: str) -> list[object]:
    if type(value) is not list:
        raise TypeError(f"{name} must be an exact list")
    return value


@dataclass(frozen=True, slots=True)
class FrozenCandidateFeatureSourceV3:
    """Minimal immutable identity for a pre-V3 preserved candidate payload."""

    candidate_id: str
    key_mode: int
    provenance: str
    candidate_payload_sha256: str
    generation_report_sha256: str

    def __post_init__(self) -> None:
        _exact_string(self.candidate_id, name="candidate_id")
        if type(self.key_mode) is not int or self.key_mode not in KEY_MODES:
            raise ValueError("key_mode is unsupported")
        _exact_string(self.provenance, name="provenance")
        _sha256(self.candidate_payload_sha256, name="candidate_payload_sha256")
        _sha256(self.generation_report_sha256, name="generation_report_sha256")

    def to_report(self) -> dict[str, object]:
        return {
            "version": "frozen-candidate-feature-source-v3",
            "candidateId": self.candidate_id,
            "keyMode": self.key_mode,
            "provenance": self.provenance,
            "candidatePayloadSha256": self.candidate_payload_sha256,
            "generationReportSha256": self.generation_report_sha256,
        }


@dataclass(frozen=True, slots=True)
class CalibrationFeatureV3:
    candidate_id: str
    audio_sha256: str
    key_mode: int
    provenance: str
    feature_schema_sha256: str
    source_evidence_sha256: str
    feature_names: tuple[str, ...]
    values: tuple[float, ...]

    def __post_init__(self) -> None:
        _exact_string(self.candidate_id, name="candidate_id")
        _sha256(self.audio_sha256, name="audio_sha256")
        if type(self.key_mode) is not int or self.key_mode not in KEY_MODES:
            raise ValueError("key_mode is unsupported")
        _exact_string(self.provenance, name="provenance")
        _sha256(self.feature_schema_sha256, name="feature_schema_sha256")
        _sha256(self.source_evidence_sha256, name="source_evidence_sha256")
        if type(self.feature_names) is not tuple or any(
            type(name) is not str or not name for name in self.feature_names
        ):
            raise TypeError("feature_names must be a tuple of exact strings")
        if len(set(self.feature_names)) != len(self.feature_names) or not self.feature_names:
            raise ValueError("feature_names must be non-empty and unique")
        if type(self.values) is not tuple or len(self.values) != len(self.feature_names):
            raise ValueError("values must align with feature_names")
        for index, value in enumerate(self.values):
            _finite_float(value, name=f"values[{index}]")

    def to_report(self) -> dict[str, object]:
        return {
            "candidateId": self.candidate_id,
            "audioSha256": self.audio_sha256,
            "keyMode": self.key_mode,
            "provenance": self.provenance,
            "featureSchemaSha256": self.feature_schema_sha256,
            "sourceEvidenceSha256": self.source_evidence_sha256,
            "featureNames": list(self.feature_names),
            "values": list(self.values),
        }

    def stable_sha256(self) -> str:
        return _canonical_sha256(self.to_report())


def _calibration_feature_schema_sha256() -> str:
    return _canonical_sha256(
        {
            "version": CALIBRATION_FEATURE_VERSION,
            "calculatorFamily": "OSU_TOOLS_MANIA",
            "calculatorVersion": 20241007,
            "featureNames": list(CALIBRATION_FEATURE_NAMES_V3),
        }
    )


def _build_calibration_feature_v3(
    *,
    candidate_id: str,
    key_mode: int,
    provenance: str,
    candidate_payload_sha256: str,
    source_name: str,
    source_report: dict[str, object],
    profile: ChartQualityProfile,
    official_star: ManiaStarEvidenceV1,
    audio_sha256: str,
) -> CalibrationFeatureV3:
    if not isinstance(profile, ChartQualityProfile):
        raise TypeError("profile must be ChartQualityProfile")
    if not isinstance(official_star, ManiaStarEvidenceV1):
        raise TypeError("official_star must be ManiaStarEvidenceV1")
    audio_digest = _sha256(audio_sha256, name="audio_sha256")
    payload_digest = _sha256(
        candidate_payload_sha256,
        name="candidate_payload_sha256",
    )
    if not official_star.authorizes_calibration_feature:
        raise ValueError("official star must come from verified pinned tool execution")
    if official_star.input_osu_sha256 != payload_digest:
        raise ValueError("official star input differs from candidate payload digest")

    difficulty = profile.difficulty
    hold = profile.hold
    vector = profile.difficulty_vector_v2
    values = tuple(
        float(value)
        for value in (
            official_star.star_rating,
            difficulty.project_rating,
            difficulty.avg_nps,
            difficulty.p95_nps,
            difficulty.peak_nps,
            difficulty.chord_ratio,
            difficulty.max_jack,
            hold.note_ratio,
            hold.time_occupancy_ratio,
            hold.mean_duration_ms,
            hold.p95_duration_ms,
            hold.max_duration_ms,
            hold.max_concurrent,
            hold.max_held_lane_ratio,
            hold.max_release_count_250ms,
            vector.density_strain,
            vector.jack_strain,
            vector.chord_load,
            vector.ln_strain,
            vector.coordination,
            vector.peak_skill,
            vector.bounded_stamina,
            vector.ordering_score,
            vector.max_section_peak,
            vector.mean_hold_beats,
            vector.p95_hold_beats,
            vector.hold_occupancy_ratio,
            vector.overlap_input_load,
            vector.release_load,
        )
    )
    source_evidence_sha256 = _canonical_sha256(
        {
            source_name: source_report,
            "profile": profile.to_report(),
            "officialStar": official_star.to_report(),
            "audioSha256": audio_digest,
        }
    )
    return CalibrationFeatureV3(
        candidate_id=candidate_id,
        audio_sha256=audio_digest,
        key_mode=key_mode,
        provenance=provenance,
        feature_schema_sha256=_calibration_feature_schema_sha256(),
        source_evidence_sha256=source_evidence_sha256,
        feature_names=CALIBRATION_FEATURE_NAMES_V3,
        values=values,
    )


def build_calibration_feature_v3(
    *,
    candidate: CandidateFamilyEvidenceV3,
    profile: ChartQualityProfile,
    official_star: ManiaStarEvidenceV1,
    audio_sha256: str,
) -> CalibrationFeatureV3:
    """Bind one fixed numeric feature vector to its immutable source evidence."""
    if not isinstance(candidate, CandidateFamilyEvidenceV3):
        raise TypeError("candidate must be CandidateFamilyEvidenceV3")
    return _build_calibration_feature_v3(
        candidate_id=candidate.candidate_id,
        key_mode=candidate.key_mode,
        provenance=candidate.provenance,
        candidate_payload_sha256=candidate.candidate_payload_sha256,
        source_name="candidate",
        source_report=candidate.to_report(),
        profile=profile,
        official_star=official_star,
        audio_sha256=audio_sha256,
    )


def build_calibration_feature_from_frozen_source_v3(
    *,
    source: FrozenCandidateFeatureSourceV3,
    profile: ChartQualityProfile,
    official_star: ManiaStarEvidenceV1,
    audio_sha256: str,
) -> CalibrationFeatureV3:
    """Build a report-only feature without pretending old evidence is live V3 evidence."""
    if not isinstance(source, FrozenCandidateFeatureSourceV3):
        raise TypeError("source must be FrozenCandidateFeatureSourceV3")
    return _build_calibration_feature_v3(
        candidate_id=source.candidate_id,
        key_mode=source.key_mode,
        provenance=source.provenance,
        candidate_payload_sha256=source.candidate_payload_sha256,
        source_name="frozenCandidateSource",
        source_report=source.to_report(),
        profile=profile,
        official_star=official_star,
        audio_sha256=audio_sha256,
    )


@dataclass(frozen=True, slots=True)
class PairwisePreferenceV3:
    audio_sha256: str
    harder_candidate_id: str
    easier_candidate_id: str
    confidence: int

    def __post_init__(self) -> None:
        _sha256(self.audio_sha256, name="audio_sha256")
        _exact_string(self.harder_candidate_id, name="harder_candidate_id")
        _exact_string(self.easier_candidate_id, name="easier_candidate_id")
        if self.harder_candidate_id == self.easier_candidate_id:
            raise ValueError("preference candidates must differ")
        if type(self.confidence) is not int:
            raise TypeError("confidence must be an exact integer")
        if not 1 <= self.confidence <= 5:
            raise ValueError("confidence must be between 1 and 5")


@dataclass(frozen=True, slots=True)
class PairwiseQualityPreferenceV3:
    audio_sha256: str
    preferred_candidate_id: str
    other_candidate_id: str
    confidence: int

    def __post_init__(self) -> None:
        _sha256(self.audio_sha256, name="audio_sha256")
        _exact_string(self.preferred_candidate_id, name="preferred_candidate_id")
        _exact_string(self.other_candidate_id, name="other_candidate_id")
        if self.preferred_candidate_id == self.other_candidate_id:
            raise ValueError("quality preference candidates must differ")
        if type(self.confidence) is not int:
            raise TypeError("confidence must be an exact integer")
        if not 1 <= self.confidence <= 5:
            raise ValueError("confidence must be between 1 and 5")


@dataclass(frozen=True, slots=True)
class PairwisePreferenceProjectionV3:
    preferences: tuple[PairwisePreferenceV3, ...]
    quality_preferences: tuple[PairwiseQualityPreferenceV3, ...]
    label_sha256: str
    quality_conflict_count: int
    difficulty_uncertain_or_tie_count: int
    quality_uncertain_or_tie_count: int
    contradiction_count: int
    quality_contradiction_count: int


def project_pairwise_preferences_v3(
    entries: tuple[tuple[PairwiseTaskV1, PairwiseLabelV1], ...],
    *,
    features: tuple[CalibrationFeatureV3, ...],
) -> PairwisePreferenceProjectionV3:
    """Project difficulty and musical-quality labels as independent preferences."""
    if type(entries) is not tuple or type(features) is not tuple:
        raise TypeError("entries and features must be tuples")
    by_id = {feature.candidate_id: feature for feature in features}
    if len(by_id) != len(features):
        raise ValueError("candidate features must have unique identities")
    difficulty_grouped: dict[
        tuple[str, int, str, str],
        list[tuple[str, int]],
    ] = {}
    quality_grouped: dict[
        tuple[str, int, str, str],
        list[tuple[str, int]],
    ] = {}
    quality_conflicts = 0
    difficulty_uncertain_or_tie = 0
    quality_uncertain_or_tie = 0
    canonical_entries: list[dict[str, object]] = []
    for task, label in entries:
        if not isinstance(task, PairwiseTaskV1) or not isinstance(label, PairwiseLabelV1):
            raise TypeError("entry must contain a pairwise task and label")
        canonical_entries.append({"task": task.to_private_report(), "label": label.to_report()})
        for binding in (task.left, task.right):
            feature = by_id.get(binding.candidate_id)
            if feature is None:
                raise ValueError("pairwise task references a missing feature")
            if feature.stable_sha256() != binding.feature_sha256:
                raise ValueError("pairwise feature digest differs from its binding")
            if feature.audio_sha256 != binding.audio_sha256 or feature.key_mode != binding.key_mode:
                raise ValueError("pairwise feature identity differs from its binding")
        harder = canonical_answer(task, label, dimension="harder")
        quality = canonical_answer(task, label, dimension="musical_quality")
        pair = tuple(sorted((task.left.candidate_id, task.right.candidate_id)))
        group_key = (task.left.audio_sha256, task.left.key_mode, pair[0], pair[1])
        if harder in {"TIE", "UNCERTAIN"}:
            difficulty_uncertain_or_tie += 1
        else:
            difficulty_grouped.setdefault(group_key, []).append(
                (harder, label.confidence)
            )
        if quality in {"TIE", "UNCERTAIN"}:
            quality_uncertain_or_tie += 1
        else:
            quality_grouped.setdefault(group_key, []).append(
                (quality, label.confidence)
            )
        if (
            harder not in {"TIE", "UNCERTAIN"}
            and quality not in {"TIE", "UNCERTAIN"}
            and quality != harder
        ):
            quality_conflicts += 1

    preferences = []
    contradictions = 0
    for (audio_sha, _key_mode, left_id, right_id), answers in sorted(
        difficulty_grouped.items()
    ):
        winners = {winner for winner, _confidence in answers}
        if len(winners) != 1:
            contradictions += 1
            continue
        winner = next(iter(winners))
        loser = right_id if winner == left_id else left_id
        preferences.append(
            PairwisePreferenceV3(
                audio_sha256=audio_sha,
                harder_candidate_id=winner,
                easier_candidate_id=loser,
                confidence=max(confidence for _winner, confidence in answers),
            )
        )
    quality_preferences = []
    quality_contradictions = 0
    for (audio_sha, _key_mode, left_id, right_id), answers in sorted(
        quality_grouped.items()
    ):
        winners = {winner for winner, _confidence in answers}
        if len(winners) != 1:
            quality_contradictions += 1
            continue
        winner = next(iter(winners))
        loser = right_id if winner == left_id else left_id
        quality_preferences.append(
            PairwiseQualityPreferenceV3(
                audio_sha256=audio_sha,
                preferred_candidate_id=winner,
                other_candidate_id=loser,
                confidence=max(confidence for _winner, confidence in answers),
            )
        )
    label_sha256 = _canonical_sha256(sorted(canonical_entries, key=_canonical_sha256))
    return PairwisePreferenceProjectionV3(
        preferences=tuple(preferences),
        quality_preferences=tuple(quality_preferences),
        label_sha256=label_sha256,
        quality_conflict_count=quality_conflicts,
        difficulty_uncertain_or_tie_count=difficulty_uncertain_or_tie,
        quality_uncertain_or_tie_count=quality_uncertain_or_tie,
        contradiction_count=contradictions,
        quality_contradiction_count=quality_contradictions,
    )


def group_disjoint_folds(
    audio_sha256: tuple[str, ...],
    *,
    n_splits: int,
    seed: str,
) -> dict[str, int]:
    if type(audio_sha256) is not tuple:
        raise TypeError("audio_sha256 must be a tuple")
    groups = tuple(sorted(set(audio_sha256)))
    if len(groups) != len(audio_sha256):
        raise ValueError("audio groups must be unique")
    for group in groups:
        _sha256(group, name="audio group")
    if type(n_splits) is not int or n_splits < 2 or n_splits > len(groups):
        raise ValueError("n_splits must be between two and the group count")
    seed_value = _exact_string(seed, name="seed")
    ordered = sorted(
        groups,
        key=lambda group: hashlib.sha256(f"{seed_value}:{group}".encode()).digest(),
    )
    assigned = {group: index % n_splits for index, group in enumerate(ordered)}
    return {group: assigned[group] for group in sorted(assigned)}


@dataclass(frozen=True, slots=True)
class DifficultyCalibrationModelV3:
    key_mode: int
    feature_schema_sha256: str
    label_sha256: str
    feature_names: tuple[str, ...]
    provenance_names: tuple[str, ...]
    means: tuple[float, ...]
    scales: tuple[float, ...]
    domain_mins: tuple[float, ...]
    domain_maxs: tuple[float, ...]
    weights: tuple[float, ...]
    training_audio_sha256: tuple[str, ...]
    validation_audio_sha256: tuple[str, ...]
    version: Literal["difficulty-pairwise-calibration-v3"] = CALIBRATION_MODEL_VERSION

    def __post_init__(self) -> None:
        if self.version != CALIBRATION_MODEL_VERSION:
            raise ValueError("unsupported calibration model version")
        if type(self.key_mode) is not int or self.key_mode not in KEY_MODES:
            raise ValueError("key_mode is unsupported")
        _sha256(self.feature_schema_sha256, name="feature_schema_sha256")
        _sha256(self.label_sha256, name="label_sha256")
        if type(self.feature_names) is not tuple or any(
            type(name) is not str or not name for name in self.feature_names
        ):
            raise TypeError("feature_names must be a tuple of exact strings")
        width = len(self.feature_names)
        if width == 0 or len(set(self.feature_names)) != width:
            raise ValueError("feature_names must be non-empty and unique")
        for values, name in (
            (self.means, "means"),
            (self.scales, "scales"),
            (self.domain_mins, "domain_mins"),
            (self.domain_maxs, "domain_maxs"),
        ):
            if type(values) is not tuple or len(values) != width:
                raise ValueError(f"{name} must align with feature_names")
            for value in values:
                _finite_float(value, name=name)
        if any(scale <= 0 for scale in self.scales) or any(
            low > high for low, high in zip(self.domain_mins, self.domain_maxs)
        ):
            raise ValueError("calibration scale or domain is invalid")
        if type(self.provenance_names) is not tuple or any(
            type(name) is not str or not name for name in self.provenance_names
        ):
            raise TypeError("provenance_names must be a tuple of exact strings")
        if self.provenance_names != tuple(sorted(set(self.provenance_names))):
            raise ValueError("provenance_names must be sorted and unique")
        if type(self.weights) is not tuple or len(self.weights) != width + len(
            self.provenance_names
        ):
            raise ValueError("weights do not match numeric and provenance features")
        for weight in self.weights:
            _finite_float(weight, name="weight")
        for values, name in (
            (self.training_audio_sha256, "training_audio_sha256"),
            (self.validation_audio_sha256, "validation_audio_sha256"),
        ):
            if type(values) is not tuple or values != tuple(sorted(set(values))):
                raise ValueError(f"{name} must be sorted and unique")
            for value in values:
                _sha256(value, name=name)
        if set(self.training_audio_sha256) & set(self.validation_audio_sha256):
            raise ValueError("training and validation audio groups overlap")

    def to_report(self) -> dict[str, object]:
        return {
            "version": self.version,
            "keyMode": self.key_mode,
            "featureSchemaSha256": self.feature_schema_sha256,
            "labelSha256": self.label_sha256,
            "featureNames": list(self.feature_names),
            "provenanceNames": list(self.provenance_names),
            "means": list(self.means),
            "scales": list(self.scales),
            "domainMins": list(self.domain_mins),
            "domainMaxs": list(self.domain_maxs),
            "weights": list(self.weights),
            "trainingAudioSha256": list(self.training_audio_sha256),
            "validationAudioSha256": list(self.validation_audio_sha256),
            "activationState": "REPORT_ONLY",
        }

    def stable_sha256(self) -> str:
        return _canonical_sha256(self.to_report())

    def predict(self, feature: CalibrationFeatureV3) -> CalibrationPredictionV3:
        if not isinstance(feature, CalibrationFeatureV3):
            raise TypeError("feature must be CalibrationFeatureV3")
        if feature.key_mode != self.key_mode:
            raise ValueError("feature key mode differs from calibration")
        if (
            feature.feature_schema_sha256 != self.feature_schema_sha256
            or feature.feature_names != self.feature_names
        ):
            raise ValueError("feature schema differs from calibration")
        if feature.provenance not in self.provenance_names or any(
            value < low or value > high
            for value, low, high in zip(feature.values, self.domain_mins, self.domain_maxs)
        ):
            return CalibrationPredictionV3(
                candidate_id=feature.candidate_id,
                state="UNKNOWN",
                score=None,
                calibration_sha256=self.stable_sha256(),
            )
        numeric = [
            (value - mean) / scale
            for value, mean, scale in zip(feature.values, self.means, self.scales)
        ]
        provenance = [float(name == feature.provenance) for name in self.provenance_names]
        score = float(np.dot(np.asarray((*numeric, *provenance)), np.asarray(self.weights)))
        return CalibrationPredictionV3(
            candidate_id=feature.candidate_id,
            state="IN_DOMAIN",
            score=score,
            calibration_sha256=self.stable_sha256(),
        )


def parse_difficulty_calibration_model_v3(value: object) -> DifficultyCalibrationModelV3:
    report = _exact_mapping(
        value,
        name="difficulty calibration model",
        keys=frozenset(
            {
                "version",
                "keyMode",
                "featureSchemaSha256",
                "labelSha256",
                "featureNames",
                "provenanceNames",
                "means",
                "scales",
                "domainMins",
                "domainMaxs",
                "weights",
                "trainingAudioSha256",
                "validationAudioSha256",
                "activationState",
            }
        ),
    )
    if report["activationState"] != "REPORT_ONLY":
        raise ValueError("calibration model must remain REPORT_ONLY")
    parsed = DifficultyCalibrationModelV3(
        version=report["version"],
        key_mode=report["keyMode"],
        feature_schema_sha256=report["featureSchemaSha256"],
        label_sha256=report["labelSha256"],
        feature_names=tuple(_exact_list(report["featureNames"], name="featureNames")),
        provenance_names=tuple(
            _exact_list(report["provenanceNames"], name="provenanceNames")
        ),
        means=tuple(_exact_list(report["means"], name="means")),
        scales=tuple(_exact_list(report["scales"], name="scales")),
        domain_mins=tuple(_exact_list(report["domainMins"], name="domainMins")),
        domain_maxs=tuple(_exact_list(report["domainMaxs"], name="domainMaxs")),
        weights=tuple(_exact_list(report["weights"], name="weights")),
        training_audio_sha256=tuple(
            _exact_list(report["trainingAudioSha256"], name="trainingAudioSha256")
        ),
        validation_audio_sha256=tuple(
            _exact_list(report["validationAudioSha256"], name="validationAudioSha256")
        ),
    )
    if parsed.to_report() != report:
        raise ValueError("difficulty calibration model projection differs")
    return parsed


@dataclass(frozen=True, slots=True)
class CalibrationFitResultV3:
    model: DifficultyCalibrationModelV3
    training_pair_count: int
    training_audio_group_count: int
    validation_audio_group_count: int
    effective_parameter_count: int
    validation_total_pair_count: int
    validation_pair_count: int
    validation_unknown_pair_count: int
    validation_disagreement_rate: float | None
    activation_state: Literal["REPORT_ONLY"] = "REPORT_ONLY"


def _design_row(
    feature: CalibrationFeatureV3,
    *,
    means: np.ndarray,
    scales: np.ndarray,
    provenance_names: tuple[str, ...],
) -> np.ndarray:
    numeric = (np.asarray(feature.values, dtype=np.float64) - means) / scales
    provenance = np.asarray(
        [float(name == feature.provenance) for name in provenance_names],
        dtype=np.float64,
    )
    return np.concatenate((numeric, provenance))


def fit_pairwise_calibration_v3(
    *,
    key_mode: int,
    features: tuple[CalibrationFeatureV3, ...],
    preferences: tuple[PairwisePreferenceV3, ...],
    validation_audio_sha256: tuple[str, ...],
    label_sha256: str,
) -> CalibrationFitResultV3:
    if type(key_mode) is not int or key_mode not in KEY_MODES:
        raise ValueError("key_mode is unsupported")
    if type(features) is not tuple or not features:
        raise ValueError("features must be a non-empty tuple")
    if type(preferences) is not tuple or not preferences:
        raise ValueError("preferences must be a non-empty tuple")
    label_digest = _sha256(label_sha256, name="label_sha256")
    validation = tuple(sorted(set(validation_audio_sha256)))
    if validation != validation_audio_sha256:
        raise ValueError("validation audio groups must be sorted and unique")
    for audio in validation:
        _sha256(audio, name="validation audio")

    by_id = {feature.candidate_id: feature for feature in features}
    if len(by_id) != len(features):
        raise ValueError("candidate features must have unique identities")
    first = features[0]
    for feature in features:
        if feature.key_mode != key_mode:
            raise ValueError("feature key mode differs from calibration")
        if (
            feature.feature_schema_sha256 != first.feature_schema_sha256
            or feature.feature_names != first.feature_names
        ):
            raise ValueError("feature schema differs within calibration input")
    for preference in preferences:
        harder = by_id.get(preference.harder_candidate_id)
        easier = by_id.get(preference.easier_candidate_id)
        if harder is None or easier is None:
            raise ValueError("preference references a missing candidate feature")
        if (
            harder.audio_sha256 != preference.audio_sha256
            or easier.audio_sha256 != preference.audio_sha256
        ):
            raise ValueError("preference audio differs from candidate features")

    train_preferences = tuple(
        preference for preference in preferences if preference.audio_sha256 not in set(validation)
    )
    validation_preferences = tuple(
        preference for preference in preferences if preference.audio_sha256 in set(validation)
    )
    if not train_preferences:
        raise ValueError("validation audio cannot train calibration; no training pairs remain")
    train_ids = {
        candidate_id
        for preference in train_preferences
        for candidate_id in (preference.harder_candidate_id, preference.easier_candidate_id)
    }
    train_features = tuple(by_id[candidate_id] for candidate_id in sorted(train_ids))
    train_values = np.asarray([feature.values for feature in train_features], dtype=np.float64)
    means = train_values.mean(axis=0)
    scales = train_values.std(axis=0)
    scales = np.where(scales == 0.0, 1.0, scales)
    domain_mins = train_values.min(axis=0)
    domain_maxs = train_values.max(axis=0)
    provenance_names = tuple(sorted({feature.provenance for feature in train_features}))
    effective_parameter_count = len(first.feature_names) + max(
        len(provenance_names) - 1,
        0,
    )
    if len(train_preferences) <= effective_parameter_count:
        raise ValueError(
            "insufficient independent pairwise evidence: "
            f"training_pairs={len(train_preferences)} must exceed "
            f"effective_parameters={effective_parameter_count}"
        )
    width = len(first.feature_names) + len(provenance_names)
    weights = np.zeros(width, dtype=np.float64)
    for _iteration in range(600):
        gradient = np.zeros(width, dtype=np.float64)
        total_weight = 0.0
        for preference in train_preferences:
            harder = _design_row(
                by_id[preference.harder_candidate_id],
                means=means,
                scales=scales,
                provenance_names=provenance_names,
            )
            easier = _design_row(
                by_id[preference.easier_candidate_id],
                means=means,
                scales=scales,
                provenance_names=provenance_names,
            )
            difference = harder - easier
            confidence_weight = preference.confidence / 5.0
            margin = float(np.dot(weights, difference))
            probability_error = 1.0 / (1.0 + math.exp(min(60.0, margin)))
            gradient -= confidence_weight * probability_error * difference
            total_weight += confidence_weight
        gradient = gradient / max(total_weight, 1.0) + 0.001 * weights
        weights -= 0.08 * gradient

    model = DifficultyCalibrationModelV3(
        key_mode=key_mode,
        feature_schema_sha256=first.feature_schema_sha256,
        label_sha256=label_digest,
        feature_names=first.feature_names,
        provenance_names=provenance_names,
        means=tuple(float(value) for value in means),
        scales=tuple(float(value) for value in scales),
        domain_mins=tuple(float(value) for value in domain_mins),
        domain_maxs=tuple(float(value) for value in domain_maxs),
        weights=tuple(float(value) for value in weights),
        training_audio_sha256=tuple(
            sorted({preference.audio_sha256 for preference in train_preferences})
        ),
        validation_audio_sha256=validation,
    )
    disagreements = 0
    comparable = 0
    unknown = 0
    for preference in validation_preferences:
        harder_prediction = model.predict(by_id[preference.harder_candidate_id])
        easier_prediction = model.predict(by_id[preference.easier_candidate_id])
        if harder_prediction.state == "UNKNOWN" or easier_prediction.state == "UNKNOWN":
            unknown += 1
            continue
        comparable += 1
        disagreements += harder_prediction.score <= easier_prediction.score
    disagreement_rate = None if comparable == 0 else float(disagreements / comparable)
    return CalibrationFitResultV3(
        model=model,
        training_pair_count=len(train_preferences),
        training_audio_group_count=len(
            {preference.audio_sha256 for preference in train_preferences}
        ),
        validation_audio_group_count=len(validation),
        effective_parameter_count=effective_parameter_count,
        validation_total_pair_count=len(validation_preferences),
        validation_pair_count=comparable,
        validation_unknown_pair_count=unknown,
        validation_disagreement_rate=disagreement_rate,
    )
