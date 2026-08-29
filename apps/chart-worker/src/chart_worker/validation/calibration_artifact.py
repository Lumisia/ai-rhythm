"""Strict cross-binding for report-only difficulty calibration activation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from chart_worker.validation.calibration_activation import (
    CalibrationActivationEvidenceV3,
    parse_calibration_activation_evidence_v3,
)
from chart_worker.validation.difficulty_calibration import (
    DifficultyCalibrationModelV3,
    parse_difficulty_calibration_model_v3,
)


def _sha256(value: object, *, name: str) -> str:
    if type(value) is not str or len(value) != 64:
        raise TypeError(f"{name} must be a lowercase SHA-256 digest")
    if any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
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
class CalibrationActivationArtifactV3:
    model: DifficultyCalibrationModelV3
    activation: CalibrationActivationEvidenceV3

    @property
    def eligible_for_manual_activation(self) -> bool:
        return True

    @property
    def automatically_enforced(self) -> bool:
        return False

    def to_report(self) -> dict[str, object]:
        return {
            "version": "difficulty-calibration-activation-artifact-v3",
            "model": self.model.to_report(),
            "modelSha256": self.model.stable_sha256(),
            "activationEvidence": self.activation.to_report(),
            "activationEvidenceSha256": self.activation.stable_sha256(),
            "eligibleForManualActivation": True,
            "automaticallyEnforced": False,
        }

    def stable_sha256(self) -> str:
        return _canonical_sha256(self.to_report())


def build_calibration_activation_artifact_v3(
    model: DifficultyCalibrationModelV3,
    activation: CalibrationActivationEvidenceV3,
) -> CalibrationActivationArtifactV3:
    if not isinstance(model, DifficultyCalibrationModelV3):
        raise TypeError("model must be DifficultyCalibrationModelV3")
    if not isinstance(activation, CalibrationActivationEvidenceV3):
        raise TypeError("activation must be CalibrationActivationEvidenceV3")
    model = parse_difficulty_calibration_model_v3(model.to_report())
    activation = parse_calibration_activation_evidence_v3(activation.to_report())
    if activation.model_sha256 != model.stable_sha256():
        raise ValueError("activation model digest differs from model")
    if activation.label_sha256 != model.label_sha256:
        raise ValueError("activation label digest differs from model")
    if activation.feature_schema_sha256 != model.feature_schema_sha256:
        raise ValueError("activation feature schema differs from model")
    if not activation.activation_eligible:
        raise ValueError("activation evidence is not activation eligible")
    return CalibrationActivationArtifactV3(model=model, activation=activation)


def parse_calibration_activation_artifact_v3(
    value: object,
    *,
    expected_label_sha256: str,
    expected_feature_schema_sha256: str,
    expected_corpus_sha256: str,
    expected_fold_assignment_sha256: str,
) -> CalibrationActivationArtifactV3:
    if type(value) is not dict or any(type(key) is not str for key in value):
        raise TypeError("calibration activation artifact must be an exact object")
    required = {
        "version",
        "model",
        "modelSha256",
        "activationEvidence",
        "activationEvidenceSha256",
        "eligibleForManualActivation",
        "automaticallyEnforced",
    }
    if set(value) != required:
        raise ValueError(
            "calibration activation artifact keys differ: "
            f"missing={sorted(required - set(value))}, extra={sorted(set(value) - required)}"
        )
    if value["version"] != "difficulty-calibration-activation-artifact-v3":
        raise ValueError("unsupported calibration activation artifact version")
    model = parse_difficulty_calibration_model_v3(value["model"])
    activation = parse_calibration_activation_evidence_v3(value["activationEvidence"])
    artifact = build_calibration_activation_artifact_v3(model, activation)
    if value != artifact.to_report():
        raise ValueError("calibration activation artifact projection differs")
    if value["modelSha256"] != model.stable_sha256():
        raise ValueError("model digest differs from artifact")
    if value["activationEvidenceSha256"] != activation.stable_sha256():
        raise ValueError("activation evidence digest differs from artifact")
    expected = (
        (_sha256(expected_label_sha256, name="expected_label_sha256"), model.label_sha256, "label"),
        (
            _sha256(expected_feature_schema_sha256, name="expected_feature_schema_sha256"),
            model.feature_schema_sha256,
            "feature schema",
        ),
        (
            _sha256(expected_corpus_sha256, name="expected_corpus_sha256"),
            activation.corpus_sha256,
            "corpus",
        ),
        (
            _sha256(
                expected_fold_assignment_sha256,
                name="expected_fold_assignment_sha256",
            ),
            activation.fold_assignment_sha256,
            "fold assignment",
        ),
    )
    for expected_digest, actual_digest, name in expected:
        if expected_digest != actual_digest:
            raise ValueError(f"expected {name} digest differs from artifact")
    return artifact
