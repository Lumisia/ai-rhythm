import pytest

from chart_worker.validation.calibration_activation import (
    HeldOutSongComparisonV3,
    evaluate_calibration_activation_v3,
)
from chart_worker.validation.calibration_artifact import (
    build_calibration_activation_artifact_v3,
    parse_calibration_activation_artifact_v3,
)
from chart_worker.validation.difficulty_calibration import DifficultyCalibrationModelV3


def _model() -> DifficultyCalibrationModelV3:
    return DifficultyCalibrationModelV3(
        key_mode=4,
        feature_schema_sha256="c" * 64,
        label_sha256="b" * 64,
        feature_names=("officialStarRating",),
        provenance_names=("PRIMARY",),
        means=(2.0,),
        scales=(1.0,),
        domain_mins=(1.0,),
        domain_maxs=(4.0,),
        weights=(1.0, 0.0),
        training_audio_sha256=("1" * 64,),
        validation_audio_sha256=tuple(f"{index:064x}" for index in range(69)),
    )


def _eligible(model: DifficultyCalibrationModelV3):
    held_out = tuple(
        HeldOutSongComparisonV3(
            audio_sha256=f"{index:064x}",
            current_disagreement=index < 20,
            calibrated_disagreement=False,
        )
        for index in range(69)
    )
    return evaluate_calibration_activation_v3(
        model_sha256=model.stable_sha256(),
        label_sha256=model.label_sha256,
        feature_schema_sha256=model.feature_schema_sha256,
        corpus_sha256="d" * 64,
        fold_assignment_sha256="e" * 64,
        held_out=held_out,
        structure_or_hold_regressions=0,
        completeness_regressions=0,
        severe_gap_or_intro_regressions=0,
        unknown_pair_count=0,
    )


def test_activation_artifact_cross_binds_model_and_recomputed_evidence():
    model = _model()
    activation = _eligible(model)
    artifact = build_calibration_activation_artifact_v3(model, activation)

    parsed = parse_calibration_activation_artifact_v3(
        artifact.to_report(),
        expected_label_sha256="b" * 64,
        expected_feature_schema_sha256="c" * 64,
        expected_corpus_sha256="d" * 64,
        expected_fold_assignment_sha256="e" * 64,
    )

    assert parsed == artifact
    assert parsed.eligible_for_manual_activation is True
    assert parsed.automatically_enforced is False


def test_activation_artifact_rejects_hash_mismatch_and_ineligible_evidence():
    model = _model()
    activation = _eligible(model)
    artifact = build_calibration_activation_artifact_v3(model, activation)

    with pytest.raises(ValueError, match="expected corpus"):
        parse_calibration_activation_artifact_v3(
            artifact.to_report(),
            expected_label_sha256="b" * 64,
            expected_feature_schema_sha256="c" * 64,
            expected_corpus_sha256="f" * 64,
            expected_fold_assignment_sha256="e" * 64,
        )
    with pytest.raises(ValueError, match="not activation eligible"):
        ineligible = evaluate_calibration_activation_v3(
            model_sha256=model.stable_sha256(),
            label_sha256=model.label_sha256,
            feature_schema_sha256=model.feature_schema_sha256,
            corpus_sha256="d" * 64,
            fold_assignment_sha256="e" * 64,
            held_out=tuple(
                HeldOutSongComparisonV3(
                    audio_sha256=f"{index:064x}",
                    current_disagreement=False,
                    calibrated_disagreement=False,
                )
                for index in range(69)
            ),
            structure_or_hold_regressions=0,
            completeness_regressions=0,
            severe_gap_or_intro_regressions=0,
            unknown_pair_count=0,
        )
        build_calibration_activation_artifact_v3(
            model,
            ineligible,
        )
