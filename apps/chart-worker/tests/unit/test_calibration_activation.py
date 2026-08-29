import pytest

from chart_worker.validation.calibration_activation import (
    HeldOutSongComparisonV3,
    evaluate_calibration_activation_v3,
    one_sided_zero_event_upper_bound,
    parse_calibration_activation_evidence_v3,
)


def _outcomes(count: int, *, improvements: int):
    return tuple(
        HeldOutSongComparisonV3(
            audio_sha256=f"{index:064x}",
            current_disagreement=(index < improvements),
            calibrated_disagreement=False,
        )
        for index in range(count)
    )


def test_zero_event_upper_bound_does_not_treat_small_sample_as_proof_of_zero_risk():
    assert one_sided_zero_event_upper_bound(12) > 0.20
    assert one_sided_zero_event_upper_bound(69) < 0.05


def test_activation_evidence_can_be_eligible_but_never_self_enforces():
    evidence = evaluate_calibration_activation_v3(
        model_sha256="a" * 64,
        label_sha256="b" * 64,
        feature_schema_sha256="c" * 64,
        corpus_sha256="d" * 64,
        fold_assignment_sha256="e" * 64,
        held_out=_outcomes(69, improvements=20),
        structure_or_hold_regressions=0,
        completeness_regressions=0,
        severe_gap_or_intro_regressions=0,
        unknown_pair_count=0,
    )

    assert evidence.activation_eligible is True
    assert evidence.activation_state == "REPORT_ONLY"
    assert evidence.blockers == ()
    assert evidence.difficulty_delta_point_estimate < 0
    assert evidence.difficulty_delta_ci_upper <= 0
    assert evidence.severe_regression_rate_upper < 0.05
    assert evidence.unknown_pair_count == 0
    assert len(evidence.to_report()["heldOut"]) == 69
    assert parse_calibration_activation_evidence_v3(evidence.to_report()) == evidence


def test_any_unknown_validation_pair_blocks_activation():
    evidence = evaluate_calibration_activation_v3(
        model_sha256="a" * 64,
        label_sha256="b" * 64,
        feature_schema_sha256="c" * 64,
        corpus_sha256="d" * 64,
        fold_assignment_sha256="e" * 64,
        held_out=_outcomes(69, improvements=20),
        structure_or_hold_regressions=0,
        completeness_regressions=0,
        severe_gap_or_intro_regressions=0,
        unknown_pair_count=1,
    )

    assert evidence.activation_eligible is False
    assert "CALIBRATION_COVERAGE_UNKNOWN" in evidence.blockers
    assert evidence.unknown_pair_count == 1


def test_any_observed_safety_or_completeness_regression_blocks_activation():
    evidence = evaluate_calibration_activation_v3(
        model_sha256="a" * 64,
        label_sha256="b" * 64,
        feature_schema_sha256="c" * 64,
        corpus_sha256="d" * 64,
        fold_assignment_sha256="e" * 64,
        held_out=_outcomes(69, improvements=20),
        structure_or_hold_regressions=1,
        completeness_regressions=1,
        severe_gap_or_intro_regressions=1,
        unknown_pair_count=0,
    )

    assert evidence.activation_eligible is False
    assert "STRUCTURE_OR_HOLD_REGRESSION" in evidence.blockers
    assert "TWELVE_SLOT_COMPLETENESS_REGRESSION" in evidence.blockers
    assert "SEVERE_GAP_OR_INTRO_REGRESSION" in evidence.blockers


def test_no_measured_difficulty_improvement_blocks_activation():
    evidence = evaluate_calibration_activation_v3(
        model_sha256="a" * 64,
        label_sha256="b" * 64,
        feature_schema_sha256="c" * 64,
        corpus_sha256="d" * 64,
        fold_assignment_sha256="e" * 64,
        held_out=_outcomes(69, improvements=0),
        structure_or_hold_regressions=0,
        completeness_regressions=0,
        severe_gap_or_intro_regressions=0,
        unknown_pair_count=0,
    )

    assert evidence.activation_eligible is False
    assert "NO_HELD_OUT_DIFFICULTY_IMPROVEMENT" in evidence.blockers


def test_activation_parser_recomputes_aggregates_instead_of_trusting_reported_ci():
    evidence = evaluate_calibration_activation_v3(
        model_sha256="a" * 64,
        label_sha256="b" * 64,
        feature_schema_sha256="c" * 64,
        corpus_sha256="d" * 64,
        fold_assignment_sha256="e" * 64,
        held_out=_outcomes(69, improvements=20),
        structure_or_hold_regressions=0,
        completeness_regressions=0,
        severe_gap_or_intro_regressions=0,
        unknown_pair_count=0,
    )
    tampered = {
        **evidence.to_report(),
        "difficultyDeltaCiUpper": evidence.difficulty_delta_ci_upper - 0.5,
    }

    with pytest.raises(ValueError, match="recalculation"):
        parse_calibration_activation_evidence_v3(tampered)
