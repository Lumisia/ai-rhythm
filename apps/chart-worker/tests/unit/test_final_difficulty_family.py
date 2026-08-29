from __future__ import annotations

from chart_worker.validation.final_difficulty_family import (
    CALIBRATION_STATES,
    DifficultyFamilyEntry,
    observe_final_difficulty_family,
)


def _entry(
    difficulty: str,
    project_rating: float | None,
    ordering_score: float | None,
    *,
    provenance: str = "PRIMARY",
) -> DifficultyFamilyEntry:
    return DifficultyFamilyEntry(
        difficulty=difficulty,
        provenance=provenance,
        project_rating=project_rating,
        ordering_score=ordering_score,
    )


def test_recovery_expert_below_hard_is_exposed_without_claiming_a_calibrated_tier() -> None:
    observed = observe_final_difficulty_family(
        4,
        (
            _entry("EASY", 1.0, 10.0),
            _entry("NORMAL", 2.0, 20.0),
            _entry("HARD", 3.68, 36.0),
            _entry("EXPERT", 1.94, 19.0, provenance="SAFE_FALLBACK"),
        ),
        calibration_state="PILOT_ONLY",
    )

    assert observed.calibration_state == "PILOT_ONLY"
    assert observed.provisional_concern == "RECOVERY_INVERSION"
    assert observed.project_rating_inversions == (("HARD", "EXPERT"),)
    assert observed.ordering_score_inversions == (("HARD", "EXPERT"),)
    assert observed.recovery_difficulties == ("EXPERT",)
    assert observed.requires_review is True
    assert observed.resolution_status == "UNRESOLVED"
    report = observed.to_report()
    assert report["version"] == "final-difficulty-family-observation-v2"
    assert report["policyState"] == "REPORTING_ENFORCED"
    assert report["mutatesSelection"] is False
    assert report["resolutionStatus"] == "UNRESOLVED"
    assert report["mutatesCharts"] is False
    assert report["mutatesQualityStatus"] is True


def test_legacy_and_v2_disagreement_is_not_promoted_to_a_certain_inversion() -> None:
    observed = observe_final_difficulty_family(
        6,
        (
            _entry("EASY", 1.0, 10.0),
            _entry("NORMAL", 2.0, 20.0),
            _entry("HARD", 5.39, 30.0),
            _entry("EXPERT", 5.36, 40.0),
        ),
        calibration_state="PILOT_ONLY",
    )

    assert observed.provisional_concern == "METRIC_DISAGREEMENT"
    assert observed.project_rating_inversions == (("HARD", "EXPERT"),)
    assert observed.ordering_score_inversions == ()


def test_clean_uncalibrated_family_is_observed_without_a_pass_claim() -> None:
    observed = observe_final_difficulty_family(
        7,
        tuple(
            _entry(difficulty, float(index), float(index * 10))
            for index, difficulty in enumerate(
                ("EASY", "NORMAL", "HARD", "EXPERT"),
                start=1,
            )
        ),
        calibration_state="PILOT_ONLY",
    )

    assert observed.provisional_concern == "NONE"
    assert observed.requires_review is False
    assert observed.resolution_status == "NO_OBSERVED_CONCERN"
    assert observed.calibration_state == "PILOT_ONLY"
    assert observed.to_report()["contractStatus"] == "UNCALIBRATED"


def test_missing_metric_is_incomplete_evidence_not_a_zero_score() -> None:
    observed = observe_final_difficulty_family(
        4,
        (
            _entry("EASY", 1.0, 10.0),
            _entry("NORMAL", 2.0, 20.0),
            _entry("HARD", 3.0, None),
            _entry("EXPERT", 4.0, 40.0),
        ),
        calibration_state="PILOT_ONLY",
    )

    assert observed.provisional_concern == "INCOMPLETE_EVIDENCE"
    assert observed.missing_metric_difficulties == ("HARD",)


def test_partial_family_is_reported_and_never_compared_across_key_modes() -> None:
    observed = observe_final_difficulty_family(
        4,
        (
            _entry("HARD", 3.0, 30.0),
            _entry("EXPERT", 4.0, 40.0),
        ),
        calibration_state="PILOT_ONLY",
    )

    assert observed.key_mode == 4
    assert observed.missing_difficulties == ("EASY", "NORMAL")
    assert observed.provisional_concern == "INCOMPLETE_EVIDENCE"


def test_calibration_state_is_explicit_and_does_not_infer_authority() -> None:
    entries = (
        _entry("EASY", 1.0, 10.0),
        _entry("NORMAL", 2.0, 20.0),
        _entry("HARD", 3.0, 30.0),
        _entry("EXPERT", 4.0, 40.0),
    )

    for state in CALIBRATION_STATES:
        observed = observe_final_difficulty_family(
            4,
            entries,
            calibration_state=state,
        )

        assert observed.calibration_state == state
        assert observed.contract_status == (
            "CALIBRATED" if state == "ENFORCED" else "UNCALIBRATED"
        )
        assert observed.to_report()["productionCalibrationEnforced"] is (
            state == "ENFORCED"
        )


def test_calibration_state_cannot_be_omitted_or_invented() -> None:
    entries = (_entry("EASY", 1.0, 10.0),)

    try:
        observe_final_difficulty_family(4, entries)  # type: ignore[call-arg]
    except TypeError:
        pass
    else:  # pragma: no cover - protects the explicit caller contract
        raise AssertionError("calibration_state must be required")

    try:
        observe_final_difficulty_family(
            4,
            entries,
            calibration_state="CALIBRATED_BY_ASSUMPTION",  # type: ignore[arg-type]
        )
    except ValueError as error:
        assert "calibration_state" in str(error)
    else:  # pragma: no cover - protects fail-closed validation
        raise AssertionError("unknown calibration states must be rejected")
