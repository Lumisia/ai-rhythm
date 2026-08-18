from __future__ import annotations

from chart_worker.validation.final_difficulty_family import (
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
    )

    assert observed.calibration_state == "UNAVAILABLE"
    assert observed.provisional_concern == "RECOVERY_INVERSION"
    assert observed.project_rating_inversions == (("HARD", "EXPERT"),)
    assert observed.ordering_score_inversions == (("HARD", "EXPERT"),)
    assert observed.recovery_difficulties == ("EXPERT",)
    assert observed.to_report()["policyState"] == "OBSERVATION_ONLY"


def test_legacy_and_v2_disagreement_is_not_promoted_to_a_certain_inversion() -> None:
    observed = observe_final_difficulty_family(
        6,
        (
            _entry("EASY", 1.0, 10.0),
            _entry("NORMAL", 2.0, 20.0),
            _entry("HARD", 5.39, 30.0),
            _entry("EXPERT", 5.36, 40.0),
        ),
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
    )

    assert observed.provisional_concern == "NONE"
    assert observed.calibration_state == "UNAVAILABLE"
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
    )

    assert observed.key_mode == 4
    assert observed.missing_difficulties == ("EASY", "NORMAL")
    assert observed.provisional_concern == "INCOMPLETE_EVIDENCE"
