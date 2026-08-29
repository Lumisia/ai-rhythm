from __future__ import annotations

from chart_worker.analysis.coverage_jury import LocalAudioGapEvidence
from chart_worker.analysis.coverage_opportunity import CoverageKind
from chart_worker.validation.coverage_family_review import (
    CoverageFamilyVerdict,
    CoverageGapMember,
    review_coverage_family,
)
from chart_worker.validation.timing_integrity import TimingIntegrityStatus


def _audio(
    *,
    active_ratio: float | None = 0.9,
    active_onsets: int = 12,
    local_attacks: int = 8,
    neighboring_ratio: float | None = 0.9,
) -> LocalAudioGapEvidence:
    return LocalAudioGapEvidence(
        version="coverage-jury-local-evidence-v1",
        start_ms=100_000,
        end_ms=110_000,
        active_frame_ratio=active_ratio,
        active_onset_count=active_onsets,
        global_strong_attack_count=3,
        local_strong_attack_count=local_attacks,
        global_threshold=0.8,
        local_threshold=0.4,
        neighboring_activity_ratio=neighboring_ratio,
    )


def _gap(
    key_mode: int,
    difficulty: str,
    *,
    start_ms: int = 100_000,
    end_ms: int = 110_000,
    model_backed: bool = True,
    hold_occupancy_ratio: float = 0.0,
    opportunity_kind: CoverageKind = CoverageKind.ATTACK_REQUIRED,
) -> CoverageGapMember:
    return CoverageGapMember(
        key_mode=key_mode,
        difficulty=difficulty,
        start_ms=start_ms,
        end_ms=end_ms,
        model_backed=model_backed,
        hold_occupancy_ratio=hold_occupancy_ratio,
        opportunity_kind=opportunity_kind,
    )


def test_shared_active_gap_with_suspect_timing_routes_to_timing_authority() -> None:
    target = _gap(4, "EXPERT")
    siblings = (
        _gap(4, "HARD"),
        _gap(6, "HARD"),
        _gap(6, "EXPERT"),
        _gap(7, "NORMAL"),
        _gap(7, "EXPERT"),
    )

    review = review_coverage_family(
        target,
        siblings,
        _audio(),
        timing_status=TimingIntegrityStatus.NEEDS_CORROBORATION,
    )

    assert review.verdict is CoverageFamilyVerdict.TIMING_AUTHORITY_SUSPECT
    assert review.independent_key_family_count == 3
    assert review.overlapping_model_chart_count == 6


def test_shared_active_gap_with_healthy_timing_is_common_model_omission() -> None:
    review = review_coverage_family(
        _gap(4, "EXPERT"),
        (
            _gap(4, "HARD"),
            _gap(6, "EXPERT"),
            _gap(7, "EXPERT"),
        ),
        _audio(),
        timing_status=TimingIntegrityStatus.HEALTHY,
    )

    assert review.verdict is CoverageFamilyVerdict.SHARED_MODEL_OMISSION
    assert review.independent_key_family_count == 3


def test_active_gap_without_sibling_overlap_is_chart_specific() -> None:
    review = review_coverage_family(
        _gap(4, "EXPERT"),
        (
            _gap(6, "EXPERT", start_ms=20_000, end_ms=30_000),
            _gap(7, "HARD", start_ms=120_000, end_ms=130_000),
        ),
        _audio(),
        timing_status=TimingIntegrityStatus.HEALTHY,
    )

    assert review.verdict is CoverageFamilyVerdict.CHART_SPECIFIC_OMISSION
    assert review.overlapping_model_chart_count == 1


def test_song_relative_expert_evidence_matches_the_frozen_active_gap() -> None:
    """Catch requiring four global attacks after local corroboration is available."""
    review = review_coverage_family(
        _gap(4, "EXPERT", start_ms=188_692, end_ms=195_451),
        (),
        LocalAudioGapEvidence(
            version="coverage-jury-local-evidence-v1",
            start_ms=188_692,
            end_ms=195_451,
            active_frame_ratio=0.651969,
            active_onset_count=7,
            global_strong_attack_count=3,
            local_strong_attack_count=3,
            global_threshold=1.0,
            local_threshold=0.966019,
            neighboring_activity_ratio=0.7153,
        ),
        timing_status=TimingIntegrityStatus.HEALTHY,
    )

    assert review.verdict is CoverageFamilyVerdict.CHART_SPECIFIC_OMISSION


def test_uncertain_source_cannot_be_promoted_by_active_family_evidence() -> None:
    """Catch family corroboration overriding a report-only source decision."""
    review = review_coverage_family(
        _gap(
            4,
            "EXPERT",
            start_ms=188_692,
            end_ms=195_451,
            opportunity_kind=CoverageKind.UNCERTAIN,
        ),
        (),
        LocalAudioGapEvidence(
            version="coverage-jury-local-evidence-v1",
            start_ms=188_692,
            end_ms=195_451,
            active_frame_ratio=0.651969,
            active_onset_count=7,
            global_strong_attack_count=3,
            local_strong_attack_count=3,
            global_threshold=1.0,
            local_threshold=0.966019,
            neighboring_activity_ratio=0.7153,
        ),
        timing_status=TimingIntegrityStatus.HEALTHY,
    )

    assert review.verdict is CoverageFamilyVerdict.AMBIGUOUS
    assert review.reasons == ("SOURCE_COVERAGE_OPPORTUNITY_IS_REPORT_ONLY",)


def test_active_interval_with_quiet_neighbors_remains_ambiguous() -> None:
    """Catch turning an isolated active bridge or stinger into mandatory notes."""
    review = review_coverage_family(
        _gap(4, "EXPERT"),
        (),
        _audio(active_ratio=0.65, active_onsets=7, local_attacks=3, neighboring_ratio=0.1),
        timing_status=TimingIntegrityStatus.HEALTHY,
    )

    assert review.verdict is CoverageFamilyVerdict.AMBIGUOUS


def test_hold_occupied_interval_is_not_authorized_for_tap_insertion() -> None:
    """Catch filling a continuous sustain with synthetic TAP attacks."""
    review = review_coverage_family(
        _gap(4, "EXPERT", hold_occupancy_ratio=0.9),
        (),
        _audio(active_ratio=0.9, active_onsets=7, local_attacks=3),
        timing_status=TimingIntegrityStatus.HEALTHY,
    )

    assert review.verdict is CoverageFamilyVerdict.AMBIGUOUS


def test_low_tier_needs_its_existing_higher_attack_floor() -> None:
    """Catch using the expert corroboration floor for an intentional EASY rest."""
    review = review_coverage_family(
        _gap(4, "EASY"),
        (),
        _audio(active_ratio=0.9, active_onsets=7, local_attacks=3),
        timing_status=TimingIntegrityStatus.HEALTHY,
    )

    assert review.verdict is CoverageFamilyVerdict.AMBIGUOUS


def test_isolated_active_gap_with_unhealthy_timing_routes_to_timing_review() -> None:
    """Catch inserting notes against a timing authority that still needs corroboration."""
    review = review_coverage_family(
        _gap(4, "EXPERT"),
        (),
        _audio(active_ratio=0.9, active_onsets=7, local_attacks=3),
        timing_status=TimingIntegrityStatus.NEEDS_CORROBORATION,
    )

    assert review.verdict is CoverageFamilyVerdict.TIMING_AUTHORITY_SUSPECT


def test_quiet_gap_is_a_musical_break_even_when_every_chart_is_empty() -> None:
    review = review_coverage_family(
        _gap(4, "EXPERT"),
        (_gap(6, "EXPERT"), _gap(7, "EXPERT")),
        _audio(active_ratio=0.05, active_onsets=0, local_attacks=0),
        timing_status=TimingIntegrityStatus.DAMAGED,
    )

    assert review.verdict is CoverageFamilyVerdict.MUSICAL_BREAK


def test_fallback_siblings_do_not_masquerade_as_independent_model_votes() -> None:
    review = review_coverage_family(
        _gap(4, "EXPERT"),
        (
            _gap(6, "EXPERT", model_backed=False),
            _gap(7, "EXPERT", model_backed=False),
        ),
        _audio(),
        timing_status=TimingIntegrityStatus.HEALTHY,
    )

    assert review.verdict is CoverageFamilyVerdict.AMBIGUOUS
    assert review.independent_key_family_count == 1
    assert review.overlapping_fallback_chart_count == 2


def test_single_stinger_or_missing_activity_remains_ambiguous() -> None:
    stinger = review_coverage_family(
        _gap(4, "EXPERT"),
        (),
        _audio(active_ratio=0.02, active_onsets=1, local_attacks=1),
        timing_status=TimingIntegrityStatus.HEALTHY,
    )
    unavailable = review_coverage_family(
        _gap(4, "EXPERT"),
        (),
        _audio(active_ratio=None, active_onsets=8, local_attacks=5),
        timing_status=TimingIntegrityStatus.HEALTHY,
    )

    assert stinger.verdict is CoverageFamilyVerdict.AMBIGUOUS
    assert unavailable.verdict is CoverageFamilyVerdict.AMBIGUOUS


def test_report_separates_correlated_chart_count_from_key_families() -> None:
    review = review_coverage_family(
        _gap(4, "EXPERT"),
        (
            _gap(4, "EASY"),
            _gap(4, "NORMAL"),
            _gap(4, "HARD"),
            _gap(6, "EXPERT"),
        ),
        _audio(),
        timing_status=TimingIntegrityStatus.HEALTHY,
    )

    report = review.to_report()
    assert report["overlappingModelChartCount"] == 5
    assert report["independentKeyFamilyCount"] == 2
    assert report["verdict"] == "SHARED_MODEL_OMISSION"
