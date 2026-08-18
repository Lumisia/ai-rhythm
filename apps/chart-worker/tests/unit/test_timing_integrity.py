from __future__ import annotations

import pytest

from chart_worker.analysis.local_timing import LocalTimingSegmentMetrics
from chart_worker.validation.local_timing_review import (
    LocalTimingAuthorityReview,
    LocalTimingSegmentReview,
)
from chart_worker.validation.recovery_preflight import (
    RecoveryPreflight,
    RecoveryPreflightAction,
)
from chart_worker.validation.timing_integrity import (
    TimingIntegrityStatus,
    assess_timing_integrity,
)
from chart_worker.validation.timing_review import TimingAuthorityAction


def _segment(
    index: int,
    start_ms: int,
    end_ms: int,
    bpm: float,
    *,
    active: bool = True,
    current_support: float = 0.8,
    neighbor_support: float = 0.8,
    pulse_conflict: bool = False,
    grid_damage: bool = False,
    quiet: bool = False,
) -> LocalTimingSegmentReview:
    metrics = LocalTimingSegmentMetrics(
        index=index,
        start_ms=start_ms,
        end_ms=end_ms,
        bpm=bpm,
        onset_count=32,
        active_onset_count=0 if quiet else 32,
        active_frame_ratio=0.05 if quiet else 0.9,
        active_confident=active and not quiet,
        current_grid_support=current_support,
        neighbor_grid_support=neighbor_support,
        current_residual_p95_ms=15.0 if current_support >= 0.55 else 180.0,
        neighbor_residual_p95_ms=15.0,
        isolated_metrical_outlier=False,
        pulse_conflict=pulse_conflict,
        phase_conflict=pulse_conflict,
        evidence_status="SUFFICIENT" if active and not quiet else "INSUFFICIENT",
        boundary_onset_distance_ms=10.0,
        boundary_supported=True,
    )
    reasons = ("QUIET_LOCAL_TIMING_SEGMENT",) if quiet else ()
    return LocalTimingSegmentReview(
        metrics=metrics,
        grid_damage=grid_damage,
        action=(
            TimingAuthorityAction.PASS
            if quiet or (not grid_damage and not pulse_conflict)
            else TimingAuthorityAction.REVIEW
        ),
        reasons=reasons,
    )


def _review(*segments: LocalTimingSegmentReview) -> LocalTimingAuthorityReview:
    action = (
        TimingAuthorityAction.REVIEW
        if any(segment.action is TimingAuthorityAction.REVIEW for segment in segments)
        else TimingAuthorityAction.PASS
    )
    return LocalTimingAuthorityReview(action=action, reasons=(), segments=segments)


def _preflight(action: RecoveryPreflightAction) -> RecoveryPreflight:
    return RecoveryPreflight(action=action, difficulties=())


def test_active_return_island_with_damaged_preflight_is_damaged() -> None:
    review = _review(
        _segment(151, 142_805, 143_735, 94.0, active=False),
        _segment(
            152,
            143_735,
            158_890,
            3.96,
            current_support=0.1,
            neighbor_support=0.8,
            pulse_conflict=True,
            grid_damage=True,
        ),
        _segment(
            153,
            158_890,
            165_881,
            9.4,
            active=False,
            current_support=0.2,
            neighbor_support=0.75,
            pulse_conflict=True,
            grid_damage=True,
        ),
        _segment(154, 165_881, 167_178, 94.0, active=False),
    )

    result = assess_timing_integrity(
        review,
        _preflight(RecoveryPreflightAction.DAMAGED),
    )

    assert result.status is TimingIntegrityStatus.DAMAGED
    assert result.reasons == (
        "ACTIVE_RETURN_TIMING_ISLAND",
        "RECOVERY_PREFLIGHT_DAMAGED",
    )
    assert len(result.islands) == 1
    assert result.islands[0].segment_indices == (152, 153)
    assert result.islands[0].active_duration_ms == 22_146


def test_extreme_active_island_without_damaged_preflight_needs_corroboration() -> None:
    review = _review(
        _segment(0, 0, 20_000, 294.0),
        _segment(1, 20_000, 28_000, 23.5, current_support=0.2, neighbor_support=0.8, pulse_conflict=True),
        _segment(2, 28_000, 36_000, 24.7, current_support=0.2, neighbor_support=0.8, pulse_conflict=True),
        _segment(3, 36_000, 44_000, 18.0, current_support=0.2, neighbor_support=0.8, pulse_conflict=True),
        _segment(4, 44_000, 80_000, 141.0),
    )

    result = assess_timing_integrity(
        review,
        _preflight(RecoveryPreflightAction.REVIEW),
    )

    assert result.status is TimingIntegrityStatus.NEEDS_CORROBORATION
    assert result.islands[0].segment_indices == (1, 2, 3)


@pytest.mark.parametrize(
    "segments",
    [
        (
            _segment(0, 0, 20_000, 90.0),
            _segment(1, 20_000, 40_000, 95.0),
            _segment(2, 40_000, 60_000, 100.0),
            _segment(3, 60_000, 80_000, 105.0),
        ),
        (
            _segment(0, 0, 40_000, 90.0),
            _segment(1, 40_000, 80_000, 140.0),
        ),
        (
            _segment(0, 0, 20_000, 90.0),
            _segment(1, 20_000, 40_000, 140.0),
            _segment(2, 40_000, 60_000, 90.0),
        ),
    ],
    ids=("gradual-ramp", "persistent-change", "supported-return"),
)
def test_supported_variable_bpm_is_healthy(
    segments: tuple[LocalTimingSegmentReview, ...],
) -> None:
    result = assess_timing_integrity(
        _review(*segments),
        _preflight(RecoveryPreflightAction.PASS),
    )

    assert result.status is TimingIntegrityStatus.HEALTHY
    assert result.islands == ()


def test_quiet_fermata_is_not_classified_as_damage() -> None:
    result = assess_timing_integrity(
        _review(
            _segment(0, 0, 20_000, 90.0),
            _segment(1, 20_000, 35_000, 8.0, active=False, quiet=True),
            _segment(2, 35_000, 60_000, 90.0),
        ),
        _preflight(RecoveryPreflightAction.DAMAGED),
    )

    assert result.status is TimingIntegrityStatus.NEEDS_CORROBORATION
    assert result.islands == ()
    assert result.reasons == ("RECOVERY_PREFLIGHT_DAMAGED",)


def test_report_preserves_units_and_evidence() -> None:
    result = assess_timing_integrity(
        _review(
            _segment(0, 0, 20_000, 120.0),
            _segment(1, 20_000, 30_000, 10.0, current_support=0.1, neighbor_support=0.8, pulse_conflict=True),
            _segment(2, 30_000, 50_000, 120.0),
        ),
        _preflight(RecoveryPreflightAction.REVIEW),
    )

    report = result.to_report()
    assert report["version"] == "timing-integrity-v1"
    assert report["status"] == "NEEDS_CORROBORATION"
    assert report["islands"][0]["startMs"] == 20_000
    assert report["islands"][0]["endMs"] == 30_000
    assert report["islands"][0]["currentGridSupport"] == 0.1
