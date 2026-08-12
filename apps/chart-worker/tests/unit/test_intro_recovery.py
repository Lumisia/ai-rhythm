from chart_worker.analysis.intro_anchor import IntroAnchorEvidence
from chart_worker.generation.intro_recovery import build_intro_recovery_plan
from chart_worker.generation.osu_parser import OsuBpmEvent


def _evidence(status: str = "CONFIRMED") -> IntroAnchorEvidence:
    return IntroAnchorEvidence(
        status=status,
        anchor_ms=21,
        anchor_grid_ms=0,
        grid_distance_ms=21,
        aggregate_percentile_rank=0.99,
        prominent_band_count=3,
        pulse_continuation_matches=3,
        pulse_continuation_opportunities=4,
    )


def test_plan_adds_measured_anchor_only_to_conditioning_timing():
    original = (OsuBpmEvent(405, 140.0), OsuBpmEvent(1_695, 140.0))

    plan = build_intro_recovery_plan(
        _evidence(),
        original,
        duration_ms=2_000,
    )

    assert plan is not None
    assert plan.partial_start_ms == 0
    assert plan.partial_end_ms == 1_736
    assert plan.conditioned_bpm_events == (
        OsuBpmEvent(21, 140.0),
        *original,
    )
    assert original == (OsuBpmEvent(405, 140.0), OsuBpmEvent(1_695, 140.0))


def test_uncertain_anchor_never_builds_a_recovery_plan():
    assert (
        build_intro_recovery_plan(
            _evidence("UNCERTAIN"),
            (OsuBpmEvent(405, 140.0),),
            duration_ms=2_000,
        )
        is None
    )

def test_anchor_at_or_after_first_event_needs_no_leading_recovery():
    evidence = IntroAnchorEvidence(
        status="CONFIRMED",
        anchor_ms=405,
        anchor_grid_ms=405,
        grid_distance_ms=0,
        aggregate_percentile_rank=0.99,
        prominent_band_count=3,
        pulse_continuation_matches=3,
        pulse_continuation_opportunities=4,
    )

    assert (
        build_intro_recovery_plan(
            evidence,
            (OsuBpmEvent(405, 140.0),),
            duration_ms=2_000,
        )
        is None
    )
