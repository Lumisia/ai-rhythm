import math
from pathlib import Path
from types import SimpleNamespace

import pytest

from chart_worker.analysis.intro_anchor import IntroAnchorEvidence
from chart_worker.generation import intro_recovery
from chart_worker.generation.candidate_state import VariantState
from chart_worker.generation.generation_control import (
    AdditionalInferenceBudget,
    RecoveryKind,
)
from chart_worker.generation.intro_recovery import (
    build_intro_recovery_plan,
    execute_intro_retry,
    intro_phrase_recovery_end_ms,
    intro_region_recovery_end_ms,
)
from chart_worker.generation.osu_parser import OsuBpmEvent
from chart_worker.generation.params import GenerationRequest


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


def test_phrase_window_uses_local_variable_bpm_for_four_beat_context():
    assert intro_phrase_recovery_end_ms(
        12_000,
        (OsuBpmEvent(0, 120.0), OsuBpmEvent(10_000, 240.0)),
        duration_ms=20_000,
    ) == 13_000


def test_phrase_window_integrates_a_bpm_change_inside_the_context() -> None:
    assert intro_phrase_recovery_end_ms(
        9_000,
        (OsuBpmEvent(0, 120.0), OsuBpmEvent(10_000, 240.0)),
        duration_ms=20_000,
    ) == 10_500


def test_phrase_window_declines_a_near_full_song_remap():
    assert (
        intro_phrase_recovery_end_ms(
            17_000,
            (OsuBpmEvent(0, 120.0),),
            duration_ms=20_000,
        )
        is None
    )


def test_region_window_adds_local_four_beat_context_without_following_late_chart():
    assert intro_region_recovery_end_ms(
        4_320,
        (OsuBpmEvent(0, 120.0),),
        duration_ms=20_000,
    ) == 6_320


def test_region_window_integrates_non_aligned_variable_bpm_change():
    assert intro_region_recovery_end_ms(
        4_000,
        (OsuBpmEvent(0, 120.0), OsuBpmEvent(4_750, 240.0)),
        duration_ms=20_000,
    ) == 5_375


def test_region_window_declines_when_tempo_authority_starts_after_intro():
    """A confirmed intro must not invent tempo before the first authority event."""

    assert (
        intro_region_recovery_end_ms(
            3_741,
            (OsuBpmEvent(6_937, 87.0),),
            duration_ms=250_039,
        )
        is None
    )


def test_region_window_accepts_a_valid_pre_zero_tempo_event():
    assert intro_region_recovery_end_ms(
        4_000,
        (OsuBpmEvent(-1_000, 120.0),),
        duration_ms=20_000,
    ) == 6_000


@pytest.mark.parametrize(
    "events",
    [
        (OsuBpmEvent(6_937, 0.0),),
        (OsuBpmEvent(6_937, math.inf),),
        (OsuBpmEvent(6_937, 120.0), OsuBpmEvent(6_937, 140.0)),
    ],
)
def test_region_window_does_not_hide_a_malformed_late_tempo_map(events):
    with pytest.raises(ValueError):
        intro_region_recovery_end_ms(
            3_741,
            events,
            duration_ms=250_039,
        )


def test_intro_retry_does_not_spend_budget_after_tail_exhaustion(monkeypatch, tmp_path: Path):
    blocked_by = {"signature": "END_BOUNDARY_CROSSES_HOLD:500:11"}
    state = VariantState(
        key_mode=4,
        difficulty="EXPERT",
        flat_index=3,
        full_length_retry_blocked_by=blocked_by,
    )
    source = SimpleNamespace(
        request=GenerationRequest(
            audio_path=Path("game.flac"),
            timing_reference_path=Path("timing.osu"),
            key_mode=4,
            difficulty="EXPERT",
            seed=3,
            duration_ms=2_000,
        )
    )
    budget = AdditionalInferenceBudget(limit=1)

    def forbidden_inference(*args, **kwargs):
        del args, kwargs
        raise AssertionError("blocked intro retry reached inference")

    monkeypatch.setattr(
        intro_recovery,
        "run_inference_with_journal",
        forbidden_inference,
    )

    result = execute_intro_retry(
        state,
        source,
        prepared=None,
        authority=SimpleNamespace(reference_path=Path("timing.osu")),
        onset_analysis=None,
        run_dir=tmp_path,
        generator=object(),
        base_seed=0,
        authority_epoch=1,
        inference_budget=budget,
        evaluate_candidate=lambda *args, **kwargs: None,
        serialize_candidate=lambda *args, **kwargs: "",
        intro_anchor_covered=lambda *args, **kwargs: None,
    )

    assert result is None
    assert budget.used == 0
    assert state.recovery.was_attempted(RecoveryKind.INTRO) is False
    assert state.attempt_evidence == [
        {
            "reason": "INTRO_CONTRACT_RETRY_SUPPRESSED_BY_TAIL_EXHAUSTION",
            "blockedBy": blocked_by,
        }
    ]
