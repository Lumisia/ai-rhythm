import numpy as np

from chart_worker.analysis.activity import AudioActivity
from chart_worker.analysis.onset import OnsetAnalysis
from chart_worker.generation.osu_parser import OsuBpmEvent
from chart_worker.validation.leading_timing_coverage import (
    review_leading_timing_coverage,
)
from chart_worker.validation.timing_review import TimingAuthorityAction


def _analysis(
    duration_ms: int,
    *,
    onset_ms: tuple[int, ...],
    active_onset_ms: tuple[int, ...],
    active: bool,
) -> OnsetAnalysis:
    frame_ms = 100.0
    frame_count = duration_ms // 100 + 1
    rms_db = np.full(frame_count, -10.0 if active else -80.0)
    return OnsetAnalysis(
        sample_rate_hz=1_000,
        hop_length=100,
        strength=np.ones(frame_count),
        band_strength=np.ones((3, frame_count)),
        onset_ms=onset_ms,
        n_fft=1,
        activity=AudioActivity(
            frame_ms=frame_ms,
            rms_db=rms_db,
            floor_db=-60.0,
            active_onset_ms=active_onset_ms,
        ),
    )


def test_long_active_leading_gap_retries_with_stable_evidence():
    onsets = tuple(range(500, 95_000, 500))

    result = review_leading_timing_coverage(
        (OsuBpmEvent(95_645, 222.0),),
        _analysis(150_000, onset_ms=onsets, active_onset_ms=onsets, active=True),
        duration_ms=150_000,
    )

    assert result.action is TimingAuthorityAction.RETRY_TIMING
    assert result.reasons == ("ACTIVE_LEADING_TIMING_GAP",)
    assert result.to_report() == {
        "action": "RETRY_TIMING",
        "reasons": ["ACTIVE_LEADING_TIMING_GAP"],
        "firstEventTimeMs": 95_645,
        "leadingDurationMs": 95_645,
        "onsetCount": 189,
        "activeOnsetCount": 189,
        "activeFrameRatio": 1.0,
        "introAnchor": {
            "status": "UNCERTAIN",
            "anchorMs": 500,
            "anchorGridMs": 510,
            "gridDistanceMs": 10,
            "aggregatePercentileRank": 0.5,
            "prominentBandCount": 0,
            "pulseContinuationMatches": 1,
            "pulseContinuationOpportunities": 4,
        },
    }


def test_short_confirmed_intro_routes_to_map_review_without_timing_retry():
    """phase가 맞는 인트로는 Timing 대신 MAP 첫 창 복구 대상으로 남긴다."""
    onsets = tuple(range(100, 2_600, 250))

    result = review_leading_timing_coverage(
        (OsuBpmEvent(2_678, 222.0),),
        _analysis(150_000, onset_ms=onsets, active_onset_ms=onsets, active=True),
        duration_ms=150_000,
    )

    assert result.action is TimingAuthorityAction.REVIEW
    assert result.reasons == ("CONFIRMED_INTRO_ANCHOR_BEFORE_FIRST_EVENT",)
    assert result.intro_anchor.status == "CONFIRMED"
    assert result.leading_duration_ms == 2_678


def test_anchor_within_half_a_beat_passes():
    """리듬 시작이 첫 timing event 반 박 이내면 인트로가 덮여 있다."""
    result = review_leading_timing_coverage(
        (OsuBpmEvent(520, 120.0),),
        _analysis(
            10_000,
            onset_ms=(500, 750, 1_000),
            active_onset_ms=(500, 750, 1_000),
            active=True,
        ),
        duration_ms=10_000,
    )

    assert result.action is TimingAuthorityAction.PASS
    assert result.reasons == ()


def test_single_intro_fx_is_not_treated_as_rhythm_start():
    """단발 FX 하나로 timing 재생성을 강제하지 않는다."""
    result = review_leading_timing_coverage(
        (OsuBpmEvent(4_000, 120.0),),
        _analysis(
            20_000,
            onset_ms=(120,),
            active_onset_ms=(120,),
            active=True,
        ),
        duration_ms=20_000,
    )

    assert result.action is TimingAuthorityAction.REVIEW
    assert result.reasons == ("UNCERTAIN_INTRO_ANCHOR",)
    assert result.intro_anchor.status == "UNCERTAIN"


def test_long_quiet_leading_gap_passes():
    onsets = tuple(range(500, 95_000, 500))

    result = review_leading_timing_coverage(
        (OsuBpmEvent(95_645, 222.0),),
        _analysis(150_000, onset_ms=onsets, active_onset_ms=(), active=False),
        duration_ms=150_000,
    )

    assert result.action is TimingAuthorityAction.PASS
    assert result.reasons == ("QUIET_LEADING_TIMING_GAP",)
    assert result.active_onset_count == 0
    assert result.active_frame_ratio == 0.0


def test_missing_activity_uses_all_onsets_as_active_fallback():
    onsets = tuple(range(500, 10_000, 500))
    analysis = OnsetAnalysis(
        sample_rate_hz=1_000,
        hop_length=100,
        strength=np.ones(201),
        band_strength=np.ones((3, 201)),
        onset_ms=onsets,
        n_fft=1,
    )

    result = review_leading_timing_coverage(
        (OsuBpmEvent(10_000, 120.0),),
        analysis,
        duration_ms=20_000,
    )

    assert result.action is TimingAuthorityAction.RETRY_TIMING
    assert result.active_onset_count == 19
    assert result.active_frame_ratio == 1.0


def test_non_positive_first_event_has_no_leading_gap():
    result = review_leading_timing_coverage(
        (OsuBpmEvent(-100, 120.0),),
        _analysis(10_000, onset_ms=(), active_onset_ms=(), active=False),
        duration_ms=10_000,
    )

    assert result.action is TimingAuthorityAction.PASS
    assert result.reasons == ()
    assert result.leading_duration_ms == 0
