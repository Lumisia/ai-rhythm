from pathlib import Path

import numpy as np

from chart_worker.analysis.activity import AudioActivity
from chart_worker.analysis.local_timing import measure_local_timing
from chart_worker.analysis.onset import OnsetAnalysis
from chart_worker.generation.osu_parser import OsuBpmEvent
from chart_worker.stages.types import SongTimingAuthority
from chart_worker.validation.local_timing_review import review_local_timing_authority
from chart_worker.validation.recovery_preflight import review_recovery_preflight
from chart_worker.validation.timing_review import TimingAuthorityAction


def _authority(*events: tuple[int, float]) -> SongTimingAuthority:
    return SongTimingAuthority(
        reference_path=Path("timing.osu"),
        sha256="timing",
        audio_sha256="audio",
        bpm_events=tuple(OsuBpmEvent(time_ms=time, bpm=bpm) for time, bpm in events),
        generator_name="test",
        seed=0,
        mode="STANDARD",
        attempt_count=1,
    )


def _analysis(
    duration_ms: int,
    onset_ms: tuple[int, ...],
    *,
    quiet_range: tuple[int, int] | None = None,
) -> OnsetAnalysis:
    frame_ms = 100
    frame_count = duration_ms // frame_ms
    strength = np.zeros(frame_count)
    for time_ms in onset_ms:
        strength[min(frame_count - 1, time_ms // frame_ms)] = 1.0
    rms = np.full(frame_count, -10.0)
    active_onsets = onset_ms
    if quiet_range is not None:
        start_ms, end_ms = quiet_range
        rms[start_ms // frame_ms : end_ms // frame_ms] = -80.0
        active_onsets = tuple(
            time_ms for time_ms in onset_ms if not start_ms <= time_ms < end_ms
        )
    return OnsetAnalysis(
        sample_rate_hz=1_000,
        hop_length=100,
        strength=strength,
        band_strength=np.vstack((strength, strength, strength)),
        onset_ms=onset_ms,
        activity=AudioActivity(
            frame_ms=frame_ms,
            rms_db=rms,
            floor_db=-40.0,
            active_onset_ms=active_onsets,
        ),
    )


def _review(authority: SongTimingAuthority, analysis: OnsetAnalysis, duration_ms: int):
    preflight = review_recovery_preflight(
        authority,
        analysis,
        duration_ms=duration_ms,
    )
    metrics = measure_local_timing(
        authority.bpm_events,
        analysis,
        duration_ms=duration_ms,
    )
    return review_local_timing_authority(metrics, preflight)


def test_active_isolated_bad_timing_segment_retries():
    duration_ms = 30_000
    authority = _authority((0, 80.0), (10_000, 5.6), (21_000, 165.0))
    analysis = _analysis(duration_ms, tuple(range(0, duration_ms, 250)))

    review = _review(authority, analysis, duration_ms)
    damaged = review.segments[1]

    assert review.action is TimingAuthorityAction.RETRY_TIMING
    assert damaged.grid_damage is True
    assert damaged.isolated_metrical_outlier is True
    assert damaged.pulse_conflict is True
    assert damaged.metrics.neighbor_residual_p95_ms == 0.0


def test_same_timing_segment_in_a_quiet_break_is_not_hard_retry():
    duration_ms = 30_000
    authority = _authority((0, 80.0), (10_000, 5.6), (21_000, 165.0))
    analysis = _analysis(
        duration_ms,
        tuple(range(0, duration_ms, 250)),
        quiet_range=(10_000, 21_000),
    )

    review = _review(authority, analysis, duration_ms)

    assert review.action is not TimingAuthorityAction.RETRY_TIMING
    assert review.segments[1].active_confident is False


def test_genuinely_slow_timing_with_supported_subdivisions_is_not_hard_retry():
    duration_ms = 30_000
    authority = _authority((0, 5.0))
    grid_onsets = set(range(0, duration_ms, 1_500))
    grid_onsets.update((750, 12_750, 24_750))
    analysis = _analysis(duration_ms, tuple(sorted(grid_onsets)))

    review = _review(authority, analysis, duration_ms)

    assert review.action is not TimingAuthorityAction.RETRY_TIMING
    assert review.segments[0].current_grid_support >= 0.80


def test_half_double_neighbor_is_not_an_isolated_metrical_outlier():
    duration_ms = 30_000
    authority = _authority((0, 80.0), (10_000, 160.0), (20_000, 80.0))
    analysis = _analysis(duration_ms, tuple(range(0, duration_ms, 125)))

    review = _review(authority, analysis, duration_ms)

    assert review.segments[1].isolated_metrical_outlier is False
    assert review.action is not TimingAuthorityAction.RETRY_TIMING


def test_insufficient_local_evidence_is_review_not_pass():
    duration_ms = 5_000
    authority = _authority((0, 120.0))
    analysis = _analysis(duration_ms, ())

    review = _review(authority, analysis, duration_ms)

    assert review.action is TimingAuthorityAction.REVIEW
    assert review.segments[0].evidence_status == "INSUFFICIENT"


def test_local_review_report_preserves_segment_threshold_inputs():
    duration_ms = 30_000
    authority = _authority((0, 80.0), (10_000, 5.6), (21_000, 165.0))
    analysis = _analysis(duration_ms, tuple(range(0, duration_ms, 250)))

    report = _review(authority, analysis, duration_ms).to_report()

    assert report["version"] == "local-timing-review-v2-duration-weighted"
    assert report["action"] == "RETRY_TIMING"
    assert report["durationEvidence"] == {
        "activeEvidenceMs": 30_000,
        "supportedActiveMs": 9_000,
        "contradictedActiveMs": 11_000,
        "insufficientActiveMs": 0,
        "quietMs": 0,
        "supportedRatio": 0.3,
        "contradictedRatio": 0.366667,
        "insufficientRatio": 0.0,
        "segmentCount": 3,
    }
    assert report["segments"][1]["startMs"] == 10_000
    assert report["segments"][1]["bpm"] == 5.6
    assert report["segments"][1]["contradictionCount"] >= 2
