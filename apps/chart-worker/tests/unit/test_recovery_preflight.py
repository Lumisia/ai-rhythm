from pathlib import Path

import numpy as np

from chart_worker.analysis.activity import AudioActivity
from chart_worker.analysis.onset import OnsetAnalysis
from chart_worker.generation.osu_parser import OsuBpmEvent
from chart_worker.stages.types import SongTimingAuthority
from chart_worker.validation.recovery_preflight import (
    RecoveryPreflightAction,
    review_recovery_preflight,
)


def _authority(bpm: float) -> SongTimingAuthority:
    return SongTimingAuthority(
        reference_path=Path("timing.osu"),
        sha256="timing",
        audio_sha256="audio",
        bpm_events=(OsuBpmEvent(time_ms=0, bpm=bpm),),
        generator_name="test",
        seed=0,
        mode="STANDARD",
        attempt_count=1,
    )


def _analysis(duration_ms: int, *, active: bool) -> OnsetAnalysis:
    onset_ms = tuple(range(0, duration_ms, 1_000))
    frame_ms = 100
    frame_count = duration_ms // frame_ms
    strength = np.zeros(frame_count)
    for time_ms in onset_ms:
        strength[min(frame_count - 1, time_ms // frame_ms)] = 1.0
    return OnsetAnalysis(
        sample_rate_hz=1_000,
        hop_length=100,
        strength=strength,
        band_strength=np.vstack((strength, strength, strength)),
        onset_ms=onset_ms,
        activity=AudioActivity(
            frame_ms=frame_ms,
            rms_db=np.full(frame_count, -10.0 if active else -80.0),
            floor_db=-40.0,
            active_onset_ms=onset_ms if active else (),
        ),
    )


def test_active_fixed_divisor_gap_is_review_when_an_alternate_divisor_works():
    report = review_recovery_preflight(
        _authority(5.0),
        _analysis(12_000, active=True),
        duration_ms=12_000,
    )

    easy = report.for_difficulty("EASY")
    assert easy.action is RecoveryPreflightAction.REVIEW
    assert easy.selected_divisor == 1
    assert easy.active_gaps
    assert 8 in easy.viable_divisors


def test_quiet_fixed_divisor_gap_is_not_timing_damage():
    report = review_recovery_preflight(
        _authority(5.0),
        _analysis(12_000, active=False),
        duration_ms=12_000,
    )

    easy = report.for_difficulty("EASY")
    assert easy.action is RecoveryPreflightAction.PASS
    assert not easy.active_gaps


def test_all_allowed_divisors_can_be_reported_as_damaged():
    report = review_recovery_preflight(
        _authority(0.25),
        _analysis(20_000, active=True),
        duration_ms=20_000,
    )

    easy = report.for_difficulty("EASY")
    assert easy.action is RecoveryPreflightAction.DAMAGED
    assert easy.active_gaps
    assert easy.viable_divisors == ()
    assert report.action is RecoveryPreflightAction.DAMAGED


def test_preflight_report_keeps_divisor_and_gap_evidence():
    report = review_recovery_preflight(
        _authority(5.0),
        _analysis(12_000, active=True),
        duration_ms=12_000,
    ).to_report()

    assert report["version"] == "recovery-preflight-v1"
    assert report["action"] == "REVIEW"
    assert report["difficulties"][0]["difficulty"] == "EASY"
    assert report["difficulties"][0]["selectedDivisor"] == 1
    assert report["difficulties"][0]["activeGaps"][0]["durationMs"] == 12_000
