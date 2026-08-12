from __future__ import annotations

import numpy as np

from chart_worker.analysis.activity import AudioActivity
from chart_worker.analysis.outro_evidence import build_outro_evidence_profile


def test_outro_profile_records_raw_and_active_evidence_for_each_tail_window():
    activity = AudioActivity(
        frame_ms=1_000.0,
        rms_db=np.asarray(
            [-70.0, -65.0, -60.0, -55.0, -50.0, -45.0, -35.0, -25.0, -50.0, -70.0]
        ),
        floor_db=-40.0,
        active_onset_ms=(6_500, 8_500),
    )

    profile = build_outro_evidence_profile(
        activity=activity,
        onset_ms=(1_500, 5_500, 6_500, 8_500, 9_500),
        duration_ms=10_000,
    )

    report = profile.to_report()
    assert report["version"] == "outro-evidence-profile-v1"
    assert report["policyState"] == "UNCALIBRATED"
    assert report["semanticClassification"] == "UNAVAILABLE"
    assert [window["windowMs"] for window in report["windows"]] == [1_000, 2_000, 5_000, 10_000]

    one, two, five, ten = report["windows"]
    assert (one["rawOnsetCount"], one["activeOnsetCount"], one["activeFrameRatio"]) == (1, 0, 0.0)
    assert (two["rawOnsetCount"], two["activeOnsetCount"], two["activeFrameRatio"]) == (2, 1, 0.0)
    assert (five["rawOnsetCount"], five["activeOnsetCount"], five["activeFrameRatio"]) == (4, 2, 0.4)
    assert (ten["rawOnsetCount"], ten["activeOnsetCount"], ten["activeFrameRatio"]) == (5, 2, 0.2)
    assert five["rmsPercentilesDb"] == {"p50": -45.0, "p90": -29.0, "p99": -25.4}
    assert isinstance(five["rmsSlopeDbPerSec"], float)


def test_outro_profile_is_observational_and_handles_missing_frames_without_a_fake_verdict():
    profile = build_outro_evidence_profile(
        activity=AudioActivity(
            frame_ms=0.0,
            rms_db=np.asarray([], dtype=np.float64),
            floor_db=-60.0,
            active_onset_ms=(),
        ),
        onset_ms=(),
        duration_ms=0,
    )

    report = profile.to_report()
    assert report["semanticClassification"] == "UNAVAILABLE"
    assert report["windows"][0]["rmsPercentilesDb"] == {
        "p50": None,
        "p90": None,
        "p99": None,
    }
    assert report["windows"][0]["rmsSlopeDbPerSec"] is None
    assert "decision" not in report


def test_outro_profile_preserves_elapsed_time_when_non_finite_frames_are_ignored():
    profile = build_outro_evidence_profile(
        activity=AudioActivity(
            frame_ms=1_000.0,
            rms_db=np.asarray([-10.0, np.nan, -30.0]),
            floor_db=-40.0,
            active_onset_ms=(),
        ),
        onset_ms=(),
        duration_ms=3_000,
    )

    # The two finite observations are two seconds apart.  Removing the NaN
    # must not compress that interval to one second.
    assert profile.to_report()["windows"][2]["rmsSlopeDbPerSec"] == -10.0
