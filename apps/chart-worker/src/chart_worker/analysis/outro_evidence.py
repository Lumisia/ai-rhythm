"""Policy-free, multi-window observations for song outro calibration."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, floor
from typing import Literal

import numpy as np

from chart_worker.analysis.activity import AudioActivity

OUTRO_EVIDENCE_WINDOWS_MS = (1_000, 2_000, 5_000, 10_000)


@dataclass(frozen=True, slots=True)
class OutroWindowEvidence:
    window_ms: int
    start_ms: int
    end_ms: int
    rms_percentiles_db: dict[str, float | None]
    active_frame_ratio: float
    raw_onset_count: int
    active_onset_count: int
    rms_slope_db_per_sec: float | None

    def to_report(self) -> dict[str, object]:
        return {
            "windowMs": self.window_ms,
            "startMs": self.start_ms,
            "endMs": self.end_ms,
            "rmsPercentilesDb": self.rms_percentiles_db,
            "activeFrameRatio": self.active_frame_ratio,
            "rawOnsetCount": self.raw_onset_count,
            "activeOnsetCount": self.active_onset_count,
            "rmsSlopeDbPerSec": self.rms_slope_db_per_sec,
        }


@dataclass(frozen=True, slots=True)
class OutroEvidenceProfile:
    version: Literal["outro-evidence-profile-v1"]
    policy_state: Literal["UNCALIBRATED"]
    semantic_classification: Literal["UNAVAILABLE"]
    duration_ms: int
    floor_db: float
    windows: tuple[OutroWindowEvidence, ...]

    def to_report(self) -> dict[str, object]:
        return {
            "version": self.version,
            "policyState": self.policy_state,
            "semanticClassification": self.semantic_classification,
            "durationMs": self.duration_ms,
            "floorDb": self.floor_db,
            "windows": [window.to_report() for window in self.windows],
        }


def _rounded(value: float) -> float:
    return round(float(value), 6)


def _rms_statistics(
    activity: AudioActivity,
    *,
    start_ms: int,
    end_ms: int,
) -> tuple[dict[str, float | None], float | None]:
    if activity.frame_ms <= 0 or activity.rms_db.size == 0 or end_ms <= start_ms:
        return {"p50": None, "p90": None, "p99": None}, None

    start_frame = max(0, floor(start_ms / activity.frame_ms))
    end_frame = min(activity.rms_db.size, ceil(end_ms / activity.frame_ms))
    raw_values = np.asarray(
        activity.rms_db[start_frame:end_frame],
        dtype=np.float64,
    )
    finite = np.isfinite(raw_values)
    values = raw_values[finite]
    if values.size == 0:
        return {"p50": None, "p90": None, "p99": None}, None

    percentiles = {
        "p50": _rounded(np.percentile(values, 50)),
        "p90": _rounded(np.percentile(values, 90)),
        "p99": _rounded(np.percentile(values, 99)),
    }
    if values.size < 2:
        return percentiles, None

    time_sec = np.flatnonzero(finite).astype(np.float64) * (
        activity.frame_ms / 1_000.0
    )
    slope = float(np.polyfit(time_sec, values, 1)[0])
    return percentiles, _rounded(slope)


def _count_in_window(values: tuple[int, ...], start_ms: int, end_ms: int) -> int:
    return sum(start_ms <= value <= end_ms for value in values)


def build_outro_evidence_profile(
    *,
    activity: AudioActivity,
    onset_ms: tuple[int, ...],
    duration_ms: int,
) -> OutroEvidenceProfile:
    """Record repeatable tail features without inferring musical meaning.

    The profile intentionally does not select a note or release boundary.  Human
    labels have not yet calibrated those semantics, so both policy and semantic
    status remain explicit instead of converting audio features into a verdict.
    """
    clipped_duration_ms = max(0, int(duration_ms))
    raw_onsets = tuple(sorted(set(onset_ms)))
    active_onsets = tuple(sorted(set(activity.active_onset_ms)))
    windows: list[OutroWindowEvidence] = []
    for window_ms in OUTRO_EVIDENCE_WINDOWS_MS:
        start_ms = max(0, clipped_duration_ms - window_ms)
        end_ms = clipped_duration_ms
        percentiles, slope = _rms_statistics(
            activity,
            start_ms=start_ms,
            end_ms=end_ms,
        )
        windows.append(
            OutroWindowEvidence(
                window_ms=window_ms,
                start_ms=start_ms,
                end_ms=end_ms,
                rms_percentiles_db=percentiles,
                active_frame_ratio=_rounded(
                    activity.active_frame_ratio(start_ms, end_ms)
                ),
                raw_onset_count=_count_in_window(raw_onsets, start_ms, end_ms),
                active_onset_count=_count_in_window(active_onsets, start_ms, end_ms),
                rms_slope_db_per_sec=slope,
            )
        )
    return OutroEvidenceProfile(
        version="outro-evidence-profile-v1",
        policy_state="UNCALIBRATED",
        semantic_classification="UNAVAILABLE",
        duration_ms=clipped_duration_ms,
        floor_db=_rounded(activity.floor_db),
        windows=tuple(windows),
    )
