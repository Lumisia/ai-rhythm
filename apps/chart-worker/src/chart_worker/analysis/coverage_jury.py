"""Policy-free local audio evidence for note-coverage gaps."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from chart_worker.analysis.coverage_opportunity import (
    RELATIVE_ATTACK_FLOOR,
    STRONG_ATTACK_QUANTILE,
    validate_onset_analysis,
)
from chart_worker.analysis.onset import OnsetAnalysis

COVERAGE_JURY_LOCAL_EVIDENCE_VERSION = "coverage-jury-local-evidence-v1"
LOCAL_CONTEXT_MIN_MS = 2_000
LOCAL_CONTEXT_MAX_MS = 8_000
LOCAL_RELATIVE_ATTACK_FLOOR = 0.25
LOCAL_ATTACK_QUANTILE = 50.0


@dataclass(frozen=True, slots=True)
class LocalAudioGapEvidence:
    version: Literal["coverage-jury-local-evidence-v1"]
    start_ms: int
    end_ms: int
    active_frame_ratio: float | None
    active_onset_count: int
    global_strong_attack_count: int
    local_strong_attack_count: int
    global_threshold: float | None
    local_threshold: float | None
    neighboring_activity_ratio: float | None

    def to_report(self) -> dict[str, object]:
        return {
            "version": self.version,
            "startMs": self.start_ms,
            "endMs": self.end_ms,
            "durationMs": self.end_ms - self.start_ms,
            "activeFrameRatio": self.active_frame_ratio,
            "activeOnsetCount": self.active_onset_count,
            "globalStrongAttackCount": self.global_strong_attack_count,
            "localStrongAttackCount": self.local_strong_attack_count,
            "globalThreshold": self.global_threshold,
            "localThreshold": self.local_threshold,
            "neighboringActivityRatio": self.neighboring_activity_ratio,
            "policyState": "OBSERVATION_ONLY",
            "mutatesGeneration": False,
        }


def _exact_non_negative_int(value: object, field: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{field} must be an exact integer")
    if value < 0:
        raise ValueError(f"{field} must be non-negative")
    return value


def _threshold(
    strengths: tuple[float, ...],
    *,
    relative_floor: float,
    quantile: float,
) -> float | None:
    if not strengths:
        return None
    values = np.asarray(strengths, dtype=np.float64)
    maximum = float(np.max(values))
    if maximum <= 0:
        return None
    return max(maximum * relative_floor, float(np.percentile(values, quantile)))


def measure_local_gap_evidence(
    onset_analysis: OnsetAnalysis,
    *,
    start_ms: int,
    end_ms: int,
) -> LocalAudioGapEvidence:
    """Measure a gap at song and local scales without deciding whether to repair it."""
    start_ms = _exact_non_negative_int(start_ms, "start_ms")
    end_ms = _exact_non_negative_int(end_ms, "end_ms")
    if end_ms <= start_ms:
        raise ValueError("end_ms must be after start_ms")
    validate_onset_analysis(onset_analysis)

    activity = onset_analysis.activity
    active_times = onset_analysis.onset_ms if activity is None else activity.active_onset_ms
    if tuple(sorted(set(active_times))) != active_times or any(
        type(time_ms) is not int or time_ms < 0 for time_ms in active_times
    ):
        raise ValueError("active onset times must be sorted non-negative exact integers")
    onset_times = set(onset_analysis.onset_ms)
    if any(time_ms not in onset_times for time_ms in active_times):
        raise ValueError("active onset times must be a subset of detected onset times")

    interval_times = tuple(time_ms for time_ms in active_times if start_ms < time_ms < end_ms)
    interval_strengths = tuple(onset_analysis.strength_at(time_ms) for time_ms in interval_times)
    global_strengths = tuple(onset_analysis.strength_at(time_ms) for time_ms in active_times)
    global_threshold = _threshold(
        global_strengths,
        relative_floor=RELATIVE_ATTACK_FLOOR,
        quantile=STRONG_ATTACK_QUANTILE,
    )

    duration_ms = end_ms - start_ms
    context_ms = min(max(duration_ms // 2, LOCAL_CONTEXT_MIN_MS), LOCAL_CONTEXT_MAX_MS)
    local_context_times = tuple(
        time_ms
        for time_ms in active_times
        if max(0, start_ms - context_ms) <= time_ms <= end_ms + context_ms
    )
    local_threshold = _threshold(
        tuple(onset_analysis.strength_at(time_ms) for time_ms in local_context_times),
        relative_floor=LOCAL_RELATIVE_ATTACK_FLOOR,
        quantile=LOCAL_ATTACK_QUANTILE,
    )

    global_strong_count = (
        0
        if global_threshold is None
        else sum(value >= global_threshold for value in interval_strengths)
    )
    local_strong_count = (
        0
        if local_threshold is None
        else sum(value >= local_threshold for value in interval_strengths)
    )

    active_frame_ratio: float | None = None
    neighboring_activity_ratio: float | None = None
    if activity is not None:
        active_frame_ratio = round(activity.active_frame_ratio(start_ms, end_ms), 6)
        available_end_ms = round(max(0, onset_analysis.frame_count - 1) * onset_analysis.frame_ms)
        neighbor_ratios = []
        if start_ms > 0:
            before_start = max(0, start_ms - duration_ms)
            neighbor_ratios.append(activity.active_frame_ratio(before_start, start_ms))
        if end_ms < available_end_ms:
            after_end = min(available_end_ms, end_ms + duration_ms)
            neighbor_ratios.append(activity.active_frame_ratio(end_ms, after_end))
        if neighbor_ratios:
            neighboring_activity_ratio = round(float(np.mean(neighbor_ratios)), 6)

    return LocalAudioGapEvidence(
        version=COVERAGE_JURY_LOCAL_EVIDENCE_VERSION,
        start_ms=start_ms,
        end_ms=end_ms,
        active_frame_ratio=active_frame_ratio,
        active_onset_count=len(interval_times),
        global_strong_attack_count=global_strong_count,
        local_strong_attack_count=local_strong_count,
        global_threshold=None if global_threshold is None else round(global_threshold, 6),
        local_threshold=None if local_threshold is None else round(local_threshold, 6),
        neighboring_activity_ratio=neighboring_activity_ratio,
    )
