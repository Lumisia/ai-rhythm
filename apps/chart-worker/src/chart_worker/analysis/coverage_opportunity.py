"""Beat-normalized evidence for distinguishing attacks from sustained coverage."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from typing import Literal

import numpy as np

from chart_worker.analysis.onset import OnsetAnalysis
from chart_worker.analysis.song_context import LocalTempoMap
from chart_worker.schema.note import Chart
from chart_worker.schema.types import DIFFICULTIES

COVERAGE_OPPORTUNITY_VERSION = "coverage-opportunity-v1"
MIN_PHRASE_BEATS = 16.0
PHRASE_BEAT_QUANTIZATION_TOLERANCE = 0.01
"""Integer-millisecond timestamps can undershoot an exact beat boundary slightly."""
MIN_PHRASE_DURATION_MS = 4_000
MIN_SUSTAIN_HOLD_OCCUPANCY = 0.80
MIN_ACTIVE_FRAME_RATIO = 0.35
RELATIVE_ATTACK_FLOOR = 0.35
STRONG_ATTACK_QUANTILE = 75.0
MIN_STRONG_ATTACKS = {
    "EASY": 8,
    "NORMAL": 6,
    "HARD": 4,
    "EXPERT": 4,
}


class CoverageKind(StrEnum):
    ATTACK_REQUIRED = "ATTACK_REQUIRED"
    SUSTAIN_REPRESENTABLE = "SUSTAIN_REPRESENTABLE"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"


@dataclass(frozen=True, slots=True)
class CoverageOpportunity:
    version: Literal["coverage-opportunity-v1"]
    start_ms: int
    end_ms: int
    beat_count: float | None
    strong_attack_count: int
    active_onset_count: int
    hold_occupancy_ratio: float
    active_frame_ratio: float
    strong_attack_threshold: float | None
    evidence_confidence: Literal["SUFFICIENT", "INSUFFICIENT"]
    kind: CoverageKind

    def to_report(self) -> dict[str, object]:
        return {
            "version": self.version,
            "startMs": self.start_ms,
            "endMs": self.end_ms,
            "durationMs": self.end_ms - self.start_ms,
            "beatCount": self.beat_count,
            "strongAttackCount": self.strong_attack_count,
            "activeOnsetCount": self.active_onset_count,
            "holdOccupancyRatio": self.hold_occupancy_ratio,
            "activeFrameRatio": self.active_frame_ratio,
            "strongAttackThreshold": self.strong_attack_threshold,
            "evidenceConfidence": self.evidence_confidence,
            "kind": self.kind.value,
        }


def _exact_non_negative_int(value: object, field: str) -> int:
    if type(value) is not int:  # bool and int subclasses are not protocol integers.
        raise TypeError(f"{field} must be an exact integer")
    if value < 0:
        raise ValueError(f"{field} must be non-negative")
    return value


def _hold_occupancy_ratio(notes: Chart, *, start_ms: int, end_ms: int) -> float:
    intervals: list[tuple[int, int]] = []
    for note in notes:
        if note.kind != "HOLD":
            continue
        hold_end_ms = note.time_ms + (note.duration_ms or 0)
        left = max(start_ms, note.time_ms)
        right = min(end_ms, hold_end_ms)
        if right > left:
            intervals.append((left, right))
    if not intervals:
        return 0.0
    intervals.sort()
    covered_ms = 0
    current_start, current_end = intervals[0]
    for left, right in intervals[1:]:
        if left <= current_end:
            current_end = max(current_end, right)
            continue
        covered_ms += current_end - current_start
        current_start, current_end = left, right
    covered_ms += current_end - current_start
    return round(covered_ms / (end_ms - start_ms), 6)


def _validate_onset_analysis(onset_analysis: OnsetAnalysis) -> None:
    if type(onset_analysis.sample_rate_hz) is not int or onset_analysis.sample_rate_hz <= 0:
        raise ValueError("onset sample_rate_hz must be a positive exact integer")
    if type(onset_analysis.hop_length) is not int or onset_analysis.hop_length <= 0:
        raise ValueError("onset hop_length must be a positive exact integer")
    strength = np.asarray(onset_analysis.strength)
    if strength.ndim != 1 or strength.size == 0:
        raise ValueError("onset strength must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(strength)):
        raise ValueError("onset strength must be finite")
    if np.any(strength < 0) or np.any(strength > 1):
        raise ValueError("onset strength must be normalized to [0, 1]")
    if tuple(sorted(set(onset_analysis.onset_ms))) != onset_analysis.onset_ms:
        raise ValueError("onset_ms must be sorted and unique")
    if any(type(time_ms) is not int or time_ms < 0 for time_ms in onset_analysis.onset_ms):
        raise ValueError("onset_ms values must be non-negative exact integers")


def _insufficient(
    *,
    start_ms: int,
    end_ms: int,
    beat_count: float | None,
    active_onset_count: int,
    hold_occupancy_ratio: float,
    active_frame_ratio: float,
    strong_attack_count: int = 0,
    strong_attack_threshold: float | None = None,
) -> CoverageOpportunity:
    return CoverageOpportunity(
        version=COVERAGE_OPPORTUNITY_VERSION,
        start_ms=start_ms,
        end_ms=end_ms,
        beat_count=beat_count,
        strong_attack_count=strong_attack_count,
        active_onset_count=active_onset_count,
        hold_occupancy_ratio=hold_occupancy_ratio,
        active_frame_ratio=active_frame_ratio,
        strong_attack_threshold=strong_attack_threshold,
        evidence_confidence="INSUFFICIENT",
        kind=CoverageKind.INSUFFICIENT_EVIDENCE,
    )


def classify_coverage_interval(
    notes: Chart,
    onset_analysis: OnsetAnalysis,
    tempo_map: LocalTempoMap | None,
    *,
    start_ms: int,
    end_ms: int,
    difficulty: str,
) -> CoverageOpportunity:
    """Classify one note-start gap without treating HOLD occupancy as an attack.

    Thresholds are deliberately song-relative and versioned.  They are calibration
    policy, not a claim that spectral flux is musical ground truth.
    """

    start_ms = _exact_non_negative_int(start_ms, "start_ms")
    end_ms = _exact_non_negative_int(end_ms, "end_ms")
    if end_ms <= start_ms:
        raise ValueError("end_ms must be after start_ms")
    if type(difficulty) is not str or difficulty not in DIFFICULTIES:
        raise ValueError(f"unsupported difficulty: {difficulty!r}")
    _validate_onset_analysis(onset_analysis)

    hold_occupancy_ratio = _hold_occupancy_ratio(
        notes,
        start_ms=start_ms,
        end_ms=end_ms,
    )
    activity = onset_analysis.activity
    active_frame_ratio = (
        0.0 if activity is None else activity.active_frame_ratio(start_ms, end_ms)
    )
    beat_count = (
        None
        if tempo_map is None
        else round(tempo_map.beats_between(start_ms, end_ms), 6)
    )
    if beat_count is not None and (not math.isfinite(beat_count) or beat_count < 0):
        raise ValueError("integrated beat count must be finite and non-negative")
    if activity is None or tempo_map is None:
        return _insufficient(
            start_ms=start_ms,
            end_ms=end_ms,
            beat_count=beat_count,
            active_onset_count=0,
            hold_occupancy_ratio=hold_occupancy_ratio,
            active_frame_ratio=active_frame_ratio,
        )

    active_times = activity.active_onset_ms
    if tuple(sorted(set(active_times))) != active_times or any(
        type(time_ms) is not int or time_ms < 0 for time_ms in active_times
    ):
        raise ValueError("active onset times must be sorted non-negative exact integers")
    onset_times = set(onset_analysis.onset_ms)
    if any(time_ms not in onset_times for time_ms in active_times):
        raise ValueError("active onset times must be a subset of detected onset times")
    if not active_times:
        return _insufficient(
            start_ms=start_ms,
            end_ms=end_ms,
            beat_count=beat_count,
            active_onset_count=0,
            hold_occupancy_ratio=hold_occupancy_ratio,
            active_frame_ratio=active_frame_ratio,
        )

    all_active_strengths = np.asarray(
        [onset_analysis.strength_at(time_ms) for time_ms in active_times],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(all_active_strengths)):
        raise ValueError("active onset strengths must be finite")
    global_max = float(np.max(all_active_strengths))
    if global_max <= 0:
        return _insufficient(
            start_ms=start_ms,
            end_ms=end_ms,
            beat_count=beat_count,
            active_onset_count=0,
            hold_occupancy_ratio=hold_occupancy_ratio,
            active_frame_ratio=active_frame_ratio,
        )
    threshold = max(
        global_max * RELATIVE_ATTACK_FLOOR,
        float(np.percentile(all_active_strengths, STRONG_ATTACK_QUANTILE)),
    )
    interval_times = tuple(time_ms for time_ms in active_times if start_ms < time_ms < end_ms)
    interval_strengths = tuple(onset_analysis.strength_at(time_ms) for time_ms in interval_times)
    strong_attack_count = sum(value >= threshold for value in interval_strengths)

    if (
        end_ms - start_ms < MIN_PHRASE_DURATION_MS
        or beat_count + PHRASE_BEAT_QUANTIZATION_TOLERANCE < MIN_PHRASE_BEATS
    ):
        return _insufficient(
            start_ms=start_ms,
            end_ms=end_ms,
            beat_count=beat_count,
            active_onset_count=len(interval_times),
            hold_occupancy_ratio=hold_occupancy_ratio,
            active_frame_ratio=active_frame_ratio,
            strong_attack_count=strong_attack_count,
            strong_attack_threshold=round(threshold, 6),
        )

    if strong_attack_count >= MIN_STRONG_ATTACKS[difficulty]:
        kind = CoverageKind.ATTACK_REQUIRED
    elif (
        hold_occupancy_ratio >= MIN_SUSTAIN_HOLD_OCCUPANCY
        and active_frame_ratio >= MIN_ACTIVE_FRAME_RATIO
    ):
        kind = CoverageKind.SUSTAIN_REPRESENTABLE
    else:
        return _insufficient(
            start_ms=start_ms,
            end_ms=end_ms,
            beat_count=beat_count,
            active_onset_count=len(interval_times),
            hold_occupancy_ratio=hold_occupancy_ratio,
            active_frame_ratio=active_frame_ratio,
            strong_attack_count=strong_attack_count,
            strong_attack_threshold=round(threshold, 6),
        )
    return CoverageOpportunity(
        version=COVERAGE_OPPORTUNITY_VERSION,
        start_ms=start_ms,
        end_ms=end_ms,
        beat_count=beat_count,
        strong_attack_count=strong_attack_count,
        active_onset_count=len(interval_times),
        hold_occupancy_ratio=hold_occupancy_ratio,
        active_frame_ratio=round(active_frame_ratio, 6),
        strong_attack_threshold=round(threshold, 6),
        evidence_confidence="SUFFICIENT",
        kind=kind,
    )
