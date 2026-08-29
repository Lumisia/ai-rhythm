"""Cross-evidence classification of attacks and sustained note coverage."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from typing import Literal

import numpy as np

from chart_worker.analysis.gameplay_occupancy import hold_occupancy_ms
from chart_worker.analysis.onset import OnsetAnalysis
from chart_worker.analysis.song_context import LocalTempoMap
from chart_worker.schema.note import Chart
from chart_worker.schema.types import DIFFICULTIES

COVERAGE_OPPORTUNITY_VERSION = "coverage-opportunity-v4"
MIN_PHRASE_DURATION_MS = 4_000
MIN_SUSTAIN_HOLD_OCCUPANCY = 0.80
MIN_ACTIVE_FRAME_RATIO = 0.35
MAX_REST_ACTIVE_FRAME_RATIO = MIN_ACTIVE_FRAME_RATIO / 2
RELATIVE_ATTACK_FLOOR = 0.35
STRONG_ATTACK_QUANTILE = 75.0
LOCAL_CONTEXT_MIN_MS = 2_000
LOCAL_CONTEXT_MAX_MS = 8_000
LOCAL_RELATIVE_ATTACK_FLOOR = 0.25
LOCAL_ATTACK_QUANTILE = 50.0
LOCAL_CORROBORATED_ACTIVE_FRAME_RATIO = 0.50
LOCAL_CORROBORATED_NEIGHBORING_ACTIVITY_RATIO = 0.50
LOCAL_CORROBORATED_STRONG_ATTACK_MIN = 2
LOCAL_CORROBORATED_MAX_HOLD_OCCUPANCY_RATIO = 0.20
MIN_STRONG_ATTACKS = {
    "EASY": 8,
    "NORMAL": 6,
    "HARD": 4,
    "EXPERT": 4,
}


class CoverageKind(StrEnum):
    ATTACK_REQUIRED = "ATTACK_REQUIRED"
    SUSTAIN_COVERED = "SUSTAIN_COVERED"
    MUSICAL_REST_OR_SIMPLIFICATION = "MUSICAL_REST_OR_SIMPLIFICATION"
    UNCERTAIN = "UNCERTAIN"

    # Source-compatible aliases. New reports emit the canonical v4 values.
    SUSTAIN_REPRESENTABLE = "SUSTAIN_COVERED"
    INSUFFICIENT_EVIDENCE = "UNCERTAIN"


@dataclass(frozen=True, slots=True)
class CoverageOpportunity:
    version: Literal["coverage-opportunity-v4"]
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
    local_strong_attack_count: int = 0
    local_strong_attack_threshold: float | None = None
    neighboring_activity_ratio: float | None = None
    attack_evidence_scope: Literal["GLOBAL", "LOCAL_CORROBORATED", "NONE"] = "NONE"

    @property
    def actionable(self) -> bool:
        return self.kind is CoverageKind.ATTACK_REQUIRED

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
            "localStrongAttackCount": self.local_strong_attack_count,
            "localStrongAttackThreshold": self.local_strong_attack_threshold,
            "neighboringActivityRatio": self.neighboring_activity_ratio,
            "attackEvidenceScope": self.attack_evidence_scope,
            "evidenceConfidence": self.evidence_confidence,
            "kind": self.kind.value,
            "actionable": self.actionable,
        }


@dataclass(frozen=True, slots=True)
class AttackEvidence:
    active_onset_count: int
    global_strong_attack_count: int
    local_strong_attack_count: int
    global_threshold: float | None
    local_threshold: float | None
    neighboring_activity_ratio: float | None


def _exact_non_negative_int(value: object, field: str) -> int:
    if type(value) is not int:  # bool and int subclasses are not protocol integers.
        raise TypeError(f"{field} must be an exact integer")
    if value < 0:
        raise ValueError(f"{field} must be non-negative")
    return value


def _hold_occupancy_ratio(notes: Chart, *, start_ms: int, end_ms: int) -> float:
    return round(
        hold_occupancy_ms(notes, start_ms=start_ms, end_ms=end_ms)
        / (end_ms - start_ms),
        6,
    )


def validate_onset_analysis(onset_analysis: OnsetAnalysis) -> None:
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


def _attack_threshold(
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


def measure_attack_evidence(
    onset_analysis: OnsetAnalysis,
    *,
    start_ms: int,
    end_ms: int,
) -> AttackEvidence:
    """Measure song-global and phrase-local attacks without deciding policy."""
    start_ms = _exact_non_negative_int(start_ms, "start_ms")
    end_ms = _exact_non_negative_int(end_ms, "end_ms")
    if end_ms <= start_ms:
        raise ValueError("end_ms must be after start_ms")
    validate_onset_analysis(onset_analysis)
    return _measure_attack_evidence_validated(
        onset_analysis,
        start_ms=start_ms,
        end_ms=end_ms,
    )


def _measure_attack_evidence_validated(
    onset_analysis: OnsetAnalysis,
    *,
    start_ms: int,
    end_ms: int,
) -> AttackEvidence:
    """Measure evidence after the caller has validated bounds and analysis."""

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
    global_threshold = _attack_threshold(
        tuple(onset_analysis.strength_at(time_ms) for time_ms in active_times),
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
    local_threshold = _attack_threshold(
        tuple(onset_analysis.strength_at(time_ms) for time_ms in local_context_times),
        relative_floor=LOCAL_RELATIVE_ATTACK_FLOOR,
        quantile=LOCAL_ATTACK_QUANTILE,
    )

    neighboring_activity_ratio: float | None = None
    if activity is not None:
        available_end_ms = round(max(0, onset_analysis.frame_count - 1) * onset_analysis.frame_ms)
        neighbor_ratios: list[float] = []
        if start_ms > 0:
            neighbor_ratios.append(
                activity.active_frame_ratio(max(0, start_ms - duration_ms), start_ms)
            )
        if end_ms < available_end_ms:
            neighbor_ratios.append(
                activity.active_frame_ratio(
                    end_ms,
                    min(available_end_ms, end_ms + duration_ms),
                )
            )
        if neighbor_ratios:
            neighboring_activity_ratio = round(float(np.mean(neighbor_ratios)), 6)

    return AttackEvidence(
        active_onset_count=len(interval_times),
        global_strong_attack_count=(
            0
            if global_threshold is None
            else sum(value >= global_threshold for value in interval_strengths)
        ),
        local_strong_attack_count=(
            0
            if local_threshold is None
            else sum(value >= local_threshold for value in interval_strengths)
        ),
        global_threshold=(
            None if global_threshold is None else round(global_threshold, 6)
        ),
        local_threshold=None if local_threshold is None else round(local_threshold, 6),
        neighboring_activity_ratio=neighboring_activity_ratio,
    )


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
    local_strong_attack_count: int = 0,
    local_strong_attack_threshold: float | None = None,
    neighboring_activity_ratio: float | None = None,
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
        kind=CoverageKind.UNCERTAIN,
        local_strong_attack_count=local_strong_attack_count,
        local_strong_attack_threshold=local_strong_attack_threshold,
        neighboring_activity_ratio=neighboring_activity_ratio,
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
    validate_onset_analysis(onset_analysis)

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
        if (
            hold_occupancy_ratio >= MIN_SUSTAIN_HOLD_OCCUPANCY
            and active_frame_ratio >= MIN_ACTIVE_FRAME_RATIO
        ):
            kind = CoverageKind.SUSTAIN_COVERED
            evidence_confidence: Literal["SUFFICIENT", "INSUFFICIENT"] = "SUFFICIENT"
        elif active_frame_ratio <= MAX_REST_ACTIVE_FRAME_RATIO:
            kind = CoverageKind.MUSICAL_REST_OR_SIMPLIFICATION
            evidence_confidence = "SUFFICIENT"
        else:
            kind = CoverageKind.UNCERTAIN
            evidence_confidence = "INSUFFICIENT"
        return CoverageOpportunity(
            version=COVERAGE_OPPORTUNITY_VERSION,
            start_ms=start_ms,
            end_ms=end_ms,
            beat_count=beat_count,
            strong_attack_count=0,
            active_onset_count=0,
            hold_occupancy_ratio=hold_occupancy_ratio,
            active_frame_ratio=round(active_frame_ratio, 6),
            strong_attack_threshold=None,
            evidence_confidence=evidence_confidence,
            kind=kind,
        )

    attack_evidence = _measure_attack_evidence_validated(
        onset_analysis,
        start_ms=start_ms,
        end_ms=end_ms,
    )
    if attack_evidence.global_threshold is None:
        return _insufficient(
            start_ms=start_ms,
            end_ms=end_ms,
            beat_count=beat_count,
            active_onset_count=attack_evidence.active_onset_count,
            hold_occupancy_ratio=hold_occupancy_ratio,
            active_frame_ratio=active_frame_ratio,
            local_strong_attack_count=attack_evidence.local_strong_attack_count,
            local_strong_attack_threshold=attack_evidence.local_threshold,
            neighboring_activity_ratio=attack_evidence.neighboring_activity_ratio,
        )
    strong_attack_count = attack_evidence.global_strong_attack_count

    if active_frame_ratio <= MAX_REST_ACTIVE_FRAME_RATIO and strong_attack_count == 0:
        return CoverageOpportunity(
            version=COVERAGE_OPPORTUNITY_VERSION,
            start_ms=start_ms,
            end_ms=end_ms,
            beat_count=beat_count,
            strong_attack_count=0,
            active_onset_count=attack_evidence.active_onset_count,
            hold_occupancy_ratio=hold_occupancy_ratio,
            active_frame_ratio=round(active_frame_ratio, 6),
            strong_attack_threshold=attack_evidence.global_threshold,
            evidence_confidence="SUFFICIENT",
            kind=CoverageKind.MUSICAL_REST_OR_SIMPLIFICATION,
            local_strong_attack_count=attack_evidence.local_strong_attack_count,
            local_strong_attack_threshold=attack_evidence.local_threshold,
            neighboring_activity_ratio=attack_evidence.neighboring_activity_ratio,
        )

    if end_ms - start_ms < MIN_PHRASE_DURATION_MS:
        return _insufficient(
            start_ms=start_ms,
            end_ms=end_ms,
            beat_count=beat_count,
            active_onset_count=attack_evidence.active_onset_count,
            hold_occupancy_ratio=hold_occupancy_ratio,
            active_frame_ratio=active_frame_ratio,
            strong_attack_count=strong_attack_count,
            strong_attack_threshold=attack_evidence.global_threshold,
            local_strong_attack_count=attack_evidence.local_strong_attack_count,
            local_strong_attack_threshold=attack_evidence.local_threshold,
            neighboring_activity_ratio=attack_evidence.neighboring_activity_ratio,
        )

    # Tempo integration is retained as diagnostic evidence, but it must not
    # veto two independent observations from the audio itself: repeated
    # spectral attacks and sustained RMS activity.  A half/double-tempo switch
    # or a pathological local BPM otherwise turns audible phrases into false
    # "insufficient evidence" gaps.
    required_active_onsets = MIN_STRONG_ATTACKS[difficulty]
    required_local_attacks = max(
        LOCAL_CORROBORATED_STRONG_ATTACK_MIN,
        required_active_onsets - 1,
    )
    globally_corroborated = (
        strong_attack_count >= MIN_STRONG_ATTACKS[difficulty]
        and active_frame_ratio >= MIN_ACTIVE_FRAME_RATIO
    )
    locally_corroborated = (
        active_frame_ratio >= LOCAL_CORROBORATED_ACTIVE_FRAME_RATIO
        and attack_evidence.neighboring_activity_ratio is not None
        and attack_evidence.neighboring_activity_ratio
        >= LOCAL_CORROBORATED_NEIGHBORING_ACTIVITY_RATIO
        and attack_evidence.active_onset_count >= required_active_onsets
        and attack_evidence.local_strong_attack_count >= required_local_attacks
        and hold_occupancy_ratio <= LOCAL_CORROBORATED_MAX_HOLD_OCCUPANCY_RATIO
    )
    if globally_corroborated or locally_corroborated:
        kind = CoverageKind.ATTACK_REQUIRED
        attack_evidence_scope: Literal["GLOBAL", "LOCAL_CORROBORATED", "NONE"] = (
            "GLOBAL" if globally_corroborated else "LOCAL_CORROBORATED"
        )
    elif (
        hold_occupancy_ratio >= MIN_SUSTAIN_HOLD_OCCUPANCY
        and active_frame_ratio >= MIN_ACTIVE_FRAME_RATIO
    ):
        kind = CoverageKind.SUSTAIN_COVERED
        attack_evidence_scope = "NONE"
    else:
        return _insufficient(
            start_ms=start_ms,
            end_ms=end_ms,
            beat_count=beat_count,
            active_onset_count=attack_evidence.active_onset_count,
            hold_occupancy_ratio=hold_occupancy_ratio,
            active_frame_ratio=active_frame_ratio,
            strong_attack_count=strong_attack_count,
            strong_attack_threshold=attack_evidence.global_threshold,
            local_strong_attack_count=attack_evidence.local_strong_attack_count,
            local_strong_attack_threshold=attack_evidence.local_threshold,
            neighboring_activity_ratio=attack_evidence.neighboring_activity_ratio,
        )
    return CoverageOpportunity(
        version=COVERAGE_OPPORTUNITY_VERSION,
        start_ms=start_ms,
        end_ms=end_ms,
        beat_count=beat_count,
        strong_attack_count=strong_attack_count,
        active_onset_count=attack_evidence.active_onset_count,
        hold_occupancy_ratio=hold_occupancy_ratio,
        active_frame_ratio=round(active_frame_ratio, 6),
        strong_attack_threshold=attack_evidence.global_threshold,
        evidence_confidence="SUFFICIENT",
        kind=kind,
        local_strong_attack_count=attack_evidence.local_strong_attack_count,
        local_strong_attack_threshold=attack_evidence.local_threshold,
        neighboring_activity_ratio=attack_evidence.neighboring_activity_ratio,
        attack_evidence_scope=attack_evidence_scope,
    )
