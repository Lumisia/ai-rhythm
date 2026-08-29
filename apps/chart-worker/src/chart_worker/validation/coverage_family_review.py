"""Correlation-aware routing for active note-coverage gaps."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum

from chart_worker.analysis.coverage_jury import LocalAudioGapEvidence
from chart_worker.analysis.coverage_opportunity import (
    LOCAL_CORROBORATED_ACTIVE_FRAME_RATIO,
    LOCAL_CORROBORATED_MAX_HOLD_OCCUPANCY_RATIO,
    LOCAL_CORROBORATED_NEIGHBORING_ACTIVITY_RATIO,
    LOCAL_CORROBORATED_STRONG_ATTACK_MIN,
    MIN_PHRASE_DURATION_MS,
    MIN_STRONG_ATTACKS,
    CoverageKind,
)
from chart_worker.schema.types import DIFFICULTIES, KEY_MODES
from chart_worker.validation.timing_integrity import TimingIntegrityStatus

COVERAGE_FAMILY_REVIEW_VERSION = "coverage-family-review-v3"
OVERLAP_RATIO_MIN = 0.5
ACTIVE_FRAME_RATIO_MIN = LOCAL_CORROBORATED_ACTIVE_FRAME_RATIO
NEIGHBORING_ACTIVITY_RATIO_MIN = LOCAL_CORROBORATED_NEIGHBORING_ACTIVITY_RATIO
LOCAL_STRONG_ATTACK_MIN = LOCAL_CORROBORATED_STRONG_ATTACK_MIN
MAX_HOLD_OCCUPANCY_RATIO = LOCAL_CORROBORATED_MAX_HOLD_OCCUPANCY_RATIO


class CoverageFamilyVerdict(StrEnum):
    TIMING_AUTHORITY_SUSPECT = "TIMING_AUTHORITY_SUSPECT"
    SHARED_MODEL_OMISSION = "SHARED_MODEL_OMISSION"
    CHART_SPECIFIC_OMISSION = "CHART_SPECIFIC_OMISSION"
    MUSICAL_BREAK = "MUSICAL_BREAK"
    AMBIGUOUS = "AMBIGUOUS"


@dataclass(frozen=True, slots=True)
class CoverageGapMember:
    key_mode: int
    difficulty: str
    start_ms: int
    end_ms: int
    model_backed: bool
    hold_occupancy_ratio: float
    opportunity_kind: CoverageKind

    def __post_init__(self) -> None:
        if type(self.key_mode) is not int or self.key_mode not in KEY_MODES:
            raise ValueError("key_mode must be one of the supported exact integers")
        if self.difficulty not in DIFFICULTIES:
            raise ValueError("difficulty is unsupported")
        if type(self.start_ms) is not int or type(self.end_ms) is not int:
            raise TypeError("gap bounds must be exact integers")
        if self.start_ms < 0 or self.end_ms <= self.start_ms:
            raise ValueError("gap bounds must define a positive interval")
        if type(self.model_backed) is not bool:
            raise TypeError("model_backed must be an exact boolean")
        if not isinstance(self.opportunity_kind, CoverageKind):
            raise TypeError("opportunity_kind must be a CoverageKind")
        if (
            isinstance(self.hold_occupancy_ratio, bool)
            or not isinstance(self.hold_occupancy_ratio, (int, float))
            or not math.isfinite(float(self.hold_occupancy_ratio))
            or not 0.0 <= float(self.hold_occupancy_ratio) <= 1.0
        ):
            raise ValueError("hold_occupancy_ratio must be finite and within [0, 1]")


@dataclass(frozen=True, slots=True)
class CoverageFamilyReview:
    verdict: CoverageFamilyVerdict
    reasons: tuple[str, ...]
    independent_key_family_count: int
    overlapping_model_chart_count: int
    overlapping_fallback_chart_count: int
    same_key_model_sibling_count: int

    def to_report(self) -> dict[str, object]:
        return {
            "version": COVERAGE_FAMILY_REVIEW_VERSION,
            "verdict": self.verdict.value,
            "reasons": list(self.reasons),
            "independentKeyFamilyCount": self.independent_key_family_count,
            "overlappingModelChartCount": self.overlapping_model_chart_count,
            "overlappingFallbackChartCount": self.overlapping_fallback_chart_count,
            "sameKeyModelSiblingCount": self.same_key_model_sibling_count,
        }


def _substantially_overlaps(
    target: CoverageGapMember,
    candidate: CoverageGapMember,
) -> bool:
    intersection_ms = min(target.end_ms, candidate.end_ms) - max(
        target.start_ms,
        candidate.start_ms,
    )
    if intersection_ms <= 0:
        return False
    shorter_ms = min(
        target.end_ms - target.start_ms,
        candidate.end_ms - candidate.start_ms,
    )
    return intersection_ms / shorter_ms >= OVERLAP_RATIO_MIN


def review_coverage_family(
    target: CoverageGapMember,
    siblings: tuple[CoverageGapMember, ...],
    audio: LocalAudioGapEvidence,
    *,
    timing_status: TimingIntegrityStatus,
) -> CoverageFamilyReview:
    """Choose a repair authority without treating 11 siblings as 11 independent votes."""
    if (audio.start_ms, audio.end_ms) != (target.start_ms, target.end_ms):
        raise ValueError("local audio evidence must match the target interval")
    if not isinstance(timing_status, TimingIntegrityStatus):
        raise TypeError("timing_status must be a TimingIntegrityStatus")

    overlapping = tuple(sibling for sibling in siblings if _substantially_overlaps(target, sibling))
    model_members = tuple(member for member in (target, *overlapping) if member.model_backed)
    fallback_members = tuple(member for member in overlapping if not member.model_backed)
    key_families = {member.key_mode for member in model_members}
    same_key_siblings = sum(
        member.model_backed and member.key_mode == target.key_mode for member in overlapping
    )

    quiet = (
        audio.active_frame_ratio is not None
        and audio.active_frame_ratio <= 0.2
        and audio.active_onset_count == 0
        and audio.local_strong_attack_count == 0
    )
    required_active_onsets = MIN_STRONG_ATTACKS[target.difficulty]
    required_local_attacks = max(
        LOCAL_STRONG_ATTACK_MIN,
        required_active_onsets - 1,
    )
    active = (
        target.end_ms - target.start_ms >= MIN_PHRASE_DURATION_MS
        and audio.active_frame_ratio is not None
        and audio.active_frame_ratio >= ACTIVE_FRAME_RATIO_MIN
        and audio.neighboring_activity_ratio is not None
        and audio.neighboring_activity_ratio >= NEIGHBORING_ACTIVITY_RATIO_MIN
        and audio.active_onset_count >= required_active_onsets
        and audio.local_strong_attack_count >= required_local_attacks
        and target.hold_occupancy_ratio <= MAX_HOLD_OCCUPANCY_RATIO
    )
    broadly_shared = len(key_families) >= 2 and len(model_members) >= 3

    if target.opportunity_kind is not CoverageKind.ATTACK_REQUIRED:
        verdict = CoverageFamilyVerdict.AMBIGUOUS
        reasons = ("SOURCE_COVERAGE_OPPORTUNITY_IS_REPORT_ONLY",)
    elif quiet:
        verdict = CoverageFamilyVerdict.MUSICAL_BREAK
        reasons = ("QUIET_INTERVAL_WITHOUT_LOCAL_ATTACKS",)
    elif not active:
        verdict = CoverageFamilyVerdict.AMBIGUOUS
        reasons = ("INSUFFICIENT_INDEPENDENT_AUDIO_EVIDENCE",)
    elif timing_status is not TimingIntegrityStatus.HEALTHY:
        verdict = CoverageFamilyVerdict.TIMING_AUTHORITY_SUSPECT
        reasons = ("TIMING_INTEGRITY_REQUIRES_CORROBORATION",)
        if broadly_shared:
            reasons = ("ACTIVE_GAP_SHARED_ACROSS_KEY_FAMILIES", *reasons)
    elif broadly_shared:
        verdict = CoverageFamilyVerdict.SHARED_MODEL_OMISSION
        reasons = ("ACTIVE_GAP_SHARED_ACROSS_KEY_FAMILIES",)
    elif len(model_members) == 1 and not fallback_members:
        verdict = CoverageFamilyVerdict.CHART_SPECIFIC_OMISSION
        reasons = ("ACTIVE_GAP_ISOLATED_TO_TARGET_CHART",)
    else:
        verdict = CoverageFamilyVerdict.AMBIGUOUS
        reasons = ("CORRELATED_FAMILY_EVIDENCE_IS_INCONCLUSIVE",)

    return CoverageFamilyReview(
        verdict=verdict,
        reasons=reasons,
        independent_key_family_count=len(key_families),
        overlapping_model_chart_count=len(model_members),
        overlapping_fallback_chart_count=len(fallback_members),
        same_key_model_sibling_count=same_key_siblings,
    )
