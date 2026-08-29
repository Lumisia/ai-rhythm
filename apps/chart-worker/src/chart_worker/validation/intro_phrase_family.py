"""Cross-difficulty review of the first two unique note rows.

This module deliberately measures a narrow family inconsistency: HARD and
EXPERT start together, HARD continues, and EXPERT alone leaves an abnormally
long gap after its first row.  It does not mutate charts or decide how a
candidate should be regenerated.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

from chart_worker.validation.intro_region_contract import IntroRegionCandidateReview

NEAR_START_MAX_MS = 1_000
NEAR_START_MAX_BEATS = 1.0
HARD_IMMEDIATE_MAX_MS = 1_500
HARD_IMMEDIATE_MAX_BEATS = 2.0
EXPERT_LONG_GAP_MIN_MS = 10_000
EXPERT_LONG_GAP_MIN_BEATS = 16.0
GAP_RATIO_MIN = 8.0

IntroPhraseStatus = Literal["PASS", "REVIEW", "DEFECT", "INSUFFICIENT"]
IntroPhraseReason = Literal[
    "CONSISTENT",
    "ISOLATED_EXPERT_FIRST_ROW",
    "EXPERT_LATE_START",
    "EXPERT_OUTSIDE_CONFIRMED_INTRO_REGION",
    "EXPERT_EARLY_GHOST",
    "SHARED_LONG_SILENCE_POSSIBLE_NORMAL",
    "INSUFFICIENT_ROWS",
]


@dataclass(frozen=True, slots=True)
class IntroPhraseChartView:
    key_mode: int
    difficulty: str
    first_row_ms: int | None
    second_row_ms: int | None
    post_first_gap_beats: float | None
    first_row_audio_supported: bool | None = None
    candidate_id: str | None = None
    seed: int | None = None
    attempt: int | None = None

    def __post_init__(self) -> None:
        if self.first_row_ms is not None and self.first_row_ms < 0:
            raise ValueError("first_row_ms must be non-negative")
        if self.second_row_ms is not None and (
            self.first_row_ms is None or self.second_row_ms < self.first_row_ms
        ):
            raise ValueError("second_row_ms must follow first_row_ms")
        if self.post_first_gap_beats is not None and self.post_first_gap_beats < 0:
            raise ValueError("post_first_gap_beats must be non-negative")

    @property
    def post_first_gap_ms(self) -> int | None:
        if self.first_row_ms is None or self.second_row_ms is None:
            return None
        return self.second_row_ms - self.first_row_ms

    def to_report(self) -> dict[str, object]:
        return {
            "keyMode": self.key_mode,
            "difficulty": self.difficulty,
            "firstRowMs": self.first_row_ms,
            "secondRowMs": self.second_row_ms,
            "postFirstGapMs": self.post_first_gap_ms,
            "postFirstGapBeats": self.post_first_gap_beats,
            "firstRowAudioSupported": self.first_row_audio_supported,
            "candidateId": self.candidate_id,
            "seed": self.seed,
            "attempt": self.attempt,
        }


@dataclass(frozen=True, slots=True)
class IntroPhraseFamilyReview:
    status: IntroPhraseStatus
    reason: IntroPhraseReason
    hard: IntroPhraseChartView
    expert: IntroPhraseChartView
    start_delta_ms: int | None
    start_delta_beats: float | None
    gap_delta_ms: int | None
    gap_ratio: float | None

    @property
    def should_recover(self) -> bool:
        return self.status == "DEFECT"

    @property
    def should_block_publication(self) -> bool:
        return self.status == "DEFECT"

    def to_report(self) -> dict[str, object]:
        return {
            "version": "intro-phrase-family-v1",
            "mode": "ACTIVE_FOR_HIGH_CONFIDENCE_DEFECT",
            "policyState": "PROVISIONAL",
            "status": self.status,
            "reason": self.reason,
            "shouldRecover": self.should_recover,
            "shouldBlockPublication": self.should_block_publication,
            "startDeltaMs": self.start_delta_ms,
            "startDeltaBeats": self.start_delta_beats,
            "gapDeltaMs": self.gap_delta_ms,
            "gapRatio": self.gap_ratio,
            "thresholds": {
                "nearStartMaxMs": NEAR_START_MAX_MS,
                "nearStartMaxBeats": NEAR_START_MAX_BEATS,
                "hardImmediateMaxMs": HARD_IMMEDIATE_MAX_MS,
                "hardImmediateMaxBeats": HARD_IMMEDIATE_MAX_BEATS,
                "expertLongGapMinMs": EXPERT_LONG_GAP_MIN_MS,
                "expertLongGapMinBeats": EXPERT_LONG_GAP_MIN_BEATS,
                "gapRatioMin": GAP_RATIO_MIN,
            },
            "hard": self.hard.to_report(),
            "expert": self.expert.to_report(),
        }


def review_intro_phrase_pair(
    hard: IntroPhraseChartView,
    expert: IntroPhraseChartView,
    *,
    start_delta_beats: float | None,
) -> IntroPhraseFamilyReview:
    """Classify one same-key HARD/EXPERT pair without mutating either chart."""
    if hard.key_mode != expert.key_mode:
        raise ValueError("HARD and EXPERT must use the same key mode")
    if hard.difficulty != "HARD" or expert.difficulty != "EXPERT":
        raise ValueError("views must be ordered as HARD then EXPERT")
    if start_delta_beats is not None and start_delta_beats < 0:
        raise ValueError("start_delta_beats must be non-negative")

    hard_gap_ms = hard.post_first_gap_ms
    expert_gap_ms = expert.post_first_gap_ms
    complete = (
        hard.first_row_ms is not None
        and expert.first_row_ms is not None
        and hard_gap_ms is not None
        and expert_gap_ms is not None
        and hard.post_first_gap_beats is not None
        and expert.post_first_gap_beats is not None
        and start_delta_beats is not None
    )
    if not complete:
        return IntroPhraseFamilyReview(
            "INSUFFICIENT",
            "INSUFFICIENT_ROWS",
            hard,
            expert,
            None,
            start_delta_beats,
            None,
            None,
        )

    # Narrowed above; aliases keep the policy expression readable.
    assert hard.first_row_ms is not None
    assert expert.first_row_ms is not None
    assert hard_gap_ms is not None
    assert expert_gap_ms is not None
    assert hard.post_first_gap_beats is not None
    assert expert.post_first_gap_beats is not None
    assert start_delta_beats is not None

    signed_start_delta_ms = expert.first_row_ms - hard.first_row_ms
    start_delta_ms = abs(signed_start_delta_ms)
    gap_delta_ms = expert_gap_ms - hard_gap_ms
    gap_ratio = round(expert_gap_ms / max(1, hard_gap_ms), 6)

    starts_near = (
        start_delta_ms <= NEAR_START_MAX_MS
        and start_delta_beats <= NEAR_START_MAX_BEATS
    )
    hard_immediate = (
        hard_gap_ms <= HARD_IMMEDIATE_MAX_MS
        and hard.post_first_gap_beats <= HARD_IMMEDIATE_MAX_BEATS
    )
    expert_long = (
        expert_gap_ms >= EXPERT_LONG_GAP_MIN_MS
        and expert.post_first_gap_beats >= EXPERT_LONG_GAP_MIN_BEATS
    )
    hard_long = (
        hard_gap_ms >= EXPERT_LONG_GAP_MIN_MS
        and hard.post_first_gap_beats >= EXPERT_LONG_GAP_MIN_BEATS
    )

    if (
        starts_near
        and hard_immediate
        and expert_long
        and gap_ratio >= GAP_RATIO_MIN
    ):
        status: IntroPhraseStatus = "DEFECT"
        reason: IntroPhraseReason = "ISOLATED_EXPERT_FIRST_ROW"
    elif starts_near and hard_long and expert_long:
        status = "REVIEW"
        reason = "SHARED_LONG_SILENCE_POSSIBLE_NORMAL"
    elif not starts_near and signed_start_delta_ms > 0:
        status = "REVIEW"
        reason = "EXPERT_LATE_START"
    elif not starts_near and signed_start_delta_ms < 0:
        status = "REVIEW"
        reason = "EXPERT_EARLY_GHOST"
    else:
        status = "PASS"
        reason = "CONSISTENT"

    return IntroPhraseFamilyReview(
        status,
        reason,
        hard,
        expert,
        start_delta_ms,
        round(start_delta_beats, 6),
        gap_delta_ms,
        gap_ratio,
    )


def corroborate_intro_phrase_review(
    review: IntroPhraseFamilyReview,
    *,
    hard_region: IntroRegionCandidateReview,
    expert_region: IntroRegionCandidateReview,
) -> IntroPhraseFamilyReview:
    """Promote only an audio-confirmed relative late start to a defect.

    The HARD chart is a control, not the source of truth: it must independently
    pass the same song-level intro-region contract.  Unknown or conflicting
    evidence preserves the provisional REVIEW result.
    """

    if type(review) is not IntroPhraseFamilyReview:
        raise TypeError("review must be an IntroPhraseFamilyReview")
    if type(hard_region) is not IntroRegionCandidateReview:
        raise TypeError("hard_region must be an IntroRegionCandidateReview")
    if type(expert_region) is not IntroRegionCandidateReview:
        raise TypeError("expert_region must be an IntroRegionCandidateReview")
    if (
        review.reason == "EXPERT_LATE_START"
        and hard_region.status == "PASS"
        and expert_region.status == "DEFECT"
        and expert_region.reason == "CONFIRMED_INTRO_REGION_MISSED"
    ):
        return replace(
            review,
            status="DEFECT",
            reason="EXPERT_OUTSIDE_CONFIRMED_INTRO_REGION",
        )
    return review
