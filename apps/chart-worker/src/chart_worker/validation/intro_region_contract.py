"""Adaptive, non-mutating first-row interval derived from audio evidence."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Literal

from chart_worker.analysis.intro_anchor import GRID_SUPPORT_WINDOW_MS
from chart_worker.analysis.leading_silence import consensus_leading_boundary_ms
from chart_worker.analysis.song_context import SongAnalysisContext

IntroRegionStatus = Literal["CONFIRMED", "UNKNOWN"]
IntroRegionReviewStatus = Literal["PASS", "REVIEW", "DEFECT", "UNKNOWN"]
IntroRegionReviewReason = Literal[
    "WITHIN_CONFIRMED_INTRO_REGION",
    "CONFIRMED_INTRO_REGION_MISSED",
    "NOTE_INSIDE_CONFIRMED_LEADING_SILENCE",
    "BEFORE_CONFIRMED_INTRO_REGION",
    "INTRO_REGION_EVIDENCE_UNAVAILABLE",
    "FIRST_ROW_UNAVAILABLE",
]


@dataclass(frozen=True, slots=True)
class IntroRegionContract:
    version: Literal["intro-region-contract-v1"]
    status: IntroRegionStatus
    allowed_first_row_ms: tuple[int, int] | None
    leading_silence_end_ms: int | None
    anchor_ms: int | None
    anchor_grid_ms: int | None
    supported_pulse_ms: tuple[int, ...]
    quantization_tolerance_ms: int
    reasons: tuple[str, ...]

    def to_report(self) -> dict[str, object]:
        return {
            "version": self.version,
            "status": self.status,
            "allowedFirstRowMs": (
                list(self.allowed_first_row_ms)
                if self.allowed_first_row_ms is not None
                else None
            ),
            "leadingSilenceEndMs": self.leading_silence_end_ms,
            "anchorMs": self.anchor_ms,
            "anchorGridMs": self.anchor_grid_ms,
            "supportedPulseMs": list(self.supported_pulse_ms),
            "quantizationToleranceMs": self.quantization_tolerance_ms,
            "reasons": list(self.reasons),
            "mutatesChart": False,
        }

    def stable_sha256(self) -> str:
        payload = json.dumps(
            self.to_report(),
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True, slots=True)
class IntroRegionCandidateReview:
    status: IntroRegionReviewStatus
    reason: IntroRegionReviewReason
    first_row_ms: int | None
    allowed_first_row_ms: tuple[int, int] | None
    lateness_ms: int | None = None

    def to_report(self) -> dict[str, object]:
        return {
            "status": self.status,
            "reason": self.reason,
            "firstRowMs": self.first_row_ms,
            "allowedFirstRowMs": (
                list(self.allowed_first_row_ms)
                if self.allowed_first_row_ms is not None
                else None
            ),
            "latenessMs": self.lateness_ms,
        }


def _unknown(
    song_context: SongAnalysisContext,
    *,
    leading_silence_end_ms: int | None,
    reason: str,
) -> IntroRegionContract:
    anchor = song_context.intro_anchor
    return IntroRegionContract(
        version="intro-region-contract-v1",
        status="UNKNOWN",
        allowed_first_row_ms=None,
        leading_silence_end_ms=leading_silence_end_ms,
        anchor_ms=anchor.anchor_ms,
        anchor_grid_ms=anchor.anchor_grid_ms,
        supported_pulse_ms=anchor.supported_pulse_ms,
        quantization_tolerance_ms=GRID_SUPPORT_WINDOW_MS,
        reasons=(reason,),
    )


def build_intro_region_contract(
    song_context: SongAnalysisContext,
) -> IntroRegionContract:
    """Build a continuous allowed region; never choose one canonical row."""

    anchor = song_context.intro_anchor
    leading_observation = song_context.onset_analysis.leading_silence
    leading_silence_end_ms = (
        consensus_leading_boundary_ms(leading_observation)
        if leading_observation is not None
        else None
    )
    if (
        anchor.status != "CONFIRMED"
        or anchor.anchor_ms is None
        or anchor.anchor_grid_ms is None
        or not anchor.supported_pulse_ms
    ):
        return _unknown(
            song_context,
            leading_silence_end_ms=leading_silence_end_ms,
            reason="CONFIRMED_RHYTHMIC_SEQUENCE_UNAVAILABLE",
        )

    lower_ms = max(
        0,
        min(anchor.anchor_ms, anchor.anchor_grid_ms) - GRID_SUPPORT_WINDOW_MS,
    )
    if leading_silence_end_ms is not None:
        lower_ms = max(lower_ms, leading_silence_end_ms)
    upper_ms = min(
        song_context.duration_ms,
        max(anchor.supported_pulse_ms) + GRID_SUPPORT_WINDOW_MS,
    )
    if lower_ms >= upper_ms:
        return _unknown(
            song_context,
            leading_silence_end_ms=leading_silence_end_ms,
            reason="INTRO_EVIDENCE_INTERVAL_CONTRADICTION",
        )

    reasons = ["CONFIRMED_AUDIO_GRID_PULSE_SEQUENCE"]
    if leading_silence_end_ms is not None:
        reasons.append("CONFIRMED_LEADING_SILENCE_LOWER_BOUND")
    return IntroRegionContract(
        version="intro-region-contract-v1",
        status="CONFIRMED",
        allowed_first_row_ms=(lower_ms, upper_ms),
        leading_silence_end_ms=leading_silence_end_ms,
        anchor_ms=anchor.anchor_ms,
        anchor_grid_ms=anchor.anchor_grid_ms,
        supported_pulse_ms=anchor.supported_pulse_ms,
        quantization_tolerance_ms=GRID_SUPPORT_WINDOW_MS,
        reasons=tuple(reasons),
    )


def review_intro_region_candidate(
    contract: IntroRegionContract,
    *,
    first_row_ms: int | None,
) -> IntroRegionCandidateReview:
    if not isinstance(contract, IntroRegionContract):
        raise TypeError("contract must be an IntroRegionContract")
    if first_row_ms is not None and (
        type(first_row_ms) is not int or first_row_ms < 0
    ):
        raise ValueError("first_row_ms must be a non-negative exact integer or None")
    allowed = contract.allowed_first_row_ms
    if contract.status != "CONFIRMED" or allowed is None:
        return IntroRegionCandidateReview(
            "UNKNOWN",
            "INTRO_REGION_EVIDENCE_UNAVAILABLE",
            first_row_ms,
            allowed,
        )
    if first_row_ms is None:
        return IntroRegionCandidateReview(
            "DEFECT",
            "FIRST_ROW_UNAVAILABLE",
            None,
            allowed,
        )

    lower_ms, upper_ms = allowed
    if first_row_ms < lower_ms:
        if (
            contract.leading_silence_end_ms is not None
            and first_row_ms < contract.leading_silence_end_ms
        ):
            return IntroRegionCandidateReview(
                "DEFECT",
                "NOTE_INSIDE_CONFIRMED_LEADING_SILENCE",
                first_row_ms,
                allowed,
            )
        return IntroRegionCandidateReview(
            "REVIEW",
            "BEFORE_CONFIRMED_INTRO_REGION",
            first_row_ms,
            allowed,
        )
    if first_row_ms > upper_ms:
        return IntroRegionCandidateReview(
            "DEFECT",
            "CONFIRMED_INTRO_REGION_MISSED",
            first_row_ms,
            allowed,
            lateness_ms=first_row_ms - upper_ms,
        )
    return IntroRegionCandidateReview(
        "PASS",
        "WITHIN_CONFIRMED_INTRO_REGION",
        first_row_ms,
        allowed,
    )
