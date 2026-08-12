"""Shadow-only cross-key review for unexpectedly early final note starts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

OUTRO_SIBLING_AGREEMENT_MS = 1_000
OUTRO_EARLY_START_REVIEW_MS = 4_000
OUTRO_SINGLE_SIBLING_EARLY_START_REVIEW_MS = 6_000
OUTRO_KEY_MODES = (4, 6, 7)


@dataclass(frozen=True, slots=True)
class OutroChartView:
    key_mode: int
    difficulty: str
    last_note_start_ms: int
    last_note_end_ms: int

    def to_report(self) -> dict[str, object]:
        return {
            "keyMode": self.key_mode,
            "difficulty": self.difficulty,
            "lastNoteStartMs": self.last_note_start_ms,
            "lastNoteEndMs": self.last_note_end_ms,
        }


@dataclass(frozen=True, slots=True)
class OutroFamilyFinding:
    reason: Literal[
        "OUTRO_FAMILY_EARLY_START",
        "OUTRO_FAMILY_EARLY_START_SINGLE_SIBLING",
    ]
    support_level: Literal[
        "TWO_SIBLING_CONSENSUS",
        "SINGLE_SIBLING_PROVISIONAL",
    ]
    key_mode: int
    difficulty: str
    target_start_ms: int
    sibling_key_modes: tuple[int, ...]
    sibling_start_ms: tuple[int, ...]
    reference_start_ms: int
    early_by_ms: int

    def to_report(self) -> dict[str, object]:
        return {
            "reason": self.reason,
            "supportLevel": self.support_level,
            "keyMode": self.key_mode,
            "difficulty": self.difficulty,
            "targetStartMs": self.target_start_ms,
            "siblingKeyModes": list(self.sibling_key_modes),
            "siblingStartMs": list(self.sibling_start_ms),
            "referenceStartMs": self.reference_start_ms,
            "earlyByMs": self.early_by_ms,
        }


@dataclass(frozen=True, slots=True)
class OutroFamilyReview:
    version: Literal["outro-family-review-v3-tiered-start-shadow"]
    mode: Literal["SHADOW"]
    policy_state: Literal["UNCALIBRATED"]
    status: Literal["PASS", "REVIEW"]
    charts: tuple[OutroChartView, ...]
    findings: tuple[OutroFamilyFinding, ...]

    def to_report(self) -> dict[str, object]:
        return {
            "version": self.version,
            "mode": self.mode,
            "policyState": self.policy_state,
            "status": self.status,
            "mutatesCharts": False,
            "additionalInferenceCalls": 0,
            "siblingAgreementMs": OUTRO_SIBLING_AGREEMENT_MS,
            "earlyStartReviewMs": OUTRO_EARLY_START_REVIEW_MS,
            "singleSiblingEarlyStartReviewMs": (
                OUTRO_SINGLE_SIBLING_EARLY_START_REVIEW_MS
            ),
            "charts": [chart.to_report() for chart in self.charts],
            "findings": [finding.to_report() for finding in self.findings],
        }


def review_outro_family(charts: tuple[OutroChartView, ...]) -> OutroFamilyReview:
    """Compare only same-difficulty 4K/6K/7K final note starts.

    Two sibling keys that agree provide the strong signal.  If the third key is
    missing, one sibling can only produce a provisional finding at the larger
    threshold.  Conflicting siblings produce no finding.  HOLD release times
    remain in the chart evidence but do not establish that a later playable
    attack exists.  The result is diagnostic only: it neither changes notes nor
    spends generation budget.
    """
    by_slot = {(chart.difficulty, chart.key_mode): chart for chart in charts}
    difficulties = sorted({chart.difficulty for chart in charts})
    findings: list[OutroFamilyFinding] = []
    for difficulty in difficulties:
        for key_mode in OUTRO_KEY_MODES:
            target = by_slot.get((difficulty, key_mode))
            if target is None:
                continue
            siblings = tuple(
                by_slot[(difficulty, sibling_key)]
                for sibling_key in OUTRO_KEY_MODES
                if sibling_key != key_mode and (difficulty, sibling_key) in by_slot
            )
            sibling_starts = tuple(
                sibling.last_note_start_ms for sibling in siblings
            )
            if len(siblings) == 2:
                if (
                    abs(sibling_starts[0] - sibling_starts[1])
                    > OUTRO_SIBLING_AGREEMENT_MS
                ):
                    continue
                reason = "OUTRO_FAMILY_EARLY_START"
                support_level = "TWO_SIBLING_CONSENSUS"
                reference_start_ms = round(sum(sibling_starts) / 2)
                review_threshold_ms = OUTRO_EARLY_START_REVIEW_MS
            elif len(siblings) == 1:
                reason = "OUTRO_FAMILY_EARLY_START_SINGLE_SIBLING"
                support_level = "SINGLE_SIBLING_PROVISIONAL"
                reference_start_ms = sibling_starts[0]
                review_threshold_ms = OUTRO_SINGLE_SIBLING_EARLY_START_REVIEW_MS
            else:
                continue
            early_by_ms = reference_start_ms - target.last_note_start_ms
            if early_by_ms < review_threshold_ms:
                continue
            findings.append(
                OutroFamilyFinding(
                    reason=reason,
                    support_level=support_level,
                    key_mode=key_mode,
                    difficulty=difficulty,
                    target_start_ms=target.last_note_start_ms,
                    sibling_key_modes=tuple(sibling.key_mode for sibling in siblings),
                    sibling_start_ms=sibling_starts,
                    reference_start_ms=reference_start_ms,
                    early_by_ms=early_by_ms,
                )
            )
    return OutroFamilyReview(
        version="outro-family-review-v3-tiered-start-shadow",
        mode="SHADOW",
        policy_state="UNCALIBRATED",
        status="REVIEW" if findings else "PASS",
        charts=charts,
        findings=tuple(findings),
    )
