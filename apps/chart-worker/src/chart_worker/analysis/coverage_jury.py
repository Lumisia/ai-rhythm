"""Policy-free local audio evidence for note-coverage gaps."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from chart_worker.analysis.coverage_opportunity import measure_attack_evidence
from chart_worker.analysis.onset import OnsetAnalysis

COVERAGE_JURY_LOCAL_EVIDENCE_VERSION = "coverage-jury-local-evidence-v1"


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


def measure_local_gap_evidence(
    onset_analysis: OnsetAnalysis,
    *,
    start_ms: int,
    end_ms: int,
) -> LocalAudioGapEvidence:
    """Measure a gap at song and local scales without deciding whether to repair it."""
    attack_evidence = measure_attack_evidence(
        onset_analysis,
        start_ms=start_ms,
        end_ms=end_ms,
    )
    activity = onset_analysis.activity
    active_frame_ratio: float | None = None
    if activity is not None:
        active_frame_ratio = round(activity.active_frame_ratio(start_ms, end_ms), 6)

    return LocalAudioGapEvidence(
        version=COVERAGE_JURY_LOCAL_EVIDENCE_VERSION,
        start_ms=start_ms,
        end_ms=end_ms,
        active_frame_ratio=active_frame_ratio,
        active_onset_count=attack_evidence.active_onset_count,
        global_strong_attack_count=attack_evidence.global_strong_attack_count,
        local_strong_attack_count=attack_evidence.local_strong_attack_count,
        global_threshold=attack_evidence.global_threshold,
        local_threshold=attack_evidence.local_threshold,
        neighboring_activity_ratio=attack_evidence.neighboring_activity_ratio,
    )
