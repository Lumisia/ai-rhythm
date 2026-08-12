"""Audio-activity review for the interval before the first timing event."""

from dataclasses import dataclass

from chart_worker.analysis.intro_anchor import (
    IntroAnchorEvidence,
    classify_intro_anchor,
)
from chart_worker.analysis.onset import OnsetAnalysis
from chart_worker.generation.osu_parser import OsuBpmEvent
from chart_worker.validation.timing_review import TimingAuthorityAction

LEADING_GAP_MIN_MS = 8_000
LEADING_GAP_MIN_ONSETS = 8
LEADING_GAP_MIN_FRAME_RATIO = 0.35

ANCHOR_REVIEW_LATENESS_BEATS = 0.5
"""확인된 anchor가 첫 event보다 반 박 이상 앞서면 MAP 복구 대상으로 남긴다."""


@dataclass(frozen=True, slots=True)
class LeadingTimingCoverage:
    action: TimingAuthorityAction
    reasons: tuple[str, ...]
    first_event_time_ms: int
    leading_duration_ms: int
    onset_count: int
    active_onset_count: int
    active_frame_ratio: float
    intro_anchor: IntroAnchorEvidence

    def to_report(self) -> dict[str, object]:
        return {
            "action": self.action.value,
            "reasons": list(self.reasons),
            "firstEventTimeMs": self.first_event_time_ms,
            "leadingDurationMs": self.leading_duration_ms,
            "onsetCount": self.onset_count,
            "activeOnsetCount": self.active_onset_count,
            "activeFrameRatio": self.active_frame_ratio,
            "introAnchor": self.intro_anchor.to_report(),
        }


def review_leading_timing_coverage(
    events: tuple[OsuBpmEvent, ...],
    analysis: OnsetAnalysis,
    *,
    duration_ms: int,
) -> LeadingTimingCoverage:
    """Classify audible coverage before the first timing event without mutation."""
    if not events:
        raise ValueError("leading timing coverage requires at least one event")
    if duration_ms < 0:
        raise ValueError("duration_ms must be non-negative")

    first_event_time_ms = events[0].time_ms
    intro_anchor = classify_intro_anchor(
        events,
        analysis,
        duration_ms=duration_ms,
    )
    leading_duration_ms = min(duration_ms, max(0, first_event_time_ms))
    onsets = tuple(
        onset for onset in analysis.onset_ms if 0 <= onset < leading_duration_ms
    )

    if analysis.activity is None:
        active_onsets = onsets
        active_frame_ratio = 1.0 if onsets else 0.0
    else:
        active = set(analysis.activity.active_onset_ms)
        active_onsets = tuple(onset for onset in onsets if onset in active)
        active_frame_ratio = round(
            analysis.activity.active_frame_ratio(0, leading_duration_ms), 6
        )

    first_beat_ms = 60_000.0 / events[0].bpm
    anchor_ms = intro_anchor.anchor_ms
    lateness_beats = (
        (first_event_time_ms - anchor_ms) / first_beat_ms
        if anchor_ms is not None and first_beat_ms > 0
        else 0.0
    )

    if leading_duration_ms == 0:
        action = TimingAuthorityAction.PASS
        reasons: tuple[str, ...] = ()
    elif intro_anchor.status == "NON_RHYTHMIC":
        # 첫 event 앞이 정말 조용하다. 의도적인 인트로로 보존한다.
        action = TimingAuthorityAction.PASS
        reasons = ("QUIET_LEADING_TIMING_GAP",)
    elif (
        len(active_onsets) >= LEADING_GAP_MIN_ONSETS
        and active_frame_ratio >= LEADING_GAP_MIN_FRAME_RATIO
        and leading_duration_ms >= LEADING_GAP_MIN_MS
    ):
        # 활발한 음악 8초 이상을 timing 이 통째로 놓쳤다 (기존 규칙 유지).
        action = TimingAuthorityAction.RETRY_TIMING
        reasons = ("ACTIVE_LEADING_TIMING_GAP",)
    elif (
        intro_anchor.status == "CONFIRMED"
        and lateness_beats > ANCHOR_REVIEW_LATENESS_BEATS
    ):
        # timing point의 phase가 anchor를 지지한다면 Timing 재생성보다
        # MAP 첫 창의 문맥 부족 문제다. 실패시키지 않고 프리롤 MAP 복구로
        # 넘긴다.
        action = TimingAuthorityAction.REVIEW
        reasons = ("CONFIRMED_INTRO_ANCHOR_BEFORE_FIRST_EVENT",)
    elif (
        intro_anchor.status == "UNCERTAIN"
        and lateness_beats > ANCHOR_REVIEW_LATENESS_BEATS
    ):
        action = TimingAuthorityAction.REVIEW
        reasons = ("UNCERTAIN_INTRO_ANCHOR",)
    else:
        # anchor 가 첫 event 반 박 이내에 있다. 인트로가 덮여 있다.
        action = TimingAuthorityAction.PASS
        reasons = ()

    return LeadingTimingCoverage(
        action=action,
        reasons=reasons,
        first_event_time_ms=first_event_time_ms,
        leading_duration_ms=leading_duration_ms,
        onset_count=len(onsets),
        active_onset_count=len(active_onsets),
        active_frame_ratio=active_frame_ratio,
        intro_anchor=intro_anchor,
    )
