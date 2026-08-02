"""S1.5: select the safer timing source before chart generation."""

from chart_worker.analysis.timing import TimingCandidate, TimingStatus, match_times
from chart_worker.errors import ErrorCode, WorkerError

_PHASE_MATCH_WINDOW_MS = 250


def candidate_phase_difference_ms(left: TimingCandidate, right: TimingCandidate) -> float:
    """Return the median signed offset of one-to-one matched projected beats."""
    return match_times(
        left.projected_beat_ms,
        right.projected_beat_ms,
        window_ms=_PHASE_MATCH_WINDOW_MS,
    ).median_signed_ms


def select_timing_candidate(
    beat_this: TimingCandidate,
    super_timing: TimingCandidate | None,
) -> TimingCandidate:
    """Select a timing candidate, preserving review-required disagreements."""
    if super_timing is None or super_timing.status is TimingStatus.FAIL:
        return beat_this

    phase = candidate_phase_difference_ms(beat_this, super_timing)
    if abs(phase) > 50.0:
        raise WorkerError(
            ErrorCode.CHART_TIMING_REVIEW_REQUIRED,
            "timing candidates disagree by more than 50ms",
            context={"phaseDifferenceMs": round(phase, 3)},
        )

    passing = [
        candidate
        for candidate in (beat_this, super_timing)
        if candidate.p95_abs_ms <= 30.0
    ]
    if not passing:
        raise WorkerError(ErrorCode.CHART_TIMING_CANDIDATE_FAILED, "no timing candidate passed")
    if len(passing) == 1:
        return passing[0]

    left, right = passing
    if abs(left.f1_20ms - right.f1_20ms) <= 0.02:
        return min(passing, key=lambda candidate: len(candidate.points))
    return max(passing, key=lambda candidate: candidate.f1_20ms)
