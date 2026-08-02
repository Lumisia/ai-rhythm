"""S1.5: select the safer timing source before chart generation."""

import math

from chart_worker.analysis.timing import TimingCandidate, TimingStatus, match_times
from chart_worker.errors import ErrorCode, WorkerError

_PHASE_MATCH_WINDOW_MS = 250
_MIN_PHASE_MATCHES = 3
_MIN_PHASE_COVERAGE = 0.80
_F1_TIE_TOLERANCE = 1e-12


def _has_sufficient_phase_coverage(*, matched_count: int, left_count: int, right_count: int) -> bool:
    """Require three matches and 80% coverage of the shorter projected-beat grid."""
    compared_count = min(left_count, right_count)
    return (
        compared_count > 0
        and matched_count >= _MIN_PHASE_MATCHES
        and matched_count / compared_count >= _MIN_PHASE_COVERAGE
    )


def candidate_phase_difference_ms(left: TimingCandidate, right: TimingCandidate) -> float:
    """Return the median signed offset of one-to-one matched projected beats."""
    matches = match_times(
        left.projected_beat_ms,
        right.projected_beat_ms,
        window_ms=_PHASE_MATCH_WINDOW_MS,
    )
    left_count = len(set(left.projected_beat_ms))
    right_count = len(set(right.projected_beat_ms))
    matched_count = len(matches.matched_pairs)
    compared_count = min(left_count, right_count)
    coverage = matched_count / compared_count if compared_count else 0.0
    if not _has_sufficient_phase_coverage(
        matched_count=matched_count,
        left_count=left_count,
        right_count=right_count,
    ):
        raise WorkerError(
            ErrorCode.CHART_TIMING_REVIEW_REQUIRED,
            "timing candidates have insufficient matched beat coverage",
            context={
                "matchedBeatCount": matched_count,
                "coverage": round(coverage, 3),
                "minimumCoverage": _MIN_PHASE_COVERAGE,
                "minimumMatchedBeatCount": _MIN_PHASE_MATCHES,
            },
        )
    return matches.median_signed_ms


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
    f1_difference = abs(left.f1_20ms - right.f1_20ms)
    if f1_difference < 0.02 or math.isclose(
        f1_difference,
        0.02,
        rel_tol=0.0,
        abs_tol=_F1_TIE_TOLERANCE,
    ):
        return min(passing, key=lambda candidate: len(candidate.points))
    return max(passing, key=lambda candidate: candidate.f1_20ms)
