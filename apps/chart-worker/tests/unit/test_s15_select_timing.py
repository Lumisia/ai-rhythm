import pytest

from chart_worker.analysis.timing import TimingCandidate, TimingPoint, TimingSource, TimingStatus
from chart_worker.errors import ErrorCode, WorkerError
from chart_worker.stages.s15_select_timing import (
    candidate_phase_difference_ms,
    select_timing_candidate,
)


def _candidate(
    source: TimingSource,
    *,
    beats: tuple[int, ...] = (100, 600, 1100),
    point_count: int = 1,
    f1_20ms: float = 0.9,
    p95_abs_ms: float = 10.0,
    status: TimingStatus = TimingStatus.PASS,
) -> TimingCandidate:
    points = tuple(
        TimingPoint(time_ms=index * 500, bpm=120.0, meter=4, start_beat_index=None)
        for index in range(point_count)
    )
    return TimingCandidate(
        source=source,
        points=points,
        projected_beat_ms=beats,
        f1_20ms=f1_20ms,
        f1_50ms=f1_20ms,
        p95_abs_ms=p95_abs_ms,
        status=status,
        reasons=(),
    )


def test_candidate_phase_difference_uses_one_to_one_matched_beats():
    left = _candidate(TimingSource.BEAT_THIS_PIECEWISE, beats=(100, 600, 1100))
    right = _candidate(TimingSource.MAPPERATORINATOR_SUPER, beats=(160, 660, 1160))

    assert candidate_phase_difference_ms(left, right) == pytest.approx(-60.0)


def test_selects_beat_this_when_super_timing_is_unavailable_or_failed():
    beat_this = _candidate(TimingSource.BEAT_THIS_PIECEWISE)
    failed = _candidate(TimingSource.MAPPERATORINATOR_SUPER, status=TimingStatus.FAIL)

    assert select_timing_candidate(beat_this, None) is beat_this
    assert select_timing_candidate(beat_this, failed) is beat_this


def test_rejects_timing_candidates_that_disagree_by_more_than_fifty_ms():
    beat_this = _candidate(TimingSource.BEAT_THIS_PIECEWISE)
    super_timing = _candidate(TimingSource.MAPPERATORINATOR_SUPER, beats=(151, 651, 1151))

    with pytest.raises(WorkerError) as caught:
        select_timing_candidate(beat_this, super_timing)

    assert caught.value.code is ErrorCode.CHART_TIMING_REVIEW_REQUIRED
    assert caught.value.context == {"phaseDifferenceMs": -51.0}


def test_selects_the_only_candidate_that_passes_the_p95_gate():
    beat_this = _candidate(TimingSource.BEAT_THIS_PIECEWISE, p95_abs_ms=31.0)
    super_timing = _candidate(TimingSource.MAPPERATORINATOR_SUPER, p95_abs_ms=30.0)

    assert select_timing_candidate(beat_this, super_timing) is super_timing


def test_rejects_when_neither_candidate_passes_the_p95_gate():
    beat_this = _candidate(TimingSource.BEAT_THIS_PIECEWISE, p95_abs_ms=31.0)
    super_timing = _candidate(TimingSource.MAPPERATORINATOR_SUPER, p95_abs_ms=31.0)

    with pytest.raises(WorkerError) as caught:
        select_timing_candidate(beat_this, super_timing)

    assert caught.value.code is ErrorCode.CHART_TIMING_CANDIDATE_FAILED


def test_prefers_fewer_points_when_passing_f1_scores_are_within_two_percent():
    beat_this = _candidate(TimingSource.BEAT_THIS_PIECEWISE, point_count=2, f1_20ms=0.90)
    super_timing = _candidate(
        TimingSource.MAPPERATORINATOR_SUPER, point_count=1, f1_20ms=0.919
    )

    assert select_timing_candidate(beat_this, super_timing) is super_timing


def test_prefers_higher_f1_when_passing_scores_are_more_than_two_percent_apart():
    beat_this = _candidate(TimingSource.BEAT_THIS_PIECEWISE, point_count=1, f1_20ms=0.90)
    super_timing = _candidate(
        TimingSource.MAPPERATORINATOR_SUPER, point_count=2, f1_20ms=0.93
    )

    assert select_timing_candidate(beat_this, super_timing) is super_timing
