import json

import pytest

from chart_worker.analysis.timing import TimingCandidate, TimingPoint, TimingSource, TimingStatus
from chart_worker.config import WorkerConfig
from chart_worker.errors import ErrorCode, WorkerError
from chart_worker.stages import s15_select_timing
from chart_worker.stages.s15_select_timing import (
    _has_sufficient_phase_coverage,
    candidate_phase_difference_ms,
    run_timing_selection,
    select_timing_candidate,
)
from tests.support import fake_analysis


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


def test_phase_coverage_requires_three_matches_and_eighty_percent_of_the_shorter_grid():
    assert not _has_sufficient_phase_coverage(matched_count=2, left_count=3, right_count=3)
    assert not _has_sufficient_phase_coverage(matched_count=3, left_count=5, right_count=5)
    assert _has_sufficient_phase_coverage(matched_count=4, left_count=5, right_count=5)


def test_selects_beat_this_when_super_timing_is_unavailable_or_failed():
    beat_this = _candidate(TimingSource.BEAT_THIS_PIECEWISE)
    failed = _candidate(TimingSource.MAPPERATORINATOR_SUPER, status=TimingStatus.FAIL)

    assert select_timing_candidate(beat_this, None) is beat_this
    assert select_timing_candidate(beat_this, failed) is beat_this


def test_rejects_failed_beat_this_when_super_timing_is_unavailable():
    beat_this = _candidate(
        TimingSource.BEAT_THIS_PIECEWISE,
        status=TimingStatus.FAIL,
    )

    with pytest.raises(WorkerError) as caught:
        select_timing_candidate(beat_this, None)

    assert caught.value.code is ErrorCode.CHART_TIMING_CANDIDATE_FAILED


def test_generated_failed_super_timing_still_runs_phase_disagreement_gate():
    beat_this = _candidate(TimingSource.BEAT_THIS_PIECEWISE)
    failed = _candidate(
        TimingSource.MAPPERATORINATOR_SUPER,
        beats=(151, 651, 1151),
        status=TimingStatus.FAIL,
    )

    with pytest.raises(WorkerError) as caught:
        select_timing_candidate(beat_this, failed)

    assert caught.value.code is ErrorCode.CHART_TIMING_REVIEW_REQUIRED


def test_generated_failed_super_timing_falls_back_after_phase_gate():
    beat_this = _candidate(TimingSource.BEAT_THIS_PIECEWISE)
    failed = _candidate(
        TimingSource.MAPPERATORINATOR_SUPER,
        beats=(110, 610, 1110),
        status=TimingStatus.FAIL,
    )

    assert select_timing_candidate(beat_this, failed) is beat_this


def test_super_timing_execution_failure_falls_back_and_persists_warning(
    tmp_path, monkeypatch
):
    source = tmp_path / "fixture.wav"
    source.write_bytes(b"source")
    analysis = fake_analysis(source, tmp_path, WorkerConfig())

    class FailingTimingGenerator:
        def __call__(self, audio_path, workdir):
            del audio_path, workdir
            raise WorkerError(
                ErrorCode.CHART_GENERATION_FAILED,
                "GPU unavailable",
                context={"device": "cuda"},
            )

    monkeypatch.setattr(
        s15_select_timing,
        "MapperatorinatorTimingGenerator",
        lambda config: FailingTimingGenerator(),
    )

    result = run_timing_selection(analysis, tmp_path, WorkerConfig(), True)
    report = json.loads(result.timing_quality_report_path.read_text(encoding="utf-8"))

    assert result.timing_candidate.source is TimingSource.BEAT_THIS_PIECEWISE
    assert report["warnings"] == [
        {
            "code": "CHART_GENERATION_FAILED",
            "message": "CHART_GENERATION_FAILED: GPU unavailable",
            "context": {"device": "cuda"},
        }
    ]


def test_super_timing_does_not_swallow_programming_errors(tmp_path, monkeypatch):
    source = tmp_path / "fixture.wav"
    source.write_bytes(b"source")
    analysis = fake_analysis(source, tmp_path, WorkerConfig())

    class BrokenTimingGenerator:
        def __call__(self, audio_path, workdir):
            del audio_path, workdir
            raise RuntimeError("bug")

    monkeypatch.setattr(
        s15_select_timing,
        "MapperatorinatorTimingGenerator",
        lambda config: BrokenTimingGenerator(),
    )

    with pytest.raises(RuntimeError, match="bug"):
        run_timing_selection(analysis, tmp_path, WorkerConfig(), True)


def test_rejects_timing_candidates_that_disagree_by_more_than_fifty_ms():
    beat_this = _candidate(TimingSource.BEAT_THIS_PIECEWISE)
    super_timing = _candidate(TimingSource.MAPPERATORINATOR_SUPER, beats=(151, 651, 1151))

    with pytest.raises(WorkerError) as caught:
        select_timing_candidate(beat_this, super_timing)

    assert caught.value.code is ErrorCode.CHART_TIMING_REVIEW_REQUIRED
    assert caught.value.context == {"phaseDifferenceMs": -51.0}


def test_rejects_candidates_when_phase_matching_finds_no_common_beats():
    """A 300ms phase shift at 60 BPM must not be mistaken for a zero offset."""
    beat_this = _candidate(TimingSource.BEAT_THIS_PIECEWISE, beats=(0, 1_000, 2_000))
    super_timing = _candidate(TimingSource.MAPPERATORINATOR_SUPER, beats=(300, 1_300, 2_300))

    with pytest.raises(WorkerError) as caught:
        select_timing_candidate(beat_this, super_timing)

    assert caught.value.code is ErrorCode.CHART_TIMING_REVIEW_REQUIRED
    assert caught.value.context["matchedBeatCount"] == 0


def test_rejects_candidates_when_phase_matching_covers_too_little_of_a_grid():
    beat_this = _candidate(TimingSource.BEAT_THIS_PIECEWISE, beats=(0, 500, 1_000, 1_500, 2_000))
    super_timing = _candidate(
        TimingSource.MAPPERATORINATOR_SUPER,
        beats=(0, 500, 1_000, 2_800, 3_300),
    )

    with pytest.raises(WorkerError) as caught:
        select_timing_candidate(beat_this, super_timing)

    assert caught.value.code is ErrorCode.CHART_TIMING_REVIEW_REQUIRED
    assert caught.value.context["matchedBeatCount"] == 3
    assert caught.value.context["coverage"] == 0.6


def test_rejects_candidates_when_all_short_grid_beats_match_but_long_grid_is_sparse():
    beat_this = _candidate(TimingSource.BEAT_THIS_PIECEWISE, beats=(0, 500, 1_000))
    super_timing = _candidate(
        TimingSource.MAPPERATORINATOR_SUPER,
        beats=tuple(range(0, 50_000, 500)),
    )

    with pytest.raises(WorkerError) as caught:
        select_timing_candidate(beat_this, super_timing)

    assert caught.value.code is ErrorCode.CHART_TIMING_REVIEW_REQUIRED
    assert caught.value.context["matchedBeatCount"] == 3
    assert caught.value.context["coverage"] == 0.03


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


def test_prefers_fewer_points_at_the_inclusive_two_percent_f1_boundary():
    beat_this = _candidate(TimingSource.BEAT_THIS_PIECEWISE, point_count=1, f1_20ms=0.90)
    super_timing = _candidate(
        TimingSource.MAPPERATORINATOR_SUPER, point_count=2, f1_20ms=0.92
    )

    assert select_timing_candidate(beat_this, super_timing) is beat_this


def test_prefers_higher_f1_when_passing_scores_are_more_than_two_percent_apart():
    beat_this = _candidate(TimingSource.BEAT_THIS_PIECEWISE, point_count=1, f1_20ms=0.90)
    super_timing = _candidate(
        TimingSource.MAPPERATORINATOR_SUPER, point_count=2, f1_20ms=0.93
    )

    assert select_timing_candidate(beat_this, super_timing) is super_timing
