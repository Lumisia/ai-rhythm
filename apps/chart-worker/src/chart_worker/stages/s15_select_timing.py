"""S1.5: select the safer timing source before chart generation."""

import dataclasses
import json
import math
from pathlib import Path

from chart_worker.analysis.timing import (
    TimingCandidate,
    TimingPoint,
    TimingSource,
    TimingStatus,
    match_times,
    project_beats,
)
from chart_worker.config import WorkerConfig
from chart_worker.errors import ErrorCode, WorkerError
from chart_worker.generation.mapperatorinator import MapperatorinatorTimingGenerator
from chart_worker.generation.osu_parser import parse_osu_timing
from chart_worker.generation.timing_osu import timing_points_to_osu
from chart_worker.stages.types import AnalysisStageResult

_PHASE_MATCH_WINDOW_MS = 250
_MIN_PHASE_MATCHES = 3
_MIN_PHASE_COVERAGE = 0.80
_F1_TIE_TOLERANCE = 1e-12


def _candidate_payload(candidate: TimingCandidate) -> dict[str, object]:
    return {
        "source": candidate.source.value,
        "points": [
            {
                "timeMs": point.time_ms,
                "bpm": point.bpm,
                "meter": point.meter,
                "startBeatIndex": point.start_beat_index,
            }
            for point in candidate.points
        ],
        "projectedBeatMs": list(candidate.projected_beat_ms),
        "f1At20Ms": candidate.f1_20ms,
        "f1At50Ms": candidate.f1_50ms,
        "p95AbsMs": candidate.p95_abs_ms,
        "status": candidate.status.value,
        "reasons": list(candidate.reasons),
    }


def _evaluate_super_timing(
    points: tuple[TimingPoint, ...], analysis: AnalysisStageResult
) -> TimingCandidate:
    projected = project_beats(points, end_ms=analysis.normalized.duration_ms)
    metrics_20 = match_times(projected, analysis.beat_grid.beat_ms, window_ms=20)
    metrics_50 = match_times(projected, analysis.beat_grid.beat_ms, window_ms=50)
    passed = bool(points) and metrics_50.p95_abs_ms <= 30.0
    return TimingCandidate(
        source=TimingSource.MAPPERATORINATOR_SUPER,
        points=points,
        projected_beat_ms=projected,
        f1_20ms=metrics_20.f1,
        f1_50ms=metrics_50.f1,
        p95_abs_ms=metrics_50.p95_abs_ms,
        status=TimingStatus.PASS if passed else TimingStatus.FAIL,
        reasons=() if passed else ("no timing points or p95 exceeds 30ms",),
    )


def run_timing_selection(
    analysis: AnalysisStageResult,
    run_dir: Path,
    config: WorkerConfig,
    enable_super_timing: bool,
) -> AnalysisStageResult:
    """Select one timing candidate and materialize its shared pipeline artifacts."""
    super_timing = None
    warnings: list[dict[str, object]] = []
    if enable_super_timing:
        try:
            text = MapperatorinatorTimingGenerator(config)(
                analysis.normalized.path,
                run_dir / "analysis" / "super-timing-work",
            )
        except WorkerError as error:
            if error.code is not ErrorCode.CHART_GENERATION_FAILED:
                raise
            warnings.append(
                {
                    "code": error.code.value,
                    "message": str(error),
                    "context": error.context,
                }
            )
        else:
            super_timing = _evaluate_super_timing(parse_osu_timing(text), analysis)

    selected = select_timing_candidate(analysis.timing_candidate, super_timing)
    timing_path = run_dir / "analysis" / "timing.osu"
    timing_path.write_text(
        timing_points_to_osu(
            selected.points,
            audio_filename=analysis.normalized.path.name,
            title=analysis.normalized.path.stem,
        ),
        encoding="utf-8",
    )
    report_path = run_dir / "analysis" / "timing-quality-v1.json"
    report_path.write_text(
        json.dumps(
            {
                "version": 1,
                "selected": _candidate_payload(selected),
                "warnings": warnings,
                "candidates": [
                    _candidate_payload(candidate)
                    for candidate in (analysis.timing_candidate, super_timing)
                    if candidate is not None
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return dataclasses.replace(
        analysis,
        timing_candidate=selected,
        timing_osu_path=timing_path,
        timing_quality_report_path=report_path,
    )


def _has_sufficient_phase_coverage(*, matched_count: int, left_count: int, right_count: int) -> bool:
    """Require three matches and 80% coverage of both projected-beat grids."""
    compared_count = max(left_count, right_count)
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
    compared_count = max(left_count, right_count)
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
    if super_timing is None:
        if beat_this.status is TimingStatus.PASS and beat_this.p95_abs_ms <= 30.0:
            return beat_this
        raise WorkerError(ErrorCode.CHART_TIMING_CANDIDATE_FAILED, "no timing candidate passed")

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
        if candidate.status is TimingStatus.PASS and candidate.p95_abs_ms <= 30.0
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
