"""Read-only onset alignment diagnostics for generated note rows."""

from dataclasses import dataclass
from itertools import pairwise
from typing import Literal

import numpy as np

from chart_worker.analysis.activity import AudioActivity
from chart_worker.schema.note import Chart

TimingStatus = Literal["PASS", "REVIEW", "INSUFFICIENT"]

SECTION_MS = 30_000
MIN_SECTION_ROWS = 8
OVERALL_PRECISION_50_MIN = 0.70
SECTION_PRECISION_50_MIN = 0.60
SECTION_PHASE_DRIFT_MAX_MS = 25.0
ACTIVE_GAP_MIN_MS = 8_000
ACTIVE_GAP_MIN_ONSETS = 8
ACTIVE_GAP_MIN_FRAME_RATIO = 0.35


@dataclass(frozen=True, slots=True)
class TimingMetrics:
    row_count: int
    precision_20: float | None
    precision_50: float | None
    signed_median_ms: float | None
    absolute_p95_ms: float | None

    def to_report(self) -> dict[str, int | float | None]:
        return {
            "rowCount": self.row_count,
            "precision20": self.precision_20,
            "precision50": self.precision_50,
            "signedMedianMs": self.signed_median_ms,
            "absoluteP95Ms": self.absolute_p95_ms,
        }


@dataclass(frozen=True, slots=True)
class TimingSection:
    start_ms: int
    end_ms: int
    status: TimingStatus
    metrics: TimingMetrics

    def to_report(self) -> dict[str, int | float | str | None]:
        return {
            "startMs": self.start_ms,
            "endMs": self.end_ms,
            "status": self.status,
            **self.metrics.to_report(),
        }


@dataclass(frozen=True, slots=True)
class TimingCoverageGap:
    start_ms: int
    end_ms: int
    onset_count: int
    active_onset_count: int
    active_frame_ratio: float

    def to_report(self) -> dict[str, int | float]:
        return {
            "startMs": self.start_ms,
            "endMs": self.end_ms,
            "durationMs": self.end_ms - self.start_ms,
            "onsetCount": self.onset_count,
            "activeOnsetCount": self.active_onset_count,
            "activeFrameRatio": self.active_frame_ratio,
        }


@dataclass(frozen=True, slots=True)
class TimingDiagnostics:
    status: TimingStatus
    onset_count: int
    active_onset_count: int
    first_note_time_ms: int | None
    max_gap_ms: int
    coverage_gaps: tuple[TimingCoverageGap, ...]
    quiet_coverage_gaps: tuple[TimingCoverageGap, ...]
    overall: TimingMetrics
    sections: tuple[TimingSection, ...]

    def to_report(self) -> dict[str, object]:
        return {
            "status": self.status,
            "onsetCount": self.onset_count,
            "activeOnsetCount": self.active_onset_count,
            "firstNoteTimeMs": self.first_note_time_ms,
            "maxGapMs": self.max_gap_ms,
            "coverageGaps": [gap.to_report() for gap in self.coverage_gaps],
            "quietCoverageGaps": [gap.to_report() for gap in self.quiet_coverage_gaps],
            "overall": self.overall.to_report(),
            "sections": [section.to_report() for section in self.sections],
        }


def _nearest_errors_ms(
    rows: np.ndarray,
    onsets: np.ndarray,
) -> np.ndarray:
    indices = np.searchsorted(onsets, rows)
    left_indices = np.clip(indices - 1, 0, onsets.size - 1)
    right_indices = np.clip(indices, 0, onsets.size - 1)
    left = onsets[left_indices]
    right = onsets[right_indices]
    left_distance = np.where(indices > 0, np.abs(rows - left), np.inf)
    right_distance = np.where(indices < onsets.size, np.abs(rows - right), np.inf)
    nearest = np.where(right_distance < left_distance, right, left)
    return rows - nearest


def _metrics(rows: tuple[int, ...], onsets: np.ndarray) -> TimingMetrics:
    if not rows or onsets.size == 0:
        return TimingMetrics(
            row_count=len(rows),
            precision_20=None,
            precision_50=None,
            signed_median_ms=None,
            absolute_p95_ms=None,
        )
    errors = _nearest_errors_ms(np.asarray(rows, dtype=np.int64), onsets)
    absolute = np.abs(errors)
    return TimingMetrics(
        row_count=len(rows),
        precision_20=round(float(np.mean(absolute <= 20)), 6),
        precision_50=round(float(np.mean(absolute <= 50)), 6),
        signed_median_ms=round(float(np.median(errors)), 6),
        absolute_p95_ms=round(float(np.percentile(absolute, 95)), 6),
    )


def _section_status(
    metrics: TimingMetrics,
    *,
    overall_median_ms: float | None,
    minimum_rows: int,
) -> TimingStatus:
    if metrics.row_count < minimum_rows or metrics.precision_50 is None:
        return "INSUFFICIENT"
    if metrics.precision_50 < SECTION_PRECISION_50_MIN:
        return "REVIEW"
    if (
        overall_median_ms is not None
        and metrics.signed_median_ms is not None
        and abs(metrics.signed_median_ms - overall_median_ms)
        > SECTION_PHASE_DRIFT_MAX_MS
    ):
        return "REVIEW"
    return "PASS"


def _coverage_gaps(
    rows: tuple[int, ...],
    onsets: np.ndarray,
    active_onsets: np.ndarray,
    *,
    duration_ms: int,
    activity: AudioActivity | None,
) -> tuple[tuple[TimingCoverageGap, ...], tuple[TimingCoverageGap, ...]]:
    boundaries = tuple(sorted({0, duration_ms, *rows}))
    active_gaps = []
    quiet_gaps = []
    for start_ms, end_ms in pairwise(boundaries):
        if end_ms - start_ms < ACTIVE_GAP_MIN_MS:
            continue
        onset_start = int(np.searchsorted(onsets, start_ms, side="right"))
        onset_end = int(np.searchsorted(onsets, end_ms, side="left"))
        onset_count = onset_end - onset_start
        if onset_count >= ACTIVE_GAP_MIN_ONSETS:
            active_start = int(np.searchsorted(active_onsets, start_ms, side="right"))
            active_end = int(np.searchsorted(active_onsets, end_ms, side="left"))
            active_onset_count = active_end - active_start
            active_frame_ratio = (
                1.0
                if activity is None
                else round(activity.active_frame_ratio(start_ms, end_ms), 6)
            )
            gap = TimingCoverageGap(
                start_ms=start_ms,
                end_ms=end_ms,
                onset_count=onset_count,
                active_onset_count=active_onset_count,
                active_frame_ratio=active_frame_ratio,
            )
            target = (
                active_gaps
                if active_onset_count >= ACTIVE_GAP_MIN_ONSETS
                and active_frame_ratio >= ACTIVE_GAP_MIN_FRAME_RATIO
                else quiet_gaps
            )
            target.append(gap)
    return tuple(active_gaps), tuple(quiet_gaps)


def diagnose_chart_timing(
    notes: Chart,
    onset_ms: tuple[int, ...],
    *,
    duration_ms: int,
    section_ms: int = SECTION_MS,
    minimum_section_rows: int = MIN_SECTION_ROWS,
    activity: AudioActivity | None = None,
) -> TimingDiagnostics:
    """Compare unique note rows with nearest onsets without changing the chart."""
    if duration_ms < 0:
        raise ValueError("duration_ms must be non-negative")
    if section_ms <= 0:
        raise ValueError("section_ms must be positive")
    if minimum_section_rows <= 0:
        raise ValueError("minimum_section_rows must be positive")

    rows = tuple(sorted({note.time_ms for note in notes}))
    onsets = np.asarray(sorted(set(onset_ms)), dtype=np.int64)
    active_onsets = (
        onsets
        if activity is None
        else np.asarray(
            sorted(set(onsets.tolist()).intersection(activity.active_onset_ms)),
            dtype=np.int64,
        )
    )
    overall = _metrics(rows, onsets)
    coverage_gaps, quiet_coverage_gaps = _coverage_gaps(
        rows,
        onsets,
        active_onsets,
        duration_ms=duration_ms,
        activity=activity,
    )
    sections = []
    for start_ms in range(0, max(1, duration_ms), section_ms):
        end_ms = min(duration_ms, start_ms + section_ms)
        is_last = end_ms == duration_ms
        section_rows = tuple(
            row
            for row in rows
            if start_ms <= row and (row <= end_ms if is_last else row < end_ms)
        )
        metrics = _metrics(section_rows, onsets)
        sections.append(
            TimingSection(
                start_ms=start_ms,
                end_ms=end_ms,
                status=_section_status(
                    metrics,
                    overall_median_ms=overall.signed_median_ms,
                    minimum_rows=minimum_section_rows,
                ),
                metrics=metrics,
            )
        )

    if overall.row_count == 0 or overall.precision_50 is None:
        status: TimingStatus = "INSUFFICIENT"
    elif (
        overall.precision_50 < OVERALL_PRECISION_50_MIN
        or coverage_gaps
        or any(section.status == "REVIEW" for section in sections)
    ):
        status = "REVIEW"
    else:
        status = "PASS"

    max_gap_ms = max((right - left for left, right in pairwise(rows)), default=0)
    return TimingDiagnostics(
        status=status,
        onset_count=int(onsets.size),
        active_onset_count=int(active_onsets.size),
        first_note_time_ms=rows[0] if rows else None,
        max_gap_ms=max_gap_ms,
        coverage_gaps=coverage_gaps,
        quiet_coverage_gaps=quiet_coverage_gaps,
        overall=overall,
        sections=tuple(sections),
    )
