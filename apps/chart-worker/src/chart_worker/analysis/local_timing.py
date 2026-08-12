"""Segment-local onset and metrical evidence for a shared timing authority."""

from dataclasses import dataclass
from math import log2

import numpy as np

from chart_worker.analysis.onset import OnsetAnalysis
from chart_worker.generation.osu_parser import OsuBpmEvent

GRID_DIVISORS = (1, 2, 3, 4, 6, 8)
METRICAL_RATIOS = (0.5, 2.0 / 3.0, 1.0, 1.5, 2.0)
GRID_SUPPORT_WINDOW_MS = 70.0
METRICAL_DISTANCE_MAX_OCTAVES = 0.08
ACTIVE_DURATION_MIN_MS = 8_000
ACTIVE_ONSET_MIN = 8
ACTIVE_FRAME_RATIO_MIN = 0.35

_GRID_POINTS = np.unique(
    np.concatenate(
        [np.arange(divisor + 1, dtype=float) / divisor for divisor in GRID_DIVISORS]
    )
)


@dataclass(frozen=True, slots=True)
class LocalTimingSegmentMetrics:
    index: int
    start_ms: int
    end_ms: int
    bpm: float
    onset_count: int
    active_onset_count: int
    active_frame_ratio: float
    active_confident: bool
    current_grid_support: float
    neighbor_grid_support: float
    current_residual_p95_ms: float | None
    neighbor_residual_p95_ms: float | None
    isolated_metrical_outlier: bool
    pulse_conflict: bool
    phase_conflict: bool
    evidence_status: str
    boundary_onset_distance_ms: float | None = None
    boundary_supported: bool | None = None


@dataclass(frozen=True, slots=True)
class LocalTimingMetrics:
    segments: tuple[LocalTimingSegmentMetrics, ...]


def _range(values: tuple[int, ...], start_ms: int, end_ms: int) -> tuple[int, ...]:
    return tuple(value for value in values if start_ms <= value < end_ms)


def _grid_residuals(
    onsets: tuple[int, ...],
    *,
    bpm: float,
    phase_ms: int,
) -> np.ndarray:
    if not onsets:
        return np.asarray([], dtype=float)
    beat_ms = 60_000.0 / bpm
    positions = ((np.asarray(onsets, dtype=float) - phase_ms) / beat_ms) % 1.0
    distances = np.min(np.abs(positions[:, None] - _GRID_POINTS[None, :]), axis=1)
    return distances * beat_ms


def _support_and_p95(
    onsets: tuple[int, ...],
    *,
    bpm: float,
    phase_ms: int,
) -> tuple[float, float | None]:
    residuals = _grid_residuals(onsets, bpm=bpm, phase_ms=phase_ms)
    if not residuals.size:
        return 0.0, None
    return (
        round(float(np.mean(residuals <= GRID_SUPPORT_WINDOW_MS)), 6),
        round(float(np.percentile(residuals, 95)), 6),
    )


def _metrical_distance(left_bpm: float, right_bpm: float) -> float:
    return min(abs(log2(left_bpm / (ratio * right_bpm))) for ratio in METRICAL_RATIOS)


def _isolated_outlier(
    events: tuple[OsuBpmEvent, ...],
    index: int,
) -> bool:
    if index == 0 or index + 1 >= len(events):
        return False
    previous = events[index - 1].bpm
    current = events[index].bpm
    following = events[index + 1].bpm
    neighbors_match = (
        _metrical_distance(previous, following) <= METRICAL_DISTANCE_MAX_OCTAVES
    )
    return neighbors_match and all(
        _metrical_distance(current, neighbor) > METRICAL_DISTANCE_MAX_OCTAVES
        for neighbor in (previous, following)
    )


def measure_local_timing(
    events: tuple[OsuBpmEvent, ...],
    analysis: OnsetAnalysis,
    *,
    duration_ms: int,
) -> LocalTimingMetrics:
    """Measure every timing segment without rejecting an absolute BPM range."""
    segments = []
    active_source = (
        analysis.activity.active_onset_ms
        if analysis.activity is not None
        else analysis.onset_ms
    )
    for index, event in enumerate(events):
        start_ms = max(0, event.time_ms)
        end_ms = min(
            duration_ms,
            events[index + 1].time_ms if index + 1 < len(events) else duration_ms,
        )
        if end_ms <= start_ms:
            continue
        onsets = _range(analysis.onset_ms, start_ms, end_ms)
        active_onsets = _range(active_source, start_ms, end_ms)
        boundary_onset_distance_ms = (
            min(abs(onset_ms - start_ms) for onset_ms in active_source)
            if index > 0 and active_source
            else None
        )
        active_frame_ratio = (
            (1.0 if active_onsets else 0.0)
            if analysis.activity is None
            else analysis.activity.active_frame_ratio(start_ms, end_ms)
        )
        active_confident = (
            end_ms - start_ms >= ACTIVE_DURATION_MIN_MS
            and len(active_onsets) >= ACTIVE_ONSET_MIN
            and active_frame_ratio >= ACTIVE_FRAME_RATIO_MIN
        )
        current_support, current_p95 = _support_and_p95(
            active_onsets,
            bpm=event.bpm,
            phase_ms=start_ms,
        )
        neighbor_results = [
            _support_and_p95(active_onsets, bpm=events[neighbor].bpm, phase_ms=start_ms)
            for neighbor in (index - 1, index + 1)
            if 0 <= neighbor < len(events)
        ]
        if neighbor_results:
            neighbor_support, neighbor_p95 = max(
                neighbor_results,
                key=lambda result: (
                    result[0],
                    -(
                        result[1]
                        if result[1] is not None
                        else float("inf")
                    ),
                ),
            )
        else:
            neighbor_support, neighbor_p95 = 0.0, None
        pulse_conflict = (
            active_confident
            and current_support < 0.55
            and neighbor_support >= 0.55
            and neighbor_support - current_support >= 0.25
        )
        phase_conflict = (
            pulse_conflict
            and current_p95 is not None
            and neighbor_p95 is not None
            and current_p95 - neighbor_p95 >= GRID_SUPPORT_WINDOW_MS
        )
        segments.append(
            LocalTimingSegmentMetrics(
                index=index,
                start_ms=start_ms,
                end_ms=end_ms,
                bpm=event.bpm,
                onset_count=len(onsets),
                active_onset_count=len(active_onsets),
                active_frame_ratio=round(active_frame_ratio, 6),
                active_confident=active_confident,
                current_grid_support=current_support,
                neighbor_grid_support=neighbor_support,
                current_residual_p95_ms=current_p95,
                neighbor_residual_p95_ms=neighbor_p95,
                isolated_metrical_outlier=_isolated_outlier(events, index),
                pulse_conflict=pulse_conflict,
                phase_conflict=phase_conflict,
                evidence_status="SUFFICIENT" if active_confident else "INSUFFICIENT",
                boundary_onset_distance_ms=boundary_onset_distance_ms,
                boundary_supported=(
                    boundary_onset_distance_ms <= GRID_SUPPORT_WINDOW_MS
                    if boundary_onset_distance_ms is not None
                    else None
                ),
            )
        )
    return LocalTimingMetrics(segments=tuple(segments))
