"""Bounded chart-candidate retry gates and deterministic ranking."""

from dataclasses import dataclass

import numpy as np

from chart_worker.analysis.onset import OnsetAnalysis
from chart_worker.analysis.timing import TimingCandidate
from chart_worker.schema.types import DIFFICULTIES

MAX_CANDIDATE_ATTEMPTS = 3
RETRY_SEED_STEP = 10_000

MAX_LONG_GAP_BARS = 2.0
MAX_RATING_ERROR = 0.35
MAX_REMOVED_RATIO = 0.45
MIN_DRUM_PRECISION = 0.70
MAX_PLAYABILITY_PASSES = 8


@dataclass(frozen=True, slots=True)
class CandidateParameters:
    seed: int
    requested_star: float
    cfg_scale: float


@dataclass(frozen=True, slots=True)
class CandidateQuality:
    long_gap_bars: float
    rating_error: float
    removed_ratio: float
    drum_precision: float | None
    playability_passes: int
    hold_ratio_error: float
    reference_pass: bool | None


def _require_difficulty(difficulty: str) -> None:
    if difficulty not in DIFFICULTIES:
        raise ValueError(f"unsupported difficulty: {difficulty}")


def needs_retry(quality: CandidateQuality, *, difficulty: str) -> bool:
    """Return whether another seed may repair this candidate's quality."""
    _require_difficulty(difficulty)
    structural_failure = (
        quality.long_gap_bars > MAX_LONG_GAP_BARS
        or quality.rating_error >= MAX_RATING_ERROR
        or quality.removed_ratio > MAX_REMOVED_RATIO
        or quality.playability_passes >= MAX_PLAYABILITY_PASSES
        or quality.reference_pass is False
    )
    drum_failure = (
        difficulty in ("HARD", "EXPERT")
        and quality.drum_precision is not None
        and quality.drum_precision < MIN_DRUM_PRECISION
    )
    return structural_failure or drum_failure


def rank_candidate(quality: CandidateQuality, *, difficulty: str) -> tuple[float, ...]:
    """Build the approved lexicographic quality key; lower is better."""
    _require_difficulty(difficulty)
    drum_rank = (
        -quality.drum_precision
        if difficulty in ("HARD", "EXPERT") and quality.drum_precision is not None
        else 0.0
    )
    return (
        float(needs_retry(quality, difficulty=difficulty)),
        abs(quality.rating_error),
        quality.removed_ratio,
        drum_rank,
        quality.hold_ratio_error,
    )


def select_candidate_index(
    qualities: tuple[CandidateQuality, ...], *, difficulty: str
) -> int:
    """Select deterministically, preserving attempt order on exact ties."""
    if not qualities:
        raise ValueError("at least one candidate quality is required")
    return min(
        range(len(qualities)),
        key=lambda index: (rank_candidate(qualities[index], difficulty=difficulty), index),
    )


def _bar_spans(timing: TimingCandidate, *, duration_ms: int) -> tuple[tuple[int, int], ...]:
    """Project meter-aware bars, resetting at each selected timing point."""
    if duration_ms <= 0:
        raise ValueError("duration_ms must be positive")
    spans: list[tuple[int, int]] = []
    beats = timing.projected_beat_ms
    for point_index, point in enumerate(timing.points):
        segment_end = (
            timing.points[point_index + 1].time_ms
            if point_index + 1 < len(timing.points)
            else duration_ms
        )
        segment_beats = tuple(
            beat for beat in beats if point.time_ms <= beat < segment_end
        )
        if not segment_beats:
            continue
        for beat_index in range(0, len(segment_beats), point.meter):
            start = segment_beats[beat_index]
            end_index = beat_index + point.meter
            end = segment_beats[end_index] if end_index < len(segment_beats) else segment_end
            if start < end:
                spans.append((start, end))
    return tuple(spans)


def _bar_index(time_ms: int, spans: tuple[tuple[int, int], ...]) -> int | None:
    for index, (start, end) in enumerate(spans):
        if start <= time_ms < end:
            return index
    return None


def longest_active_bar_gap(
    *,
    onsets: OnsetAnalysis,
    timing: TimingCandidate,
    duration_ms: int,
    note_times: tuple[int, ...],
) -> float:
    """Count the longest note-empty bar run bounded by active bars."""
    spans = _bar_spans(timing, duration_ms=duration_ms)
    if not spans or not onsets.onset_ms:
        return 0.0

    onset_strengths = np.asarray(
        [onsets.strength_at(time_ms) for time_ms in onsets.onset_ms],
        dtype=np.float64,
    )
    threshold = float(np.percentile(onset_strengths, 75))
    active = sorted(
        {
            index
            for time_ms, strength in zip(onsets.onset_ms, onset_strengths, strict=True)
            if strength >= threshold
            if (index := _bar_index(time_ms, spans)) is not None
        }
    )
    if len(active) < 2:
        return 0.0

    occupied = {
        index
        for time_ms in set(note_times)
        if (index := _bar_index(time_ms, spans)) is not None
    }
    longest = 0
    run = 0
    for bar_index in range(active[0] + 1, active[-1]):
        if bar_index in occupied:
            run = 0
        else:
            run += 1
            longest = max(longest, run)
    return float(longest)
