"""Bounded chart-candidate retry gates and deterministic ranking."""

from dataclasses import dataclass

import numpy as np

from chart_worker.analysis.onset import OnsetAnalysis
from chart_worker.analysis.timing import TimingCandidate
from chart_worker.generation.params import REQUESTED_STAR
from chart_worker.schema.types import DIFFICULTIES, KEY_MODES

MAX_CANDIDATE_ATTEMPTS = 3
RETRY_SEED_STEP = 10_000

MAX_LONG_GAP_BARS = 2.0
MAX_RATING_ERROR = 0.35
MAX_REMOVED_RATIO = 0.45
MIN_AUDIO_ONSET_PRECISION = 0.70
AUDIO_ONSET_WINDOW_MS = 50


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
    audio_onset_precision: float | None
    drum_precision: float | None
    playability_passes: int
    playability_violations: int
    hold_ratio_error: float
    reference_pass: bool | None
    requested_star: float | None = None
    cfg_scale: float | None = None


def _require_difficulty(difficulty: str) -> None:
    if difficulty not in DIFFICULTIES:
        raise ValueError(f"unsupported difficulty: {difficulty}")


def requested_star_candidates(difficulty: str) -> tuple[float, float, float]:
    """Return the bounded half-star schedule for one difficulty."""
    _require_difficulty(difficulty)
    current = REQUESTED_STAR[difficulty]
    return (current, current - 0.5, current - 1.0)


def candidate_parameters(
    base_seed: int,
    combination_index: int,
    attempt: int,
    previous: CandidateQuality | None,
) -> CandidateParameters:
    """Choose one of three deterministic seed, star, and CFG combinations."""
    if not 0 <= combination_index < len(KEY_MODES) * len(DIFFICULTIES):
        raise ValueError(f"combination_index out of range: {combination_index}")
    if not 1 <= attempt <= MAX_CANDIDATE_ATTEMPTS:
        raise ValueError(f"attempt must be within 1..{MAX_CANDIDATE_ATTEMPTS}")
    if attempt > 1 and previous is None:
        raise ValueError("previous quality is required for retry attempts")

    difficulty = DIFFICULTIES[combination_index % len(DIFFICULTIES)]
    star_candidates = requested_star_candidates(difficulty)
    requested_star = star_candidates[0]
    if previous is not None:
        previous_star = (
            previous.requested_star
            if previous.requested_star is not None
            else star_candidates[0]
        )
        requested_star = previous_star
        if attempt == 3 and (
            previous.rating_error > MAX_RATING_ERROR
            or previous.removed_ratio > MAX_REMOVED_RATIO
        ):
            requested_star = max(star_candidates[-1], previous_star - 0.5)

    return CandidateParameters(
        seed=base_seed + combination_index + (attempt - 1) * RETRY_SEED_STEP,
        requested_star=requested_star,
        cfg_scale=1.25 if attempt == 1 else 1.0,
    )


def needs_retry(quality: CandidateQuality, *, difficulty: str) -> bool:
    """Return whether another seed may repair this candidate's quality."""
    _require_difficulty(difficulty)
    structural_failure = (
        quality.long_gap_bars > MAX_LONG_GAP_BARS
        or abs(quality.rating_error) >= MAX_RATING_ERROR
        or quality.removed_ratio > MAX_REMOVED_RATIO
        or quality.playability_violations > 0
        or quality.reference_pass is False
    )
    onset_failure = (
        difficulty in ("HARD", "EXPERT")
        and quality.audio_onset_precision is not None
        and quality.audio_onset_precision < MIN_AUDIO_ONSET_PRECISION
    )
    return structural_failure or onset_failure


def rank_candidate(quality: CandidateQuality, *, difficulty: str) -> tuple[float, ...]:
    """Build the approved lexicographic quality key; lower is better."""
    _require_difficulty(difficulty)
    onset_rank = (
        -quality.audio_onset_precision
        if difficulty in ("HARD", "EXPERT")
        and quality.audio_onset_precision is not None
        else 0.0
    )
    return (
        float(needs_retry(quality, difficulty=difficulty)),
        quality.long_gap_bars,
        abs(quality.rating_error),
        quality.removed_ratio,
        onset_rank,
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
