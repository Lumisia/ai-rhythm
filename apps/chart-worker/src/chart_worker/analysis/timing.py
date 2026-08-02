"""Deterministic Beat This timing fitting and timing-quality measurements."""

from __future__ import annotations

import json
from dataclasses import dataclass
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from chart_worker.schema.chart import BpmEvent

if TYPE_CHECKING:
    from chart_worker.analysis.beat import BeatGrid


MIN_SEGMENT_BEATS = 8
P95_LIMIT_MS = 30.0
MAX_LIMIT_MS = 50.0
MIN_SSE_IMPROVEMENT = 0.35
MAX_TIMING_POINTS = 32
DEFAULT_METER = 4


class TimingSource(str, Enum):
    BEAT_THIS_PIECEWISE = "BEAT_THIS_PIECEWISE"
    MAPPERATORINATOR_SUPER = "MAPPERATORINATOR_SUPER"


class TimingStatus(str, Enum):
    PASS = "PASS"
    REVIEW_REQUIRED = "REVIEW_REQUIRED"
    FAIL = "FAIL"


@dataclass(frozen=True, slots=True)
class TimingPoint:
    time_ms: int
    bpm: float
    meter: int
    start_beat_index: int | None


@dataclass(frozen=True, slots=True)
class MatchMetrics:
    matched_pairs: tuple[tuple[int, int], ...]
    precision: float
    recall: float
    f1: float
    median_signed_ms: float
    p95_abs_ms: float
    p99_abs_ms: float
    max_abs_ms: float

    @classmethod
    def from_pairs(
        cls,
        pairs: tuple[tuple[int, int], ...],
        *,
        predicted_count: int,
        reference_count: int,
    ) -> MatchMetrics:
        matched_count = len(pairs)
        precision = matched_count / predicted_count if predicted_count else 0.0
        recall = matched_count / reference_count if reference_count else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        if not pairs:
            return cls(pairs, precision, recall, f1, 0.0, 0.0, 0.0, 0.0)

        signed_errors = np.array([predicted - reference for predicted, reference in pairs])
        abs_errors = np.abs(signed_errors)
        return cls(
            pairs,
            precision,
            recall,
            f1,
            float(np.median(signed_errors)),
            float(np.percentile(abs_errors, 95)),
            float(np.percentile(abs_errors, 99)),
            float(np.max(abs_errors)),
        )


@dataclass(frozen=True, slots=True)
class TimingCandidate:
    source: TimingSource
    points: tuple[TimingPoint, ...]
    projected_beat_ms: tuple[int, ...]
    f1_20ms: float
    f1_50ms: float
    p95_abs_ms: float
    status: TimingStatus
    reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ReferenceChart:
    key_mode: int
    difficulty: str
    sections: dict[str, tuple[int, ...]]


@dataclass(frozen=True, slots=True)
class ReferenceQuality:
    macro_f1_20ms: float
    phase_abs_ms: float
    p95_abs_ms: float
    passes: bool


@dataclass(frozen=True, slots=True)
class _Segment:
    start: int
    end: int
    bpm: float
    sse: float
    p95_abs_ms: float
    max_abs_ms: float


def _fit_segment(beat_ms: tuple[int, ...], start: int, end: int) -> _Segment:
    times = np.asarray(beat_ms[start:end], dtype=np.float64)
    indices = np.arange(times.size, dtype=np.float64)
    slope, intercept = np.polyfit(indices, times, 1)
    residual = times - (slope * indices + intercept)
    return _Segment(
        start=start,
        end=end,
        bpm=60_000.0 / float(slope),
        sse=float(np.square(residual).sum()),
        p95_abs_ms=float(np.percentile(np.abs(residual), 95)),
        max_abs_ms=float(np.abs(residual).max()),
    )


def _exceeds_error_limits(p95_abs_ms: float, max_abs_ms: float) -> bool:
    return p95_abs_ms > P95_LIMIT_MS or max_abs_ms > MAX_LIMIT_MS


def _sse_improvement_is_sufficient(unsplit_sse: float, split_sse: float) -> bool:
    return unsplit_sse > 0 and (unsplit_sse - split_sse) / unsplit_sse >= MIN_SSE_IMPROVEMENT


def _bps_are_mergeable(first_bpm: float, second_bpm: float) -> bool:
    return abs(Decimal(str(second_bpm)) - Decimal(str(first_bpm))) * 200 < Decimal(str(first_bpm))


def _split_segment(
    beat_ms: tuple[int, ...],
    downbeat_indices: tuple[int, ...],
    segment: _Segment,
) -> tuple[_Segment, _Segment] | None:
    if not _exceeds_error_limits(segment.p95_abs_ms, segment.max_abs_ms):
        return None

    candidates: list[tuple[float, _Segment, _Segment]] = []
    for split_index in downbeat_indices:
        if split_index - segment.start < MIN_SEGMENT_BEATS:
            continue
        if segment.end - split_index < MIN_SEGMENT_BEATS:
            continue
        left = _fit_segment(beat_ms, segment.start, split_index)
        right = _fit_segment(beat_ms, split_index, segment.end)
        candidates.append((left.sse + right.sse, left, right))
    if not candidates:
        return None

    split_sse, left, right = min(candidates, key=lambda candidate: candidate[0])
    if not _sse_improvement_is_sufficient(segment.sse, split_sse):
        return None
    return left, right


def _merge_adjacent(beat_ms: tuple[int, ...], segments: list[_Segment]) -> list[_Segment]:
    merged: list[_Segment] = []
    for segment in segments:
        if merged and _bps_are_mergeable(merged[-1].bpm, segment.bpm):
            previous = merged.pop()
            merged.append(_fit_segment(beat_ms, previous.start, segment.end))
        else:
            merged.append(segment)
    return merged


def fit_piecewise_timing(grid: BeatGrid) -> tuple[TimingPoint, ...]:
    """Fit BPM segments, splitting only where Beat This marked a downbeat."""
    if len(grid.beat_ms) < 2:
        raise ValueError("at least two beats are required to fit timing")

    start_index = grid.downbeat_indices[0] if grid.downbeat_indices else 0
    if len(grid.beat_ms) - start_index < 2:
        start_index = 0
    segments: list[_Segment] = [_fit_segment(grid.beat_ms, start_index, len(grid.beat_ms))]
    index = 0
    while index < len(segments) and len(segments) < MAX_TIMING_POINTS:
        split = _split_segment(grid.beat_ms, grid.downbeat_indices, segments[index])
        if split is None:
            index += 1
            continue
        segments[index : index + 1] = split

    segments = _merge_adjacent(grid.beat_ms, segments)
    meter = grid.beats_per_bar or DEFAULT_METER
    return tuple(
        TimingPoint(
            time_ms=grid.beat_ms[segment.start],
            bpm=segment.bpm,
            meter=meter,
            start_beat_index=segment.start,
        )
        for segment in segments
    )


def project_beats(points: tuple[TimingPoint, ...], *, end_ms: int) -> tuple[int, ...]:
    """Project a timing-point sequence onto integer millisecond beat positions."""
    projected: list[int] = []
    for index, point in enumerate(points):
        if point.bpm <= 0:
            raise ValueError("timing point has a non-positive bpm")
        next_time = points[index + 1].time_ms if index + 1 < len(points) else end_ms
        beat_length_ms = 60_000.0 / point.bpm
        beat_index = 0
        while True:
            time_ms = round(point.time_ms + beat_index * beat_length_ms)
            if time_ms >= next_time or time_ms >= end_ms:
                break
            projected.append(time_ms)
            beat_index += 1
    return tuple(projected)


def match_times(
    predicted_ms: tuple[int, ...], reference_ms: tuple[int, ...], *, window_ms: int
) -> MatchMetrics:
    candidates = sorted(
        (abs(predicted - reference), predicted, reference)
        for predicted in sorted(set(predicted_ms))
        for reference in sorted(set(reference_ms))
        if abs(predicted - reference) <= window_ms
    )
    used_predicted: set[int] = set()
    used_reference: set[int] = set()
    pairs: list[tuple[int, int]] = []
    for _, predicted, reference in candidates:
        if predicted in used_predicted or reference in used_reference:
            continue
        used_predicted.add(predicted)
        used_reference.add(reference)
        pairs.append((predicted, reference))
    return MatchMetrics.from_pairs(
        tuple(sorted(pairs)),
        predicted_count=len(set(predicted_ms)),
        reference_count=len(set(reference_ms)),
    )


def _require_object(value: object, *, name: str) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{name} must be an object")
    return value


def _require_exact_fields(value: dict[str, object], expected: set[str], *, name: str) -> None:
    if set(value) != expected:
        raise ValueError(f"{name} must contain exactly {sorted(expected)}")


def _require_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")  # noqa: TRY004 - schema contract
    return value


def load_reference_onsets(path: Path) -> dict[tuple[int, str], ReferenceChart]:
    """Load the versioned human-reference onset contract without coercion."""
    root = _require_object(json.loads(path.read_text(encoding="utf-8")), name="reference")
    _require_exact_fields(root, {"version", "charts"}, name="reference")
    if _require_int(root["version"], name="version") != 1:
        raise ValueError("unsupported reference onset version")
    charts_value = root["charts"]
    if not isinstance(charts_value, list):
        raise ValueError("charts must be an array")  # noqa: TRY004 - schema contract

    loaded: dict[tuple[int, str], ReferenceChart] = {}
    for chart_index, chart_value in enumerate(charts_value):
        chart = _require_object(chart_value, name=f"charts[{chart_index}]")
        _require_exact_fields(
            chart,
            {"keyMode", "difficulty", "sections"},
            name=f"charts[{chart_index}]",
        )
        key_mode = _require_int(chart["keyMode"], name="keyMode")
        if key_mode not in (4, 6, 7):
            raise ValueError("keyMode must be 4, 6, or 7")
        difficulty = chart["difficulty"]
        if difficulty not in ("EASY", "NORMAL", "HARD", "EXPERT"):
            raise ValueError("difficulty must be EASY, NORMAL, HARD, or EXPERT")
        combo = (key_mode, difficulty)
        if combo in loaded:
            raise ValueError(f"duplicate chart reference for {key_mode}K {difficulty}")

        sections_value = chart["sections"]
        if not isinstance(sections_value, list):
            raise ValueError("sections must be an array")  # noqa: TRY004 - schema contract
        sections: dict[str, tuple[int, ...]] = {}
        for section_index, section_value in enumerate(sections_value):
            section = _require_object(
                section_value,
                name=f"charts[{chart_index}].sections[{section_index}]",
            )
            _require_exact_fields(
                section,
                {"id", "onsetMs"},
                name=f"charts[{chart_index}].sections[{section_index}]",
            )
            section_id = section["id"]
            if not isinstance(section_id, str) or not section_id:
                raise ValueError("section id must be a non-empty string")
            if section_id in sections:
                raise ValueError(f"duplicate section id: {section_id}")
            onset_value = section["onsetMs"]
            if not isinstance(onset_value, list):
                raise ValueError("onsetMs must be an array")  # noqa: TRY004 - schema contract
            if not onset_value:
                raise ValueError("reference section must not be empty")
            onsets = tuple(_require_int(onset, name="onsetMs") for onset in onset_value)
            if any(onset < 0 for onset in onsets):
                raise ValueError("onsetMs must be non-negative")
            if list(onsets) != sorted(onsets):
                raise ValueError("onsetMs must be sorted")
            sections[section_id] = onsets
        if not sections:
            raise ValueError("reference chart must contain at least one section")
        loaded[combo] = ReferenceChart(key_mode, difficulty, sections)
    return loaded


def evaluate_reference(
    reference: ReferenceChart | None,
    predicted: tuple[int, ...],
) -> ReferenceQuality | None:
    """Compare unique final note starts with human labels, section by section."""
    if reference is None:
        return None

    predicted_unique = tuple(sorted(set(predicted)))
    section_f1: list[float] = []
    all_pairs: list[tuple[int, int]] = []
    for reference_ms in reference.sections.values():
        first_ms, last_ms = reference_ms[0], reference_ms[-1]
        section_predicted = tuple(
            time_ms for time_ms in predicted_unique if first_ms <= time_ms <= last_ms
        )
        f1_metrics = match_times(section_predicted, reference_ms, window_ms=20)
        error_metrics = match_times(section_predicted, reference_ms, window_ms=50)
        section_f1.append(f1_metrics.f1)
        all_pairs.extend(error_metrics.matched_pairs)

    signed_errors = np.asarray(
        [predicted_ms - reference_ms for predicted_ms, reference_ms in all_pairs],
        dtype=np.float64,
    )
    absolute_errors = np.abs(signed_errors)
    phase_abs_ms = abs(float(np.median(signed_errors))) if signed_errors.size else 0.0
    p95_abs_ms = float(np.percentile(absolute_errors, 95)) if absolute_errors.size else 0.0
    macro_f1 = float(np.mean(section_f1))
    return ReferenceQuality(
        macro_f1_20ms=macro_f1,
        phase_abs_ms=phase_abs_ms,
        p95_abs_ms=p95_abs_ms,
        passes=macro_f1 >= 0.70 and phase_abs_ms <= 15.0 and p95_abs_ms <= 30.0,
    )


def bpm_events_of(points: tuple[TimingPoint, ...]) -> list[BpmEvent]:
    """Convert timing points to chart-v1 BPM events without losing later changes."""
    return [
        BpmEvent(time_ms=0 if index == 0 else point.time_ms, bpm=point.bpm)
        for index, point in enumerate(points)
    ]
