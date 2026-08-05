"""Immutable, section-level evidence for chart quality review."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from itertools import pairwise
from types import MappingProxyType

import numpy as np

from chart_worker.analysis.activity import AudioActivity
from chart_worker.postprocess.patterns import detect_patterns, rows_of
from chart_worker.rating.project_rating import measure_rating
from chart_worker.schema.note import Chart

SECTION_MS = 15_000
RELEASE_WINDOW_MS = 250
ROW_NGRAM_SIZE = 4
ACTIVE_FRAME_RATIO = 0.35


@dataclass(frozen=True, slots=True)
class HoldProfile:
    note_ratio: float
    time_occupancy_ratio: float
    mean_duration_ms: float
    p95_duration_ms: float
    max_duration_ms: int
    max_concurrent: int
    max_held_lane_ratio: float
    max_release_count_250ms: int
    section_hold_counts: tuple[int, ...]
    section_occupancy_ratios: tuple[float, ...]
    section_release_counts_250ms: tuple[int, ...]

    def to_report(self) -> dict[str, object]:
        return {
            "noteRatio": self.note_ratio,
            "timeOccupancyRatio": self.time_occupancy_ratio,
            "meanDurationMs": self.mean_duration_ms,
            "p95DurationMs": self.p95_duration_ms,
            "maxDurationMs": self.max_duration_ms,
            "maxConcurrent": self.max_concurrent,
            "maxHeldLaneRatio": self.max_held_lane_ratio,
            "maxReleaseCount250Ms": self.max_release_count_250ms,
            "sectionHoldCounts": list(self.section_hold_counts),
            "sectionOccupancyRatios": list(self.section_occupancy_ratios),
            "sectionReleaseCounts250Ms": list(self.section_release_counts_250ms),
        }


@dataclass(frozen=True, slots=True)
class PatternProfile:
    histogram: Mapping[str, int]
    sections: tuple[Mapping[str, int], ...]
    transition_counts: Mapping[str, int]
    longest_row_ngram_repeat: int
    lane_usage_ratios: tuple[float, ...]
    section_lane_imbalances: tuple[float, ...]
    section_longest_row_ngram_repeats: tuple[int, ...]

    def to_report(self) -> dict[str, object]:
        return {
            "histogram": dict(self.histogram),
            "sections": [dict(section) for section in self.sections],
            "transitionCounts": dict(self.transition_counts),
            "longestRowNgramRepeat": self.longest_row_ngram_repeat,
            "laneUsageRatios": list(self.lane_usage_ratios),
            "sectionLaneImbalances": list(self.section_lane_imbalances),
            "sectionLongestRowNgramRepeats": list(
                self.section_longest_row_ngram_repeats
            ),
        }


@dataclass(frozen=True, slots=True)
class DifficultyProfile:
    project_rating: float
    avg_nps: float
    p95_nps: float
    peak_nps: float
    chord_ratio: float
    max_jack: int
    section_peak_nps: tuple[float, ...]

    def to_report(self) -> dict[str, object]:
        return {
            "projectRating": self.project_rating,
            "avgNps": self.avg_nps,
            "p95Nps": self.p95_nps,
            "peakNps": self.peak_nps,
            "chordRatio": self.chord_ratio,
            "maxJack": self.max_jack,
            "sectionPeakNps": list(self.section_peak_nps),
        }


@dataclass(frozen=True, slots=True)
class ChartQualityProfile:
    hold: HoldProfile
    pattern: PatternProfile
    difficulty: DifficultyProfile
    active_section_mask: tuple[bool, ...]

    def to_report(self) -> dict[str, object]:
        return {
            "holdProfile": self.hold.to_report(),
            "patternProfile": self.pattern.to_report(),
            "difficultyProfile": self.difficulty.to_report(),
            "activeSectionMask": list(self.active_section_mask),
        }


def _section_bounds(duration_ms: int) -> tuple[tuple[int, int], ...]:
    return tuple(
        (start_ms, min(start_ms + SECTION_MS, duration_ms))
        for start_ms in range(0, duration_ms, SECTION_MS)
    )


def _freeze_counts(counts: Mapping[str, int]) -> Mapping[str, int]:
    return MappingProxyType(dict(counts))


def _merged_lane_intervals(
    notes: Chart, *, key_mode: int
) -> tuple[tuple[tuple[int, int], ...], ...]:
    by_lane: list[list[tuple[int, int]]] = [[] for _ in range(key_mode)]
    for note in notes:
        if note.kind == "HOLD":
            by_lane[note.lane].append(
                (note.time_ms, note.time_ms + (note.duration_ms or 0))
            )

    merged_by_lane: list[tuple[tuple[int, int], ...]] = []
    for intervals in by_lane:
        merged: list[tuple[int, int]] = []
        for start_ms, end_ms in sorted(intervals):
            if merged and start_ms <= merged[-1][1]:
                previous_start, previous_end = merged[-1]
                merged[-1] = (previous_start, max(previous_end, end_ms))
            else:
                merged.append((start_ms, end_ms))
        merged_by_lane.append(tuple(merged))
    return tuple(merged_by_lane)


def _max_concurrent_lanes(
    merged_by_lane: tuple[tuple[tuple[int, int], ...], ...],
) -> int:
    events = sorted(
        (time_ms, delta)
        for intervals in merged_by_lane
        for start_ms, end_ms in intervals
        for time_ms, delta in ((start_ms, 1), (end_ms, -1))
    )
    current = 0
    maximum = 0
    for _time_ms, delta in events:
        current += delta
        maximum = max(maximum, current)
    return maximum


def _max_count_in_window(times: list[int], *, window_ms: int) -> int:
    ordered = sorted(times)
    left = 0
    maximum = 0
    for right, time_ms in enumerate(ordered):
        while time_ms - ordered[left] > window_ms:
            left += 1
        maximum = max(maximum, right - left + 1)
    return maximum


def _overlap_ms(start_ms: int, end_ms: int, section_start: int, section_end: int) -> int:
    return max(0, min(end_ms, section_end) - max(start_ms, section_start))


def _longest_row_ngram_repeat(lanes: tuple[tuple[int, ...], ...]) -> int:
    if len(lanes) < ROW_NGRAM_SIZE:
        return 0
    longest = 1
    last_start = len(lanes) - ROW_NGRAM_SIZE
    for start in range(last_start + 1):
        ngram = lanes[start : start + ROW_NGRAM_SIZE]
        count = 1
        cursor = start + ROW_NGRAM_SIZE
        while cursor + ROW_NGRAM_SIZE <= len(lanes):
            if lanes[cursor : cursor + ROW_NGRAM_SIZE] != ngram:
                break
            count += 1
            cursor += ROW_NGRAM_SIZE
        longest = max(longest, count)
    return longest


def _peak_nps(times: list[int]) -> float:
    ordered = sorted(times)
    left = 0
    maximum = 0
    for right, time_ms in enumerate(ordered):
        while time_ms - ordered[left] >= 1_000:
            left += 1
        maximum = max(maximum, right - left + 1)
    return float(maximum)


def _hold_profile(
    notes: Chart,
    *,
    key_mode: int,
    duration_ms: int,
    sections: tuple[tuple[int, int], ...],
) -> HoldProfile:
    holds = [note for note in notes if note.kind == "HOLD"]
    durations = [note.duration_ms or 0 for note in holds]
    releases = [note.time_ms + (note.duration_ms or 0) for note in holds]
    merged_by_lane = _merged_lane_intervals(notes, key_mode=key_mode)
    held_by_lane = tuple(
        sum(end_ms - start_ms for start_ms, end_ms in intervals)
        for intervals in merged_by_lane
    )
    total_held_ms = sum(held_by_lane)

    section_hold_counts: list[int] = []
    section_occupancies: list[float] = []
    section_release_counts: list[int] = []
    for start_ms, end_ms in sections:
        section_hold_counts.append(
            sum(start_ms <= note.time_ms < end_ms for note in holds)
        )
        section_held_ms = sum(
            _overlap_ms(interval_start, interval_end, start_ms, end_ms)
            for intervals in merged_by_lane
            for interval_start, interval_end in intervals
        )
        section_occupancies.append(
            section_held_ms / ((end_ms - start_ms) * key_mode)
        )
        section_release_counts.append(
            _max_count_in_window(
                [
                    release
                    for release in releases
                    if start_ms <= release < end_ms
                    or (end_ms == duration_ms and release == end_ms)
                ],
                window_ms=RELEASE_WINDOW_MS,
            )
        )

    return HoldProfile(
        note_ratio=len(holds) / len(notes) if notes else 0.0,
        time_occupancy_ratio=total_held_ms / (duration_ms * key_mode),
        mean_duration_ms=float(np.mean(durations)) if durations else 0.0,
        p95_duration_ms=float(np.percentile(durations, 95)) if durations else 0.0,
        max_duration_ms=max(durations, default=0),
        max_concurrent=_max_concurrent_lanes(merged_by_lane),
        max_held_lane_ratio=max(held_by_lane, default=0) / duration_ms,
        max_release_count_250ms=_max_count_in_window(
            releases, window_ms=RELEASE_WINDOW_MS
        ),
        section_hold_counts=tuple(section_hold_counts),
        section_occupancy_ratios=tuple(section_occupancies),
        section_release_counts_250ms=tuple(section_release_counts),
    )


def _lane_ratios(notes: Chart, key_mode: int) -> tuple[float, ...]:
    counts = Counter(note.lane for note in notes)
    total = len(notes)
    return tuple(counts[lane] / total if total else 0.0 for lane in range(key_mode))


def _pattern_profile(
    notes: Chart,
    *,
    key_mode: int,
    beat_ms: float,
    sections: tuple[tuple[int, int], ...],
) -> PatternProfile:
    instances = detect_patterns(notes, key_mode=key_mode, beat_ms=beat_ms)
    histogram = Counter(instance.kind.value for instance in instances)
    transition_counts = Counter(
        f"{first.kind.value}>{second.kind.value}"
        for first, second in pairwise(instances)
    )
    row_lanes = tuple(row.lanes for row in rows_of(notes))

    section_histograms: list[Mapping[str, int]] = []
    section_lane_imbalances: list[float] = []
    section_row_repeats: list[int] = []
    for start_ms, end_ms in sections:
        section_histograms.append(
            _freeze_counts(
                Counter(
                    instance.kind.value
                    for instance in instances
                    if start_ms <= instance.start_ms < end_ms
                )
            )
        )
        section_notes = [note for note in notes if start_ms <= note.time_ms < end_ms]
        ratios = _lane_ratios(section_notes, key_mode)
        section_lane_imbalances.append(max(ratios) - min(ratios) if ratios else 0.0)
        section_rows = tuple(
            row.lanes for row in rows_of(section_notes)
        )
        section_row_repeats.append(_longest_row_ngram_repeat(section_rows))

    return PatternProfile(
        histogram=_freeze_counts(histogram),
        sections=tuple(section_histograms),
        transition_counts=_freeze_counts(transition_counts),
        longest_row_ngram_repeat=_longest_row_ngram_repeat(row_lanes),
        lane_usage_ratios=_lane_ratios(notes, key_mode),
        section_lane_imbalances=tuple(section_lane_imbalances),
        section_longest_row_ngram_repeats=tuple(section_row_repeats),
    )


def _difficulty_profile(
    notes: Chart,
    *,
    duration_ms: int,
    sections: tuple[tuple[int, int], ...],
) -> DifficultyProfile:
    rating = measure_rating(notes, duration_ms)
    return DifficultyProfile(
        project_rating=rating.rating,
        avg_nps=rating.avg_nps,
        p95_nps=rating.p95_nps,
        peak_nps=rating.peak_nps,
        chord_ratio=rating.chord_ratio,
        max_jack=rating.max_jack,
        section_peak_nps=tuple(
            _peak_nps(
                [note.time_ms for note in notes if start_ms <= note.time_ms < end_ms]
            )
            for start_ms, end_ms in sections
        ),
    )


def build_chart_quality_profile(
    notes: Chart,
    *,
    key_mode: int,
    duration_ms: int,
    beat_ms: float,
    activity: AudioActivity | None,
) -> ChartQualityProfile:
    """Build non-mutating whole-chart and 15-second section evidence."""
    if key_mode <= 0:
        raise ValueError("key_mode must be positive")
    if duration_ms <= 0:
        raise ValueError("duration_ms must be positive")
    if beat_ms <= 0:
        raise ValueError("beat_ms must be positive")
    if any(note.lane >= key_mode for note in notes):
        raise ValueError("note lane must be less than key_mode")

    sections = _section_bounds(duration_ms)
    active_section_mask = tuple(
        activity.active_frame_ratio(start_ms, end_ms) >= ACTIVE_FRAME_RATIO
        if activity is not None
        else any(
            start_ms <= note.time_ms < end_ms
            or (
                note.kind == "HOLD"
                and _overlap_ms(
                    note.time_ms,
                    note.time_ms + (note.duration_ms or 0),
                    start_ms,
                    end_ms,
                )
                > 0
            )
            for note in notes
        )
        for start_ms, end_ms in sections
    )
    return ChartQualityProfile(
        hold=_hold_profile(
            notes,
            key_mode=key_mode,
            duration_ms=duration_ms,
            sections=sections,
        ),
        pattern=_pattern_profile(
            notes,
            key_mode=key_mode,
            beat_ms=beat_ms,
            sections=sections,
        ),
        difficulty=_difficulty_profile(notes, duration_ms=duration_ms, sections=sections),
        active_section_mask=active_section_mask,
    )
