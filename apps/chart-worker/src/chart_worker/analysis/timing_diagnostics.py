"""Read-only onset alignment diagnostics for generated note rows."""

from dataclasses import dataclass
from itertools import pairwise
from typing import Literal

import numpy as np

from chart_worker.analysis.activity import AudioActivity
from chart_worker.analysis.coverage_jury import (
    LocalAudioGapEvidence,
    measure_local_gap_evidence,
)
from chart_worker.analysis.coverage_opportunity import (
    MIN_PHRASE_DURATION_MS,
    MIN_STRONG_ATTACKS,
    CoverageKind,
    CoverageOpportunity,
    classify_coverage_interval,
)
from chart_worker.analysis.event_matching import maximum_ordered_match_count
from chart_worker.analysis.onset import OnsetAnalysis
from chart_worker.analysis.song_context import LocalTempoMap
from chart_worker.generation.osu_parser import OsuBpmEvent
from chart_worker.schema.note import Chart

TimingStatus = Literal["PASS", "REVIEW", "INSUFFICIENT"]

SECTION_MS = 15_000
SECTION_BEATS = 32.0
MIN_BEAT_SECTION_MS = 8_000
MAX_BEAT_SECTION_MS = 30_000
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
    absolute_p99_ms: float | None
    matched_count_50: int = 0
    matched_precision_50: float | None = None
    matched_recall_50: float | None = None
    matched_f1_50: float | None = None
    onset_reuse_inflation_50: float | None = None

    def to_report(self) -> dict[str, int | float | None]:
        return {
            "rowCount": self.row_count,
            "precision20": self.precision_20,
            "precision50": self.precision_50,
            "signedMedianMs": self.signed_median_ms,
            "absoluteP95Ms": self.absolute_p95_ms,
            "absoluteP99Ms": self.absolute_p99_ms,
            "matchedCount50": self.matched_count_50,
            "matchedPrecision50": self.matched_precision_50,
            "matchedRecall50": self.matched_recall_50,
            "matchedF150": self.matched_f1_50,
            "onsetReuseInflation50": self.onset_reuse_inflation_50,
        }


@dataclass(frozen=True, slots=True)
class TimingSection:
    start_ms: int
    end_ms: int
    status: TimingStatus
    metrics: TimingMetrics
    phase_delta_ms: float | None

    def to_report(self) -> dict[str, int | float | str | None]:
        return {
            "startMs": self.start_ms,
            "endMs": self.end_ms,
            "status": self.status,
            "phaseDeltaMs": self.phase_delta_ms,
            **self.metrics.to_report(),
        }


@dataclass(frozen=True, slots=True)
class TimingCoverageGap:
    start_ms: int
    end_ms: int
    onset_count: int
    active_onset_count: int
    active_frame_ratio: float
    position: Literal["LEADING", "POST_FIRST", "MIDDLE", "TRAILING"]
    opportunity: CoverageOpportunity | None = None
    local_audio_evidence: LocalAudioGapEvidence | None = None

    def to_report(self) -> dict[str, object]:
        report: dict[str, object] = {
            "startMs": self.start_ms,
            "endMs": self.end_ms,
            "durationMs": self.end_ms - self.start_ms,
            "onsetCount": self.onset_count,
            "activeOnsetCount": self.active_onset_count,
            "activeFrameRatio": self.active_frame_ratio,
            "position": self.position,
        }
        if self.opportunity is not None:
            report["opportunity"] = self.opportunity.to_report()
        if self.local_audio_evidence is not None:
            report["localAudioEvidence"] = self.local_audio_evidence.to_report()
        return report


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


def _metrics(
    rows: tuple[int, ...],
    onsets: np.ndarray,
    *,
    matching_onsets: np.ndarray | None = None,
) -> TimingMetrics:
    if not rows or onsets.size == 0:
        return TimingMetrics(
            row_count=len(rows),
            precision_20=None,
            precision_50=None,
            signed_median_ms=None,
            absolute_p95_ms=None,
            absolute_p99_ms=None,
        )
    row_array = np.asarray(rows, dtype=np.int64)
    errors = _nearest_errors_ms(row_array, onsets)
    absolute = np.abs(errors)
    match_reference = onsets if matching_onsets is None else matching_onsets
    matched_count = maximum_ordered_match_count(
        row_array,
        match_reference,
        window_ms=50,
    )
    matched_precision = matched_count / row_array.size
    matched_recall = (
        matched_count / match_reference.size if match_reference.size else None
    )
    matched_f1 = (
        2 * matched_precision * matched_recall / (matched_precision + matched_recall)
        if matched_recall is not None and matched_precision + matched_recall > 0
        else 0.0
        if matched_recall is not None
        else None
    )
    nearest_precision_50 = float(np.mean(absolute <= 50))
    return TimingMetrics(
        row_count=len(rows),
        precision_20=round(float(np.mean(absolute <= 20)), 6),
        precision_50=round(nearest_precision_50, 6),
        signed_median_ms=round(float(np.median(errors)), 6),
        absolute_p95_ms=round(float(np.percentile(absolute, 95)), 6),
        absolute_p99_ms=round(float(np.percentile(absolute, 99)), 6),
        matched_count_50=matched_count,
        matched_precision_50=round(matched_precision, 6),
        matched_recall_50=(
            round(matched_recall, 6) if matched_recall is not None else None
        ),
        matched_f1_50=(round(matched_f1, 6) if matched_f1 is not None else None),
        onset_reuse_inflation_50=round(
            max(0.0, nearest_precision_50 - matched_precision),
            6,
        ),
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
    notes: Chart,
    rows: tuple[int, ...],
    onsets: np.ndarray,
    active_onsets: np.ndarray,
    *,
    duration_ms: int,
    activity: AudioActivity | None,
    onset_analysis: OnsetAnalysis | None,
    tempo_map: LocalTempoMap | None,
    difficulty: str | None,
) -> tuple[tuple[TimingCoverageGap, ...], tuple[TimingCoverageGap, ...]]:
    coverage_rows = tuple(row for row in rows if 0 <= row <= duration_ms)
    boundaries = tuple(sorted({0, duration_ms, *coverage_rows}))
    first_row_ms = coverage_rows[0] if coverage_rows else None
    second_row_ms = coverage_rows[1] if len(coverage_rows) >= 2 else None
    active_gaps = []
    quiet_gaps = []
    for start_ms, end_ms in pairwise(boundaries):
        onset_start = int(np.searchsorted(onsets, start_ms, side="right"))
        onset_end = int(np.searchsorted(onsets, end_ms, side="left"))
        onset_count = onset_end - onset_start
        active_start = int(np.searchsorted(active_onsets, start_ms, side="right"))
        active_end = int(np.searchsorted(active_onsets, end_ms, side="left"))
        active_onset_count = active_end - active_start
        active_frame_ratio = (
            1.0
            if activity is None
            else round(activity.active_frame_ratio(start_ms, end_ms), 6)
        )
        local_audio_evidence = (
            measure_local_gap_evidence(
                onset_analysis,
                start_ms=start_ms,
                end_ms=end_ms,
            )
            if onset_analysis is not None
            and end_ms - start_ms >= MIN_PHRASE_DURATION_MS
            else None
        )
        opportunity = None
        if (
            onset_analysis is not None
            and activity is not None
            and tempo_map is not None
            and difficulty is not None
            and end_ms - start_ms >= MIN_PHRASE_DURATION_MS
            and active_onset_count >= min(MIN_STRONG_ATTACKS.values())
            and active_frame_ratio >= ACTIVE_GAP_MIN_FRAME_RATIO
        ):
            opportunity = classify_coverage_interval(
                notes,
                onset_analysis,
                tempo_map,
                start_ms=start_ms,
                end_ms=end_ms,
                difficulty=difficulty,
            )
            # Insufficient evidence must not erase a legacy active gap.  The
            # quality gate can downgrade it to REVIEW, while recovery/timing
            # preflight still sees that the generated grid has not proved
            # coverage.  Only positive sustain evidence moves a gap to quiet.
            target = (
                quiet_gaps
                if opportunity.kind is CoverageKind.SUSTAIN_REPRESENTABLE
                else active_gaps
            )
            target.append(
                TimingCoverageGap(
                    start_ms=start_ms,
                    end_ms=end_ms,
                    onset_count=onset_count,
                    active_onset_count=active_onset_count,
                    active_frame_ratio=active_frame_ratio,
                    position=(
                        "POST_FIRST"
                        if start_ms == first_row_ms and end_ms == second_row_ms
                        else "LEADING"
                        if start_ms == 0 and end_ms == first_row_ms
                        else "TRAILING"
                        if end_ms == duration_ms
                        else "MIDDLE"
                    ),
                    opportunity=opportunity,
                    local_audio_evidence=local_audio_evidence,
                )
            )
            continue
        if end_ms - start_ms < ACTIVE_GAP_MIN_MS:
            continue
        if onset_count >= ACTIVE_GAP_MIN_ONSETS:
            gap = TimingCoverageGap(
                start_ms=start_ms,
                end_ms=end_ms,
                onset_count=onset_count,
                active_onset_count=active_onset_count,
                active_frame_ratio=active_frame_ratio,
                position=(
                    "POST_FIRST"
                    if start_ms == first_row_ms and end_ms == second_row_ms
                    else "LEADING"
                    if start_ms == 0 and end_ms == first_row_ms
                    else "TRAILING"
                    if end_ms == duration_ms
                    else "MIDDLE"
                ),
                local_audio_evidence=local_audio_evidence,
            )
            target = (
                active_gaps
                if active_onset_count >= ACTIVE_GAP_MIN_ONSETS
                and active_frame_ratio >= ACTIVE_GAP_MIN_FRAME_RATIO
                else quiet_gaps
            )
            target.append(gap)
    return tuple(active_gaps), tuple(quiet_gaps)


def _time_after_beats(
    tempo_map: LocalTempoMap,
    *,
    start_ms: int,
    beat_count: float,
    duration_ms: int,
) -> int:
    """Return the wall-clock time after integrating ``beat_count`` local beats."""
    cursor = start_ms
    remaining_beats = beat_count
    while cursor < duration_ms:
        event = tempo_map.at(cursor)
        event_index = max(0, tempo_map.times.index(event.time_ms))
        next_event_ms = (
            tempo_map.times[event_index + 1]
            if event_index + 1 < len(tempo_map.times)
            else duration_ms
        )
        segment_end = min(duration_ms, max(cursor + 1, next_event_ms))
        available_beats = (segment_end - cursor) * event.bpm / 60_000.0
        if available_beats >= remaining_beats:
            return min(
                duration_ms,
                round(cursor + remaining_beats * 60_000.0 / event.bpm),
            )
        remaining_beats -= available_beats
        cursor = segment_end
    return duration_ms


def _section_boundaries(
    *,
    duration_ms: int,
    section_ms: int | None,
    bpm_events: tuple[OsuBpmEvent, ...] | None,
) -> tuple[tuple[int, int], ...]:
    if duration_ms == 0:
        return ((0, 0),)
    if section_ms is not None or not bpm_events:
        fixed_ms = SECTION_MS if section_ms is None else section_ms
        return tuple(
            (start_ms, min(duration_ms, start_ms + fixed_ms))
            for start_ms in range(0, duration_ms, fixed_ms)
        )

    tempo_map = LocalTempoMap(bpm_events)
    sections = []
    start_ms = 0
    while start_ms < duration_ms:
        beat_end_ms = _time_after_beats(
            tempo_map,
            start_ms=start_ms,
            beat_count=SECTION_BEATS,
            duration_ms=duration_ms,
        )
        end_ms = min(
            duration_ms,
            max(
                start_ms + MIN_BEAT_SECTION_MS,
                min(start_ms + MAX_BEAT_SECTION_MS, beat_end_ms),
            ),
        )
        sections.append((start_ms, end_ms))
        start_ms = end_ms
    return tuple(sections)


def diagnose_chart_timing(
    notes: Chart,
    onset_ms: tuple[int, ...],
    *,
    duration_ms: int,
    coverage_end_ms: int | None = None,
    section_ms: int | None = None,
    bpm_events: tuple[OsuBpmEvent, ...] | None = None,
    minimum_section_rows: int = MIN_SECTION_ROWS,
    activity: AudioActivity | None = None,
    onset_analysis: OnsetAnalysis | None = None,
    difficulty: str | None = None,
) -> TimingDiagnostics:
    """Compare unique note rows with nearest onsets without changing the chart."""
    if duration_ms < 0:
        raise ValueError("duration_ms must be non-negative")
    if coverage_end_ms is None:
        coverage_end_ms = duration_ms
    if not 0 <= coverage_end_ms <= duration_ms:
        raise ValueError("coverage_end_ms must be within duration_ms")
    if section_ms is not None and section_ms <= 0:
        raise ValueError("section_ms must be positive")
    if minimum_section_rows <= 0:
        raise ValueError("minimum_section_rows must be positive")
    if onset_analysis is not None:
        if onset_analysis.onset_ms != onset_ms:
            raise ValueError("onset_analysis.onset_ms must match onset_ms")
        if onset_analysis.activity is not activity:
            raise ValueError("onset_analysis.activity must be the supplied activity")
        if difficulty is None:
            raise ValueError("difficulty is required with onset_analysis")
    elif difficulty is not None:
        raise ValueError("onset_analysis is required with difficulty")

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
        notes,
        rows,
        onsets,
        active_onsets,
        duration_ms=coverage_end_ms,
        activity=activity,
        onset_analysis=onset_analysis,
        tempo_map=(LocalTempoMap(bpm_events) if bpm_events else None),
        difficulty=difficulty,
    )
    sections = []
    for start_ms, end_ms in _section_boundaries(
        duration_ms=duration_ms,
        section_ms=section_ms,
        bpm_events=bpm_events,
    ):
        is_last = end_ms == duration_ms
        section_rows = tuple(
            row
            for row in rows
            if start_ms <= row and (row <= end_ms if is_last else row < end_ms)
        )
        section_onsets = onsets[
            (onsets >= start_ms)
            & (onsets <= end_ms if is_last else onsets < end_ms)
        ]
        metrics = _metrics(
            section_rows,
            onsets,
            matching_onsets=section_onsets,
        )
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
                phase_delta_ms=(
                    None
                    if (
                        metrics.signed_median_ms is None
                        or overall.signed_median_ms is None
                    )
                    else round(metrics.signed_median_ms - overall.signed_median_ms, 6)
                ),
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
