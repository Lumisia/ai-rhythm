"""Multi-axis mania difficulty evidence with official strain constants."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from chart_worker.analysis.chart_events import ChartEventIndex
from chart_worker.analysis.song_context import LocalTempoMap
from chart_worker.schema.note import NoteEvent

DIFFICULTY_VECTOR_VERSION = "difficulty-vector-v2"

# Official osu!lazer StrainSkill defaults:
# https://github.com/ppy/osu/blob/master/osu.Game/Rulesets/Difficulty/Skills/StrainSkill.cs
STRAIN_SECTION_MS = 400
SECTION_DECAY_WEIGHT = 0.90

# Official osu!lazer mania Strain decay bases:
# https://github.com/ppy/osu/blob/master/osu.Game.Rulesets.Mania/Difficulty/Skills/Strain.cs
INDIVIDUAL_DECAY_BASE = 0.125
OVERALL_DECAY_BASE = 0.30

# Official osu!lazer mania evaluator constants:
# https://github.com/ppy/osu/blob/master/osu.Game.Rulesets.Mania/Difficulty/Evaluators/OverallStrainEvaluator.cs
HOLD_FACTOR = 1.25
RELEASE_THRESHOLD_MS = 30.0
RELEASE_LOGISTIC_MULTIPLIER = 0.27

_ROLES_BY_KEY_MODE: dict[int, tuple[str, ...]] = {
    4: ("L", "L", "R", "R"),
    6: ("L", "L", "L", "R", "R", "R"),
    7: ("L", "L", "L", "C", "R", "R", "R"),
}


def _rounded(value: float) -> float:
    return round(float(value), 6)


def _weighted_peak_sum(peaks: tuple[float, ...]) -> float:
    weight = 1.0
    total = 0.0
    for peak in sorted((value for value in peaks if value > 0), reverse=True):
        total += peak * weight
        weight *= SECTION_DECAY_WEIGHT
    return total


def _decay(value: float, delta_ms: int, base: float) -> float:
    return value * math.pow(base, max(0, delta_ms) / 1_000.0)


def _release_logistic(distance_ms: float) -> float:
    exponent = -RELEASE_LOGISTIC_MULTIPLIER * (
        distance_ms - RELEASE_THRESHOLD_MS
    )
    return 1.0 / (1.0 + math.exp(exponent))


def _end_ms(note: NoteEvent) -> int:
    return note.time_ms + (note.duration_ms or 0)


def _official_hold_terms(
    note: NoteEvent,
    previous_by_lane: tuple[NoteEvent | None, ...],
) -> tuple[float, float]:
    """Return osu!lazer's containment factor and crossing-release addition."""
    start_ms = note.time_ms
    end_ms = _end_ms(note)
    hold_factor = 1.0
    is_overlapping = False
    closest_end_ms = abs(end_ms - start_ms)
    for previous in previous_by_lane:
        if previous is None:
            continue
        previous_end_ms = _end_ms(previous)
        if previous_end_ms > end_ms + 1 and start_ms > previous.time_ms + 1:
            hold_factor = HOLD_FACTOR
        is_overlapping |= (
            previous_end_ms > start_ms + 1
            and end_ms > previous_end_ms + 1
            and start_ms > previous.time_ms + 1
        )
        closest_end_ms = min(closest_end_ms, abs(end_ms - previous_end_ms))
    hold_addition = _release_logistic(closest_end_ms) if is_overlapping else 0.0
    return hold_factor, hold_addition


@dataclass(frozen=True, slots=True)
class DifficultyCalibration:
    key_mode: int
    axis_names: tuple[str, ...]
    medians: tuple[float, ...]
    iqrs: tuple[float, ...]
    weights: tuple[float, ...]
    complete_song_count: int

    def __post_init__(self) -> None:
        lengths = {
            len(self.axis_names),
            len(self.medians),
            len(self.iqrs),
            len(self.weights),
        }
        if len(lengths) != 1:
            raise ValueError("calibration arrays must have the same length")
        if not self.axis_names:
            raise ValueError("calibration requires at least one axis")
        if any(iqr <= 0 or not math.isfinite(iqr) for iqr in self.iqrs):
            raise ValueError("calibration IQRs must be finite and positive")
        if any(weight < 0 or not math.isfinite(weight) for weight in self.weights):
            raise ValueError("calibration weights must be finite and non-negative")
        if self.complete_song_count <= 0:
            raise ValueError("complete_song_count must be positive")

    def to_report(self) -> dict[str, object]:
        return {
            "keyMode": self.key_mode,
            "axisNames": list(self.axis_names),
            "medians": list(self.medians),
            "iqrs": list(self.iqrs),
            "weights": list(self.weights),
            "completeSongCount": self.complete_song_count,
        }

    @classmethod
    def from_report(cls, value: dict[str, object]) -> DifficultyCalibration:
        return cls(
            key_mode=int(value["keyMode"]),
            axis_names=tuple(str(item) for item in value["axisNames"]),
            medians=tuple(float(item) for item in value["medians"]),
            iqrs=tuple(float(item) for item in value["iqrs"]),
            weights=tuple(float(item) for item in value["weights"]),
            complete_song_count=int(value["completeSongCount"]),
        )


@dataclass(frozen=True, slots=True)
class DifficultyVectorV2:
    version: str
    density_strain: float
    jack_strain: float
    chord_load: float
    ln_strain: float
    coordination: float
    peak_skill: float
    bounded_stamina: float
    ordering_score: float
    section_peaks: tuple[float, ...]
    weighted_peak_sum: float
    max_section_peak: float
    mean_hold_beats: float
    p95_hold_beats: float
    hold_occupancy_ratio: float
    overlap_input_load: float
    release_load: float

    @classmethod
    def empty(cls, section_count: int) -> DifficultyVectorV2:
        if section_count < 0:
            raise ValueError("section_count must not be negative")
        return cls(
            version=DIFFICULTY_VECTOR_VERSION,
            density_strain=0.0,
            jack_strain=0.0,
            chord_load=0.0,
            ln_strain=0.0,
            coordination=0.0,
            peak_skill=0.0,
            bounded_stamina=0.0,
            ordering_score=0.0,
            section_peaks=(0.0,) * section_count,
            weighted_peak_sum=0.0,
            max_section_peak=0.0,
            mean_hold_beats=0.0,
            p95_hold_beats=0.0,
            hold_occupancy_ratio=0.0,
            overlap_input_load=0.0,
            release_load=0.0,
        )

    def numeric_axes(self) -> tuple[float, ...]:
        return (
            self.density_strain,
            self.jack_strain,
            self.chord_load,
            self.ln_strain,
            self.coordination,
            self.peak_skill,
            self.bounded_stamina,
            self.ordering_score,
        )

    def to_report(self) -> dict[str, object]:
        return {
            "version": self.version,
            "densityStrain": self.density_strain,
            "jackStrain": self.jack_strain,
            "chordLoad": self.chord_load,
            "lnStrain": self.ln_strain,
            "coordination": self.coordination,
            "peakSkill": self.peak_skill,
            "boundedStamina": self.bounded_stamina,
            "orderingScore": self.ordering_score,
            "sectionPeaks": list(self.section_peaks),
            "weightedPeakSum": self.weighted_peak_sum,
            "maxSectionPeak": self.max_section_peak,
            "meanHoldBeats": self.mean_hold_beats,
            "p95HoldBeats": self.p95_hold_beats,
            "holdOccupancyRatio": self.hold_occupancy_ratio,
            "overlapInputLoad": self.overlap_input_load,
            "releaseLoad": self.release_load,
        }


def _coordination_section_peaks(index: ChartEventIndex) -> tuple[float, ...]:
    roles = _ROLES_BY_KEY_MODE.get(index.key_mode)
    if roles is None:
        raise ValueError(f"unsupported key_mode: {index.key_mode}")
    peaks = [0.0] * len(index.section_rows)
    previous_roles: frozenset[str] | None = None
    previous_time_ms: int | None = None
    strain = 0.0
    role_count = len(set(roles))
    for row in index.rows:
        current_roles = frozenset(roles[lane] for lane in row.lanes)
        simultaneous = (
            (len(current_roles) - 1) / (role_count - 1)
            if role_count > 1
            else 0.0
        )
        transition = (
            len(current_roles ^ previous_roles) / len(current_roles | previous_roles)
            if previous_roles and current_roles | previous_roles
            else 0.0
        )
        delta_ms = (
            row.time_ms - previous_time_ms
            if previous_time_ms is not None
            else row.time_ms
        )
        # Use the official mania overall decay base so repeated hand changes grow
        # strain while isolated changes decay. A per-section binary maximum
        # saturates to the geometric limit on ordinary multi-minute charts.
        strain = _decay(strain, delta_ms, OVERALL_DECAY_BASE)
        strain += simultaneous + transition
        section = min(len(peaks) - 1, row.time_ms // STRAIN_SECTION_MS)
        peaks[section] = max(peaks[section], strain)
        previous_roles = current_roles
        previous_time_ms = row.time_ms
    return tuple(peaks)


def _ordering_score(
    vector_values: dict[str, float],
    peak_skill: float,
    calibration: DifficultyCalibration | None,
    *,
    key_mode: int,
) -> float:
    if calibration is None:
        return peak_skill
    if calibration.key_mode != key_mode:
        raise ValueError("calibration key_mode does not match chart key_mode")
    standardized = []
    for name, median, iqr in zip(
        calibration.axis_names,
        calibration.medians,
        calibration.iqrs,
        strict=True,
    ):
        if name not in vector_values:
            raise ValueError(f"unsupported calibration axis: {name}")
        standardized.append((vector_values[name] - median) / iqr)
    return sum(
        value * weight
        for value, weight in zip(
            standardized,
            calibration.weights,
            strict=True,
        )
    )


def measure_difficulty_vector(
    index: ChartEventIndex,
    tempo_map: LocalTempoMap,
    *,
    calibration: DifficultyCalibration | None = None,
) -> DifficultyVectorV2:
    """Measure official-style strain plus transparent project diagnostic axes."""
    section_count = len(index.section_rows)
    if not index.notes:
        return DifficultyVectorV2.empty(section_count)

    total_peaks = [0.0] * section_count
    overall_peaks = [0.0] * section_count
    individual_peaks = [0.0] * section_count
    ln_peaks = [0.0] * section_count
    chord_peaks = [0.0] * section_count
    individual_strains = [0.0] * index.key_mode
    previous_by_lane: list[NoteEvent | None] = [None] * index.key_mode
    previous_start_by_lane: list[int | None] = [None] * index.key_mode
    overall_strain = 1.0
    highest_individual = 0.0
    previous_time: int | None = None
    max_overlap_input = 0.0
    max_release_load = 0.0

    row_sizes = {row.time_ms: row.size for row in index.rows}
    for note in index.notes:
        hold_factor, hold_addition = _official_hold_terms(
            note,
            tuple(previous_by_lane),
        )
        lane_previous = previous_start_by_lane[note.lane]
        column_delta = note.time_ms - lane_previous if lane_previous is not None else note.time_ms
        individual_strains[note.lane] = _decay(
            individual_strains[note.lane],
            column_delta,
            INDIVIDUAL_DECAY_BASE,
        ) + 2.0 * hold_factor
        delta_ms = note.time_ms - previous_time if previous_time is not None else 0
        if previous_time is not None and delta_ms <= 1:
            highest_individual = max(
                highest_individual,
                individual_strains[note.lane],
            )
        else:
            highest_individual = individual_strains[note.lane]
        overall_strain = _decay(
            overall_strain,
            delta_ms,
            OVERALL_DECAY_BASE,
        ) + (1.0 + hold_addition) * hold_factor
        total_strain = highest_individual + overall_strain
        section = min(section_count - 1, note.time_ms // STRAIN_SECTION_MS)
        total_peaks[section] = max(total_peaks[section], total_strain)
        overall_peaks[section] = max(overall_peaks[section], overall_strain)
        individual_peaks[section] = max(
            individual_peaks[section],
            highest_individual,
        )
        ln_peaks[section] = max(
            ln_peaks[section],
            (hold_factor - 1.0) + hold_addition,
        )
        chord_peaks[section] = max(
            chord_peaks[section],
            (row_sizes[note.time_ms] - 1) / max(1, index.key_mode - 1),
        )
        max_overlap_input = max(max_overlap_input, hold_factor - 1.0)
        max_release_load = max(max_release_load, hold_addition)
        previous_by_lane[note.lane] = note
        previous_start_by_lane[note.lane] = note.time_ms
        previous_time = note.time_ms

    full_peaks = tuple(total_peaks)
    peak_skill = _weighted_peak_sum(full_peaks)
    max_peak = max(full_peaks, default=0.0)
    stamina = peak_skill / max_peak if max_peak > 0 else 0.0
    hold_beats = tuple(
        tempo_map.beats_between(hold.start_ms, hold.end_ms)
        for hold in index.holds
    )
    mean_hold_beats = float(np.mean(hold_beats)) if hold_beats else 0.0
    p95_hold_beats = float(np.percentile(hold_beats, 95)) if hold_beats else 0.0
    hold_occupancy = (
        sum(hold.end_ms - hold.start_ms for hold in index.holds)
        / (index.duration_ms * index.key_mode)
    )
    coordination = _weighted_peak_sum(_coordination_section_peaks(index))
    raw_values = {
        "density_strain": _weighted_peak_sum(tuple(overall_peaks)),
        "jack_strain": _weighted_peak_sum(tuple(individual_peaks)),
        "chord_load": _weighted_peak_sum(tuple(chord_peaks)),
        "ln_strain": _weighted_peak_sum(tuple(ln_peaks)),
        "coordination": coordination,
        "peak_skill": peak_skill,
        "bounded_stamina": stamina,
        "mean_hold_beats": mean_hold_beats,
        "p95_hold_beats": p95_hold_beats,
        "hold_occupancy_ratio": hold_occupancy,
        "overlap_input_load": max_overlap_input,
        "release_load": max_release_load,
    }
    ordering = _ordering_score(
        raw_values,
        peak_skill,
        calibration,
        key_mode=index.key_mode,
    )
    return DifficultyVectorV2(
        version=DIFFICULTY_VECTOR_VERSION,
        density_strain=_rounded(raw_values["density_strain"]),
        jack_strain=_rounded(raw_values["jack_strain"]),
        chord_load=_rounded(raw_values["chord_load"]),
        ln_strain=_rounded(raw_values["ln_strain"]),
        coordination=_rounded(coordination),
        peak_skill=_rounded(peak_skill),
        bounded_stamina=_rounded(min(10.0, stamina)),
        ordering_score=_rounded(ordering),
        section_peaks=tuple(_rounded(value) for value in full_peaks),
        weighted_peak_sum=_rounded(peak_skill),
        max_section_peak=_rounded(max_peak),
        mean_hold_beats=_rounded(mean_hold_beats),
        p95_hold_beats=_rounded(p95_hold_beats),
        hold_occupancy_ratio=_rounded(hold_occupancy),
        overlap_input_load=_rounded(max_overlap_input),
        release_load=_rounded(max_release_load),
    )
