"""Read-only timing-grid diagnostics for generated timing candidates."""

from collections.abc import Iterable
from dataclasses import dataclass
from math import ceil
from typing import Literal

import numpy as np

from chart_worker.analysis.onset import OnsetAnalysis
from chart_worker.generation.osu_parser import OsuBpmEvent

_DIVISORS = (1, 2, 3, 4, 6, 8)
_ALTERNATIVES = ("HALF", "DOUBLE")
MIN_PERIODICITY_SUPPORT = 0.10
Alternative = Literal["HALF", "DOUBLE"]
EvidenceStatus = Literal["SUFFICIENT", "INSUFFICIENT"]


@dataclass(frozen=True, slots=True)
class NoteGridMetrics:
    unique_row_count: int
    clean_row_count: int
    clean_rate: float
    absolute_p95_beats: float


@dataclass(frozen=True, slots=True)
class TempoCandidateMetrics:
    base_pulse_support: float
    half_pulse_support: float
    double_pulse_support: float
    base_supported_pulses: int
    half_supported_pulses: int
    double_supported_pulses: int
    pulse_best_alternative: Alternative | None
    pulse_alternative_margin: float
    base_periodicity_support: float
    half_periodicity_support: float
    double_periodicity_support: float
    periodicity_frame_count: int
    periodicity_best_alternative: Alternative | None
    periodicity_margin: float
    evidence_agrees: bool
    evidence_status: EvidenceStatus

    def to_report(self) -> dict[str, object]:
        """Serialize the stable timing-candidate evidence schema."""
        return {
            "basePulseSupport": self.base_pulse_support,
            "halfPulseSupport": self.half_pulse_support,
            "doublePulseSupport": self.double_pulse_support,
            "baseSupportedPulses": self.base_supported_pulses,
            "halfSupportedPulses": self.half_supported_pulses,
            "doubleSupportedPulses": self.double_supported_pulses,
            "pulseBestAlternative": self.pulse_best_alternative,
            "pulseAlternativeMargin": self.pulse_alternative_margin,
            "basePeriodicitySupport": self.base_periodicity_support,
            "halfPeriodicitySupport": self.half_periodicity_support,
            "doublePeriodicitySupport": self.double_periodicity_support,
            "periodicityFrameCount": self.periodicity_frame_count,
            "periodicityBestAlternative": self.periodicity_best_alternative,
            "periodicityMargin": self.periodicity_margin,
            "evidenceAgrees": self.evidence_agrees,
            "evidenceStatus": self.evidence_status,
        }


def _event_index(events: tuple[OsuBpmEvent, ...], time_ms: int) -> int:
    return int(np.searchsorted([event.time_ms for event in events], time_ms, side="right")) - 1


def measure_note_grid_alignment(
    rows: Iterable[int], bpm_events: tuple[OsuBpmEvent, ...]
) -> NoteGridMetrics:
    """Measure note rows against supported subdivisions without changing any row."""
    unique_rows = sorted(set(rows))
    grid = np.unique(
        np.concatenate([np.arange(divisor + 1, dtype=float) / divisor for divisor in _DIVISORS])
    )
    errors: list[float] = []
    for row in unique_rows:
        index = _event_index(bpm_events, row)
        if index < 0:
            errors.append(0.5)
            continue
        event = bpm_events[index]
        beat_position = ((row - event.time_ms) * event.bpm / 60_000.0) % 1.0
        errors.append(float(np.min(np.abs(grid - beat_position))))
    clean_count = sum(error <= 0.025 for error in errors)
    return NoteGridMetrics(
        unique_row_count=len(unique_rows),
        clean_row_count=clean_count,
        clean_rate=clean_count / len(errors) if errors else 0.0,
        absolute_p95_beats=float(np.percentile(errors, 95)) if errors else 0.0,
    )


def _candidate_periods(beat_ms: float) -> dict[str, float]:
    return {"BASE": beat_ms, "HALF": beat_ms * 2.0, "DOUBLE": beat_ms / 2.0}


def _analysis_duration_ms(analysis: OnsetAnalysis) -> float:
    return max(0.0, (analysis.frame_count - 1) * analysis.frame_ms)


def _segments(
    events: tuple[OsuBpmEvent, ...], analysis: OnsetAnalysis
) -> Iterable[tuple[OsuBpmEvent, float, float]]:
    duration_ms = _analysis_duration_ms(analysis)
    for index, event in enumerate(events):
        next_event_ms = events[index + 1].time_ms if index + 1 < len(events) else duration_ms
        start_ms = max(0.0, float(event.time_ms))
        end_ms = min(duration_ms, float(next_event_ms))
        if end_ms > start_ms:
            yield event, start_ms, end_ms


def _pulses_in_segment(
    *, phase_ms: float, start_ms: float, end_ms: float, period_ms: float
) -> np.ndarray:
    first = phase_ms + ceil((start_ms - phase_ms) / period_ms) * period_ms
    return np.arange(first, end_ms, period_ms)


def _pulse_support(
    analysis: OnsetAnalysis, events: tuple[OsuBpmEvent, ...]
) -> tuple[dict[str, float], dict[str, int]]:
    weighted_support = {name: 0.0 for name in ("BASE", "HALF", "DOUBLE")}
    pulse_counts = {name: 0 for name in weighted_support}
    supported_counts = {name: 0 for name in weighted_support}
    for event, start_ms, end_ms in _segments(events, analysis):
        for name, period_ms in _candidate_periods(60_000.0 / event.bpm).items():
            pulses = _pulses_in_segment(
                phase_ms=event.time_ms,
                start_ms=start_ms,
                end_ms=end_ms,
                period_ms=period_ms,
            )
            values = [analysis.strength_at(round(float(pulse))) for pulse in pulses]
            weighted_support[name] += float(np.sum(values))
            pulse_counts[name] += len(values)
            supported_counts[name] += sum(value >= 0.25 for value in values)
    support = {
        name: weighted_support[name] / pulse_counts[name] if pulse_counts[name] else 0.0
        for name in weighted_support
    }
    return support, supported_counts


def _normalized_autocorrelation(values: np.ndarray, lag: int) -> float | None:
    if lag <= 0 or values.size <= lag:
        return None
    centered = values - float(np.mean(values))
    if float(np.linalg.norm(centered)) <= np.finfo(float).eps:
        return None
    left = centered[:-lag]
    right = centered[lag:]
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= 0.0:
        return None
    return float(np.clip(np.dot(left, right) / denominator, 0.0, 1.0))


def _periodicity_support(
    analysis: OnsetAnalysis, events: tuple[OsuBpmEvent, ...]
) -> tuple[dict[str, float], int]:
    total = {name: 0.0 for name in ("BASE", "HALF", "DOUBLE")}
    frames = 0
    frame_times = np.arange(analysis.frame_count) * analysis.frame_ms
    for event, start_ms, end_ms in _segments(events, analysis):
        periods = _candidate_periods(60_000.0 / event.bpm)
        if end_ms - start_ms < 4.0 * periods["HALF"]:
            continue
        mask = (frame_times >= start_ms) & (frame_times < end_ms)
        segment = np.asarray(analysis.strength[mask], dtype=float)
        if not segment.size or float(np.linalg.norm(segment)) == 0.0:
            continue
        values: dict[str, float] = {}
        for name, period_ms in periods.items():
            correlation = _normalized_autocorrelation(
                segment, round(period_ms / analysis.frame_ms)
            )
            if correlation is None:
                values = {}
                break
            values[name] = correlation
        if not values:
            continue
        for name, value in values.items():
            total[name] += value * segment.size
        frames += int(segment.size)
    return (
        {name: total[name] / frames if frames else 0.0 for name in total},
        frames,
    )


def _best_alternative(
    support: dict[str, float]
) -> tuple[Alternative | None, float]:
    alternative = max(_ALTERNATIVES, key=lambda name: support[name])
    margin = support[alternative] - support["BASE"]
    if np.isclose(support["HALF"], support["DOUBLE"]):
        return None, margin
    return (alternative if margin > 0.0 else None), margin


def measure_tempo_candidates(
    bpm_events: tuple[OsuBpmEvent, ...], analysis: OnsetAnalysis
) -> TempoCandidateMetrics:
    """Compare full-beat base, half, and double pulse/periodicity evidence."""
    pulse, pulse_counts = _pulse_support(analysis, bpm_events)
    periodicity, periodicity_frames = _periodicity_support(analysis, bpm_events)
    pulse_best, pulse_margin = _best_alternative(pulse)
    periodicity_best, periodicity_margin = _best_alternative(periodicity)
    winner_supported = (
        pulse_counts[pulse_best] if pulse_best is not None else pulse_counts["BASE"]
    )
    status: EvidenceStatus = (
        "SUFFICIENT"
        if (
            periodicity_frames
            and max(periodicity.values()) >= MIN_PERIODICITY_SUPPORT
            and pulse_counts["BASE"] >= 16
            and winner_supported >= 16
        )
        else "INSUFFICIENT"
    )
    return TempoCandidateMetrics(
        base_pulse_support=pulse["BASE"],
        half_pulse_support=pulse["HALF"],
        double_pulse_support=pulse["DOUBLE"],
        base_supported_pulses=pulse_counts["BASE"],
        half_supported_pulses=pulse_counts["HALF"],
        double_supported_pulses=pulse_counts["DOUBLE"],
        pulse_best_alternative=pulse_best,
        pulse_alternative_margin=pulse_margin,
        base_periodicity_support=periodicity["BASE"],
        half_periodicity_support=periodicity["HALF"],
        double_periodicity_support=periodicity["DOUBLE"],
        periodicity_frame_count=periodicity_frames,
        periodicity_best_alternative=periodicity_best,
        periodicity_margin=periodicity_margin,
        evidence_agrees=pulse_best is not None and pulse_best == periodicity_best,
        evidence_status=status,
    )
