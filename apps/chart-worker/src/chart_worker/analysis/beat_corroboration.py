"""Compare a timing candidate with an independent detected beat sequence."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import Literal

import numpy as np

from chart_worker.analysis.beat import BeatGrid
from chart_worker.analysis.event_matching import maximum_ordered_match_count
from chart_worker.generation.osu_parser import OsuBpmEvent

MetricalLevel = Literal["HALF", "BASE", "DOUBLE"]
METRICAL_FACTORS: dict[MetricalLevel, float] = {
    "HALF": 2.0,
    "BASE": 1.0,
    "DOUBLE": 0.5,
}

# mir_eval.beat defaults: ignore beats before 5 seconds and use a 70 ms
# F-measure window.  Keeping the same convention makes the diagnostic
# interpretable and avoids a project-specific magic tolerance.
BEAT_TRIM_START_MS = 5_000
BEAT_MATCH_WINDOW_MS = 70


@dataclass(frozen=True, slots=True)
class BeatCandidateCorroboration:
    best_metrical_level: MetricalLevel
    best_f1: float
    f1_by_level: dict[MetricalLevel, float]
    precision_by_level: dict[MetricalLevel, float]
    recall_by_level: dict[MetricalLevel, float]
    beat_count: int

    def to_report(self) -> dict[str, object]:
        return {
            "bestMetricalLevel": self.best_metrical_level,
            "bestF1": self.best_f1,
            "f1ByLevel": self.f1_by_level,
            "precisionByLevel": self.precision_by_level,
            "recallByLevel": self.recall_by_level,
            "beatCount": self.beat_count,
        }


def _candidate_pulses(
    events: tuple[OsuBpmEvent, ...],
    *,
    duration_ms: int,
    factor: float,
) -> np.ndarray:
    pulses: list[int] = []
    for index, event in enumerate(events):
        start_ms = max(BEAT_TRIM_START_MS, event.time_ms)
        end_ms = events[index + 1].time_ms if index + 1 < len(events) else duration_ms
        if end_ms <= start_ms:
            continue
        period_ms = 60_000.0 * factor / event.bpm
        first_step = ceil((start_ms - event.time_ms) / period_ms)
        pulse_ms = event.time_ms + first_step * period_ms
        while pulse_ms < end_ms:
            pulses.append(round(pulse_ms))
            pulse_ms += period_ms
    return np.asarray(sorted(set(pulses)), dtype=np.int64)


def measure_beat_corroboration(
    events: tuple[OsuBpmEvent, ...],
    beat_grid: BeatGrid,
    *,
    duration_ms: int,
) -> BeatCandidateCorroboration:
    if not events:
        raise ValueError("beat corroboration requires at least one timing event")
    if duration_ms <= 0:
        raise ValueError("duration_ms must be positive")
    beats = np.asarray(
        [beat for beat in beat_grid.beat_ms if BEAT_TRIM_START_MS <= beat < duration_ms],
        dtype=np.int64,
    )
    if beats.size < 2:
        raise ValueError("beat corroboration requires at least two trimmed beats")

    f1_by_level: dict[MetricalLevel, float] = {}
    precision_by_level: dict[MetricalLevel, float] = {}
    recall_by_level: dict[MetricalLevel, float] = {}
    for level, factor in METRICAL_FACTORS.items():
        pulses = _candidate_pulses(events, duration_ms=duration_ms, factor=factor)
        matched = maximum_ordered_match_count(
            pulses,
            beats,
            window_ms=BEAT_MATCH_WINDOW_MS,
        )
        precision = matched / pulses.size if pulses.size else 0.0
        recall = matched / beats.size
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        precision_by_level[level] = round(float(precision), 6)
        recall_by_level[level] = round(float(recall), 6)
        f1_by_level[level] = round(float(f1), 6)
    best_level = max(
        METRICAL_FACTORS,
        key=lambda level: (f1_by_level[level], level == "BASE"),
    )
    return BeatCandidateCorroboration(
        best_metrical_level=best_level,
        best_f1=f1_by_level[best_level],
        f1_by_level=f1_by_level,
        precision_by_level=precision_by_level,
        recall_by_level=recall_by_level,
        beat_count=int(beats.size),
    )
