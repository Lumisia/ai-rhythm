"""Song-level analysis inputs shared by every generated chart."""

from __future__ import annotations

import math
from bisect import bisect_right
from dataclasses import dataclass, field
from typing import Protocol

from chart_worker.analysis.intro_anchor import (
    IntroAnchorEvidence,
    classify_intro_anchor,
)
from chart_worker.analysis.onset import OnsetAnalysis
from chart_worker.generation.osu_parser import OsuBpmEvent


class TimingAuthorityLike(Protocol):
    bpm_events: tuple[OsuBpmEvent, ...]


@dataclass(frozen=True, slots=True)
class LocalTempoMap:
    """Normalized uninherited timing points with logarithmic lookup."""

    events: tuple[OsuBpmEvent, ...]
    times: tuple[int, ...] = field(init=False)

    def __post_init__(self) -> None:
        if not self.events:
            raise ValueError("tempo map requires at least one BPM event")
        by_time: dict[int, OsuBpmEvent] = {}
        for event in self.events:
            if not math.isfinite(event.bpm) or event.bpm <= 0:
                raise ValueError("bpm must be finite and positive")
            by_time[event.time_ms] = event
        normalized = tuple(by_time[time_ms] for time_ms in sorted(by_time))
        object.__setattr__(self, "events", normalized)
        object.__setattr__(
            self,
            "times",
            tuple(event.time_ms for event in normalized),
        )

    def at(self, time_ms: int) -> OsuBpmEvent:
        index = max(0, bisect_right(self.times, time_ms) - 1)
        return self.events[index]

    def beats_between(self, start_ms: int, end_ms: int) -> float:
        """Integrate beat count over every local BPM segment."""
        if end_ms < start_ms:
            raise ValueError("end_ms must be greater than or equal to start_ms")
        cursor = start_ms
        beats = 0.0
        while cursor < end_ms:
            event_index = max(0, bisect_right(self.times, cursor) - 1)
            next_index = event_index + 1
            boundary = (
                self.times[next_index]
                if next_index < len(self.times)
                else end_ms
            )
            segment_end = min(end_ms, max(cursor, boundary))
            if segment_end == cursor:
                segment_end = end_ms
            beats += (
                (segment_end - cursor)
                * self.events[event_index].bpm
                / 60_000.0
            )
            cursor = segment_end
        return beats


@dataclass(frozen=True, slots=True)
class SongAnalysisContext:
    """Expensive song evidence computed once and reused by all 12 charts."""

    duration_ms: int
    tempo_map: LocalTempoMap
    onset_analysis: OnsetAnalysis
    intro_anchor: IntroAnchorEvidence

    @classmethod
    def build(
        cls,
        authority: TimingAuthorityLike,
        onset_analysis: OnsetAnalysis,
        *,
        duration_ms: int,
        intro_anchor: IntroAnchorEvidence | None = None,
    ) -> SongAnalysisContext:
        if duration_ms <= 0:
            raise ValueError("duration_ms must be positive")
        tempo_map = LocalTempoMap(authority.bpm_events)
        return cls(
            duration_ms=duration_ms,
            tempo_map=tempo_map,
            onset_analysis=onset_analysis,
            intro_anchor=(
                intro_anchor
                if intro_anchor is not None
                else classify_intro_anchor(
                    tempo_map.events,
                    onset_analysis,
                    duration_ms=duration_ms,
                )
            ),
        )
