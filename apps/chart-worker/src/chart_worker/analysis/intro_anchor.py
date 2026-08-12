"""Evidence-based classification of audible rhythm before the timing origin.

The classifier deliberately does not use an absolute loudness or onset-strength
cutoff.  It combines song-relative attack rank, per-band rank, continuation on
the selected timing grid, and the timing-grid phase itself.  A lone effect is
therefore evidence to review, never enough by itself to rewrite generation.
"""

from dataclasses import dataclass
from typing import Literal

import numpy as np

from chart_worker.analysis.onset import OnsetAnalysis
from chart_worker.generation.osu_parser import OsuBpmEvent

IntroAnchorStatus = Literal["CONFIRMED", "UNCERTAIN", "NON_RHYTHMIC"]

GRID_SUPPORT_WINDOW_MS = 70
"""Same evidence window used by local timing diagnostics."""

CONTINUATION_STEPS = 4
"""Inspect the next four half-beat positions (two beats)."""


@dataclass(frozen=True, slots=True)
class IntroAnchorEvidence:
    status: IntroAnchorStatus
    anchor_ms: int | None
    anchor_grid_ms: int | None
    grid_distance_ms: int | None
    aggregate_percentile_rank: float | None
    prominent_band_count: int
    pulse_continuation_matches: int
    pulse_continuation_opportunities: int

    def to_report(self) -> dict[str, object]:
        return {
            "status": self.status,
            "anchorMs": self.anchor_ms,
            "anchorGridMs": self.anchor_grid_ms,
            "gridDistanceMs": self.grid_distance_ms,
            "aggregatePercentileRank": self.aggregate_percentile_rank,
            "prominentBandCount": self.prominent_band_count,
            "pulseContinuationMatches": self.pulse_continuation_matches,
            "pulseContinuationOpportunities": self.pulse_continuation_opportunities,
        }


def _percentile_rank(values: np.ndarray, value: float) -> float:
    if values.size == 0:
        return 0.0
    # Mid-rank prevents a flat envelope (all values tied) from pretending that
    # every onset is in the top percentile.
    lower = np.count_nonzero(values < value)
    equal = np.count_nonzero(values == value)
    return float((lower + 0.5 * equal) / values.size)


def _nearest_half_beat(
    time_ms: int, first_event: OsuBpmEvent
) -> tuple[int, int, float]:
    half_beat_ms = 30_000.0 / first_event.bpm
    steps = round((time_ms - first_event.time_ms) / half_beat_ms)
    raw_grid_ms = round(first_event.time_ms + steps * half_beat_ms)
    # Negative note coordinates cannot be represented by our chart schema.  An
    # onset close to the extrapolated negative tick belongs to the allowed 0 ms
    # pre-roll boundary instead (Ignite: -24 ms tick, 21 ms detected attack).
    grid_ms = max(0, raw_grid_ms)
    return grid_ms, abs(time_ms - grid_ms), half_beat_ms


def _has_onset_near(onsets: tuple[int, ...], target_ms: float) -> bool:
    return any(abs(onset - target_ms) <= GRID_SUPPORT_WINDOW_MS for onset in onsets)


def classify_intro_anchor(
    events: tuple[OsuBpmEvent, ...],
    analysis: OnsetAnalysis,
    *,
    duration_ms: int,
) -> IntroAnchorEvidence:
    """Classify the earliest supported rhythmic attack before the first event."""
    if not events:
        raise ValueError("intro anchor classification requires at least one event")
    if duration_ms < 0:
        raise ValueError("duration_ms must be non-negative")

    first_event = events[0]
    leading_end_ms = min(duration_ms, max(0, first_event.time_ms))
    leading_onsets = tuple(
        onset for onset in analysis.onset_ms if 0 <= onset < leading_end_ms
    )
    if analysis.activity is None:
        active_onsets = leading_onsets
    else:
        active = set(analysis.activity.active_onset_ms)
        active_onsets = tuple(onset for onset in leading_onsets if onset in active)
    if not active_onsets:
        return IntroAnchorEvidence(
            status="NON_RHYTHMIC",
            anchor_ms=None,
            anchor_grid_ms=None,
            grid_distance_ms=None,
            aggregate_percentile_rank=None,
            prominent_band_count=0,
            pulse_continuation_matches=0,
            pulse_continuation_opportunities=0,
        )

    measured_onsets = tuple(
        onset for onset in analysis.onset_ms if 0 <= onset < duration_ms
    )
    aggregate_values = np.asarray(
        [analysis.strength_at(onset) for onset in measured_onsets], dtype=np.float64
    )
    band_values = np.asarray(
        [
            analysis.band_strength[:, analysis.window_of(onset)].max(axis=1)
            for onset in measured_onsets
        ],
        dtype=np.float64,
    )

    uncertain: IntroAnchorEvidence | None = None
    for anchor_ms in active_onsets:
        window = analysis.window_of(anchor_ms)
        aggregate = analysis.strength_at(anchor_ms)
        aggregate_rank = _percentile_rank(aggregate_values, aggregate)
        anchor_bands = analysis.band_strength[:, window].max(axis=1)
        prominent_band_count = sum(
            _percentile_rank(band_values[:, band], float(anchor_bands[band])) >= 0.9
            for band in range(anchor_bands.size)
        )
        anchor_grid_ms, grid_distance_ms, half_beat_ms = _nearest_half_beat(
            anchor_ms, first_event
        )
        continuation_matches = sum(
            _has_onset_near(
                measured_onsets,
                anchor_ms + step * half_beat_ms,
            )
            for step in range(1, CONTINUATION_STEPS + 1)
        )

        evidence = IntroAnchorEvidence(
            status="UNCERTAIN",
            anchor_ms=anchor_ms,
            anchor_grid_ms=anchor_grid_ms,
            grid_distance_ms=grid_distance_ms,
            aggregate_percentile_rank=round(aggregate_rank, 6),
            prominent_band_count=prominent_band_count,
            pulse_continuation_matches=continuation_matches,
            pulse_continuation_opportunities=CONTINUATION_STEPS,
        )
        if uncertain is None:
            uncertain = evidence

        grid_supported = grid_distance_ms <= GRID_SUPPORT_WINDOW_MS
        sequence_supported = continuation_matches >= 2
        broadband_attack = aggregate_rank >= 0.9 and prominent_band_count >= 2
        if grid_supported and (sequence_supported or broadband_attack):
            return IntroAnchorEvidence(
                status="CONFIRMED",
                anchor_ms=evidence.anchor_ms,
                anchor_grid_ms=evidence.anchor_grid_ms,
                grid_distance_ms=evidence.grid_distance_ms,
                aggregate_percentile_rank=evidence.aggregate_percentile_rank,
                prominent_band_count=evidence.prominent_band_count,
                pulse_continuation_matches=evidence.pulse_continuation_matches,
                pulse_continuation_opportunities=evidence.pulse_continuation_opportunities,
            )

    assert uncertain is not None
    return uncertain
