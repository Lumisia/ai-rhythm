"""Evidence-based classification of the earliest audible rhythmic phrase.

The classifier deliberately does not use an absolute loudness or onset-strength
cutoff.  It combines song-relative attack rank, per-band rank, continuation on
the selected timing grid, and the timing-grid phase itself.  A lone effect is
therefore evidence to review, never enough by itself to rewrite generation.
"""

from dataclasses import dataclass
from typing import Literal

import numpy as np

from chart_worker.analysis.leading_silence import consensus_leading_boundary_ms
from chart_worker.analysis.onset import OnsetAnalysis
from chart_worker.generation.osu_parser import OsuBpmEvent

IntroAnchorStatus = Literal["CONFIRMED", "UNCERTAIN", "NON_RHYTHMIC"]

GRID_SUPPORT_WINDOW_MS = 70
"""Same evidence window used by local timing diagnostics."""

CONTINUATION_STEPS = 4
"""Inspect the next four half-beat positions (two beats)."""

INTRO_SEARCH_HALF_BEAT_STEPS = 16
"""Search at most eight local beats beyond the non-negative timing origin."""


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
    supported_pulse_ms: tuple[int, ...] = ()

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
            "supportedPulseMs": list(self.supported_pulse_ms),
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
    time_ms: int, events: tuple[OsuBpmEvent, ...]
) -> tuple[int, int, float]:
    """Return the nearest tick on the timing segment active at ``time_ms``.

    An uninherited osu! timing point establishes both a local BPM and a local
    phase.  Reusing the first event after a later timing point can therefore
    snap a valid variable-BPM intro to the wrong half-beat family.
    """

    local_event = events[0]
    for event in events:
        if event.time_ms > time_ms:
            break
        local_event = event
    half_beat_ms = 30_000.0 / local_event.bpm
    steps = round((time_ms - local_event.time_ms) / half_beat_ms)
    raw_grid_ms = round(local_event.time_ms + steps * half_beat_ms)
    # Negative note coordinates cannot be represented by our chart schema.  An
    # onset close to the extrapolated negative tick belongs to the allowed 0 ms
    # pre-roll boundary instead (Ignite: -24 ms tick, 21 ms detected attack).
    grid_ms = max(0, raw_grid_ms)
    return grid_ms, abs(time_ms - grid_ms), half_beat_ms


def _has_onset_near(onsets: tuple[int, ...], target_ms: float) -> bool:
    return any(abs(onset - target_ms) <= GRID_SUPPORT_WINDOW_MS for onset in onsets)


def _time_after_half_beats(
    start_ms: int,
    half_beat_steps: int,
    events: tuple[OsuBpmEvent, ...],
) -> int:
    """Advance over local BPM boundaries instead of freezing the first BPM."""

    current_ms = float(start_ms)
    remaining_beats = half_beat_steps * 0.5
    event_index = max(
        (index for index, event in enumerate(events) if event.time_ms <= start_ms),
        default=0,
    )
    while remaining_beats > 0:
        event = events[event_index]
        beat_ms = 60_000.0 / event.bpm
        if event_index + 1 >= len(events):
            current_ms += remaining_beats * beat_ms
            break
        next_event_ms = float(events[event_index + 1].time_ms)
        if next_event_ms <= current_ms:
            event_index += 1
            continue
        available_beats = (next_event_ms - current_ms) / beat_ms
        if remaining_beats <= available_beats:
            current_ms += remaining_beats * beat_ms
            break
        remaining_beats -= available_beats
        current_ms = next_event_ms
        event_index += 1
    return round(current_ms)


def _local_half_beat_targets(
    start_ms: int,
    count: int,
    events: tuple[OsuBpmEvent, ...],
) -> tuple[int, ...]:
    """Enumerate grid opportunities while respecting timing-point phase resets."""

    event_index = max(
        (index for index, event in enumerate(events) if event.time_ms <= start_ms),
        default=0,
    )
    cursor_ms = float(start_ms)
    targets = []
    while len(targets) < count:
        event = events[event_index]
        next_tick_ms = cursor_ms + 30_000.0 / event.bpm
        if (
            event_index + 1 < len(events)
            and events[event_index + 1].time_ms <= next_tick_ms
        ):
            event_index += 1
            cursor_ms = float(events[event_index].time_ms)
        else:
            cursor_ms = next_tick_ms
        targets.append(round(cursor_ms))
    return tuple(targets)


def classify_intro_anchor(
    events: tuple[OsuBpmEvent, ...],
    analysis: OnsetAnalysis,
    *,
    duration_ms: int,
) -> IntroAnchorEvidence:
    """Classify the earliest supported attack in a bounded local-beat horizon."""
    if not events:
        raise ValueError("intro anchor classification requires at least one event")
    if duration_ms < 0:
        raise ValueError("duration_ms must be non-negative")

    first_event = events[0]
    leading_silence_end_ms = (
        consensus_leading_boundary_ms(analysis.leading_silence)
        if analysis.leading_silence is not None
        else None
    )
    search_origin_ms = max(
        0,
        first_event.time_ms,
        leading_silence_end_ms or 0,
    )
    leading_end_ms = min(
        duration_ms,
        _time_after_half_beats(
            search_origin_ms,
            INTRO_SEARCH_HALF_BEAT_STEPS,
            events,
        ),
    )
    leading_onsets = tuple(
        onset for onset in analysis.onset_ms if 0 <= onset < leading_end_ms
    )
    if analysis.activity is None:
        candidate_onsets = leading_onsets
    else:
        active = set(analysis.activity.active_onset_ms)
        active_leading = tuple(onset for onset in leading_onsets if onset in active)
        if not active_leading:
            candidate_onsets = ()
        else:
            candidate_onsets = tuple(
                onset
                for onset in leading_onsets
                if onset in active
                or analysis.activity.active_frame_ratio(
                    max(0, onset - GRID_SUPPORT_WINDOW_MS),
                    onset + GRID_SUPPORT_WINDOW_MS,
                )
                > 0.0
            )
    if not candidate_onsets:
        return IntroAnchorEvidence(
            status="NON_RHYTHMIC",
            anchor_ms=None,
            anchor_grid_ms=None,
            grid_distance_ms=None,
            aggregate_percentile_rank=None,
            prominent_band_count=0,
            pulse_continuation_matches=0,
            pulse_continuation_opportunities=0,
            supported_pulse_ms=(),
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
    for anchor_ms in candidate_onsets:
        window = analysis.window_of(anchor_ms)
        aggregate = analysis.strength_at(anchor_ms)
        aggregate_rank = _percentile_rank(aggregate_values, aggregate)
        anchor_bands = analysis.band_strength[:, window].max(axis=1)
        prominent_band_count = sum(
            _percentile_rank(band_values[:, band], float(anchor_bands[band])) >= 0.9
            for band in range(anchor_bands.size)
        )
        anchor_grid_ms, grid_distance_ms, _half_beat_ms = _nearest_half_beat(
            anchor_ms, events
        )
        region_pulse_targets = _local_half_beat_targets(
            anchor_grid_ms,
            INTRO_SEARCH_HALF_BEAT_STEPS,
            events,
        )
        grid_supported = _has_onset_near(measured_onsets, anchor_grid_ms)
        supported_pulses = (
            ((anchor_grid_ms,) if grid_supported else ())
            + tuple(
                target_ms
                for target_ms in region_pulse_targets
                if _has_onset_near(measured_onsets, target_ms)
            )
        )
        confirmation_targets = set(region_pulse_targets[:CONTINUATION_STEPS])
        continuation_matches = sum(
            target_ms in confirmation_targets
            for target_ms in supported_pulses
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
            supported_pulse_ms=supported_pulses,
        )
        if uncertain is None:
            uncertain = evidence

        sequence_supported = (
            analysis.activity is not None and continuation_matches >= 2
        )
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
                supported_pulse_ms=evidence.supported_pulse_ms,
            )

    assert uncertain is not None
    return uncertain
