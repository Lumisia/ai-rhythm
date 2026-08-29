"""Policy-free observation of consecutive silence at canonical-audio start."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import Literal

import numpy as np

from chart_worker.analysis.audio_io import AudioSignal
from chart_worker.analysis.terminal_silence import DEFAULT_THRESHOLDS_DB

MIN_CONSENSUS_PREFIX_MS = 3_000
"""Ignore short production lead-ins and codec padding as a hard lower bound."""


@dataclass(frozen=True, slots=True)
class LeadingThresholdCandidate:
    rms_db: float
    peak_db: float
    prefix_end_ms: int | None
    prefix_duration_ms: int


@dataclass(frozen=True, slots=True)
class LeadingSilenceObservation:
    version: Literal["leading-silence-observation-v1"]
    duration_ms: int
    frame_ms: int
    channel_count: int
    candidates: tuple[LeadingThresholdCandidate, ...]
    candidate_spread_ms: int | None
    first_onset_ms: int | None

    def to_report(self) -> dict[str, object]:
        return {
            "version": self.version,
            "durationMs": self.duration_ms,
            "frameMs": self.frame_ms,
            "channelCount": self.channel_count,
            "candidates": [
                {
                    "rmsDb": candidate.rms_db,
                    "peakDb": candidate.peak_db,
                    "prefixEndMs": candidate.prefix_end_ms,
                    "prefixDurationMs": candidate.prefix_duration_ms,
                }
                for candidate in self.candidates
            ],
            "candidateSpreadMs": self.candidate_spread_ms,
            "firstOnsetMs": self.first_onset_ms,
            "policyState": "OBSERVATION_ONLY",
            "mutatesGeneration": False,
        }


def consensus_leading_boundary_ms(
    observation: LeadingSilenceObservation,
) -> int | None:
    """Return the end of the silent prefix shared by every fixed detector.

    Prefixes all begin at zero, so their temporal intersection ends at the
    earliest candidate end.  This is intentionally only a lower bound for
    chart objects; it is not a claim about the first musical beat.
    """

    if not isinstance(observation, LeadingSilenceObservation):
        raise TypeError("observation must be a LeadingSilenceObservation")
    if type(observation.duration_ms) is not int or observation.duration_ms < 0:
        raise ValueError("leading duration must be a non-negative exact integer")
    if type(observation.frame_ms) is not int or observation.frame_ms <= 0:
        raise ValueError("leading frame size must be a positive exact integer")
    expected_thresholds = set(DEFAULT_THRESHOLDS_DB)
    actual_thresholds = {
        (candidate.rms_db, candidate.peak_db)
        for candidate in observation.candidates
    }
    if (
        len(observation.candidates) != len(DEFAULT_THRESHOLDS_DB)
        or actual_thresholds != expected_thresholds
    ):
        return None

    ends: list[int] = []
    for candidate in observation.candidates:
        if (
            type(candidate.prefix_end_ms) is not int
            or type(candidate.prefix_duration_ms) is not int
            or candidate.prefix_end_ms < 0
            or candidate.prefix_end_ms > observation.duration_ms
            or candidate.prefix_duration_ms != candidate.prefix_end_ms
        ):
            return None
        ends.append(candidate.prefix_end_ms)
    common_end_ms = min(ends)
    if common_end_ms < MIN_CONSENSUS_PREFIX_MS:
        return None
    if observation.first_onset_ms is not None:
        if (
            type(observation.first_onset_ms) is not int
            or observation.first_onset_ms < 0
            or observation.first_onset_ms > observation.duration_ms
        ):
            return None
        if observation.first_onset_ms < common_end_ms:
            return None
    return common_end_ms


def _amplitude_db(value: float) -> float:
    if value <= 0.0:
        return float("-inf")
    return 20.0 * float(np.log10(value))


def observe_leading_silence(
    signal: AudioSignal,
    *,
    frame_ms: int = 20,
    thresholds_db: tuple[tuple[float, float], ...] = DEFAULT_THRESHOLDS_DB,
    first_onset_ms: int | None = None,
) -> LeadingSilenceObservation:
    """Measure only silence contiguous with the start of every audio channel."""

    if type(frame_ms) is not int or frame_ms <= 0:
        raise ValueError("frame_ms must be an exact positive integer")
    if type(thresholds_db) is not tuple or not thresholds_db:
        raise TypeError("thresholds_db must be a non-empty tuple")
    thresholds: list[tuple[float, float]] = []
    for threshold in thresholds_db:
        if type(threshold) is not tuple or len(threshold) != 2:
            raise TypeError("each threshold must be an exact pair")
        rms_db, peak_db = threshold
        if type(rms_db) not in {int, float} or type(peak_db) not in {int, float}:
            raise TypeError("threshold values must be exact numbers")
        if not np.isfinite(rms_db) or not np.isfinite(peak_db):
            raise ValueError("threshold values must be finite")
        thresholds.append((float(rms_db), float(peak_db)))
    if len(set(thresholds)) != len(thresholds):
        raise ValueError("threshold candidates must be unique")

    duration_ms = signal.duration_ms
    if first_onset_ms is not None:
        if type(first_onset_ms) is not int:
            raise TypeError("first_onset_ms must be an exact integer or None")
        if first_onset_ms < 0 or first_onset_ms > duration_ms:
            raise ValueError("first_onset_ms must lie within the audio duration")
    samples = np.asarray(signal.samples)
    if samples.ndim != 2 or samples.shape[0] == 0 or samples.shape[1] == 0:
        raise ValueError("audio samples must be a non-empty (frames, channels) array")
    if not np.issubdtype(samples.dtype, np.number) or not np.all(np.isfinite(samples)):
        raise ValueError("audio samples must be finite numbers")

    frame_samples = max(1, round(signal.sample_rate_hz * frame_ms / 1_000))
    frame_count = ceil(samples.shape[0] / frame_samples)
    rms_db = np.empty(frame_count, dtype=np.float64)
    peak_db = np.empty(frame_count, dtype=np.float64)
    for frame_index in range(frame_count):
        start = frame_index * frame_samples
        stop = min(samples.shape[0], start + frame_samples)
        window = samples[start:stop]
        per_channel_rms = np.sqrt(np.mean(np.square(window), axis=0))
        per_channel_peak = np.max(np.abs(window), axis=0)
        rms_db[frame_index] = _amplitude_db(float(np.max(per_channel_rms)))
        peak_db[frame_index] = _amplitude_db(float(np.max(per_channel_peak)))

    candidates: list[LeadingThresholdCandidate] = []
    for rms_threshold, peak_threshold in thresholds:
        silent = (rms_db <= rms_threshold) & (peak_db <= peak_threshold)
        if not bool(silent[0]):
            prefix_end_ms = None
            prefix_duration_ms = 0
        else:
            non_silent = np.flatnonzero(~silent)
            end_frame = int(non_silent[0]) if non_silent.size else frame_count
            end_sample = min(samples.shape[0], end_frame * frame_samples)
            prefix_end_ms = round(end_sample * 1_000 / signal.sample_rate_hz)
            prefix_duration_ms = prefix_end_ms
        candidates.append(
            LeadingThresholdCandidate(
                rms_db=rms_threshold,
                peak_db=peak_threshold,
                prefix_end_ms=prefix_end_ms,
                prefix_duration_ms=prefix_duration_ms,
            )
        )

    ends = [
        candidate.prefix_end_ms
        for candidate in candidates
        if candidate.prefix_end_ms is not None
    ]
    spread = max(ends) - min(ends) if ends else None
    return LeadingSilenceObservation(
        version="leading-silence-observation-v1",
        duration_ms=duration_ms,
        frame_ms=frame_ms,
        channel_count=signal.channels,
        candidates=tuple(candidates),
        candidate_spread_ms=spread,
        first_onset_ms=first_onset_ms,
    )
