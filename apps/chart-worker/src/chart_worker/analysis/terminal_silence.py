"""Policy-free observation of consecutive silence at the end of canonical audio."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import Literal

import numpy as np

from chart_worker.analysis.audio_io import AudioSignal

DEFAULT_THRESHOLDS_DB: tuple[tuple[float, float], ...] = (
    (-72.0, -60.0),
    (-66.0, -54.0),
    (-60.0, -48.0),
)
"""Observation grid only; these values are not an enforcement policy."""

MIN_CONSENSUS_SUFFIX_MS = 3_000
"""Ignore codec padding and short production tails when enforcing a boundary."""


@dataclass(frozen=True, slots=True)
class TerminalThresholdCandidate:
    rms_db: float
    peak_db: float
    suffix_start_ms: int | None
    suffix_duration_ms: int


@dataclass(frozen=True, slots=True)
class TerminalSilenceObservation:
    version: Literal["terminal-silence-observation-v1"]
    duration_ms: int
    frame_ms: int
    channel_count: int
    candidates: tuple[TerminalThresholdCandidate, ...]
    candidate_spread_ms: int | None
    last_onset_ms: int | None

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
                    "suffixStartMs": candidate.suffix_start_ms,
                    "suffixDurationMs": candidate.suffix_duration_ms,
                }
                for candidate in self.candidates
            ],
            "candidateSpreadMs": self.candidate_spread_ms,
            "lastOnsetMs": self.last_onset_ms,
            "policyState": "OBSERVATION_ONLY",
            "mutatesGeneration": False,
        }


def consensus_terminal_boundary_ms(
    observation: TerminalSilenceObservation,
) -> int | None:
    """Return one conservative boundary only when every fixed detector agrees.

    This is deliberately narrower than semantic music-end detection.  It only
    recognizes a long, contiguous file-ending suffix whose stereo-aware RMS
    and peak measurements agree across the frozen threshold grid.
    """
    if not isinstance(observation, TerminalSilenceObservation):
        raise TypeError("observation must be a TerminalSilenceObservation")
    if type(observation.duration_ms) is not int or observation.duration_ms < 0:
        raise ValueError("terminal duration must be a non-negative exact integer")
    if type(observation.frame_ms) is not int or observation.frame_ms <= 0:
        raise ValueError("terminal frame size must be a positive exact integer")
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
    starts: list[int] = []
    for candidate in observation.candidates:
        if (
            type(candidate.suffix_start_ms) is not int
            or type(candidate.suffix_duration_ms) is not int
            or candidate.suffix_start_ms < 0
            or candidate.suffix_start_ms > observation.duration_ms
            or candidate.suffix_duration_ms
            != observation.duration_ms - candidate.suffix_start_ms
        ):
            return None
        starts.append(candidate.suffix_start_ms)
    latest_start_ms = max(starts)
    if max(starts) - min(starts) > observation.frame_ms * 2:
        return None
    if observation.duration_ms - latest_start_ms < MIN_CONSENSUS_SUFFIX_MS:
        return None
    if observation.last_onset_ms is not None:
        if type(observation.last_onset_ms) is not int:
            return None
        if observation.last_onset_ms > latest_start_ms:
            return None
    return latest_start_ms


def _amplitude_db(value: float) -> float:
    if value <= 0.0:
        return float("-inf")
    return 20.0 * float(np.log10(value))


def _validate_thresholds(
    thresholds_db: tuple[tuple[float, float], ...],
) -> tuple[tuple[float, float], ...]:
    if type(thresholds_db) is not tuple or not thresholds_db:
        raise TypeError("thresholds_db must be a non-empty tuple")
    observed: list[tuple[float, float]] = []
    for threshold in thresholds_db:
        if type(threshold) is not tuple or len(threshold) != 2:
            raise TypeError("each threshold must be an exact (rms_db, peak_db) tuple")
        rms_db, peak_db = threshold
        if type(rms_db) not in {int, float} or type(peak_db) not in {int, float}:
            raise TypeError("threshold values must be exact integers or floats")
        if not np.isfinite(rms_db) or not np.isfinite(peak_db):
            raise ValueError("threshold values must be finite")
        observed.append((float(rms_db), float(peak_db)))
    if len(set(observed)) != len(observed):
        raise ValueError("threshold candidates must be unique")
    return tuple(observed)


def observe_terminal_silence(
    signal: AudioSignal,
    *,
    frame_ms: int = 20,
    thresholds_db: tuple[tuple[float, float], ...] = DEFAULT_THRESHOLDS_DB,
    last_onset_ms: int | None = None,
) -> TerminalSilenceObservation:
    """Measure terminal silence candidates without deciding a music boundary.

    RMS and peak are computed per channel, then aggregated with ``max``.  This
    intentionally preserves a tail present in only one stereo channel and
    considers only the silence run contiguous with the end of the file.
    """
    if type(frame_ms) is not int or frame_ms <= 0:
        raise ValueError("frame_ms must be an exact positive integer")
    duration_ms = signal.duration_ms
    if last_onset_ms is not None:
        if type(last_onset_ms) is not int:
            raise TypeError("last_onset_ms must be an exact integer or None")
        if last_onset_ms < 0 or last_onset_ms > duration_ms:
            raise ValueError("last_onset_ms must lie within the audio duration")
    samples = np.asarray(signal.samples)
    if samples.ndim != 2 or samples.shape[0] == 0 or samples.shape[1] == 0:
        raise ValueError("audio samples must be a non-empty (frames, channels) array")
    if not np.issubdtype(samples.dtype, np.number) or not np.all(np.isfinite(samples)):
        raise ValueError("audio samples must be finite numbers")
    thresholds = _validate_thresholds(thresholds_db)
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

    candidates: list[TerminalThresholdCandidate] = []
    for rms_threshold, peak_threshold in thresholds:
        silent = (rms_db <= rms_threshold) & (peak_db <= peak_threshold)
        if not bool(silent[-1]):
            suffix_start_ms = None
            suffix_duration_ms = 0
        else:
            non_silent = np.flatnonzero(~silent)
            start_frame = int(non_silent[-1] + 1) if non_silent.size else 0
            start_sample = min(samples.shape[0], start_frame * frame_samples)
            suffix_start_ms = round(start_sample * 1_000 / signal.sample_rate_hz)
            suffix_duration_ms = max(0, duration_ms - suffix_start_ms)
        candidates.append(
            TerminalThresholdCandidate(
                rms_db=rms_threshold,
                peak_db=peak_threshold,
                suffix_start_ms=suffix_start_ms,
                suffix_duration_ms=suffix_duration_ms,
            )
        )

    starts = [
        candidate.suffix_start_ms
        for candidate in candidates
        if candidate.suffix_start_ms is not None
    ]
    spread = max(starts) - min(starts) if starts else None
    return TerminalSilenceObservation(
        version="terminal-silence-observation-v1",
        duration_ms=duration_ms,
        frame_ms=frame_ms,
        channel_count=signal.channels,
        candidates=tuple(candidates),
        candidate_spread_ms=spread,
        last_onset_ms=last_onset_ms,
    )
