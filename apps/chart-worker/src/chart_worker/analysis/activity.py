"""Song-relative audio activity used by timing diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, floor

import numpy as np

SILENCE_DB = -60.0
RMS_FLOOR_PERCENTILE = 20.0
ONSET_FLOOR_PERCENTILE = 25.0


@dataclass(frozen=True, slots=True)
class AudioActivity:
    frame_ms: float
    rms_db: np.ndarray
    floor_db: float
    active_onset_ms: tuple[int, ...]

    def active_frame_ratio(self, start_ms: int, end_ms: int) -> float:
        """Return the active RMS-frame share after clipping to available audio."""
        if self.frame_ms <= 0 or self.rms_db.size == 0 or end_ms <= start_ms:
            return 0.0
        start = max(0, floor(start_ms / self.frame_ms))
        end = min(self.rms_db.size, ceil(end_ms / self.frame_ms))
        if end <= start:
            return 0.0
        window = self.rms_db[start:end]
        return float(np.count_nonzero(window > self.floor_db) / window.size)


def _window_bounds(
    frame: int,
    frame_count: int,
    *,
    n_fft: int | None,
    hop_length: int | None,
) -> tuple[int, int]:
    if n_fft is None or hop_length is None:
        return frame, frame + 1
    ahead = max(1, ceil(n_fft / hop_length))
    return max(0, frame - 1), min(frame_count, frame + ahead + 1)


def build_audio_activity(
    *,
    rms_db: np.ndarray,
    normalized_strength: np.ndarray,
    onset_frames: np.ndarray,
    frame_ms: float,
    silence_db: float = SILENCE_DB,
    n_fft: int | None = None,
    hop_length: int | None = None,
) -> AudioActivity:
    """Classify detected onsets using song-relative RMS and onset floors."""
    rms = np.asarray(rms_db, dtype=np.float64).reshape(-1)
    strength = np.asarray(normalized_strength, dtype=np.float64).reshape(-1)
    frame_count = min(rms.size, strength.size)
    rms = rms[:frame_count]
    strength = strength[:frame_count]

    non_silent = rms[rms > silence_db]
    floor_db = (
        float(np.percentile(non_silent, RMS_FLOOR_PERCENTILE))
        if non_silent.size
        else float(silence_db)
    )

    candidates: list[tuple[int, float, float]] = []
    for raw_frame in np.asarray(onset_frames, dtype=np.int64).reshape(-1):
        frame = int(raw_frame)
        if frame < 0 or frame >= frame_count:
            continue
        start, end = _window_bounds(
            frame,
            frame_count,
            n_fft=n_fft,
            hop_length=hop_length,
        )
        candidates.append(
            (
                frame,
                float(np.max(rms[start:end])),
                float(np.max(strength[start:end])),
            )
        )

    if not candidates or not non_silent.size:
        active_onset_ms: tuple[int, ...] = ()
    else:
        onset_floor = float(
            np.percentile(
                np.asarray([candidate[2] for candidate in candidates]),
                ONSET_FLOOR_PERCENTILE,
            )
        )
        active_onset_ms = tuple(
            dict.fromkeys(
                round(frame * frame_ms)
                for frame, window_rms, window_strength in candidates
                if window_rms > floor_db
                and window_strength > 0
                and window_strength >= onset_floor
            )
        )

    return AudioActivity(
        frame_ms=float(frame_ms),
        rms_db=rms,
        floor_db=floor_db,
        active_onset_ms=active_onset_ms,
    )
