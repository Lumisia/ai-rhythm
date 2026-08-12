"""Lazy, reusable Beat This file analyzer.

The dependency is optional.  Import and checkpoint loading happen only after
the timing selector has identified a genuinely close Standard/Super pair.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Protocol

import numpy as np

from chart_worker.analysis.beat import BeatGrid, build_beat_grid


class BeatThisModel(Protocol):
    def __call__(self, audio_path: Path) -> tuple[np.ndarray, np.ndarray]: ...


ModelFactory = Callable[[str, str, bool], BeatThisModel]


@lru_cache(maxsize=4)
def _cached_model(checkpoint: str, device: str, float16: bool) -> BeatThisModel:
    from beat_this.inference import File2Beats

    return File2Beats(
        checkpoint_path=checkpoint,
        device=device,
        float16=float16,
        dbn=False,
    )


@dataclass(slots=True)
class BeatThisFileAnalyzer:
    checkpoint: str = "final0"
    device: str = "cpu"
    float16: bool = False
    model_factory: ModelFactory = _cached_model
    _model: BeatThisModel | None = field(default=None, init=False, repr=False)

    def __call__(self, audio_path: Path) -> BeatGrid:
        if self._model is None:
            self._model = self.model_factory(
                self.checkpoint,
                self.device,
                self.float16,
            )
        beat_sec, downbeat_sec = self._model(audio_path)
        return build_beat_grid(
            np.asarray(beat_sec, dtype=np.float64),
            np.asarray(downbeat_sec, dtype=np.float64),
        )
