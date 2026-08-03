"""S1: normalize source audio without deriving replacement timing data."""

from collections.abc import Callable
from pathlib import Path
from typing import Any

from chart_worker.audio.normalize import NormalizedAudio, normalize_audio
from chart_worker.config import WorkerConfig
from chart_worker.stages.types import PreparedAudio

NormalizeAudio = Callable[..., NormalizedAudio]


def run_prepare(
    source: Path,
    run_dir: Path,
    *,
    config: WorkerConfig,
    run: Any = None,
    normalizer: NormalizeAudio = normalize_audio,
) -> PreparedAudio:
    """Normalize once; Mapperatorinator remains the timing and note authority."""
    target = run_dir / "audio" / "game.flac"
    normalized = normalizer(source, target, config=config, run=run)
    return PreparedAudio(normalized=normalized)
