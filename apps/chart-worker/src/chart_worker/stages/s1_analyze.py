"""S1: 표준화, Beat This!, librosa, timing osu."""

from collections.abc import Callable
from pathlib import Path
from typing import Any

from chart_worker.analysis.audio_io import AudioSignal, load_audio
from chart_worker.analysis.beat import BeatBackend, analyze_beats, beat_this_backend
from chart_worker.analysis.onset import OnsetBackend, analyze_onsets, librosa_backend
from chart_worker.audio.normalize import NormalizedAudio, normalize_audio
from chart_worker.audio.runner import CommandResult
from chart_worker.config import WorkerConfig
from chart_worker.generation.timing_osu import beat_grid_to_timing_osu
from chart_worker.stages.types import AnalysisStageResult

NormalizeFn = Callable[..., NormalizedAudio]
AudioLoader = Callable[[Path], AudioSignal]
RunCommand = Callable[[list[str]], CommandResult]


def run_analysis(
    source: Path,
    run_dir: Path,
    *,
    config: WorkerConfig,
    run: RunCommand | None = None,
    normalizer: NormalizeFn = normalize_audio,
    audio_loader: AudioLoader = load_audio,
    beat_backend: BeatBackend | None = None,
    onset_backend: OnsetBackend | None = None,
) -> AnalysisStageResult:
    target = run_dir / "audio" / "game.flac"
    normalized = normalizer(source, target, config=config, run=run)
    signal = audio_loader(normalized.path)
    grid = analyze_beats(
        signal,
        backend=beat_backend or beat_this_backend(checkpoint="final0", device="cpu"),
    )
    onsets = analyze_onsets(signal, backend=onset_backend or librosa_backend())

    timing_path = run_dir / "analysis" / "timing.osu"
    timing_path.parent.mkdir(parents=True, exist_ok=True)
    timing_path.write_text(
        beat_grid_to_timing_osu(
            grid,
            audio_filename=normalized.path.name,
            title=source.stem,
        ),
        encoding="utf-8",
    )
    return AnalysisStageResult(normalized, signal, grid, onsets, timing_path)
