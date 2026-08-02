"""S1: 표준화, Beat This!, librosa, timing osu."""

from collections.abc import Callable
from pathlib import Path

from chart_worker.analysis.audio_io import AudioSignal, load_audio
from chart_worker.analysis.beat import BeatBackend, analyze_beats, beat_this_backend
from chart_worker.analysis.onset import OnsetBackend, analyze_onsets, librosa_backend
from chart_worker.analysis.timing import (
    TimingCandidate,
    TimingSource,
    TimingStatus,
    fit_piecewise_timing,
    match_times,
    project_beats,
)
from chart_worker.audio.normalize import NormalizedAudio, normalize_audio
from chart_worker.audio.runner import CommandResult
from chart_worker.config import WorkerConfig
from chart_worker.generation.timing_osu import timing_points_to_osu
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
    points = fit_piecewise_timing(grid)
    projected = project_beats(points, end_ms=normalized.duration_ms)
    metrics_20 = match_times(projected, grid.beat_ms, window_ms=20)
    metrics_50 = match_times(projected, grid.beat_ms, window_ms=50)
    passed = metrics_50.p95_abs_ms <= 30.0
    candidate = TimingCandidate(
        source=TimingSource.BEAT_THIS_PIECEWISE,
        points=points,
        projected_beat_ms=projected,
        f1_20ms=metrics_20.f1,
        f1_50ms=metrics_50.f1,
        p95_abs_ms=metrics_50.p95_abs_ms,
        status=TimingStatus.PASS if passed else TimingStatus.FAIL,
        reasons=() if passed else ("p95 exceeds 30ms",),
    )

    timing_path = run_dir / "analysis" / "timing.osu"
    timing_path.parent.mkdir(parents=True, exist_ok=True)
    timing_path.write_text(
        timing_points_to_osu(
            candidate.points,
            audio_filename=normalized.path.name,
            title=source.stem,
        ),
        encoding="utf-8",
    )
    return AnalysisStageResult(
        normalized=normalized,
        signal=signal,
        beat_grid=grid,
        onsets=onsets,
        timing_candidate=candidate,
        timing_osu_path=timing_path,
        timing_quality_report_path=run_dir / "analysis" / "timing-quality-v1.json",
    )
