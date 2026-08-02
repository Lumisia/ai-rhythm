"""파이프라인 단계 사이의 명시적 데이터 계약."""

from dataclasses import dataclass
from pathlib import Path

from chart_worker.analysis.audio_io import AudioSignal
from chart_worker.analysis.beat import BeatGrid
from chart_worker.analysis.onset import OnsetAnalysis
from chart_worker.audio.normalize import NormalizedAudio
from chart_worker.generation.mapperatorinator import GeneratedChart


@dataclass(frozen=True, slots=True)
class AnalysisStageResult:
    normalized: NormalizedAudio
    signal: AudioSignal
    beat_grid: BeatGrid
    onsets: OnsetAnalysis
    timing_osu_path: Path


@dataclass(frozen=True, slots=True)
class GeneratedVariant:
    key_mode: int
    difficulty: str
    requested_star: float
    raw_osu_path: Path
    generated: GeneratedChart
