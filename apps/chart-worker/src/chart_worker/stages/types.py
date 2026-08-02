"""파이프라인 단계 사이의 명시적 데이터 계약."""

from dataclasses import dataclass
from pathlib import Path

from chart_worker.analysis.audio_io import AudioSignal
from chart_worker.analysis.beat import BeatGrid
from chart_worker.analysis.onset import OnsetAnalysis
from chart_worker.analysis.timing import TimingCandidate
from chart_worker.audio.normalize import NormalizedAudio
from chart_worker.generation.mapperatorinator import GeneratedChart
from chart_worker.postprocess.difficulty_solver import SolveResult
from chart_worker.postprocess.lane_conversion import ConversionResult
from chart_worker.report.alignment import AlignmentReport
from chart_worker.schema.chart import ChartDocument
from chart_worker.schema.keysound import KeysoundManifest
from chart_worker.schema.playtest_run import AudioFileRef
from chart_worker.validation.playability import PlayabilityResult


@dataclass(frozen=True, slots=True)
class AnalysisStageResult:
    normalized: NormalizedAudio
    signal: AudioSignal
    beat_grid: BeatGrid
    onsets: OnsetAnalysis
    timing_candidate: TimingCandidate
    timing_osu_path: Path
    timing_quality_report_path: Path


@dataclass(frozen=True, slots=True)
class GeneratedVariant:
    key_mode: int
    difficulty: str
    requested_star: float
    raw_osu_path: Path
    generated: GeneratedChart


@dataclass(frozen=True, slots=True)
class StemStageResult:
    game_ref: AudioFileRef
    no_drums_ref: AudioFileRef | None
    keys_ref: AudioFileRef | None
    drum_onsets: tuple[int, ...]
    keysound_manifest: KeysoundManifest | None
    keysound_manifest_path: Path | None


@dataclass(frozen=True, slots=True)
class PostprocessReports:
    conversion: ConversionResult
    difficulty: SolveResult
    playability: PlayabilityResult
    alignment: AlignmentReport


@dataclass(frozen=True, slots=True)
class PostprocessedVariant:
    document: ChartDocument
    path: Path
    sha256: str
    reports: PostprocessReports
