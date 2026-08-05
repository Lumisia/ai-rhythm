"""Immutable contracts between the direct generation stages."""

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from chart_worker.analysis.grid_alignment import TempoCandidateMetrics
from chart_worker.audio.normalize import NormalizedAudio
from chart_worker.generation.mapperatorinator import GeneratedChart
from chart_worker.generation.osu_parser import OsuBpmEvent
from chart_worker.schema.chart import ChartDocument
from chart_worker.validation.timing_review import TimingAuthorityReview

if TYPE_CHECKING:
    from chart_worker.validation.quality_gate import ChartAcceptance


@dataclass(frozen=True, slots=True)
class PreparedAudio:
    normalized: NormalizedAudio


@dataclass(frozen=True, slots=True)
class SongTimingAuthority:
    reference_path: Path
    sha256: str
    audio_sha256: str
    bpm_events: tuple[OsuBpmEvent, ...]
    generator_name: str
    seed: int | None
    mode: Literal["STANDARD", "SUPER_TIMING"]
    attempt_count: int
    tempo_metrics: TempoCandidateMetrics | None = None
    review: TimingAuthorityReview | None = None


@dataclass(frozen=True, slots=True)
class GeneratedVariant:
    key_mode: int
    difficulty: str
    requested_star: float
    raw_osu_path: Path
    generated: GeneratedChart
    acceptance: "ChartAcceptance"
    cfg_scale: float = 1.0
    attempt: int = 1
    attempt_errors: tuple[str, ...] = ()
    attempt_evidence: tuple[dict[str, object], ...] = ()
    timing_authority_sha256: str = ""


@dataclass(frozen=True, slots=True)
class ExportedVariant:
    document: ChartDocument
    path: Path
    sha256: str
