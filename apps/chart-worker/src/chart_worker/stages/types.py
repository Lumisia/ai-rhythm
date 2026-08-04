"""Immutable contracts between the direct generation stages."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from chart_worker.audio.normalize import NormalizedAudio
from chart_worker.generation.mapperatorinator import GeneratedChart
from chart_worker.generation.osu_parser import OsuBpmEvent
from chart_worker.schema.chart import ChartDocument


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


@dataclass(frozen=True, slots=True)
class GeneratedVariant:
    key_mode: int
    difficulty: str
    requested_star: float
    raw_osu_path: Path
    generated: GeneratedChart
    cfg_scale: float = 1.0
    attempt: int = 1
    attempt_errors: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ExportedVariant:
    document: ChartDocument
    path: Path
    sha256: str
