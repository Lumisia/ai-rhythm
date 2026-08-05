"""테스트 생성 결과를 재파싱 가능한 osu!mania 문서로 직렬화한다."""

import math

from chart_worker.generation.osu_parser import OsuBpmEvent
from chart_worker.schema.note import Chart
from chart_worker.schema.types import KEY_MODES


def _safe_title(title: str) -> str:
    return title.replace("\r", " ").replace("\n", " ")


def notes_to_osu_mania(
    notes: Chart,
    *,
    key_mode: int,
    bpm: float,
    offset_ms: int,
    audio_filename: str,
    title: str,
    bpm_events: tuple[OsuBpmEvent, ...] | None = None,
) -> str:
    if key_mode not in KEY_MODES:
        raise ValueError(f"unsupported key_mode: {key_mode}")
    if bpm <= 0:
        raise ValueError("bpm must be positive")

    hit_objects = []
    for note in sorted(notes, key=lambda item: (item.time_ms, item.lane)):
        if note.lane >= key_mode:
            raise ValueError(f"note lane {note.lane} is outside {key_mode}K")
        x = min(511, round((note.lane + 0.5) * 512 / key_mode))
        if note.kind == "HOLD":
            end_ms = note.time_ms + (note.duration_ms or 0)
            hit_objects.append(f"{x},192,{note.time_ms},128,0,{end_ms}:0:0:0:0:")
        else:
            hit_objects.append(f"{x},192,{note.time_ms},1,0,0:0:0:0:")

    timing_events = bpm_events or (OsuBpmEvent(offset_ms, bpm),)
    timing_points = []
    for event in timing_events:
        if not math.isfinite(event.bpm) or event.bpm <= 0:
            raise ValueError("bpm must be positive and finite")
        timing_points.append(f"{event.time_ms},{60_000.0 / event.bpm:.12f},4,2,0,60,1,0")

    safe_title = _safe_title(title)
    lines = [
        "osu file format v14",
        "",
        "[General]",
        f"AudioFilename: {audio_filename}",
        "Mode: 3",
        "",
        "[Metadata]",
        f"Title:{safe_title}",
        "Artist:ai-rhythm",
        "Version:generated",
        "",
        "[Difficulty]",
        f"CircleSize:{key_mode}",
        "OverallDifficulty:8",
        "",
        "[TimingPoints]",
        *timing_points,
        "",
        "[HitObjects]",
        *hit_objects,
        "",
    ]
    return "\n".join(lines)


def timing_to_osu_mania(
    bpm_events: tuple[OsuBpmEvent, ...], *, audio_filename: str, title: str
) -> str:
    """Serialize a timing-only 4K osu!mania reference without hit objects."""
    timing_points = []
    for event in bpm_events:
        if not math.isfinite(event.bpm) or event.bpm <= 0:
            raise ValueError("bpm must be positive and finite")
        timing_points.append(f"{event.time_ms},{60_000.0 / event.bpm:.12f},4,2,0,60,1,0")

    lines = [
        "osu file format v14",
        "",
        "[General]",
        f"AudioFilename: {audio_filename}",
        "Mode: 3",
        "",
        "[Metadata]",
        f"Title:{_safe_title(title)}",
        "Artist:ai-rhythm",
        "Version:timing-reference",
        "",
        "[Difficulty]",
        "CircleSize:4",
        "OverallDifficulty:8",
        "",
        "[TimingPoints]",
        *timing_points,
        "",
        "[HitObjects]",
        "",
    ]
    return "\n".join(lines)
