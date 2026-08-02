"""Write Beat This timing points as Mapperatorinator timing-only beatmaps."""

from chart_worker.analysis.beat import BeatGrid
from chart_worker.analysis.timing import TimingPoint, fit_piecewise_timing


def beat_grid_to_timing_osu(
    grid: BeatGrid,
    *,
    audio_filename: str,
    title: str = "timing",
    artist: str = "ai-rhythm",
    creator: str = "ai-rhythm",
) -> str:
    """Compatibility wrapper for callers that still provide a BeatGrid."""
    if not grid.beat_ms:
        raise ValueError("beat grid has no beats")
    return timing_points_to_osu(
        fit_piecewise_timing(grid),
        audio_filename=audio_filename,
        title=title,
        artist=artist,
        creator=creator,
    )


def timing_points_to_osu(
    points: tuple[TimingPoint, ...],
    *,
    audio_filename: str,
    title: str = "timing",
    artist: str = "ai-rhythm",
    creator: str = "ai-rhythm",
) -> str:
    if not points:
        raise ValueError("timing has no points")
    header_lines = [
        "osu file format v14",
        "",
        "[General]",
        f"AudioFilename: {audio_filename}",
        "Mode: 3",
        "",
        "[Metadata]",
        f"Title:{title}",
        f"Artist:{artist}",
        f"Creator:{creator}",
        "Version:timing",
        "",
        "[Difficulty]",
        "HPDrainRate:5",
        "CircleSize:4",
        "OverallDifficulty:8",
        "",
    ]
    timing_lines = [
        f"{point.time_ms},{60_000.0 / point.bpm:.6f},{point.meter},2,0,60,1,0"
        for point in points
    ]
    return "\n".join([*header_lines, "[TimingPoints]", *timing_lines, "", "[HitObjects]", ""])
