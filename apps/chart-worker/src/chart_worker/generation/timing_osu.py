"""Beat This! 격자를 타이밍만 든 .osu 로 쓴다.

Mapperatorinator 에 `in_context=[TIMING]` 으로 넘기면 생성 12회가 각자
타이밍을 다시 추정하는 일이 없어지고, 12개 채보가 모두 같은 시간축을 본다.
"""

from chart_worker.analysis.beat import BeatGrid

DEFAULT_METER = 4


def beat_grid_to_timing_osu(
    grid: BeatGrid,
    *,
    audio_filename: str,
    title: str = "timing",
    artist: str = "ai-rhythm",
    creator: str = "ai-rhythm",
) -> str:
    """비트 격자를 uninherited 타이밍 포인트 하나짜리 .osu 로 만든다.

    노트를 넣지 않는다. 이 파일은 참조용이고 노트를 만드는 도구는
    Mapperatorinator 뿐이다.
    """
    if not grid.beat_ms:
        raise ValueError("beat grid has no beats")
    if grid.bpm <= 0:
        raise ValueError(f"beat grid has a non-positive bpm: {grid.bpm}")

    offset_ms = grid.beat_ms[0]
    beat_length_ms = 60_000.0 / grid.bpm
    meter = grid.beats_per_bar or DEFAULT_METER
    # 첫 다운비트가 첫 비트가 아니면 마디 첫 박에 맞춰 앞으로 당긴다.
    if grid.downbeat_indices:
        offset_ms = grid.beat_ms[grid.downbeat_indices[0]]

    return "\n".join(
        [
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
            "[TimingPoints]",
            f"{offset_ms},{beat_length_ms:.6f},{meter},2,0,60,1,0",
            "",
            "[HitObjects]",
            "",
        ]
    )
