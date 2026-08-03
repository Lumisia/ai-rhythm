import math
from dataclasses import dataclass
from pathlib import Path

from chart_worker.schema.note import Chart, NoteEvent
from chart_worker.schema.types import KEY_MODES

_HOLD_BIT = 128
_CIRCLE_BIT = 1
_AUXILIARY_TYPE_BITS = 4 | 16 | 32 | 64


@dataclass(frozen=True, slots=True)
class OsuBpmEvent:
    time_ms: int
    bpm: float


@dataclass(frozen=True, slots=True)
class OsuBeatmap:
    key_mode: int
    notes: Chart
    bpm_events: tuple[OsuBpmEvent, ...]


def _sections(text: str) -> dict[str, list[str]]:
    sections: dict[str, list[str]] = {}
    current: str | None = None
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("//"):
            continue
        if line.startswith("[") and line.endswith("]"):
            current = line[1:-1]
            sections[current] = []
        elif current is not None:
            sections[current].append(line)
    return sections


def _key_value(lines: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in lines:
        if ":" in line:
            key, _, value = line.partition(":")
            result[key.strip()] = value.strip()
    return result


def parse_osu_mania(text: str) -> OsuBeatmap:
    sections = _sections(text)
    general = _key_value(sections.get("General", []))
    if general.get("Mode") != "3":
        raise ValueError(f"not an osu!mania beatmap (Mode={general.get('Mode')!r})")

    difficulty = _key_value(sections.get("Difficulty", []))
    raw_key_mode = difficulty.get("CircleSize")
    if raw_key_mode is None:
        raise ValueError("missing CircleSize for osu!mania key mode")
    key_mode_number = float(raw_key_mode)
    if not key_mode_number.is_integer() or int(key_mode_number) not in KEY_MODES:
        raise ValueError(f"unsupported key_mode: {raw_key_mode}")
    key_mode = int(key_mode_number)

    bpm_events: list[OsuBpmEvent] = []
    for line in sections.get("TimingPoints", []):
        parts = line.split(",")
        if len(parts) < 7 or parts[6].strip() != "1":
            continue
        beat_length = float(parts[1])
        if beat_length > 0:
            bpm_events.append(
                OsuBpmEvent(time_ms=round(float(parts[0])), bpm=60000.0 / beat_length)
            )
    bpm_events.sort(key=lambda event: event.time_ms)

    notes: Chart = []
    for line in sections.get("HitObjects", []):
        parts = line.split(",")
        if len(parts) < 5:
            raise ValueError(f"malformed HitObject: {line!r}")
        try:
            x = int(float(parts[0]))
            time_ms = round(float(parts[2]))
            object_type = int(parts[3])
        except ValueError as error:
            raise ValueError(f"malformed HitObject: {line!r}") from error
        if not 0 <= x <= 512:
            raise ValueError(
                f"mania x coordinate {x} is outside the osu! playfield 0..512"
            )
        base_type = object_type & ~_AUXILIARY_TYPE_BITS
        lane = min(key_mode - 1, max(0, math.floor(x * key_mode / 512)))
        if base_type == _HOLD_BIT:
            if len(parts) < 6 or not parts[5].split(":", 1)[0]:
                raise ValueError(f"mania hold at {time_ms}ms is missing an end time")
            try:
                end_ms = round(float(parts[5].split(":", 1)[0]))
            except ValueError as error:
                raise ValueError(f"malformed HitObject: {line!r}") from error
            duration_ms = end_ms - time_ms
            if duration_ms <= 0:
                raise ValueError(
                    f"mania hold at {time_ms}ms has end {end_ms}ms "
                    "that is not after its start"
                )
            notes.append(
                NoteEvent(
                    time_ms=time_ms,
                    lane=lane,
                    kind="HOLD",
                    duration_ms=duration_ms,
                )
            )
        elif base_type == _CIRCLE_BIT:
            notes.append(NoteEvent(time_ms=time_ms, lane=lane))
        else:
            raise ValueError(f"unsupported HitObject type: {object_type}")

    notes.sort(key=lambda note: (note.time_ms, note.lane))
    return OsuBeatmap(key_mode=key_mode, notes=notes, bpm_events=tuple(bpm_events))


def parse_osu_file(path: Path) -> OsuBeatmap:
    return parse_osu_mania(Path(path).read_text(encoding="utf-8-sig"))
