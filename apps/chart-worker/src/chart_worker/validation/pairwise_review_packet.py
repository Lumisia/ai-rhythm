"""Build metadata-blinded osu!mania payloads for offline human comparison."""

from __future__ import annotations

import hashlib
import re
from pathlib import PurePath

_SECTION = re.compile(r"^\[([A-Za-z]+)\]$")
_PLAYABLE_SECTIONS = ("Difficulty", "TimingPoints", "HitObjects")


def _metadata_text(value: object, *, name: str, maximum: int) -> str:
    if type(value) is not str or not value:
        raise TypeError(f"{name} must be a non-empty exact string")
    if len(value) > maximum or "\r" in value or "\n" in value:
        raise ValueError(f"{name} is too long or contains a line break")
    return value


def _stable_song_audio_filename(value: object) -> str:
    filename = _metadata_text(value, name="audio_filename", maximum=120)
    if PurePath(filename).name != filename or any(character in filename for character in "/\\:"):
        raise ValueError("audio_filename must be a plain file name")
    if PurePath(filename).suffix.lower() not in {".mp3", ".ogg"}:
        raise ValueError("audio_filename must use osu! stable song audio (.mp3 or .ogg)")
    return filename


def _decode(source: object) -> tuple[str, str]:
    if type(source) is not bytes or not source:
        raise TypeError("source must be non-empty exact bytes")
    try:
        text = source.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("source must be UTF-8") from error
    if not text.startswith("osu file format v"):
        raise ValueError("source is not an osu! beatmap")
    newline = "\r\n" if "\r\n" in text else "\n"
    if newline == "\r\n" and text.replace("\r\n", "").find("\n") >= 0:
        raise ValueError("source mixes line endings")
    return text, newline


def _section_ranges(text: str) -> dict[str, tuple[int, int]]:
    lines = text.splitlines(keepends=True)
    starts: list[tuple[str, int]] = []
    offset = 0
    for line in lines:
        match = _SECTION.fullmatch(line.rstrip("\r\n"))
        if match:
            starts.append((match.group(1), offset))
        offset += len(line)
    names = [name for name, _offset in starts]
    if len(set(names)) != len(names):
        raise ValueError("source contains a duplicate section")
    ranges = {}
    for index, (name, start) in enumerate(starts):
        end = starts[index + 1][1] if index + 1 < len(starts) else len(text)
        ranges[name] = (start, end)
    required = {"General", "Metadata", *_PLAYABLE_SECTIONS}
    if not required <= set(ranges):
        raise ValueError("source is missing a required mania section")
    for name in ("TimingPoints", "HitObjects"):
        start, end = ranges[name]
        body = text[start:end].splitlines()[1:]
        if not any(line.strip() and not line.lstrip().startswith("//") for line in body):
            raise ValueError(f"source {name} section is empty")
    general_start, general_end = ranges["General"]
    general = text[general_start:general_end]
    mode_lines = [
        line.split(":", 1)[1].strip()
        for line in general.splitlines()[1:]
        if line.split(":", 1)[0].strip() == "Mode" and ":" in line
    ]
    if mode_lines != ["3"]:
        raise ValueError("source must declare mania Mode: 3 exactly once")
    return ranges


def _section_blob(text: str, ranges: dict[str, tuple[int, int]], name: str) -> str:
    start, end = ranges[name]
    return text[start:end]


def playable_sections_sha256(source: bytes) -> str:
    """Hash exact difficulty, timing and hit-object section bytes."""
    text, _newline = _decode(source)
    ranges = _section_ranges(text)
    payload = "".join(_section_blob(text, ranges, name) for name in _PLAYABLE_SECTIONS)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def normalized_playable_sections_sha256(source: bytes) -> str:
    """Hash playable sections after normalizing only CRLF/LF serialization."""
    text, newline = _decode(source)
    if newline == "\r\n":
        text = text.replace("\r\n", "\n")
    ranges = _section_ranges(text)
    payload = "".join(_section_blob(text, ranges, name) for name in _PLAYABLE_SECTIONS)
    return hashlib.sha256(payload.encode()).hexdigest()


def replace_song_audio_filename_v1(source: bytes, *, audio_filename: str) -> bytes:
    """Replace one General AudioFilename without changing playable section bytes."""
    filename = _stable_song_audio_filename(audio_filename)
    text, _newline = _decode(source)
    ranges = _section_ranges(text)
    before = playable_sections_sha256(source)
    start, end = ranges["General"]
    general = text[start:end]
    pattern = re.compile(r"^AudioFilename:[^\r\n]*(?=\r?$)", re.MULTILINE)
    if len(pattern.findall(general)) != 1:
        raise ValueError("source must declare AudioFilename exactly once")
    rewritten = pattern.sub(f"AudioFilename: {filename}", general)
    output = (text[:start] + rewritten + text[end:]).encode("utf-8")
    if playable_sections_sha256(output) != before:
        raise ValueError("audio filename migration changed playable beatmap sections")
    return output


def build_blind_osu_payload_v1(
    source: bytes,
    *,
    audio_filename: str,
    blind_title: str,
    blind_version: str,
) -> bytes:
    """Remove generation identity while preserving playable section bytes exactly."""
    filename = _stable_song_audio_filename(audio_filename)
    title = _metadata_text(blind_title, name="blind_title", maximum=120)
    version = _metadata_text(blind_version, name="blind_version", maximum=80)
    text, newline = _decode(source)
    ranges = _section_ranges(text)
    before = playable_sections_sha256(source)

    general = newline.join(
        (
            "[General]",
            f"AudioFilename: {filename}",
            "AudioLeadIn: 0",
            "PreviewTime: -1",
            "Countdown: 0",
            "SampleSet: Normal",
            "StackLeniency: 0.7",
            "Mode: 3",
            "LetterboxInBreaks: 0",
            "WidescreenStoryboard: 1",
            "",
            "",
        )
    )
    metadata = newline.join(
        (
            "[Metadata]",
            f"Title:{title}",
            f"TitleUnicode:{title}",
            "Artist:Blind Review",
            "ArtistUnicode:Blind Review",
            "Creator:ai-rhythm blind packet",
            f"Version:{version}",
            "Source:",
            "Tags:pairwise_blind_v1",
            "",
            "",
        )
    )
    events = newline.join(
        (
            "[Events]",
            "// Background and storyboard intentionally removed for blinded review",
            "",
            "",
        )
    )
    difficulty = _section_blob(text, ranges, "Difficulty")
    timing = _section_blob(text, ranges, "TimingPoints")
    hit_objects = _section_blob(text, ranges, "HitObjects")
    colours = _section_blob(text, ranges, "Colours") if "Colours" in ranges else ""
    header = text.splitlines()[0]
    output = (
        header
        + newline
        + newline
        + general
        + metadata
        + difficulty
        + events
        + timing
        + colours
        + hit_objects
    ).encode("utf-8")
    if playable_sections_sha256(output) != before:
        raise ValueError("blinding changed playable beatmap sections")
    return output
