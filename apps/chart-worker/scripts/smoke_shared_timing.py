"""Run one shared-timing Mapperatorinator smoke verification."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from time import perf_counter
from typing import Any

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from chart_worker.analysis.onset import analyze_canonical_audio
from chart_worker.config import WorkerConfig, load_config
from chart_worker.generation.mapperatorinator import (
    ChartGenerator,
    GeneratedChart,
    MapperatorinatorGenerator,
    build_map_command,
)
from chart_worker.generation.osu_parser import OsuBeatmap, OsuBpmEvent, parse_osu_mania
from chart_worker.generation.osu_writer import notes_to_osu_mania
from chart_worker.generation.params import GenerationRequest
from chart_worker.hashing import sha256_file
from chart_worker.stages.s1_prepare import run_prepare
from chart_worker.stages.s2_timing import run_timing_generation
from chart_worker.stages.types import PreparedAudio, SongTimingAuthority
from chart_worker.validation.generated_chart import validate_generated_chart
from chart_worker.validation.timing_authority import validate_timing_identity


@dataclass(frozen=True, slots=True)
class SmokeDependencies:
    """Replace external work for unit-level orchestration verification."""

    config: Callable[[], WorkerConfig]
    prepare: Callable[..., PreparedAudio]
    analyze: Callable[[Path], Any]
    timing: Callable[..., SongTimingAuthority]
    generator: Callable[[WorkerConfig], ChartGenerator]
    map_command: Callable[[WorkerConfig, GenerationRequest, Path], list[str]]


def _dependencies() -> SmokeDependencies:
    return SmokeDependencies(
        config=load_config,
        prepare=run_prepare,
        analyze=analyze_canonical_audio,
        timing=run_timing_generation,
        generator=MapperatorinatorGenerator,
        map_command=build_map_command,
    )


def _timing_events(events: tuple[OsuBpmEvent, ...]) -> list[list[int | float]]:
    return [[event.time_ms, event.bpm] for event in events]


def _note_fields(beatmap: OsuBeatmap) -> list[tuple[int, int, str, int | None]]:
    return [
        (note.time_ms, note.lane, note.kind, note.duration_ms) for note in beatmap.notes
    ]


def _generated_note_fields(chart: GeneratedChart) -> list[tuple[int, int, str, int | None]]:
    return [
        (note.time_ms, note.lane, note.kind, note.duration_ms) for note in chart.notes
    ]


def _validate_raw_lanes(raw_text: str, *, key_mode: int) -> None:
    """Check every source X coordinate and its parsed lane stay in the requested mode."""
    hit_objects = raw_text.partition("[HitObjects]")[2]
    for line in hit_objects.splitlines():
        if not line.strip():
            continue
        try:
            x = int(float(line.split(",", 1)[0]))
        except (IndexError, ValueError) as error:
            raise ValueError(f"invalid raw mania X coordinate: {line!r}") from error
        lane = min(key_mode - 1, max(0, x * key_mode // 512))
        if not 0 <= x <= 512 or not 0 <= lane < key_mode:
            raise ValueError(f"raw X={x} does not map to a {key_mode}K lane")


def _serialized_map(chart: GeneratedChart, prepared: PreparedAudio) -> str:
    if chart.osu_text:
        return chart.osu_text
    first_timing = chart.bpm_events[0]
    return notes_to_osu_mania(
        chart.notes,
        key_mode=chart.key_mode,
        bpm=first_timing.bpm,
        offset_ms=first_timing.time_ms,
        audio_filename=prepared.normalized.path.name,
        title=prepared.normalized.path.stem,
        bpm_events=chart.bpm_events,
    )


def _hydra_path(value: str) -> Path:
    return Path(value.strip().strip("'").replace("\\'", "'"))


def _verified_map_command_contract(
    command: list[str],
    *,
    request: GenerationRequest,
) -> dict[str, str]:
    pairs = dict(item.split("=", 1) for item in command if "=" in item)
    expected_reference = request.timing_reference_path.resolve()
    try:
        beatmap_path = _hydra_path(pairs["beatmap_path"]).resolve()
        contract = {
            "keycount": pairs["keycount"],
            "beatmapPath": str(beatmap_path),
            "outputType": pairs["output_type"],
            "inContext": pairs["in_context"],
        }
    except KeyError as error:
        raise ValueError(f"MAP command contract is missing {error.args[0]}") from error
    if (
        contract["keycount"] != str(request.key_mode)
        or beatmap_path != expected_reference
        or contract["outputType"] != "[MAP]"
        or contract["inContext"] != "[TIMING]"
    ):
        raise ValueError("MAP command contract does not use the shared timing reference")
    return contract


def run_smoke(
    *,
    source: Path,
    output: Path,
    key_mode: int,
    difficulty: str,
    seed: int = 0,
    dependencies: SmokeDependencies | None = None,
) -> dict[str, object]:
    """Prepare, time once, generate one MAP, and persist immutable evidence."""
    dependencies = dependencies or _dependencies()
    config = dependencies.config()
    output.mkdir(parents=True, exist_ok=True)
    prepared = dependencies.prepare(source, output, config=config)
    dependencies.analyze(prepared.normalized.path)
    generator = dependencies.generator(config)

    timing_started = perf_counter()
    authority = dependencies.timing(prepared, output, generator=generator, seed=seed)
    elapsed_timing_seconds = perf_counter() - timing_started

    request = GenerationRequest(
        audio_path=prepared.normalized.path,
        timing_reference_path=authority.reference_path,
        key_mode=key_mode,
        difficulty=difficulty,
        seed=seed,
        duration_ms=prepared.normalized.duration_ms,
    )
    map_workdir = output / "map" / "work" / "attempt-1"
    map_command_contract = _verified_map_command_contract(
        dependencies.map_command(config, request, map_workdir), request=request
    )
    map_started = perf_counter()
    generated = generator.generate_map(request, map_workdir)
    elapsed_map_seconds = perf_counter() - map_started
    final_reference_sha = sha256_file(authority.reference_path)
    if final_reference_sha != authority.sha256:
        raise ValueError("timing authority reference hash changed during MAP generation")

    raw_text = _serialized_map(generated, prepared)
    raw = parse_osu_mania(raw_text)
    _validate_raw_lanes(raw_text, key_mode=key_mode)
    if raw.key_mode != key_mode:
        raise ValueError(f"requested {key_mode}K but raw MAP is {raw.key_mode}K")
    if any(not 0 <= note.lane < key_mode for note in raw.notes):
        raise ValueError(f"raw MAP contains a lane outside requested {key_mode}K")

    reference = parse_osu_mania(authority.reference_path.read_text(encoding="utf-8-sig"))
    validate_timing_identity(reference.bpm_events, authority.bpm_events)
    validate_timing_identity(raw.bpm_events, reference.bpm_events)
    if _generated_note_fields(generated) != _note_fields(raw):
        raise ValueError("note fields changed between raw parse and final accepted chart")
    accepted = GeneratedChart(
        notes=raw.notes,
        key_mode=raw.key_mode,
        osu_text=raw_text,
        generator_name=generated.generator_name,
        seed=generated.seed,
        bpm_events=raw.bpm_events,
    )
    validate_generated_chart(
        accepted,
        key_mode=key_mode,
        duration_ms=prepared.normalized.duration_ms,
    )

    raw_path = output / "raw" / f"{key_mode}k-{difficulty.lower()}.osu"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_text(raw_text, encoding="utf-8")
    times = sorted({note.time_ms for note in accepted.notes})
    first_note = accepted.notes[0]
    summary: dict[str, object] = {
        "elapsedTimingSeconds": elapsed_timing_seconds,
        "elapsedMapSeconds": elapsed_map_seconds,
        "timingReferenceSha256": final_reference_sha,
        "referenceBpmEvents": _timing_events(reference.bpm_events),
        "mapBpmEvents": _timing_events(raw.bpm_events),
        "noteCount": len(accepted.notes),
        "holdCount": sum(note.kind == "HOLD" for note in accepted.notes),
        "firstNote": {
            "timeMs": first_note.time_ms,
            "lane": first_note.lane,
            "kind": first_note.kind,
            "durationMs": first_note.duration_ms,
        },
        "maximumGapMs": max((right - left for left, right in pairwise(times)), default=0),
        "mapCommandContract": map_command_contract,
    }
    (output / "shared-timing-smoke.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--key-mode", required=True, type=int, choices=(4, 6, 7))
    parser.add_argument(
        "--difficulty", required=True, choices=("EASY", "NORMAL", "HARD", "EXPERT")
    )
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    summary = run_smoke(
        source=args.source,
        output=args.output,
        key_mode=args.key_mode,
        difficulty=args.difficulty,
        seed=args.seed,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
