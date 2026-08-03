"""S2: generate each Mapperatorinator key/difficulty map exactly once."""

from pathlib import Path

from chart_worker.generation.mapperatorinator import ChartGenerator
from chart_worker.generation.osu_writer import notes_to_osu_mania
from chart_worker.generation.params import GenerationRequest
from chart_worker.schema.types import DIFFICULTIES, KEY_MODES
from chart_worker.stages.types import GeneratedVariant, PreparedAudio
from chart_worker.validation.generated_chart import validate_generated_chart


def run_generation(
    prepared: PreparedAudio,
    run_dir: Path,
    *,
    generator: ChartGenerator,
    seed: int,
) -> tuple[GeneratedVariant, ...]:
    """Generate all 4K/6K/7K difficulty combinations with no retries."""
    variants = []
    combinations = (
        (key_mode, difficulty)
        for key_mode in KEY_MODES
        for difficulty in DIFFICULTIES
    )
    for index, (key_mode, difficulty) in enumerate(combinations):
        request = GenerationRequest(
            audio_path=prepared.normalized.path,
            key_mode=key_mode,
            difficulty=difficulty,
            seed=seed + index,
            duration_ms=prepared.normalized.duration_ms,
        )
        workdir = run_dir / "raw" / "work" / f"{key_mode}k-{difficulty.lower()}"
        generated = generator(request, workdir)
        validate_generated_chart(
            generated,
            key_mode=key_mode,
            duration_ms=prepared.normalized.duration_ms,
        )
        first_timing = generated.bpm_events[0]
        osu_text = generated.osu_text or notes_to_osu_mania(
            generated.notes,
            key_mode=key_mode,
            bpm=first_timing.bpm,
            offset_ms=first_timing.time_ms,
            audio_filename=prepared.normalized.path.name,
            title=prepared.normalized.path.stem,
        )
        raw_path = run_dir / "raw" / f"{key_mode}k-{difficulty.lower()}.osu"
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw_path.write_text(osu_text, encoding="utf-8")
        variants.append(
            GeneratedVariant(
                key_mode=key_mode,
                difficulty=difficulty,
                requested_star=request.requested_star,
                raw_osu_path=raw_path,
                generated=generated,
                cfg_scale=request.cfg_scale,
                attempt=1,
            )
        )
    return tuple(variants)
