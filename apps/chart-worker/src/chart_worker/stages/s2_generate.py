"""S2: generate each map and retry only invalid Mapperatorinator output."""

from pathlib import Path

from chart_worker.errors import ErrorCode, WorkerError
from chart_worker.generation.mapperatorinator import ChartGenerator
from chart_worker.generation.osu_writer import notes_to_osu_mania
from chart_worker.generation.params import GenerationRequest
from chart_worker.schema.types import DIFFICULTIES, KEY_MODES
from chart_worker.stages.types import GeneratedVariant, PreparedAudio
from chart_worker.validation.generated_chart import (
    GeneratedChartValidationError,
    validate_generated_chart,
)

MAX_VARIANT_ATTEMPTS = 3
_VARIANT_COUNT = len(KEY_MODES) * len(DIFFICULTIES)
_RETRYABLE_GENERATION_CODES = {
    ErrorCode.CHART_GENERATION_FAILED,
    ErrorCode.CHART_OSU_PARSE_FAILED,
}


def _should_retry(error: Exception) -> bool:
    if isinstance(error, GeneratedChartValidationError):
        return True
    return isinstance(error, WorkerError) and error.code in _RETRYABLE_GENERATION_CODES


def run_generation(
    prepared: PreparedAudio,
    run_dir: Path,
    *,
    generator: ChartGenerator,
    seed: int,
) -> tuple[GeneratedVariant, ...]:
    """Generate every variant, retrying only the variant with invalid output."""
    variants = []
    combinations = (
        (key_mode, difficulty)
        for key_mode in KEY_MODES
        for difficulty in DIFFICULTIES
    )
    for index, (key_mode, difficulty) in enumerate(combinations):
        attempt_errors: list[str] = []
        attempted_seeds: list[int] = []
        for attempt in range(1, MAX_VARIANT_ATTEMPTS + 1):
            attempt_seed = seed + index + (attempt - 1) * _VARIANT_COUNT
            attempted_seeds.append(attempt_seed)
            request = GenerationRequest(
                audio_path=prepared.normalized.path,
                key_mode=key_mode,
                difficulty=difficulty,
                seed=attempt_seed,
                duration_ms=prepared.normalized.duration_ms,
            )
            workdir = (
                run_dir
                / "raw"
                / "work"
                / f"{key_mode}k-{difficulty.lower()}"
                / f"attempt-{attempt}"
            )
            try:
                generated = generator(request, workdir)
                validate_generated_chart(
                    generated,
                    key_mode=key_mode,
                    duration_ms=prepared.normalized.duration_ms,
                )
            except (GeneratedChartValidationError, WorkerError) as error:
                if not _should_retry(error):
                    raise
                attempt_errors.append(str(error))
                if attempt < MAX_VARIANT_ATTEMPTS:
                    continue
                raise WorkerError(
                    ErrorCode.CHART_CANDIDATES_EXHAUSTED,
                    f"{key_mode}K {difficulty} failed {MAX_VARIANT_ATTEMPTS} attempts",
                    context={
                        "key_mode": key_mode,
                        "difficulty": difficulty,
                        "seeds": attempted_seeds,
                        "errors": attempt_errors,
                    },
                ) from error
            break
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
                attempt=attempt,
                attempt_errors=tuple(attempt_errors),
            )
        )
    return tuple(variants)
