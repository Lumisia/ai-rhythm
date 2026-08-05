"""S2: generate each map and retry only invalid Mapperatorinator output."""

from pathlib import Path

from chart_worker.errors import ErrorCode, WorkerError
from chart_worker.generation.mapperatorinator import ChartGenerator
from chart_worker.generation.osu_parser import parse_osu_mania
from chart_worker.generation.osu_writer import notes_to_osu_mania
from chart_worker.generation.params import GenerationRequest
from chart_worker.hashing import sha256_file
from chart_worker.schema.types import DIFFICULTIES, KEY_MODES
from chart_worker.stages.types import GeneratedVariant, PreparedAudio, SongTimingAuthority
from chart_worker.validation.generated_chart import (
    GeneratedChartValidationError,
    validate_generated_chart,
)
from chart_worker.validation.timing_authority import (
    TimingAuthorityValidationError,
    validate_timing_identity,
)

MAX_VARIANT_ATTEMPTS = 3
_VARIANT_COUNT = len(KEY_MODES) * len(DIFFICULTIES)
_RETRYABLE_GENERATION_CODES = {
    ErrorCode.CHART_GENERATION_FAILED,
    ErrorCode.CHART_OSU_PARSE_FAILED,
}


def _should_retry(error: Exception) -> bool:
    if isinstance(error, (GeneratedChartValidationError, TimingAuthorityValidationError)):
        return True
    return isinstance(error, WorkerError) and error.code in _RETRYABLE_GENERATION_CODES


def _require_timing_authority(
    prepared: PreparedAudio, authority: SongTimingAuthority
) -> None:
    if authority.audio_sha256 != prepared.normalized.sha256:
        raise WorkerError(
            ErrorCode.ASSET_HASH_MISMATCH,
            "timing authority belongs to different canonical audio",
            context={
                "expected": prepared.normalized.sha256,
                "actual": authority.audio_sha256,
            },
        )
    actual_sha = (
        sha256_file(authority.reference_path)
        if authority.reference_path.is_file()
        else None
    )
    if actual_sha != authority.sha256:
        raise WorkerError(
            ErrorCode.ASSET_HASH_MISMATCH,
            "timing authority reference changed before MAP generation",
            context={
                "path": str(authority.reference_path),
                "expected": authority.sha256,
                "actual": actual_sha,
            },
        )


def _validate_serialized_timing(
    osu_text: str, authority: SongTimingAuthority
) -> None:
    try:
        bpm_events = parse_osu_mania(osu_text).bpm_events
    except ValueError as error:
        raise WorkerError(
            ErrorCode.CHART_OSU_PARSE_FAILED,
            "serialized MAP is not valid osu!mania",
        ) from error
    validate_timing_identity(bpm_events, authority.bpm_events)


def run_generation(
    prepared: PreparedAudio,
    authority: SongTimingAuthority,
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
            _require_timing_authority(prepared, authority)
            attempt_seed = seed + index + (attempt - 1) * _VARIANT_COUNT
            attempted_seeds.append(attempt_seed)
            request = GenerationRequest(
                audio_path=prepared.normalized.path,
                timing_reference_path=authority.reference_path,
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
            raw_path = run_dir / "raw" / f"{key_mode}k-{difficulty.lower()}.osu"
            try:
                generated = generator.generate_map(request, workdir)
                validate_timing_identity(generated.bpm_events, authority.bpm_events)
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
                    bpm_events=generated.bpm_events,
                )
                _validate_serialized_timing(osu_text, authority)
                raw_path.parent.mkdir(parents=True, exist_ok=True)
                raw_path.write_text(osu_text, encoding="utf-8")
                try:
                    _validate_serialized_timing(
                        raw_path.read_text(encoding="utf-8-sig"), authority
                    )
                except (TimingAuthorityValidationError, WorkerError):
                    raw_path.unlink(missing_ok=True)
                    raise
            except (
                GeneratedChartValidationError,
                TimingAuthorityValidationError,
                WorkerError,
            ) as error:
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
                timing_authority_sha256=authority.sha256,
            )
        )
    return tuple(variants)
