"""S2 timing: promote one validated timing reference for the canonical song audio."""

from pathlib import Path

from chart_worker.errors import ErrorCode, WorkerError
from chart_worker.generation.mapperatorinator import ChartGenerator, GeneratedTiming
from chart_worker.generation.osu_parser import parse_osu_file
from chart_worker.generation.osu_writer import timing_to_osu_mania
from chart_worker.generation.params import TimingGenerationRequest
from chart_worker.hashing import sha256_file
from chart_worker.stages.types import PreparedAudio, SongTimingAuthority
from chart_worker.validation.timing_authority import (
    TimingAuthorityValidationError,
    validate_timing_events,
    validate_timing_identity,
)


def _validate_and_promote(
    generated: GeneratedTiming, prepared: PreparedAudio, reference_path: Path
) -> None:
    validate_timing_events(generated.bpm_events, prepared.normalized.duration_ms)
    reference_path.parent.mkdir(parents=True, exist_ok=True)
    reference_path.write_text(
        timing_to_osu_mania(
            generated.bpm_events,
            audio_filename=prepared.normalized.path.name,
            title=prepared.normalized.path.stem,
        ),
        encoding="utf-8",
    )
    validate_timing_identity(parse_osu_file(reference_path).bpm_events, generated.bpm_events)


def run_timing_generation(
    prepared: PreparedAudio,
    run_dir: Path,
    *,
    generator: ChartGenerator,
    seed: int,
) -> SongTimingAuthority:
    """Generate one standard timing candidate, with one structural Super Timing fallback."""
    reference_path = run_dir / "audio" / "timing-reference.osu"
    errors: list[str] = []
    seeds: list[int] = []

    for attempt_count, super_timing in enumerate((False, True), start=1):
        request = TimingGenerationRequest(
            audio_path=prepared.normalized.path,
            duration_ms=prepared.normalized.duration_ms,
            seed=seed,
            super_timing=super_timing,
        )
        seeds.append(seed)
        workdir = run_dir / "timing" / "work" / f"attempt-{attempt_count}"
        try:
            generated = generator.generate_timing(request, workdir)
            _validate_and_promote(generated, prepared, reference_path)
        except TimingAuthorityValidationError as error:
            errors.append(str(error))
            if not super_timing:
                continue
            raise WorkerError(
                ErrorCode.CHART_TIMING_CANDIDATE_FAILED,
                "standard and Super Timing candidates failed structural validation",
                context={"errors": errors, "seeds": seeds},
            ) from error

        return SongTimingAuthority(
            reference_path=reference_path,
            sha256=sha256_file(reference_path),
            audio_sha256=prepared.normalized.sha256,
            bpm_events=generated.bpm_events,
            generator_name=generated.generator_name,
            seed=generated.seed,
            mode=generated.mode,
            attempt_count=attempt_count,
        )

    raise AssertionError("timing generation attempts were unexpectedly exhausted")
