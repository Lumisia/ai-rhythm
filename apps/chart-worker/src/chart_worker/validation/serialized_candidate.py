"""Validate that serialized osu!mania text preserves an accepted candidate."""

from chart_worker.errors import ErrorCode, WorkerError
from chart_worker.generation.mapperatorinator import GeneratedChart
from chart_worker.generation.osu_parser import parse_osu_mania
from chart_worker.stages.types import PreparedAudio, SongTimingAuthority
from chart_worker.validation.generated_chart import (
    GeneratedChartValidationError,
    validate_generated_chart,
)
from chart_worker.validation.timing_authority import validate_timing_identity


def _note_projection(generated: GeneratedChart) -> tuple[tuple[object, ...], ...]:
    return tuple(
        (note.time_ms, note.lane, note.kind, note.duration_ms)
        for note in generated.notes
    )


def validate_serialized_candidate(
    osu_text: str,
    generated: GeneratedChart,
    authority: SongTimingAuthority,
    prepared: PreparedAudio,
    key_mode: int,
) -> None:
    """Reject serialization that changes timing, structure, lanes, or HOLDs."""

    try:
        parsed = parse_osu_mania(osu_text)
    except ValueError as error:
        raise WorkerError(
            ErrorCode.CHART_OSU_PARSE_FAILED,
            "serialized MAP is not valid osu!mania",
        ) from error
    validate_timing_identity(parsed.bpm_events, authority.bpm_events)
    parsed_chart = GeneratedChart(
        notes=parsed.notes,
        key_mode=parsed.key_mode,
        osu_text=osu_text,
        generator_name=generated.generator_name,
        seed=generated.seed,
        bpm_events=parsed.bpm_events,
    )
    validate_generated_chart(
        parsed_chart,
        key_mode=key_mode,
        duration_ms=prepared.normalized.duration_ms,
    )
    if parsed.key_mode != generated.key_mode:
        raise GeneratedChartValidationError(
            "serialized MAP key mode differs from generated object",
            reason_code="SERIALIZED_KEY_MODE_MISMATCH",
            context={
                "serializedKeyMode": parsed.key_mode,
                "generatedKeyMode": generated.key_mode,
            },
        )
    if _note_projection(parsed_chart) != _note_projection(generated):
        raise GeneratedChartValidationError(
            "serialized MAP note fields differ from generated object",
            reason_code="SERIALIZED_NOTE_MISMATCH",
        )
