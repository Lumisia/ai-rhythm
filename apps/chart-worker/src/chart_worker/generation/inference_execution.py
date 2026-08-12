"""One audited boundary around an external chart-generator invocation.

The caller owns retry, recovery, selection, and publication policy.  This
module owns only input immutability checks, the adapter call, and append-only
attempt evidence so every inference path has the same causal record.
"""

import json
from pathlib import Path

from chart_worker.errors import ErrorCode, WorkerError
from chart_worker.generation.candidate_state import VariantState
from chart_worker.generation.mapperatorinator import ChartGenerator, GeneratedChart
from chart_worker.generation.params import GenerationRequest
from chart_worker.hashing import sha256_file
from chart_worker.stages.types import PreparedAudio, SongTimingAuthority
from chart_worker.validation.quality_gate import ChartAcceptance


def error_report(error: Exception) -> dict[str, object]:
    code = getattr(error, "code", None)
    return {
        "type": type(error).__name__,
        "code": getattr(code, "value", None),
        "message": str(error),
        "context": getattr(error, "context", None),
    }


def error_report_json(error: Exception) -> str:
    """Serialize structured error evidence, including subprocess context."""
    return json.dumps(
        error_report(error),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def require_generation_inputs_unchanged(
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
    actual_audio_sha = (
        sha256_file(prepared.normalized.path)
        if prepared.normalized.path.is_file()
        else None
    )
    if actual_audio_sha != prepared.normalized.sha256:
        raise WorkerError(
            ErrorCode.ASSET_HASH_MISMATCH,
            "canonical audio changed during MAP generation",
            context={
                "path": str(prepared.normalized.path),
                "expected": prepared.normalized.sha256,
                "actual": actual_audio_sha,
            },
        )


def record_journal_event(
    state: VariantState,
    *,
    event_type: str,
    authority_epoch: int,
    attempt: int | None,
    seed: int | None,
    payload: dict[str, object] | None = None,
) -> None:
    if state.journal is None:
        return
    state.journal.append(
        event_type=event_type,
        authority_epoch=authority_epoch,
        key_mode=state.key_mode,
        difficulty=state.difficulty,
        attempt=attempt,
        seed=seed,
        payload=payload,
    )


def run_inference_with_journal(
    state: VariantState,
    *,
    generator: ChartGenerator,
    request: GenerationRequest,
    workdir: Path,
    run_dir: Path,
    prepared: PreparedAudio,
    authority: SongTimingAuthority,
    authority_epoch: int,
    attempt: int,
    seed: int,
    purpose: str,
) -> GeneratedChart:
    workdir_name = workdir.relative_to(run_dir).as_posix()
    common_payload = {"purpose": purpose, "workdir": workdir_name}
    record_journal_event(
        state,
        event_type="INFERENCE_STARTED",
        authority_epoch=authority_epoch,
        attempt=attempt,
        seed=seed,
        payload=common_payload,
    )
    try:
        require_generation_inputs_unchanged(prepared, authority)
        try:
            generated = generator.generate_map(request, workdir)
        finally:
            require_generation_inputs_unchanged(prepared, authority)
    except Exception as error:
        record_journal_event(
            state,
            event_type="INFERENCE_FAILED",
            authority_epoch=authority_epoch,
            attempt=attempt,
            seed=seed,
            payload={**common_payload, "error": error_report(error)},
        )
        raise
    record_journal_event(
        state,
        event_type="INFERENCE_COMPLETED",
        authority_epoch=authority_epoch,
        attempt=attempt,
        seed=seed,
        payload={
            **common_payload,
            "noteCount": len(generated.notes),
            "generator": generated.generator_name,
        },
    )
    return generated


def record_gate_event(
    state: VariantState,
    *,
    authority_epoch: int,
    attempt: int,
    seed: int,
    purpose: str,
    acceptance: ChartAcceptance,
) -> None:
    record_journal_event(
        state,
        event_type="GATE_EVALUATED",
        authority_epoch=authority_epoch,
        attempt=attempt,
        seed=seed,
        payload={
            "purpose": purpose,
            "action": acceptance.action.value,
            "gateReport": acceptance.to_report(),
        },
    )


def record_candidate_event(
    state: VariantState,
    *,
    admitted: bool,
    authority_epoch: int,
    attempt: int,
    seed: int,
    purpose: str,
    reason: str,
    acceptance: ChartAcceptance | None = None,
) -> None:
    payload: dict[str, object] = {"purpose": purpose, "reason": reason}
    if acceptance is not None:
        payload["action"] = acceptance.action.value
    record_journal_event(
        state,
        event_type="CANDIDATE_ADMITTED" if admitted else "CANDIDATE_REJECTED",
        authority_epoch=authority_epoch,
        attempt=attempt,
        seed=seed,
        payload=payload,
    )
