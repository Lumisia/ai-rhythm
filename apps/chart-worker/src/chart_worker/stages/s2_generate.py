"""S2: generate each map and retry only invalid Mapperatorinator output."""

import json
from dataclasses import dataclass, field
from itertools import product
from pathlib import Path

from chart_worker.analysis.onset import OnsetAnalysis
from chart_worker.errors import ErrorCode, WorkerError
from chart_worker.generation.mapperatorinator import ChartGenerator, GeneratedChart
from chart_worker.generation.osu_parser import parse_osu_mania
from chart_worker.generation.osu_writer import notes_to_osu_mania
from chart_worker.generation.params import GenerationRequest
from chart_worker.hashing import sha256_file
from chart_worker.schema.types import DIFFICULTIES, KEY_MODES
from chart_worker.stages.types import GeneratedVariant, PreparedAudio, SongTimingAuthority
from chart_worker.validation.difficulty_order import (
    DifficultyOrderReview,
    review_difficulty_order,
)
from chart_worker.validation.generated_chart import (
    GeneratedChartValidationError,
    validate_generated_chart,
)
from chart_worker.validation.quality_gate import (
    ChartAcceptance,
    GateAction,
    evaluate_chart_candidate,
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


def _note_projection(generated: GeneratedChart) -> tuple[tuple[object, ...], ...]:
    return tuple(
        (note.time_ms, note.lane, note.kind, note.duration_ms)
        for note in generated.notes
    )


def _validate_serialized_candidate(
    osu_text: str,
    generated: GeneratedChart,
    authority: SongTimingAuthority,
    prepared: PreparedAudio,
    key_mode: int,
) -> None:
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
            "serialized MAP key mode differs from generated object"
        )
    if _note_projection(parsed_chart) != _note_projection(generated):
        raise GeneratedChartValidationError(
            "serialized MAP note fields differ from generated object"
        )


@dataclass(frozen=True, slots=True)
class _Candidate:
    request: GenerationRequest
    generated: GeneratedChart
    acceptance: ChartAcceptance
    osu_text: str
    workdir: Path
    attempt: int
    seed: int


@dataclass(slots=True)
class _VariantState:
    key_mode: int
    difficulty: str
    flat_index: int
    next_attempt: int = 1
    attempt_errors: list[str] = field(default_factory=list)
    attempt_evidence: list[dict[str, object]] = field(default_factory=list)
    attempted_seeds: list[int] = field(default_factory=list)
    pool: list[_Candidate] = field(default_factory=list)


def _exhausted_error(state: _VariantState) -> WorkerError:
    return WorkerError(
        ErrorCode.CHART_CANDIDATES_EXHAUSTED,
        f"{state.key_mode}K {state.difficulty} failed {MAX_VARIANT_ATTEMPTS} attempts",
        context={
            "key_mode": state.key_mode,
            "difficulty": state.difficulty,
            "seeds": list(state.attempted_seeds),
            "errors": list(state.attempt_errors),
            "attempts": list(state.attempt_evidence),
        },
    )


def _generate_next_pass(
    state: _VariantState,
    *,
    prepared: PreparedAudio,
    authority: SongTimingAuthority,
    onset_analysis: OnsetAnalysis,
    run_dir: Path,
    generator: ChartGenerator,
    base_seed: int,
) -> _Candidate:
    last_error: Exception | None = None
    while state.next_attempt <= MAX_VARIANT_ATTEMPTS:
        attempt = state.next_attempt
        state.next_attempt += 1
        _require_timing_authority(prepared, authority)
        attempt_seed = base_seed + state.flat_index + (attempt - 1) * _VARIANT_COUNT
        state.attempted_seeds.append(attempt_seed)
        request = GenerationRequest(
            audio_path=prepared.normalized.path,
            timing_reference_path=authority.reference_path,
            key_mode=state.key_mode,
            difficulty=state.difficulty,
            seed=attempt_seed,
            duration_ms=prepared.normalized.duration_ms,
        )
        workdir = (
            run_dir
            / "raw"
            / "work"
            / f"{state.key_mode}k-{state.difficulty.lower()}"
            / f"attempt-{attempt}"
        )
        try:
            try:
                generated = generator.generate_map(request, workdir)
            finally:
                _require_timing_authority(prepared, authority)
            acceptance = evaluate_chart_candidate(
                generated,
                authority,
                onset_analysis,
                requested_key_mode=state.key_mode,
                requested_difficulty=state.difficulty,
                duration_ms=prepared.normalized.duration_ms,
            )
            gate_report = acceptance.to_report()
            evidence = {
                "seed": attempt_seed,
                "workdir": workdir.relative_to(run_dir).as_posix(),
                "gateReport": gate_report,
            }
            if acceptance.action is GateAction.RETRY_MAP:
                state.attempt_evidence.append(evidence)
                state.attempt_errors.append(
                    json.dumps(
                        gate_report,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                )
                continue

            validate_timing_identity(generated.bpm_events, authority.bpm_events)
            validate_generated_chart(
                generated,
                key_mode=state.key_mode,
                duration_ms=prepared.normalized.duration_ms,
            )
            first_timing = generated.bpm_events[0]
            osu_text = generated.osu_text or notes_to_osu_mania(
                generated.notes,
                key_mode=state.key_mode,
                bpm=first_timing.bpm,
                offset_ms=first_timing.time_ms,
                audio_filename=prepared.normalized.path.name,
                title=prepared.normalized.path.stem,
                bpm_events=generated.bpm_events,
            )
            _validate_serialized_candidate(
                osu_text,
                generated,
                authority,
                prepared,
                state.key_mode,
            )
            return _Candidate(
                request=request,
                generated=generated,
                acceptance=acceptance,
                osu_text=osu_text,
                workdir=workdir,
                attempt=attempt,
                seed=attempt_seed,
            )
        except (
            GeneratedChartValidationError,
            TimingAuthorityValidationError,
            WorkerError,
        ) as error:
            if not _should_retry(error):
                raise
            last_error = error
            state.attempt_errors.append(str(error))

    error = _exhausted_error(state)
    if last_error is not None:
        raise error from last_error
    raise error


def _review_candidates(candidates: tuple[_Candidate, ...]) -> DifficultyOrderReview:
    profiles = {}
    for candidate in candidates:
        if candidate.acceptance.profile is None:
            raise ValueError("PASS candidate must carry a chart quality profile")
        profiles[candidate.request.difficulty] = candidate.acceptance.profile.difficulty
    return review_difficulty_order(profiles)


def _select_earliest_monotonic(
    states: dict[str, _VariantState],
) -> tuple[tuple[_Candidate, ...], DifficultyOrderReview] | None:
    ordered_pools = tuple(
        tuple(sorted(states[difficulty].pool, key=lambda candidate: candidate.seed))
        for difficulty in DIFFICULTIES
    )
    for candidates in product(*ordered_pools):
        review = _review_candidates(candidates)
        if review.status != "RETRY":
            return candidates, review
    return None


def _record_order_retry(
    state: _VariantState,
    review: DifficultyOrderReview,
    *,
    run_dir: Path,
) -> None:
    candidate = state.pool[-1]
    evidence = _candidate_evidence(
        candidate,
        reason="DIFFICULTY_ORDER_INVERTED",
        run_dir=run_dir,
    )
    evidence["difficultyOrder"] = review.to_report()
    state.attempt_evidence.append(evidence)
    state.attempt_errors.append(
        json.dumps(
            evidence,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    )


def _candidate_evidence(
    candidate: _Candidate,
    *,
    reason: str,
    run_dir: Path,
) -> dict[str, object]:
    return {
        "seed": candidate.seed,
        "attempt": candidate.attempt,
        "workdir": candidate.workdir.relative_to(run_dir).as_posix(),
        "reason": reason,
        "gateReport": candidate.acceptance.to_report(),
        "serializationValidated": True,
    }


def _record_unselected_candidates(
    state: _VariantState,
    selected: _Candidate,
    review: DifficultyOrderReview,
    *,
    run_dir: Path,
) -> None:
    for candidate in state.pool:
        if candidate is selected:
            continue
        evidence = _candidate_evidence(
            candidate,
            reason="NOT_SELECTED_EARLIEST_MONOTONIC_COMBINATION",
            run_dir=run_dir,
        )
        evidence["selectedDifficultyOrder"] = review.to_report()
        state.attempt_evidence.append(evidence)


def _promote_key_mode(
    candidates: tuple[_Candidate, ...],
    *,
    prepared: PreparedAudio,
    authority: SongTimingAuthority,
    run_dir: Path,
) -> tuple[Path, ...]:
    raw_paths = tuple(
        run_dir / "raw" / f"{candidate.request.key_mode}k-{candidate.request.difficulty.lower()}.osu"
        for candidate in candidates
    )
    raw_paths[0].parent.mkdir(parents=True, exist_ok=True)
    try:
        for candidate, raw_path in zip(candidates, raw_paths, strict=True):
            raw_path.write_text(candidate.osu_text, encoding="utf-8")
            _validate_serialized_candidate(
                raw_path.read_text(encoding="utf-8-sig"),
                candidate.generated,
                authority,
                prepared,
                candidate.request.key_mode,
            )
    except (
        GeneratedChartValidationError,
        TimingAuthorityValidationError,
        WorkerError,
        OSError,
    ) as error:
        for raw_path in raw_paths:
            raw_path.unlink(missing_ok=True)
        cause_code = (
            error.code.value if isinstance(error, WorkerError) else type(error).__name__
        )
        raise WorkerError(
            ErrorCode.CHART_CANDIDATES_EXHAUSTED,
            f"{candidates[0].request.key_mode}K selected charts failed stable promotion",
            context={
                "key_mode": candidates[0].request.key_mode,
                "failure_stage": "PROMOTION",
                "paths": [raw_path.relative_to(run_dir).as_posix() for raw_path in raw_paths],
                "selected_seeds": [candidate.seed for candidate in candidates],
                "cause_code": cause_code,
                "cause": str(error),
            },
        ) from error
    return raw_paths


def run_generation(
    prepared: PreparedAudio,
    authority: SongTimingAuthority,
    onset_analysis: OnsetAnalysis,
    run_dir: Path,
    *,
    generator: ChartGenerator,
    seed: int,
) -> tuple[GeneratedVariant, ...]:
    """Generate key-mode pools and retry only failed or inverted labels."""
    variants: list[GeneratedVariant] = []
    for key_index, key_mode in enumerate(KEY_MODES):
        states = {
            difficulty: _VariantState(
                key_mode=key_mode,
                difficulty=difficulty,
                flat_index=key_index * len(DIFFICULTIES) + difficulty_index,
            )
            for difficulty_index, difficulty in enumerate(DIFFICULTIES)
        }
        for difficulty in DIFFICULTIES:
            state = states[difficulty]
            state.pool.append(
                _generate_next_pass(
                    state,
                    prepared=prepared,
                    authority=authority,
                    onset_analysis=onset_analysis,
                    run_dir=run_dir,
                    generator=generator,
                    base_seed=seed,
                )
            )

        while True:
            selected = _select_earliest_monotonic(states)
            if selected is not None:
                candidates, order_review = selected
                break

            latest_candidates = tuple(
                states[difficulty].pool[-1] for difficulty in DIFFICULTIES
            )
            latest_review = _review_candidates(latest_candidates)
            retry_selection = None
            for difficulty in DIFFICULTIES:
                if difficulty not in latest_review.retry_difficulties:
                    continue
                state = states[difficulty]
                _record_order_retry(state, latest_review, run_dir=run_dir)
                state.pool.append(
                    _generate_next_pass(
                        state,
                        prepared=prepared,
                        authority=authority,
                        onset_analysis=onset_analysis,
                        run_dir=run_dir,
                        generator=generator,
                        base_seed=seed,
                    )
                )
                retry_selection = _select_earliest_monotonic(states)
                if retry_selection is not None:
                    break
            if retry_selection is not None:
                candidates, order_review = retry_selection
                break

        raw_paths = _promote_key_mode(
            candidates,
            prepared=prepared,
            authority=authority,
            run_dir=run_dir,
        )
        for candidate, raw_path in zip(candidates, raw_paths, strict=True):
            state = states[candidate.request.difficulty]
            _record_unselected_candidates(
                state,
                candidate,
                order_review,
                run_dir=run_dir,
            )
            variants.append(
                GeneratedVariant(
                    key_mode=key_mode,
                    difficulty=candidate.request.difficulty,
                    requested_star=candidate.request.requested_star,
                    raw_osu_path=raw_path,
                    generated=candidate.generated,
                    cfg_scale=candidate.request.cfg_scale,
                    attempt=candidate.attempt,
                    attempt_errors=tuple(state.attempt_errors),
                    attempt_evidence=tuple(state.attempt_evidence),
                    timing_authority_sha256=authority.sha256,
                    acceptance=candidate.acceptance,
                    candidate_count=len(state.pool),
                    generation_attempt_count=state.next_attempt - 1,
                    selected_seed=candidate.seed,
                    difficulty_order=order_review,
                )
            )
    return tuple(variants)
