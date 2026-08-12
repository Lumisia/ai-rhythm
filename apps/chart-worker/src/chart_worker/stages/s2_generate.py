"""S2: generate each map and retry only invalid Mapperatorinator output."""

import json
from bisect import bisect_right
from pathlib import Path

from chart_worker.analysis.activity import (
    SongBoundaryContract,
    build_song_boundary_contract,
    estimate_music_end_ms,
)
from chart_worker.analysis.grid_alignment import measure_note_grid_alignment
from chart_worker.analysis.onset import OnsetAnalysis
from chart_worker.analysis.song_context import SongAnalysisContext
from chart_worker.errors import ErrorCode, WorkerError
from chart_worker.generation.attempt_journal import AttemptJournal
from chart_worker.generation.candidate_state import (
    Candidate as _Candidate,
)
from chart_worker.generation.candidate_state import (
    VariantState as _VariantState,
)
from chart_worker.generation.candidate_state import (
    candidate_evidence as _candidate_evidence,
)
from chart_worker.generation.family_selection import (
    candidate_stable_id as _family_candidate_stable_id,
)
from chart_worker.generation.family_selection import (
    compare_difficulty_selection as _family_compare_difficulty_selection,
)
from chart_worker.generation.family_selection import (
    compare_song_selection_shadow as _family_compare_song_selection_shadow,
)
from chart_worker.generation.family_selection import (
    family_score as _family_selection_score,
)
from chart_worker.generation.family_selection import (
    first_row_ms as _family_first_row_ms,
)
from chart_worker.generation.family_selection import (
    has_complete_model_family as _family_has_complete_model_family,
)
from chart_worker.generation.family_selection import (
    review_candidates as _family_review_candidates,
)
from chart_worker.generation.family_selection import (
    select_family as _family_select_family,
)
from chart_worker.generation.family_selection import (
    song_selection_context_id as _family_song_selection_context_id,
)
from chart_worker.generation.generation_control import (
    MAX_CRASH_ATTEMPTS,
    MAX_VARIANT_ATTEMPTS,
    AdditionalInferenceBudget,
    RecoveryKind,
)
from chart_worker.generation.inference_execution import (
    error_report_json as _error_report_json,
)
from chart_worker.generation.inference_execution import (
    record_candidate_event as _record_candidate_event,
)
from chart_worker.generation.inference_execution import (
    record_gate_event as _record_gate_event,
)
from chart_worker.generation.inference_execution import (
    require_generation_inputs_unchanged as _require_timing_authority,
)
from chart_worker.generation.inference_execution import (
    run_inference_with_journal as _run_inference_with_journal,
)
from chart_worker.generation.intro_exact_reselection import (
    apply_intro_start_contract as _apply_intro_start_contract,
)
from chart_worker.generation.intro_family_recovery import (
    apply_intro_phrase_family_recovery as _apply_intro_phrase_family_recovery_impl,
)
from chart_worker.generation.intro_family_recovery import (
    intro_candidate_view as _intro_view,
)
from chart_worker.generation.intro_recovery import (
    covers_intro_anchor,
)
from chart_worker.generation.intro_recovery import (
    execute_intro_retry as _execute_intro_retry,
)
from chart_worker.generation.mapperatorinator import ChartGenerator, GeneratedChart
from chart_worker.generation.osu_writer import notes_to_osu_mania
from chart_worker.generation.params import GenerationRequest
from chart_worker.generation.partial_recovery import (
    execute_partial_repair as _execute_partial_repair,
)
from chart_worker.generation.partial_recovery import (
    plan_partial_repair as _plan_partial_repair,
)
from chart_worker.generation.publication_assembler import assemble_publication
from chart_worker.generation.recovery_router import (
    RecoveryRequest,
    intro_phrase_recovery_request,
    plan_recoveries,
    timing_family_recovery_request,
)
from chart_worker.generation.timing_family_recovery import (
    apply_timing_family_recovery as _apply_timing_family_recovery,
)
from chart_worker.generation.timing_family_recovery import (
    timing_family_reviews as _timing_family_reviews,
)
from chart_worker.schema.types import DIFFICULTIES, KEY_MODES
from chart_worker.stages.timing_feedback import (
    MapTimingFailureSignature,
    RetryTimingSignal,
    TimingFailureFamily,
    record_timing_failure,
)
from chart_worker.stages.types import (
    GenerationOutcome,
    PreparedAudio,
    SongTimingAuthority,
)
from chart_worker.validation.difficulty_order import DifficultyOrderReview
from chart_worker.validation.difficulty_selector import (
    DifficultySelectionComparison,
    SelectionMode,
)
from chart_worker.validation.generated_chart import (
    GeneratedChartValidationError,
    validate_generated_chart,
)
from chart_worker.validation.intro_phrase_family import (
    IntroPhraseFamilyReview,
)
from chart_worker.validation.intro_start_contract import (
    IntroStartContract,
    build_intro_start_contract,
    validate_exact_first_row,
)
from chart_worker.validation.outro_family_review import (
    OutroChartView,
    review_outro_family,
)
from chart_worker.validation.quality_gate import (
    ChartAcceptance,
    GateAction,
    GateAxis,
    evaluate_chart_candidate,
)
from chart_worker.validation.serialized_candidate import (
    validate_serialized_candidate as _validate_serialized_candidate,
)
from chart_worker.validation.song_family_selector import SongSelectionComparison
from chart_worker.validation.timing_authority import (
    TimingAuthorityValidationError,
    validate_timing_identity,
)

_VARIANT_COUNT = len(KEY_MODES) * len(DIFFICULTIES)
_RETRYABLE_GENERATION_CODES = {
    ErrorCode.CHART_GENERATION_FAILED,
    ErrorCode.CHART_OSU_PARSE_FAILED,
}


def _should_retry(error: Exception) -> bool:
    if isinstance(error, (GeneratedChartValidationError, TimingAuthorityValidationError)):
        return True
    return isinstance(error, WorkerError) and error.code in _RETRYABLE_GENERATION_CODES


def _is_crash(error: Exception) -> bool:
    """출력물이 없거나 쓸 수 없는 실패. 품질 예산 대신 크래시 예산을 쓴다."""
    return (
        isinstance(error, WorkerError)
        and error.code is ErrorCode.CHART_GENERATION_FAILED
    )


_Selection = tuple[
    dict[str, _VariantState],
    dict[str, _Candidate | None],
    DifficultyOrderReview | None,
]


def _exhausted_error(state: _VariantState) -> WorkerError:
    return WorkerError(
        ErrorCode.CHART_CANDIDATES_EXHAUSTED,
        (
            f"{state.key_mode}K {state.difficulty} exhausted its attempt budget "
            f"(quality {state.budget.quality_attempts}/{MAX_VARIANT_ATTEMPTS}, "
            f"crash {state.budget.crash_attempts}/{MAX_CRASH_ATTEMPTS})"
        ),
        context={
            "key_mode": state.key_mode,
            "difficulty": state.difficulty,
            "seeds": list(state.budget.attempted_seeds),
            "qualityAttempts": state.budget.quality_attempts,
            "crashAttempts": state.budget.crash_attempts,
            "totalAttempts": state.budget.next_attempt - 1,
            "errors": list(state.attempt_errors),
            "attempts": list(state.attempt_evidence),
        },
    )


def _serialized_candidate_text(
    generated: GeneratedChart,
    *,
    authority: SongTimingAuthority,
    prepared: PreparedAudio,
    key_mode: int,
) -> str:
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
    _validate_serialized_candidate(
        osu_text,
        generated,
        authority,
        prepared,
        key_mode,
    )
    return osu_text


def _intro_anchor_covered(
    generated: GeneratedChart,
    authority: SongTimingAuthority,
) -> bool | None:
    leading = authority.leading_coverage
    if leading is None or leading.intro_anchor.status != "CONFIRMED":
        return None
    return covers_intro_anchor(generated.notes, leading.intro_anchor)


def _is_localized_coverage_failure(acceptance: ChartAcceptance) -> bool:
    retry_axes = {
        decision.axis
        for decision in acceptance.decisions
        if decision.action is GateAction.RETRY_MAP
    }
    return retry_axes == {GateAxis.COVERAGE} and bool(acceptance.timing.coverage_gaps)


def _timing_segment_id(authority: SongTimingAuthority, time_ms: int) -> int | None:
    index = bisect_right(
        [event.time_ms for event in authority.bpm_events],
        time_ms,
    ) - 1
    return index if index >= 0 else None


def _timing_failure_signature(
    state: _VariantState,
    authority: SongTimingAuthority,
    *,
    seed: int,
    family: TimingFailureFamily,
    time_ms: int,
    alignment_time_ms: int | None = None,
    evidence: dict[str, object] | None = None,
) -> MapTimingFailureSignature | None:
    segment_id = _timing_segment_id(authority, time_ms)
    if segment_id is None:
        return None
    alignment = measure_note_grid_alignment(
        (alignment_time_ms if alignment_time_ms is not None else time_ms,),
        authority.bpm_events,
    )
    return MapTimingFailureSignature(
        authority_sha256=authority.sha256,
        key_mode=state.key_mode,
        difficulty=state.difficulty,
        seed=seed,
        timing_segment_id=segment_id,
        failure_family=family,
        time_ms=time_ms,
        grid_aligned=alignment.clean_rate == 1.0,
        evidence=evidence,
    )


def _local_segment_has_grid_damage(
    authority: SongTimingAuthority,
    segment_id: int,
) -> bool:
    return authority.local_review is not None and any(
        segment.metrics.index == segment_id and segment.grid_damage
        for segment in authority.local_review.segments
    )


def _acceptance_timing_failures(
    state: _VariantState,
    authority: SongTimingAuthority,
    generated: GeneratedChart,
    acceptance: ChartAcceptance,
    *,
    seed: int,
) -> tuple[MapTimingFailureSignature, ...]:
    signatures = []
    structure_error = acceptance.structure_error
    if structure_error is not None and structure_error["reasonCode"] == "DUPLICATE_NOTE":
        context = structure_error["context"]
        time_ms = context.get("timeMs") if isinstance(context, dict) else None
        lane = context.get("lane") if isinstance(context, dict) else None
        if isinstance(time_ms, int) and isinstance(lane, int):
            matching_collisions = tuple(
                collision.lane == lane and collision.post_time_ms == time_ms
                for collision in generated.resnap_diagnostics.collisions
            )
            observed_resnap = any(matching_collisions)
            collision_evidence = [
                collision.to_report()
                for collision, matches in zip(
                    generated.resnap_diagnostics.collisions,
                    matching_collisions,
                    strict=True,
                )
                if matches
            ]
            signature = _timing_failure_signature(
                state,
                authority,
                seed=seed,
                family=(
                    "RESNAP_COLLISION" if observed_resnap else "DUPLICATE_NOTE"
                ),
                time_ms=time_ms,
                evidence=(
                    {"resnapCollisions": collision_evidence}
                    if collision_evidence
                    else None
                ),
            )
            if signature is not None:
                signatures.append(signature)
    for gap in acceptance.timing.coverage_gaps:
        if gap.position != "MIDDLE":
            continue
        signature = _timing_failure_signature(
            state,
            authority,
            seed=seed,
            family="ACTIVE_MIDDLE_GAP",
            time_ms=(gap.start_ms + gap.end_ms) // 2,
            alignment_time_ms=gap.start_ms,
        )
        if signature is not None and _local_segment_has_grid_damage(
            authority,
            signature.timing_segment_id,
        ):
            signatures.append(signature)
    return tuple(signatures)


def _capture_raw_candidate(
    state: _VariantState,
    *,
    request: GenerationRequest,
    generated: GeneratedChart,
    acceptance: ChartAcceptance,
    workdir: Path,
    attempt: int,
    seed: int,
    prepared: PreparedAudio,
    authority: SongTimingAuthority,
) -> None:
    """품질 축만 거절된 모델 원본 출력을 raw fallback 으로 보존한다.

    룰 기반 채보 생성은 금지 정책이다. 구조와 timing identity 를 통과한
    출력이라면 음악성 판정(COVERAGE/PATTERN/TIMING_ALIGNMENT)이 거절해도
    "쌩 Mapperatorinator" 결과로 발행할 수 있게 남겨 둔다. 선택기는 이
    후보를 항상 정상 후보보다 뒤 순위에 둔다.
    """
    if acceptance.decision(GateAxis.STRUCTURE).action is not GateAction.PASS:
        return
    if acceptance.decision(GateAxis.TIMING_IDENTITY).action is not GateAction.PASS:
        return
    try:
        osu_text = _serialized_candidate_text(
            generated,
            authority=authority,
            prepared=prepared,
            key_mode=state.key_mode,
        )
    except (
        GeneratedChartValidationError,
        TimingAuthorityValidationError,
        WorkerError,
    ):
        # 직렬화까지 통과하지 못하면 플레이 가능성을 보장할 수 없다.
        return
    state.candidates.reject(
        _Candidate(
            request=request,
            generated=generated,
            acceptance=acceptance,
            osu_text=osu_text,
            workdir=workdir,
            attempt=attempt,
            seed=seed,
            provenance="RAW_UNVERIFIED",
            recovery_reason="QUALITY_GATE_REJECTED",
            intro_anchor_covered=_intro_anchor_covered(generated, authority),
        )
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
    authority_epoch: int,
    allow_timing_escalation: bool = True,
) -> _Candidate:
    last_error: Exception | None = None
    boundary = (
        build_song_boundary_contract(
            onset_analysis.activity,
            prepared.normalized.duration_ms,
            enforcement_mode=prepared.boundary_policy_mode,
        )
        if onset_analysis.activity is not None
        else None
    )
    music_end_ms = (
        prepared.normalized.duration_ms
        if boundary is not None and boundary.effective_source == "FULL_DURATION_BASELINE"
        else (
            estimate_music_end_ms(
                onset_analysis.activity,
                prepared.normalized.duration_ms,
            )
            if onset_analysis.activity is not None
            else None
        )
    )
    while state.budget_left:
        attempt = state.budget.next_attempt
        _require_timing_authority(prepared, authority)
        attempt_seed = base_seed + state.flat_index + (attempt - 1) * _VARIANT_COUNT
        state.budget.reserve_attempt(seed=attempt_seed)
        request = GenerationRequest(
            audio_path=prepared.normalized.path,
            timing_reference_path=authority.reference_path,
            key_mode=state.key_mode,
            difficulty=state.difficulty,
            seed=attempt_seed,
            duration_ms=prepared.normalized.duration_ms,
            music_end_ms=music_end_ms,
            generation_end_ms=(boundary.generation_end_ms if boundary else None),
            last_attack_ms=(boundary.last_attack_ms if boundary else None),
            max_note_start_ms=(boundary.max_note_start_ms if boundary else None),
        )
        workdir = (
            run_dir
            / "raw"
            / "work"
            / f"epoch-{authority_epoch}"
            / f"{state.key_mode}k-{state.difficulty.lower()}"
            / f"attempt-{attempt}"
        )
        inference_completed = False
        try:
            generated = _run_inference_with_journal(
                state,
                generator=generator,
                request=request,
                workdir=workdir,
                run_dir=run_dir,
                prepared=prepared,
                authority=authority,
                authority_epoch=authority_epoch,
                attempt=attempt,
                seed=attempt_seed,
                purpose="PRIMARY_ATTEMPT",
            )
            inference_completed = True
            acceptance = evaluate_chart_candidate(
                generated,
                authority,
                onset_analysis,
                requested_key_mode=state.key_mode,
                requested_difficulty=state.difficulty,
                duration_ms=prepared.normalized.duration_ms,
                boundary_policy_mode=prepared.boundary_policy_mode,
            )
            gate_report = acceptance.to_report()
            _record_gate_event(
                state,
                authority_epoch=authority_epoch,
                attempt=attempt,
                seed=attempt_seed,
                purpose="PRIMARY_ATTEMPT",
                acceptance=acceptance,
            )
            evidence = {
                "seed": attempt_seed,
                "workdir": workdir.relative_to(run_dir).as_posix(),
                "gateReport": gate_report,
            }
            if generated.resnap_diagnostics.status != "UNOBSERVED":
                evidence["resnapDiagnostics"] = (
                    generated.resnap_diagnostics.to_report()
                )
            if acceptance.action is GateAction.RETRY_MAP:
                state.budget.record_quality_attempt()
                timing_failures = _acceptance_timing_failures(
                    state,
                    authority,
                    generated,
                    acceptance,
                    seed=attempt_seed,
                )
                if timing_failures:
                    evidence["timingFailureSignatures"] = [
                        signature.to_report() for signature in timing_failures
                    ]
                state.attempt_evidence.append(evidence)
                state.attempt_errors.append(
                    json.dumps(
                        gate_report,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                )
                _record_candidate_event(
                    state,
                    admitted=False,
                    authority_epoch=authority_epoch,
                    attempt=attempt,
                    seed=attempt_seed,
                    purpose="PRIMARY_ATTEMPT",
                    reason="QUALITY_GATE_RETRY",
                    acceptance=acceptance,
                )
                for signature in timing_failures:
                    try:
                        record_timing_failure(state.timing_failures, signature)
                    except RetryTimingSignal:
                        if allow_timing_escalation:
                            raise
                        # authority 가 이미 Super Timing 이라 재선택이
                        # 불가능하다. 곡 전체를 버리는 대신 이 조합의
                        # 재시도만 계속하고, 끝내 실패하면 조합 단위로
                        # 격리한다 (missing → PARTIAL).
                # 구조·identity 를 통과한 거절 후보는 전부 raw fallback
                # 으로도 보존한다. 국소 커버리지 실패는 부분 재생성
                # 소스로도 쓴다 (부분 재생성이 실패해도 raw 는 남는다).
                _capture_raw_candidate(
                    state,
                    request=request,
                    generated=generated,
                    acceptance=acceptance,
                    workdir=workdir,
                    attempt=attempt,
                    seed=attempt_seed,
                    prepared=prepared,
                    authority=authority,
                )
                if _is_localized_coverage_failure(acceptance):
                    osu_text = _serialized_candidate_text(
                        generated,
                        authority=authority,
                        prepared=prepared,
                        key_mode=state.key_mode,
                    )
                    state.candidates.remember_partial_source(
                        _Candidate(
                            request=request,
                            generated=generated,
                            acceptance=acceptance,
                            osu_text=osu_text,
                            workdir=workdir,
                            attempt=attempt,
                            seed=attempt_seed,
                            provenance="RETRY",
                            intro_anchor_covered=_intro_anchor_covered(
                                generated, authority
                            ),
                        )
                    )
                continue

            validate_timing_identity(generated.bpm_events, authority.bpm_events)
            validate_generated_chart(
                generated,
                key_mode=state.key_mode,
                duration_ms=prepared.normalized.duration_ms,
                max_note_start_ms=(boundary.max_note_start_ms if boundary else None),
            )
            osu_text = _serialized_candidate_text(
                generated,
                authority=authority,
                prepared=prepared,
                key_mode=state.key_mode,
            )
            # 성공한 호출도 예산을 쓴다. 사다리 재생성이 같은 조합을
            # 무한히 다시 뽑지 않도록 총 모델 호출을 묶어 둔다.
            state.budget.record_quality_attempt()
            _record_candidate_event(
                state,
                admitted=True,
                authority_epoch=authority_epoch,
                attempt=attempt,
                seed=attempt_seed,
                purpose="PRIMARY_ATTEMPT",
                reason="PRIMARY" if attempt == 1 else "RETRY",
                acceptance=acceptance,
            )
            return _Candidate(
                request=request,
                generated=generated,
                acceptance=acceptance,
                osu_text=osu_text,
                workdir=workdir,
                attempt=attempt,
                seed=attempt_seed,
                provenance="PRIMARY" if attempt == 1 else "RETRY",
                intro_anchor_covered=_intro_anchor_covered(generated, authority),
            )
        except (
            GeneratedChartValidationError,
            TimingAuthorityValidationError,
            WorkerError,
        ) as error:
            if inference_completed:
                _record_candidate_event(
                    state,
                    admitted=False,
                    authority_epoch=authority_epoch,
                    attempt=attempt,
                    seed=attempt_seed,
                    purpose="PRIMARY_ATTEMPT",
                    reason="VALIDATION_ERROR",
                )
            if not _should_retry(error):
                raise
            if _is_crash(error):
                state.budget.record_crash_attempt()
            else:
                state.budget.record_quality_attempt()
            last_error = error
            state.attempt_errors.append(_error_report_json(error))

    error = _exhausted_error(state)
    if last_error is not None:
        raise error from last_error
    raise error


def _review_candidates(candidates: tuple[_Candidate, ...]) -> DifficultyOrderReview:
    return _family_review_candidates(candidates)


def _family_score(
    assignment: tuple[_Candidate | None, ...],
    review: DifficultyOrderReview | None,
) -> tuple[int, int, int, int, tuple[tuple[int, int], ...]]:
    return _family_selection_score(assignment, review)


def _has_complete_model_family(states: dict[str, _VariantState]) -> bool:
    return _family_has_complete_model_family(states)


def _select_family(
    states: dict[str, _VariantState],
) -> tuple[dict[str, _Candidate | None], DifficultyOrderReview | None]:
    return _family_select_family(states)


def _compare_difficulty_selection(
    states: dict[str, _VariantState],
    assignment: dict[str, _Candidate | None],
    *,
    mode: SelectionMode,
) -> tuple[dict[str, _Candidate | None], DifficultySelectionComparison]:
    return _family_compare_difficulty_selection(states, assignment, mode=mode)


def _song_selection_context_id(
    prepared: PreparedAudio,
    authority: SongTimingAuthority,
    boundary: SongBoundaryContract | None,
    intro_contract: IntroStartContract | None = None,
) -> str:
    return _family_song_selection_context_id(
        prepared,
        authority,
        boundary,
        intro_contract,
    )


def _candidate_stable_id(
    candidate: _Candidate,
    *,
    key_mode: int,
    difficulty: str,
    run_dir: Path | None = None,
) -> str:
    return _family_candidate_stable_id(
        candidate,
        key_mode=key_mode,
        difficulty=difficulty,
        run_dir=run_dir,
    )


def _compare_song_selection_shadow(
    selections: list[
        tuple[
            dict[str, _VariantState],
            dict[str, _Candidate | None],
            DifficultyOrderReview | None,
        ]
    ],
    *,
    prepared: PreparedAudio,
    authority: SongTimingAuthority,
    run_dir: Path,
    intro_contract: IntroStartContract,
    boundary: SongBoundaryContract | None,
) -> SongSelectionComparison:
    return _family_compare_song_selection_shadow(
        selections,
        prepared=prepared,
        authority=authority,
        run_dir=run_dir,
        intro_contract=intro_contract,
        boundary=boundary,
    )


def _first_row_ms(candidate: _Candidate) -> int | None:
    return _family_first_row_ms(candidate)


def _selected_candidates(
    selections: list[
        tuple[
            dict[str, _VariantState],
            dict[str, _Candidate | None],
            DifficultyOrderReview | None,
        ]
    ],
) -> tuple[_Candidate, ...]:
    return tuple(
        candidate
        for _states, assignment, _review in selections
        for difficulty in DIFFICULTIES
        if (candidate := assignment[difficulty]) is not None
    )


def _family_review(
    assignment: dict[str, _Candidate | None],
) -> DifficultyOrderReview | None:
    chosen = tuple(
        candidate
        for difficulty in DIFFICULTIES
        if (candidate := assignment[difficulty]) is not None
    )
    return _review_candidates(chosen) if chosen else None


def _apply_intro_phrase_family_recovery(
    selections: list[_Selection],
    song_context: SongAnalysisContext,
    *,
    prepared: PreparedAudio,
    authority: SongTimingAuthority,
    onset_analysis: OnsetAnalysis,
    run_dir: Path,
    generator: ChartGenerator,
    base_seed: int,
    authority_epoch: int,
    inference_budget: AdditionalInferenceBudget,
    allow_model_retry: bool = True,
    block_unresolved: bool = True,
) -> tuple[list[_Selection], tuple[IntroPhraseFamilyReview, ...]]:
    return _apply_intro_phrase_family_recovery_impl(
        selections,
        song_context,
        prepared=prepared,
        authority=authority,
        onset_analysis=onset_analysis,
        run_dir=run_dir,
        generator=generator,
        base_seed=base_seed,
        authority_epoch=authority_epoch,
        inference_budget=inference_budget,
        evaluate_candidate=evaluate_chart_candidate,
        serialize_candidate=_serialized_candidate_text,
        intro_anchor_covered=_intro_anchor_covered,
        allow_model_retry=allow_model_retry,
        block_unresolved=block_unresolved,
    )


def _try_intro_contract_retry(
    state: _VariantState,
    source: _Candidate,
    *,
    prepared: PreparedAudio,
    authority: SongTimingAuthority,
    onset_analysis: OnsetAnalysis,
    run_dir: Path,
    generator: ChartGenerator,
    base_seed: int,
    authority_epoch: int,
    inference_budget: AdditionalInferenceBudget,
    recovery_reason: str = "INTRO_START_CONTRACT",
    evidence_prefix: str = "INTRO_CONTRACT",
    workdir_name: str = "intro-contract-retry",
) -> _Candidate | None:
    return _execute_intro_retry(
        state,
        source,
        prepared=prepared,
        authority=authority,
        onset_analysis=onset_analysis,
        run_dir=run_dir,
        generator=generator,
        base_seed=base_seed,
        authority_epoch=authority_epoch,
        inference_budget=inference_budget,
        evaluate_candidate=evaluate_chart_candidate,
        serialize_candidate=_serialized_candidate_text,
        intro_anchor_covered=_intro_anchor_covered,
        recovery_reason=recovery_reason,
        evidence_prefix=evidence_prefix,
        workdir_name=workdir_name,
    )


def _record_order_retry(
    state: _VariantState,
    review: DifficultyOrderReview,
    *,
    run_dir: Path,
) -> None:
    candidate = state.candidates.admitted[-1]
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


def run_generation(
    prepared: PreparedAudio,
    authority: SongTimingAuthority,
    onset_analysis: OnsetAnalysis,
    run_dir: Path,
    *,
    generator: ChartGenerator,
    seed: int,
    authority_epoch: int = 1,
) -> GenerationOutcome:
    """Generate key-mode pools and retry only failed or inverted labels.

    실패 격리 원칙: 한 조합의 실패는 그 조합에만 갇힌다. 룰 기반 채보는
    만들지 않는다. 게이트를 통과한 후보가 없으면 raw 모델 출력
    (RAW_UNVERIFIED)을 쓰고, 그것도 없으면 그 조합만 missing 으로
    보고한다. 단조성 제약은 후보가 존재하는 난이도 사이에서만 적용한다.
    """
    # timing 재선택 escalation 은 더 나은 authority 후보가 실제로 남아
    # 있을 때만 의미가 있다. 이미 Super Timing 이면 같은 seed 로 같은
    # 결과를 다시 뽑을 뿐이므로, 실패를 조합 단위로 격리한다.
    if authority_epoch < 1:
        raise ValueError("authority_epoch must be positive")
    inference_budget = AdditionalInferenceBudget(limit=1)
    song_context = SongAnalysisContext.build(
        authority,
        onset_analysis,
        duration_ms=prepared.normalized.duration_ms,
        intro_anchor=(
            authority.leading_coverage.intro_anchor
            if authority.leading_coverage is not None
            else None
        ),
    )
    # Standard is the only authority with a genuinely independent timing escalation left.
    # Super Timing and Beat This fallback are already terminal timing authorities.
    allow_timing_escalation = authority.mode == "STANDARD"
    attempt_journal = AttemptJournal(run_dir / "attempt-journal.jsonl")
    state_families: list[dict[str, _VariantState]] = []
    for key_index, key_mode in enumerate(KEY_MODES):
        states = {
            difficulty: _VariantState(
                key_mode=key_mode,
                difficulty=difficulty,
                flat_index=key_index * len(DIFFICULTIES) + difficulty_index,
                journal=attempt_journal,
            )
            for difficulty_index, difficulty in enumerate(DIFFICULTIES)
        }
        for difficulty in DIFFICULTIES:
            state = states[difficulty]
            try:
                state.candidates.admit(
                    _generate_next_pass(
                        state,
                        prepared=prepared,
                        authority=authority,
                        onset_analysis=onset_analysis,
                        run_dir=run_dir,
                        generator=generator,
                        base_seed=seed,
                        authority_epoch=authority_epoch,
                        allow_timing_escalation=allow_timing_escalation,
                    )
                )
            except WorkerError as error:
                if error.code is not ErrorCode.CHART_CANDIDATES_EXHAUSTED:
                    raise
                state.exhausted_error = error

        # 역전 재생성: 게이트 통과 후보가 있는 난이도들 사이에서만.
        # 후보가 하나도 없는 난이도는 기준점으로 쓰지 않는다. 망가진
        # 조합에 맞추려고 멀쩡한 조합을 끌어내리는 일이 없어야 한다.
        while not _has_complete_model_family(states):
            present = tuple(
                states[difficulty].candidates.admitted[-1]
                for difficulty in DIFFICULTIES
                if states[difficulty].candidates.admitted
            )
            if len(present) < 2:
                break
            latest_review = _review_candidates(present)
            if latest_review.status != "RETRY":
                break
            progressed = False
            for difficulty in DIFFICULTIES:
                if difficulty not in latest_review.retry_difficulties:
                    continue
                state = states[difficulty]
                if not state.candidates.admitted or not state.budget_left:
                    continue
                try:
                    candidate = _generate_next_pass(
                        state,
                        prepared=prepared,
                        authority=authority,
                        onset_analysis=onset_analysis,
                        run_dir=run_dir,
                        generator=generator,
                        base_seed=seed,
                        authority_epoch=authority_epoch,
                        allow_timing_escalation=allow_timing_escalation,
                    )
                except WorkerError as error:
                    if error.code is not ErrorCode.CHART_CANDIDATES_EXHAUSTED:
                        raise
                    continue
                _record_order_retry(state, latest_review, run_dir=run_dir)
                state.candidates.admit(candidate)
                progressed = True
                if _has_complete_model_family(states):
                    break
            if not progressed:
                break
        state_families.append(states)

    partial_plans = tuple(
        plan
        for states in state_families
        for state in states.values()
        if not state.candidates.admitted
        if (
            plan := _plan_partial_repair(
                state,
                authority=authority,
                duration_ms=prepared.normalized.duration_ms,
            )
        )
        is not None
    )
    partial_by_request_id = {plan.request.request_id: plan for plan in partial_plans}
    partial_router_plan = plan_recoveries(
        tuple(plan.request for plan in partial_plans),
        available_slots=inference_budget.remaining,
    )
    if partial_plans:
        router_report = partial_router_plan.to_report()
        for plan in partial_plans:
            plan.state.attempt_evidence.append(
                {
                    "reason": "RECOVERY_ROUTER_DECISION",
                    "requestId": plan.request.request_id,
                    "plan": router_report,
                }
            )
    for request in partial_router_plan.selected:
        if not inference_budget.consume():
            raise AssertionError("recovery router exceeded additional inference budget")
        partial_plan = partial_by_request_id[request.request_id]
        repaired = _execute_partial_repair(
            partial_plan,
            prepared=prepared,
            authority=authority,
            onset_analysis=onset_analysis,
            run_dir=run_dir,
            generator=generator,
            base_seed=seed,
            authority_epoch=authority_epoch,
            evaluate_candidate=evaluate_chart_candidate,
            serialize_candidate=_serialized_candidate_text,
            intro_anchor_covered=_intro_anchor_covered,
            should_retry=_should_retry,
        )
        if repaired is not None:
            partial_plan.state.candidates.admit(repaired)

    selections: list[
        tuple[
            dict[str, _VariantState],
            dict[str, _Candidate | None],
            DifficultyOrderReview | None,
        ]
    ] = []
    for states in state_families:
        assignment, order_review = _select_family(states)
        selections.append((states, assignment, order_review))

    selection_shadows: tuple[DifficultySelectionComparison, ...] = ()
    selector_mode = prepared.difficulty_selector_mode
    if selector_mode != "CURRENT":
        compared = tuple(
            (
                states,
                *_compare_difficulty_selection(
                    states,
                    assignment,
                    mode=selector_mode,
                ),
            )
            for states, assignment, _review in selections
        )
        selections = []
        for states, assignment, _comparison in compared:
            chosen = tuple(
                candidate
                for difficulty in DIFFICULTIES
                if (candidate := assignment[difficulty]) is not None
            )
            selections.append(
                (
                    states,
                    assignment,
                    _review_candidates(chosen) if chosen else None,
                )
            )
        selection_shadows = tuple(
            comparison for _states, _assignment, comparison in compared
        )

    # First apply every zero-cost reselection. Only then compare the remaining
    # model-backed recoveries so source-code order cannot spend the song budget.
    selections, intro_phrase_family_reviews = _apply_intro_phrase_family_recovery(
        selections,
        song_context,
        prepared=prepared,
        authority=authority,
        onset_analysis=onset_analysis,
        run_dir=run_dir,
        generator=generator,
        base_seed=seed,
        authority_epoch=authority_epoch,
        inference_budget=inference_budget,
        allow_model_retry=False,
        block_unresolved=False,
    )
    selections, timing_family_reviews = _apply_timing_family_recovery(
        selections,
        prepared=prepared,
        authority=authority,
        onset_analysis=onset_analysis,
        run_dir=run_dir,
        generator=generator,
        base_seed=seed,
        authority_epoch=authority_epoch,
        inference_budget=inference_budget,
        evaluate_candidate=evaluate_chart_candidate,
        serialize_candidate=_serialized_candidate_text,
        intro_anchor_covered=_intro_anchor_covered,
        allow_model_retry=False,
    )
    unresolved_intro = sorted(
        (review for review in intro_phrase_family_reviews if review.status == "DEFECT"),
        key=lambda review: (
            review.gap_delta_ms or 0,
            review.gap_ratio or 0.0,
            -review.hard.key_mode,
        ),
        reverse=True,
    )
    timing_outliers = sorted(
        (review for review in timing_family_reviews if review.status == "OUTLIER"),
        key=lambda review: (
            review.overall_sibling_gap or 0.0,
            review.longest_local_outlier_run,
        ),
        reverse=True,
    )
    family_requests: list[RecoveryRequest] = []
    if unresolved_intro:
        family_requests.append(
            intro_phrase_recovery_request(
                key_mode=unresolved_intro[0].hard.key_mode,
                song_duration_ms=prepared.normalized.duration_ms,
            )
        )
    if timing_outliers:
        timing_target = timing_outliers[0]
        if (
            timing_target.target_key_mode is not None
            and timing_target.difficulty is not None
        ):
            family_requests.append(
                timing_family_recovery_request(
                    key_mode=timing_target.target_key_mode,
                    difficulty=timing_target.difficulty,
                    song_duration_ms=prepared.normalized.duration_ms,
                )
            )
    family_router_plan = plan_recoveries(
        tuple(family_requests),
        available_slots=inference_budget.remaining,
    )
    if family_requests:
        router_report = family_router_plan.to_report()
        for request in family_requests:
            state = next(
                states[request.difficulty]
                for states, _assignment, _review in selections
                if states[request.difficulty].key_mode == request.key_mode
            )
            state.attempt_evidence.append(
                {
                    "reason": "RECOVERY_ROUTER_DECISION",
                    "requestId": request.request_id,
                    "plan": router_report,
                }
            )
    selected_family_request = next(iter(family_router_plan.selected), None)
    if selected_family_request is not None:
        if selected_family_request.kind is RecoveryKind.INTRO:
            selections, intro_phrase_family_reviews = (
                _apply_intro_phrase_family_recovery(
                    selections,
                    song_context,
                    prepared=prepared,
                    authority=authority,
                    onset_analysis=onset_analysis,
                    run_dir=run_dir,
                    generator=generator,
                    base_seed=seed,
                    authority_epoch=authority_epoch,
                    inference_budget=inference_budget,
                    block_unresolved=False,
                )
            )
        elif selected_family_request.kind is RecoveryKind.TIMING_FAMILY:
            selections, timing_family_reviews = _apply_timing_family_recovery(
                selections,
                prepared=prepared,
                authority=authority,
                onset_analysis=onset_analysis,
                run_dir=run_dir,
                generator=generator,
                base_seed=seed,
                authority_epoch=authority_epoch,
                inference_budget=inference_budget,
                evaluate_candidate=evaluate_chart_candidate,
                serialize_candidate=_serialized_candidate_text,
                intro_anchor_covered=_intro_anchor_covered,
            )
    selections, intro_start_contract, intro_contract_review = (
        _apply_intro_start_contract(
            selections,
            song_context,
        )
    )
    # Exact first-row equality remains free. Revalidate the phrase contract
    # afterwards, but do not start a second inference layer.
    selections, intro_phrase_family_reviews = _apply_intro_phrase_family_recovery(
        selections,
        song_context,
        prepared=prepared,
        authority=authority,
        onset_analysis=onset_analysis,
        run_dir=run_dir,
        generator=generator,
        base_seed=seed,
        authority_epoch=authority_epoch,
        inference_budget=inference_budget,
        allow_model_retry=False,
    )
    final_intro_candidates = _selected_candidates(selections)
    final_intro_views = tuple(
        _intro_view(candidate, song_context) for candidate in final_intro_candidates
    )
    intro_start_contract = build_intro_start_contract(
        song_context,
        final_intro_views,
    )
    intro_contract_review = validate_exact_first_row(
        intro_start_contract,
        final_intro_views,
        corrected_count=intro_contract_review.corrected_count,
        correction_reasons=intro_contract_review.correction_reasons,
    )
    timing_family_reviews = _timing_family_reviews(selections)
    song_selection_shadow = _compare_song_selection_shadow(
        selections,
        prepared=prepared,
        authority=authority,
        run_dir=run_dir,
        intro_contract=intro_start_contract,
        boundary=(
            build_song_boundary_contract(
                onset_analysis.activity,
                prepared.normalized.duration_ms,
                enforcement_mode=prepared.boundary_policy_mode,
            )
            if onset_analysis.activity is not None
            else None
        ),
    )
    publication = assemble_publication(
        selections,
        prepared=prepared,
        authority=authority,
        run_dir=run_dir,
    )
    variants = publication.variants
    missing = publication.missing
    outro_family_review = review_outro_family(
        tuple(
            OutroChartView(
                key_mode=variant.key_mode,
                difficulty=variant.difficulty,
                last_note_start_ms=max(
                    (note.time_ms for note in variant.generated.notes),
                    default=0,
                ),
                last_note_end_ms=max(
                    (
                        note.time_ms
                        + (note.duration_ms if note.kind == "HOLD" else 0)
                        for note in variant.generated.notes
                    ),
                    default=0,
                ),
            )
            for variant in variants
        )
    )
    return GenerationOutcome(
        variants=tuple(variants),
        missing=tuple(missing),
        difficulty_selection_shadows=selection_shadows,
        song_selection_shadow=song_selection_shadow,
        intro_start_contract=intro_start_contract,
        intro_contract_review=intro_contract_review,
        intro_phrase_family_reviews=intro_phrase_family_reviews,
        timing_family_reviews=timing_family_reviews,
        outro_family_review=outro_family_review,
        additional_inference_calls=inference_budget.used,
    )
