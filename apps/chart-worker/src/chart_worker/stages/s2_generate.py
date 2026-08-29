"""S2: generate each map and retry only invalid Mapperatorinator output."""

import json
from bisect import bisect_right
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from time import perf_counter

from chart_worker.analysis.activity import (
    SongBoundaryContract,
    build_song_boundary_contract,
    estimate_music_end_ms,
)
from chart_worker.analysis.coverage_opportunity import CoverageKind
from chart_worker.analysis.grid_alignment import measure_note_grid_alignment
from chart_worker.analysis.onset import OnsetAnalysis
from chart_worker.analysis.song_context import SongAnalysisContext
from chart_worker.analysis.timing_diagnostics import TimingCoverageGap
from chart_worker.errors import ErrorCode, WorkerError
from chart_worker.generation.attempt_journal import AttemptJournal
from chart_worker.generation.candidate_payload_store import (
    persist_candidate_payload,
    verify_candidate_payload,
)
from chart_worker.generation.candidate_state import (
    Candidate as _Candidate,
)
from chart_worker.generation.candidate_state import (
    VariantState as _VariantState,
)
from chart_worker.generation.candidate_state import (
    candidate_quality_snapshot as _candidate_quality_snapshot,
)
from chart_worker.generation.coverage_repair import build_coverage_repair_chart
from chart_worker.generation.difficulty_family_compiler import (
    DifficultyFamilyCompilerDecision,
    DifficultyFamilyCompilerSlot,
    compile_difficulty_family_shadow,
    materialize_compiled_family,
    persist_difficulty_family_compiler_payloads,
)
from chart_worker.generation.difficulty_shadow_challenger import (
    bind_required_gameplay_interval,
    build_required_gameplay_evidence_for_shadow,
    choose_difficulty_shadow_target,
    difficulty_shadow_slots,
    difficulty_shadow_target_failure_reason,
    execute_difficulty_shadow_challenger,
    execute_difficulty_shadow_full_map_challenger,
    plan_difficulty_shadow_full_map,
    plan_difficulty_shadow_partial_repair,
)
from chart_worker.generation.family_selection import (
    apply_safe_family_assignment as _family_apply_safe_family_assignment,
)
from chart_worker.generation.family_selection import (
    build_song_selection_evidence_v3 as _build_song_selection_evidence_v3,
)
from chart_worker.generation.family_selection import (
    candidate_stable_id as _family_candidate_stable_id,
)
from chart_worker.generation.family_selection import (
    compare_difficulty_selection as _family_compare_difficulty_selection,
)
from chart_worker.generation.family_selection import (
    compare_song_selection as _family_compare_song_selection,
)
from chart_worker.generation.family_selection import (
    family_score as _family_selection_score,
)
from chart_worker.generation.family_selection import (
    first_row_ms as _family_first_row_ms,
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
    error_report as _error_report,
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
from chart_worker.generation.intro_family_recovery import (
    intro_phrase_family_reviews as _intro_phrase_family_reviews,
)
from chart_worker.generation.intro_family_recovery import (
    intro_recovery_targets as _intro_recovery_targets,
)
from chart_worker.generation.intro_recovery import (
    covers_intro_anchor,
)
from chart_worker.generation.intro_recovery import (
    execute_intro_retry as _execute_intro_retry,
)
from chart_worker.generation.intro_recovery import (
    intro_phrase_recovery_end_ms as _intro_phrase_recovery_end_ms,
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
from chart_worker.generation.recovery import (
    build_recovery_chart,
    select_recovery_plan,
)
from chart_worker.generation.recovery_router import (
    RecoveryRequest,
    difficulty_shadow_recovery_request,
    intro_region_recovery_request,
    plan_recoveries,
    timing_family_recovery_request,
)
from chart_worker.generation.required_gameplay_interval import (
    RequiredGameplayIntervalMode,
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
from chart_worker.validation.candidate_replacement import (
    decide_candidate_replacement as _decide_candidate_replacement,
)
from chart_worker.validation.coverage_family_review import (
    CoverageFamilyVerdict,
    CoverageGapMember,
    review_coverage_family,
)
from chart_worker.validation.difficulty_order import DifficultyOrderReview
from chart_worker.validation.difficulty_selector import (
    DifficultySelectionComparison,
    SelectionMode,
)
from chart_worker.validation.final_difficulty_family import (
    CURRENT_DIFFICULTY_FAMILY_CALIBRATION_STATE,
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
    CandidateDisposition,
    ChartAcceptance,
    GateAction,
    GateAxis,
    evaluate_chart_candidate,
)
from chart_worker.validation.safe_family_assignment import (
    SafeFamilyAssignmentDecision,
)
from chart_worker.validation.serialized_candidate import (
    validate_serialized_candidate as _validate_serialized_candidate,
)
from chart_worker.validation.song_family_selector import (
    MATCHED_F1_EPSILON,
    MATCHED_PRECISION_EPSILON,
    SongSelectionComparison,
)
from chart_worker.validation.song_family_selector_v3 import (
    evaluate_shadow_v3_proposal,
)
from chart_worker.validation.timing_authority import (
    TimingAuthorityValidationError,
    validate_timing_identity,
)
from chart_worker.validation.timing_integrity import TimingIntegrityStatus

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
    return isinstance(error, WorkerError) and error.code is ErrorCode.CHART_GENERATION_FAILED


def _is_tail_repair_exhausted(error: Exception) -> bool:
    return isinstance(error, WorkerError) and error.code is ErrorCode.MANIA_TAIL_REPAIR_EXHAUSTED


def _block_full_length_retry_after_tail_exhaustion(
    state: _VariantState,
    error: WorkerError,
    *,
    partial_attempt: int | None = None,
    partial_seed: int | None = None,
) -> None:
    if (partial_attempt is None) != (partial_seed is None):
        raise AssertionError("partial attempt and seed must be reported together")
    if partial_attempt is not None and partial_seed is not None:
        state.partial_attempts.append(partial_attempt)
        state.partial_seeds.append(partial_seed)
    state.full_length_retry_blocked_by = dict(error.context)
    state.publication_block_reason = error.code.value
    state.attempt_errors.append(_error_report_json(error))
    evidence: dict[str, object] = {
        "reason": error.code.value,
        "context": dict(error.context),
        "outerFullLengthRetrySuppressed": True,
        "totalAttempts": state.budget.next_attempt - 1 + len(state.partial_attempts),
        "qualityAttempts": state.budget.quality_attempts,
        "crashAttempts": state.budget.crash_attempts,
    }
    if partial_attempt is not None and partial_seed is not None:
        evidence["partialAttempt"] = partial_attempt
        evidence["partialSeed"] = partial_seed
    state.attempt_evidence.append(evidence)


_Selection = tuple[
    dict[str, _VariantState],
    dict[str, _Candidate | None],
    DifficultyOrderReview | None,
]


def _exhausted_error(state: _VariantState) -> WorkerError:
    tail_blocked = state.full_length_retry_blocked_by is not None
    stop_reason = (
        state.full_length_retry_blocked_by.get("reason")
        if state.full_length_retry_blocked_by is not None
        else None
    )
    stop_description = {
        "HARD_SAFE_RAW_AVAILABLE": "stopped after preserving a hard-safe raw result",
        "SERIALIZATION_CONTRACT_FAILED": "stopped after deterministic serialization failure",
    }.get(stop_reason, "stopped after bounded Mania tail repair")
    context: dict[str, object] = {
        "key_mode": state.key_mode,
        "difficulty": state.difficulty,
        "seeds": [*state.budget.attempted_seeds, *state.partial_seeds],
        "qualityAttempts": state.budget.quality_attempts,
        "crashAttempts": state.budget.crash_attempts,
        "totalAttempts": state.budget.next_attempt - 1 + len(state.partial_attempts),
        "errors": list(state.attempt_errors),
        "attempts": list(state.attempt_evidence),
    }
    if state.partial_attempts:
        context["partialAttempts"] = list(state.partial_attempts)
        context["partialSeeds"] = list(state.partial_seeds)
    if tail_blocked:
        context["fullLengthRetryBlockedBy"] = state.full_length_retry_blocked_by
    return WorkerError(
        ErrorCode.CHART_CANDIDATES_EXHAUSTED,
        (
            f"{state.key_mode}K {state.difficulty} "
            f"{stop_description if tail_blocked else 'exhausted its attempt budget'} "
            f"(quality {state.budget.quality_attempts}/{MAX_VARIANT_ATTEMPTS}, "
            f"crash {state.budget.crash_attempts}/{MAX_CRASH_ATTEMPTS})"
        ),
        context=context,
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
    index = (
        bisect_right(
            [event.time_ms for event in authority.bpm_events],
            time_ms,
        )
        - 1
    )
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
                family=("RESNAP_COLLISION" if observed_resnap else "DUPLICATE_NOTE"),
                time_ms=time_ms,
                evidence=({"resnapCollisions": collision_evidence} if collision_evidence else None),
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
) -> bool:
    """Keep a model result only when every playability-critical axis passed."""
    if not _is_hard_safe_playtest_candidate(acceptance):
        return False
    osu_text = _serialized_candidate_text(
        generated,
        authority=authority,
        prepared=prepared,
        key_mode=state.key_mode,
    )
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
    return True


def _block_full_length_retry(
    state: _VariantState,
    *,
    reason: str,
    seed: int,
    attempt: int,
) -> None:
    state.full_length_retry_blocked_by = {
        "reason": reason,
        "policyVersion": "FULL_LENGTH_RETRY_SUPPRESSION_V1",
        "seed": seed,
        "attempt": attempt,
    }
    state.publication_block_reason = (
        "QUALITY_GATE_REJECTED" if reason == "HARD_SAFE_RAW_AVAILABLE" else reason
    )


def _is_hard_safe_playtest_candidate(acceptance: ChartAcceptance) -> bool:
    return all(
        acceptance.decision(axis).action is GateAction.PASS
        for axis in (
            GateAxis.STRUCTURE,
            GateAxis.TIMING_IDENTITY,
            GateAxis.SONG_BOUNDS,
        )
    )


def _raw_playtest_score(
    candidate: _Candidate,
) -> tuple[int, int, int, float, int, int]:
    acceptance = candidate.acceptance
    noncoverage_retries = sum(
        decision.action is GateAction.RETRY_MAP
        for decision in acceptance.decisions
        if decision.axis not in {GateAxis.STRUCTURE, GateAxis.TIMING_IDENTITY, GateAxis.COVERAGE}
    )
    attack_gaps = tuple(
        gap
        for gap in acceptance.timing.coverage_gaps
        if gap.opportunity is None or gap.opportunity.kind.value == "ATTACK_REQUIRED"
    )
    precision = acceptance.timing.overall.matched_precision_50
    return (
        noncoverage_retries,
        len(attack_gaps),
        sum(gap.end_ms - gap.start_ms for gap in attack_gaps),
        -precision if precision is not None else 0.0,
        candidate.attempt,
        candidate.seed,
    )


def _attack_gap_summary(acceptance: ChartAcceptance) -> tuple[int, int]:
    gaps = tuple(
        gap
        for gap in acceptance.timing.coverage_gaps
        if gap.opportunity is None or gap.opportunity.kind.value == "ATTACK_REQUIRED"
    )
    return len(gaps), sum(gap.end_ms - gap.start_ms for gap in gaps)


def _gate_action_rank(action: GateAction) -> int:
    return {
        GateAction.PASS: 0,
        GateAction.REVIEW: 1,
        GateAction.RETRY_MAP: 2,
    }[action]


def _playtest_candidate_score(
    candidate: _Candidate,
) -> tuple[int, int, int, int, int, int, int]:
    attack_count, attack_total_ms = _attack_gap_summary(candidate.acceptance)
    provenance_rank = {
        "COVERAGE_REPAIR": 0,
        "SAFE_FALLBACK": 1,
        "RAW_UNVERIFIED": 2,
    }.get(candidate.provenance, 0)
    return (
        attack_count,
        attack_total_ms,
        _gate_action_rank(candidate.acceptance.decision(GateAxis.TIMING_ALIGNMENT).action),
        provenance_rank,
        _gate_action_rank(candidate.acceptance.decision(GateAxis.COVERAGE).action),
        candidate.attempt,
        candidate.seed,
    )


def _coverage_repair_preserves_timing(
    source: ChartAcceptance,
    repaired: ChartAcceptance,
) -> bool:
    """Reject deterministic repair that degrades its own source candidate.

    This is intentionally a source-local guard.  It does not claim that F1 is
    musical quality or compare unrelated fallback patterns with different
    absolute difficulty; it only prevents a mutation from making its input's
    one-to-one onset agreement materially worse.
    """

    source_metrics = source.timing.overall
    repaired_metrics = repaired.timing.overall
    for baseline, challenger, epsilon in (
        (
            source_metrics.matched_f1_50,
            repaired_metrics.matched_f1_50,
            MATCHED_F1_EPSILON,
        ),
        (
            source_metrics.matched_precision_50,
            repaired_metrics.matched_precision_50,
            MATCHED_PRECISION_EPSILON,
        ),
    ):
        if baseline is not None and (
            challenger is None or challenger < baseline - epsilon
        ):
            return False
    return True


def _interval_substantially_overlaps_gap(
    interval: tuple[int, int],
    gap: TimingCoverageGap,
) -> bool:
    start_ms, end_ms = interval
    intersection_ms = min(end_ms, gap.end_ms) - max(start_ms, gap.start_ms)
    if intersection_ms <= 0:
        return False
    shorter_ms = min(end_ms - start_ms, gap.end_ms - gap.start_ms)
    return intersection_ms / shorter_ms >= 0.5


def _coverage_repair_candidate(
    state: _VariantState,
    source: _Candidate,
    *,
    prepared: PreparedAudio,
    authority: SongTimingAuthority,
    onset_analysis: OnsetAnalysis,
    approved_intervals: tuple[tuple[int, int], ...] = (),
) -> _Candidate:
    repaired, plan = build_coverage_repair_chart(
        source.request,
        source.generated,
        source.acceptance,
        authority,
        onset_analysis,
        approved_intervals=approved_intervals,
    )
    acceptance = evaluate_chart_candidate(
        repaired,
        authority,
        onset_analysis,
        requested_key_mode=state.key_mode,
        requested_difficulty=state.difficulty,
        duration_ms=prepared.normalized.duration_ms,
        boundary_policy_mode=prepared.boundary_policy_mode,
    )
    if not _is_hard_safe_playtest_candidate(acceptance):
        raise ValueError("coverage repair failed hard safety gates")
    source_summary = _attack_gap_summary(source.acceptance)
    repaired_summary = _attack_gap_summary(acceptance)
    if approved_intervals:
        if any(
            _interval_substantially_overlaps_gap(interval, gap)
            for interval in approved_intervals
            for gap in acceptance.timing.coverage_gaps
        ):
            raise ValueError("coverage repair did not remove the jury-approved gap")
        for axis in (GateAxis.TIMING_ALIGNMENT, GateAxis.COVERAGE, GateAxis.PATTERN):
            if _gate_action_rank(acceptance.decision(axis).action) > _gate_action_rank(
                source.acceptance.decision(axis).action
            ):
                raise ValueError(f"coverage repair worsened the {axis.value} gate")
    elif repaired_summary >= source_summary:
        raise ValueError("coverage repair did not strictly improve active gaps")
    if not _coverage_repair_preserves_timing(source.acceptance, acceptance):
        raise ValueError("coverage repair materially worsened source timing metrics")
    osu_text = _serialized_candidate_text(
        repaired,
        authority=authority,
        prepared=prepared,
        key_mode=state.key_mode,
    )
    candidate = _Candidate(
        request=source.request,
        generated=repaired,
        acceptance=acceptance,
        osu_text=osu_text,
        workdir=source.workdir / "coverage-repair",
        attempt=source.attempt,
        seed=source.seed,
        provenance="COVERAGE_REPAIR",
        recovery_reason=(
            "JURY_APPROVED_ACTIVE_GAP_REPAIRED"
            if approved_intervals
            else "ACTIVE_COVERAGE_GAP_REPAIRED"
        ),
        intro_anchor_covered=_intro_anchor_covered(repaired, authority),
        coverage_repair_gap_count=plan.repaired_gap_count,
    )
    state.candidates.add_safe_fallback(candidate)
    state.attempt_evidence.append(
        {
            "reason": "COVERAGE_REPAIR_CREATED",
            "sourceSeed": source.seed,
            "sourceAttackGapCount": source_summary[0],
            "sourceAttackGapTotalMs": source_summary[1],
            "resultAttackGapCount": repaired_summary[0],
            "resultAttackGapTotalMs": repaired_summary[1],
            "juryApprovedIntervals": [
                {"startMs": start_ms, "endMs": end_ms} for start_ms, end_ms in approved_intervals
            ],
            "repairPlan": plan.to_report(),
        }
    )
    return candidate


def _get_or_create_coverage_repair_candidate(
    state: _VariantState,
    source: _Candidate,
    *,
    prepared: PreparedAudio,
    authority: SongTimingAuthority,
    onset_analysis: OnsetAnalysis,
) -> _Candidate | None:
    existing = next(
        (
            candidate
            for candidate in state.candidates.safe_fallbacks
            if candidate.provenance == "COVERAGE_REPAIR"
        ),
        None,
    )
    if existing is not None:
        return existing
    if any(
        evidence.get("reason") == "COVERAGE_REPAIR_REJECTED"
        and evidence.get("sourceSeed") == source.seed
        for evidence in state.attempt_evidence
    ):
        return None
    try:
        return _coverage_repair_candidate(
            state,
            source,
            prepared=prepared,
            authority=authority,
            onset_analysis=onset_analysis,
        )
    except (
        GeneratedChartValidationError,
        TimingAuthorityValidationError,
        TypeError,
        ValueError,
        WorkerError,
    ) as error:
        state.attempt_evidence.append(
            {
                "reason": "COVERAGE_REPAIR_REJECTED",
                "sourceSeed": source.seed,
                "errorType": type(error).__name__,
                "message": str(error),
            }
        )
        return None


def _safe_fallback_candidate(
    state: _VariantState,
    *,
    prepared: PreparedAudio,
    authority: SongTimingAuthority,
    onset_analysis: OnsetAnalysis,
    run_dir: Path,
    base_seed: int,
    authority_epoch: int,
    recovery_reason: str = "NO_STRUCTURALLY_SAFE_MODEL_CANDIDATE",
    register: bool = True,
) -> _Candidate:
    seed = base_seed + state.flat_index
    boundary, music_end_ms = _generation_bounds(prepared, onset_analysis)
    request = GenerationRequest(
        audio_path=prepared.normalized.path,
        timing_reference_path=authority.reference_path,
        key_mode=state.key_mode,
        difficulty=state.difficulty,
        seed=seed,
        duration_ms=prepared.normalized.duration_ms,
        music_end_ms=music_end_ms,
        generation_end_ms=(boundary.generation_end_ms if boundary else None),
        last_attack_ms=(boundary.last_attack_ms if boundary else None),
        max_note_start_ms=(boundary.max_note_start_ms if boundary else None),
    )
    plan = select_recovery_plan(request, authority, onset_analysis)
    generated = build_recovery_chart(
        request,
        authority,
        onset_analysis,
        plan=plan,
    )
    acceptance = evaluate_chart_candidate(
        generated,
        authority,
        onset_analysis,
        requested_key_mode=state.key_mode,
        requested_difficulty=state.difficulty,
        duration_ms=prepared.normalized.duration_ms,
        boundary_policy_mode=prepared.boundary_policy_mode,
    )
    if not _is_hard_safe_playtest_candidate(acceptance):
        raise WorkerError(
            ErrorCode.CHART_CANDIDATES_EXHAUSTED,
            f"{state.key_mode}K {state.difficulty} safe fallback failed hard gates",
            context={
                "key_mode": state.key_mode,
                "difficulty": state.difficulty,
                "gateReport": acceptance.to_report(),
                "recoveryPlan": plan.to_report(),
            },
        )
    osu_text = _serialized_candidate_text(
        generated,
        authority=authority,
        prepared=prepared,
        key_mode=state.key_mode,
    )
    candidate = _Candidate(
        request=request,
        generated=generated,
        acceptance=acceptance,
        osu_text=osu_text,
        workdir=(
            run_dir
            / "raw"
            / "work"
            / f"epoch-{authority_epoch}"
            / f"{state.key_mode}k-{state.difficulty.lower()}"
            / "safe-fallback"
        ),
        attempt=max(1, state.budget.next_attempt - 1),
        seed=seed,
        provenance="SAFE_FALLBACK",
        recovery_reason=recovery_reason,
        intro_anchor_covered=_intro_anchor_covered(generated, authority),
    )
    if register:
        state.candidates.add_safe_fallback(candidate)
        state.attempt_evidence.append(
            {
                "reason": "SAFE_FALLBACK_CREATED",
                "recoveryPlan": plan.to_report(),
                "modelAttemptCount": state.budget.next_attempt - 1,
                "sourceFailure": state.publication_block_reason,
            }
        )
    return candidate


def _complete_playtest_selections(
    selections: list[_Selection],
    *,
    prepared: PreparedAudio,
    authority: SongTimingAuthority,
    onset_analysis: OnsetAnalysis,
    run_dir: Path,
    base_seed: int,
    authority_epoch: int,
) -> list[_Selection]:
    """Fill every absent slot without another model invocation.

    Trust order is admitted model output, hard-safe raw model output, then a
    deterministic canonical-timing fallback. Difficulty-order defects remain
    visible in the family review instead of deleting an otherwise playable map.
    """
    completed: list[_Selection] = []
    for states, original_assignment, _review in selections:
        assignment = dict(original_assignment)
        for difficulty in DIFFICULTIES:
            if assignment[difficulty] is not None:
                continue
            state = states[difficulty]
            admitted = sorted(
                state.candidates.admitted,
                key=lambda candidate: (candidate.attempt, candidate.seed),
            )
            raw = sorted(state.candidates.raw_rejected, key=_raw_playtest_score)
            if admitted:
                candidate = admitted[0]
            elif raw:
                best_raw = raw[0]
                if best_raw.acceptance.decision(GateAxis.COVERAGE).action is GateAction.RETRY_MAP:
                    alternatives: list[_Candidate] = [best_raw]
                    coverage_repair = _get_or_create_coverage_repair_candidate(
                        state,
                        best_raw,
                        prepared=prepared,
                        authority=authority,
                        onset_analysis=onset_analysis,
                    )
                    if coverage_repair is not None:
                        alternatives.append(coverage_repair)
                    try:
                        alternatives.append(
                            _safe_fallback_candidate(
                                state,
                                prepared=prepared,
                                authority=authority,
                                onset_analysis=onset_analysis,
                                run_dir=run_dir,
                                base_seed=base_seed,
                                authority_epoch=authority_epoch,
                                recovery_reason="COVERAGE_REPAIR_UNAVAILABLE",
                            )
                        )
                    except (TypeError, ValueError, WorkerError) as error:
                        state.attempt_evidence.append(
                            {
                                "reason": "COVERAGE_SAFE_FALLBACK_REJECTED",
                                "errorType": type(error).__name__,
                                "message": str(error),
                            }
                        )
                    candidate = min(alternatives, key=_playtest_candidate_score)
                else:
                    candidate = best_raw
            else:
                candidate = _safe_fallback_candidate(
                    state,
                    prepared=prepared,
                    authority=authority,
                    onset_analysis=onset_analysis,
                    run_dir=run_dir,
                    base_seed=base_seed,
                    authority_epoch=authority_epoch,
                )
            assignment[difficulty] = candidate
        completed.append((states, assignment, _family_review(assignment)))
    return completed


def _coverage_gap_member(
    candidate: _Candidate,
    difficulty: str,
    gap: TimingCoverageGap,
) -> CoverageGapMember:
    opportunity = gap.opportunity
    return CoverageGapMember(
        key_mode=candidate.generated.key_mode,
        difficulty=difficulty,
        start_ms=gap.start_ms,
        end_ms=gap.end_ms,
        model_backed=candidate.provenance != "SAFE_FALLBACK",
        hold_occupancy_ratio=(opportunity.hold_occupancy_ratio if opportunity is not None else 0.0),
        opportunity_kind=(
            opportunity.kind if opportunity is not None else CoverageKind.UNCERTAIN
        ),
    )


def _apply_chart_specific_coverage_recovery(
    selections: list[_Selection],
    *,
    prepared: PreparedAudio,
    authority: SongTimingAuthority,
    onset_analysis: OnsetAnalysis,
) -> list[_Selection]:
    """Repair only a frozen-family, audio-corroborated chart-specific omission."""

    frozen: list[
        tuple[
            int,
            str,
            _VariantState,
            _Candidate,
            TimingCoverageGap,
            CoverageGapMember,
        ]
    ] = []
    for selection_index, (states, assignment, _review) in enumerate(selections):
        for difficulty in DIFFICULTIES:
            candidate = assignment[difficulty]
            if candidate is None:
                continue
            for gap in (
                *candidate.acceptance.timing.coverage_gaps,
                *candidate.acceptance.timing.uncertain_coverage_gaps,
            ):
                frozen.append(
                    (
                        selection_index,
                        difficulty,
                        states[difficulty],
                        candidate,
                        gap,
                        _coverage_gap_member(candidate, difficulty, gap),
                    )
                )

    timing_status = (
        authority.timing_integrity.status
        if authority.timing_integrity is not None
        else TimingIntegrityStatus.NEEDS_CORROBORATION
    )
    approved: dict[tuple[int, str], set[tuple[int, int]]] = {}
    for (
        selection_index,
        difficulty,
        state,
        candidate,
        gap,
        target,
    ) in frozen:
        opportunity = gap.opportunity
        local_audio = gap.local_audio_evidence
        if (
            candidate.acceptance.decision(GateAxis.COVERAGE).action is not GateAction.REVIEW
            or gap.position != "MIDDLE"
            or opportunity is None
            or opportunity.kind is not CoverageKind.UNCERTAIN
            or local_audio is None
        ):
            continue
        siblings = tuple(
            member
            for _index, _difficulty, _state, sibling_candidate, _gap, member in frozen
            if sibling_candidate is not candidate
        )
        review = review_coverage_family(
            target,
            siblings,
            local_audio,
            timing_status=timing_status,
        )
        state.attempt_evidence.append(
            {
                "reason": "COVERAGE_FAMILY_REVIEW_DECISION",
                "targetInterval": {
                    "startMs": gap.start_ms,
                    "endMs": gap.end_ms,
                },
                "decision": review.to_report(),
                "mutatesSelection": (
                    review.verdict is CoverageFamilyVerdict.CHART_SPECIFIC_OMISSION
                ),
            }
        )
        if review.verdict is CoverageFamilyVerdict.CHART_SPECIFIC_OMISSION:
            approved.setdefault((selection_index, difficulty), set()).add(
                (gap.start_ms, gap.end_ms)
            )

    recovered: list[_Selection] = []
    for selection_index, (states, original_assignment, _review) in enumerate(selections):
        assignment = dict(original_assignment)
        for difficulty in DIFFICULTIES:
            intervals = tuple(sorted(approved.get((selection_index, difficulty), ())))
            if not intervals:
                continue
            source = assignment[difficulty]
            if source is None:
                continue
            state = states[difficulty]
            try:
                assignment[difficulty] = _coverage_repair_candidate(
                    state,
                    source,
                    prepared=prepared,
                    authority=authority,
                    onset_analysis=onset_analysis,
                    approved_intervals=intervals,
                )
            except (
                GeneratedChartValidationError,
                TimingAuthorityValidationError,
                TypeError,
                ValueError,
                WorkerError,
            ) as error:
                state.attempt_evidence.append(
                    {
                        "reason": "JURY_APPROVED_COVERAGE_REPAIR_REJECTED",
                        "sourceSeed": source.seed,
                        "approvedIntervals": [
                            {"startMs": start_ms, "endMs": end_ms} for start_ms, end_ms in intervals
                        ],
                        "errorType": type(error).__name__,
                        "message": str(error),
                        "originalSelectionPreserved": True,
                    }
                )
        recovered.append((states, assignment, _family_review(assignment)))
    return recovered


def _generation_bounds(
    prepared: PreparedAudio,
    onset_analysis: OnsetAnalysis,
) -> tuple[SongBoundaryContract | None, int | None]:
    boundary = (
        build_song_boundary_contract(
            onset_analysis.activity,
            prepared.normalized.duration_ms,
            enforcement_mode=prepared.boundary_policy_mode,
            terminal_silence=onset_analysis.terminal_silence,
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
    return boundary, music_end_ms


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
    boundary, music_end_ms = _generation_bounds(prepared, onset_analysis)
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
                evidence["resnapDiagnostics"] = generated.resnap_diagnostics.to_report()
            if acceptance.action is GateAction.RETRY_MAP:
                state.budget.record_quality_attempt()
                evidence.update(
                    {
                        "reason": (
                            "HARD_REJECT_RETRY"
                            if acceptance.disposition is CandidateDisposition.HARD_REJECT
                            else "QUALITY_DEFECT_ROUTED_TO_RECOVERY"
                        ),
                        "candidateDisposition": acceptance.disposition.value,
                        "repairEligible": acceptance.repair_eligible,
                        "repairAxes": [axis.value for axis in acceptance.repair_axes],
                    }
                )
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
                try:
                    raw_captured = _capture_raw_candidate(
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
                except (
                    GeneratedChartValidationError,
                    TimingAuthorityValidationError,
                    WorkerError,
                ) as error:
                    state.attempt_errors.append(_error_report_json(error))
                    _block_full_length_retry(
                        state,
                        reason="SERIALIZATION_CONTRACT_FAILED",
                        seed=attempt_seed,
                        attempt=attempt,
                    )
                    last_error = error
                    break
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
                            intro_anchor_covered=_intro_anchor_covered(generated, authority),
                        )
                    )
                if raw_captured:
                    _block_full_length_retry(
                        state,
                        reason="HARD_SAFE_RAW_AVAILABLE",
                        seed=attempt_seed,
                        attempt=attempt,
                    )
                    break
                continue

            validate_timing_identity(generated.bpm_events, authority.bpm_events)
            validate_generated_chart(
                generated,
                key_mode=state.key_mode,
                duration_ms=prepared.normalized.duration_ms,
                max_note_start_ms=(boundary.max_note_start_ms if boundary else None),
            )
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
            ) as error:
                state.budget.record_quality_attempt()
                state.attempt_errors.append(_error_report_json(error))
                state.attempt_evidence.append(
                    {
                        "seed": attempt_seed,
                        "workdir": workdir.relative_to(run_dir).as_posix(),
                        "reason": "SERIALIZATION_CONTRACT_FAILED",
                        "error": _error_report_json(error),
                    }
                )
                _record_candidate_event(
                    state,
                    admitted=False,
                    authority_epoch=authority_epoch,
                    attempt=attempt,
                    seed=attempt_seed,
                    purpose="PRIMARY_ATTEMPT",
                    reason="SERIALIZATION_CONTRACT_FAILED",
                )
                _block_full_length_retry(
                    state,
                    reason="SERIALIZATION_CONTRACT_FAILED",
                    seed=attempt_seed,
                    attempt=attempt,
                )
                last_error = error
                break
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
            if _is_tail_repair_exhausted(error):
                _block_full_length_retry_after_tail_exhaustion(state, error)
                raise _exhausted_error(state) from error
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


def _persist_selection_candidate_payloads(
    selections,
    *,
    run_dir: Path,
) -> None:
    for states, _assignment, _review in selections:
        for state in states.values():
            for candidate in state.candidates.evidence_candidates:
                persist_candidate_payload(
                    run_dir=run_dir,
                    osu_text=candidate.osu_text,
                )


def _verify_selection_candidate_payloads(
    selections,
    *,
    run_dir: Path,
) -> None:
    for states, _assignment, _review in selections:
        for state in states.values():
            for candidate in state.candidates.evidence_candidates:
                verify_candidate_payload(
                    run_dir=run_dir,
                    osu_text=candidate.osu_text,
                )


def _compare_song_selection(
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
    mode: str,
) -> tuple[
    list[
        tuple[
            dict[str, _VariantState],
            dict[str, _Candidate | None],
            DifficultyOrderReview | None,
        ]
    ],
    SongSelectionComparison,
]:
    return _family_compare_song_selection(
        selections,
        prepared=prepared,
        authority=authority,
        run_dir=run_dir,
        intro_contract=intro_contract,
        boundary=boundary,
        mode=mode,
    )


def _apply_safe_family_assignment(
    selections: list[_Selection],
    *,
    run_dir: Path,
    intro_contract: IntroStartContract,
    boundary: SongBoundaryContract | None,
    song_context: SongAnalysisContext,
    post_resolution_ordering: bool = False,
):
    return _family_apply_safe_family_assignment(
        selections,
        run_dir=run_dir,
        intro_contract=intro_contract,
        boundary=boundary,
        song_context=song_context,
        post_resolution_ordering=post_resolution_ordering,
    )


def _unresolved_family_key_modes(
    decisions: tuple[SafeFamilyAssignmentDecision, ...],
) -> set[int]:
    """Return only families that still require a new candidate.

    Compiler status is intentionally absent from this decision.  A failed
    compiler may be harmless when preserved candidates already form a valid
    family, while a successful compiler may still leave only candidates that
    the safety selector must reject.
    """

    return {
        decision.key_mode
        for decision in decisions
        if decision.family_feasibility_status == "UNAVAILABLE"
    }


def _is_canonical_safe_fallback(candidate: _Candidate) -> bool:
    return (
        candidate.provenance == "SAFE_FALLBACK"
        and candidate.recovery_reason != "DIFFICULTY_FAMILY_COMPILER_V1"
    )


def _ensure_coherent_safe_fallback_family_candidates(
    selections: list[_Selection],
    *,
    unresolved_key_modes: set[int],
    prepared: PreparedAudio,
    authority: SongTimingAuthority,
    onset_analysis: OnsetAnalysis,
    run_dir: Path,
    base_seed: int,
    authority_epoch: int,
) -> set[int]:
    """Supply all four deterministic fallback candidates as one key-family.

    Building is staged so a failure in one difficulty cannot leave a partial
    family in the repositories. Existing canonical fallbacks are reused;
    compiler proposals do not count because they are derived from a model
    anchor rather than the shared recovery authority.
    """

    supplied: set[int] = set()
    for states, _assignment, _review in selections:
        key_mode = next(iter(states.values())).key_mode
        if key_mode not in unresolved_key_modes:
            continue
        staged: dict[str, _Candidate] = {}
        try:
            for difficulty in DIFFICULTIES:
                state = states[difficulty]
                if any(
                    _is_canonical_safe_fallback(candidate)
                    for candidate in state.candidates.safe_fallbacks
                ):
                    continue
                staged[difficulty] = _safe_fallback_candidate(
                    state,
                    prepared=prepared,
                    authority=authority,
                    onset_analysis=onset_analysis,
                    run_dir=run_dir,
                    base_seed=base_seed,
                    authority_epoch=authority_epoch,
                    recovery_reason="ATOMIC_DIFFICULTY_FAMILY_FALLBACK",
                    register=False,
                )
            for candidate in staged.values():
                persist_candidate_payload(
                    run_dir=run_dir,
                    osu_text=candidate.osu_text,
                )
        except (OSError, TypeError, ValueError, WorkerError) as error:
            evidence = {
                "reason": "ATOMIC_DIFFICULTY_FAMILY_FALLBACK_UNAVAILABLE",
                "keyMode": key_mode,
                "errorType": type(error).__name__,
                "message": str(error),
                "createdDifficulties": [],
                "mutatesSelection": False,
            }
            for state in states.values():
                state.attempt_evidence.append(dict(evidence))
            continue

        for difficulty, candidate in staged.items():
            state = states[difficulty]
            state.candidates.add_safe_fallback(candidate)
            state.attempt_evidence.append(
                {
                    "reason": "ATOMIC_DIFFICULTY_FAMILY_FALLBACK_CREATED",
                    "keyMode": key_mode,
                    "difficulty": difficulty,
                    "modelAttemptCount": state.budget.next_attempt - 1,
                    "sourceFailure": state.publication_block_reason,
                    "mutatesSelection": False,
                }
            )
        supplied.add(key_mode)
    return supplied


def _compile_difficulty_family_shadows(
    selections: list[_Selection],
    *,
    prepared: PreparedAudio,
    authority: SongTimingAuthority,
    onset_analysis: OnsetAnalysis,
    run_dir: Path,
    clock: Callable[[], float] | None = None,
    enforce: bool = False,
) -> tuple[DifficultyFamilyCompilerDecision, ...]:
    """Evaluate deterministic family proposals and optionally admit them for selection.

    Even in ENFORCED mode the compiler does not own the published assignment.
    It only registers structurally verified proposals as safe fallbacks; the
    final safe-family selector applies target-local non-regression constraints.
    """

    if clock is None:
        clock = perf_counter
    decisions = []
    for states, assignment, _review in selections:
        key_mode = next(iter(states.values())).key_mode
        selected = tuple(assignment[difficulty] for difficulty in DIFFICULTIES)
        compiler_started = clock()
        decision = None
        try:
            if any(candidate is None for candidate in selected):
                decision = DifficultyFamilyCompilerDecision(
                    key_mode=key_mode,
                    status="UNAVAILABLE",
                    reason="INCOMPLETE_SELECTED_FAMILY",
                    anchor_candidate_id=None,
                    anchor_source_difficulty=None,
                    proposals=(),
                    proposals_evaluated=0,
                )
            else:
                candidates = tuple(candidate for candidate in selected if candidate is not None)
                source_candidates: dict[str, _Candidate] = {}
                slots = tuple(
                    DifficultyFamilyCompilerSlot(
                        difficulty=difficulty,
                        candidate_id=_candidate_stable_id(
                            candidate,
                            key_mode=key_mode,
                            difficulty=difficulty,
                            run_dir=run_dir,
                        ),
                        candidate_payload_sha256=persist_candidate_payload(
                            run_dir=run_dir,
                            osu_text=candidate.osu_text,
                        ).sha256,
                        generated=candidate.generated,
                        acceptance=candidate.acceptance,
                        osu_text=candidate.osu_text,
                        provenance=candidate.provenance,
                    )
                    for difficulty, candidate in zip(
                        DIFFICULTIES,
                        candidates,
                        strict=True,
                    )
                )
                source_candidates = {
                    slot.candidate_id: candidate
                    for slot, candidate in zip(slots, candidates, strict=True)
                }

                def evaluate_candidate(
                    generated: GeneratedChart,
                    target_difficulty: str,
                    _key_mode: int = key_mode,
                ) -> ChartAcceptance:
                    return evaluate_chart_candidate(
                        generated,
                        authority,
                        onset_analysis,
                        requested_key_mode=_key_mode,
                        requested_difficulty=target_difficulty,
                        duration_ms=prepared.normalized.duration_ms,
                        boundary_policy_mode=prepared.boundary_policy_mode,
                    )

                def serialize_candidate(
                    generated: GeneratedChart,
                    _key_mode: int = key_mode,
                ) -> str:
                    return _serialized_candidate_text(
                        generated,
                        authority=authority,
                        prepared=prepared,
                        key_mode=_key_mode,
                    )

                decision = compile_difficulty_family_shadow(
                    slots,
                    authority=authority,
                    onset_analysis=onset_analysis,
                    duration_ms=prepared.normalized.duration_ms,
                    boundary_policy_mode=prepared.boundary_policy_mode,
                    evaluate_candidate=evaluate_candidate,
                    serialize_candidate=serialize_candidate,
                    clock=clock,
                )
                decision = persist_difficulty_family_compiler_payloads(
                    decision,
                    run_dir=run_dir,
                    clock=clock,
                )
                if enforce and decision.status == "COMPILED":
                    decision = replace(decision, publication_mode="ENFORCED")
                    compiled_assignment = materialize_compiled_family(
                        decision,
                        source_candidates=source_candidates,
                        current_assignment=assignment,
                    )
                    for difficulty, candidate in compiled_assignment.items():
                        states[difficulty].candidates.add_safe_fallback(candidate)
        except Exception as error:  # noqa: BLE001 - optional SHADOW must not abort output
            failure_total_ms = max(0.0, (clock() - compiler_started) * 1_000.0)
            decision = DifficultyFamilyCompilerDecision(
                key_mode=key_mode,
                status="UNAVAILABLE",
                reason="COMPILER_EXECUTION_FAILED",
                anchor_candidate_id=None,
                anchor_source_difficulty=None,
                proposals=(),
                proposals_evaluated=0,
                failure_type=type(error).__name__,
                failure_message=str(error),
                solver_wall_ms=(
                    decision.solver_wall_ms if decision is not None else round(failure_total_ms, 6)
                ),
                candidate_evaluation_wall_ms=(
                    decision.candidate_evaluation_wall_ms if decision is not None else 0.0
                ),
                payload_persistence_wall_ms=(
                    max(0.0, round(failure_total_ms - decision.solver_wall_ms, 6))
                    if decision is not None
                    else 0.0
                ),
            )
        evidence = {
            "reason": (
                "DIFFICULTY_FAMILY_COMPILER_ENFORCED_DECISION"
                if decision.publication_mode == "ENFORCED"
                else "DIFFICULTY_FAMILY_COMPILER_SHADOW_DECISION"
            ),
            "decision": decision.to_report(),
            "mutatesSelection": False,
        }
        for state in states.values():
            state.attempt_evidence.append(dict(evidence))
        decisions.append(decision)
    return tuple(decisions)


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
    model_retry_target: tuple[int, str] | None = None,
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
        model_retry_target=model_retry_target,
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
    rows = sorted({note.time_ms for note in source.generated.notes})
    partial_end_ms = _intro_phrase_recovery_end_ms(
        rows[1] if len(rows) >= 2 else None,
        authority.bpm_events,
        duration_ms=prepared.normalized.duration_ms,
    )
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
        partial_end_ms=partial_end_ms,
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

    Failure isolation principle: one variant cannot consume another variant's
    model budget. After bounded model recovery, every absent slot is completed
    without inference by hard-safe raw output or a canonical safe fallback.
    """
    # timing 재선택 escalation 은 더 나은 authority 후보가 실제로 남아
    # 있을 때만 의미가 있다. 이미 Super Timing 이면 같은 seed 로 같은
    # 결과를 다시 뽑을 뿐이므로, 실패를 조합 단위로 격리한다.
    if authority_epoch < 1:
        raise ValueError("authority_epoch must be positive")
    inference_budget = AdditionalInferenceBudget(
        limit=1,
        work_limit_ms=prepared.normalized.duration_ms,
    )
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
    model_inference_disabled_by: dict[str, object] | None = None
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
            if model_inference_disabled_by is not None:
                state.publication_block_reason = "INFERENCE_COMPLETION_UNKNOWN"
                state.full_length_retry_blocked_by = {
                    "reason": "INFERENCE_COMPLETION_UNKNOWN",
                    "policyVersion": "UNKNOWN_COMPLETION_NO_REEXECUTION_V1",
                    "triggerVariant": model_inference_disabled_by["triggerVariant"],
                }
                state.attempt_evidence.append(
                    {
                        "reason": "MODEL_INFERENCE_SUPPRESSED_AFTER_UNKNOWN_COMPLETION",
                        **model_inference_disabled_by,
                    }
                )
                continue
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
                if error.code is ErrorCode.INFERENCE_COMPLETION_UNKNOWN:
                    model_inference_disabled_by = {
                        "triggerVariant": {
                            "keyMode": state.key_mode,
                            "difficulty": state.difficulty,
                        },
                        "error": _error_report(error),
                    }
                    state.publication_block_reason = "INFERENCE_COMPLETION_UNKNOWN"
                    state.full_length_retry_blocked_by = {
                        "reason": "INFERENCE_COMPLETION_UNKNOWN",
                        "policyVersion": "UNKNOWN_COMPLETION_NO_REEXECUTION_V1",
                        "triggerVariant": model_inference_disabled_by["triggerVariant"],
                    }
                    state.attempt_errors.append(_error_report_json(error))
                    state.attempt_evidence.append(
                        {
                            "reason": "MODEL_INFERENCE_DISABLED_AFTER_UNKNOWN_COMPLETION",
                            **model_inference_disabled_by,
                        }
                    )
                    continue
                if error.code is not ErrorCode.CHART_CANDIDATES_EXHAUSTED:
                    raise
                state.exhausted_error = error

        state_families.append(states)

    # A deterministic, revalidated TAP-only repair is cheaper and safer than
    # another decoder invocation.  Build it first; partial model generation is
    # reserved for gaps that this zero-inference path cannot repair.
    if model_inference_disabled_by is None:
        for states in state_families:
            for state in states.values():
                if state.candidates.admitted:
                    continue
                raw = sorted(state.candidates.raw_rejected, key=_raw_playtest_score)
                if not raw:
                    continue
                best_raw = raw[0]
                if best_raw.acceptance.decision(GateAxis.COVERAGE).action is GateAction.RETRY_MAP:
                    _get_or_create_coverage_repair_candidate(
                        state,
                        best_raw,
                        prepared=prepared,
                        authority=authority,
                        onset_analysis=onset_analysis,
                    )

    partial_plans = (
        ()
        if model_inference_disabled_by is not None
        else tuple(
            plan
            for states in state_families
            for state in states.values()
            if not state.candidates.admitted
            if not any(
                candidate.provenance == "COVERAGE_REPAIR"
                and candidate.acceptance.action is not GateAction.RETRY_MAP
                for candidate in state.candidates.safe_fallbacks
            )
            if (
                plan := _plan_partial_repair(
                    state,
                    authority=authority,
                    duration_ms=prepared.normalized.duration_ms,
                )
            )
            is not None
        )
    )
    partial_by_request_id = {plan.request.request_id: plan for plan in partial_plans}
    partial_router_plan = plan_recoveries(
        tuple(plan.request for plan in partial_plans),
        available_generation_ms=(
            inference_budget.remaining_work_ms
            if inference_budget.remaining_work_ms is not None
            else 0
        ),
        available_calls=inference_budget.remaining,
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
        if not inference_budget.consume(request.estimated_generation_ms):
            raise AssertionError("recovery router exceeded additional inference budget")
        partial_plan = partial_by_request_id[request.request_id]
        try:
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
        except WorkerError as error:
            if not _is_tail_repair_exhausted(error):
                raise
            partial_attempt = max(
                partial_plan.state.budget.next_attempt,
                partial_plan.source.attempt + 1,
            )
            partial_seed = (
                seed + partial_plan.state.flat_index + MAX_VARIANT_ATTEMPTS * _VARIANT_COUNT
            )
            _block_full_length_retry_after_tail_exhaustion(
                partial_plan.state,
                error,
                partial_attempt=partial_attempt,
                partial_seed=partial_seed,
            )
            partial_plan.state.exhausted_error = _exhausted_error(partial_plan.state)
            continue
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
        selection_shadows = tuple(comparison for _states, _assignment, comparison in compared)

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
    intro_targets = _intro_recovery_targets(
        selections,
        song_context=song_context,
        authority=authority,
        duration_ms=prepared.normalized.duration_ms,
        run_dir=run_dir,
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
    if (
        intro_targets
        and model_inference_disabled_by is None
    ):
        intro_target = intro_targets[0]
        family_requests.append(
            intro_region_recovery_request(
                key_mode=intro_target.key_mode,
                difficulty=intro_target.difficulty,
                window_ms=intro_target.partial_end_ms,
            )
        )
    if timing_outliers and model_inference_disabled_by is None:
        timing_target = timing_outliers[0]
        if timing_target.target_key_mode is not None and timing_target.difficulty is not None:
            family_requests.append(
                timing_family_recovery_request(
                    key_mode=timing_target.target_key_mode,
                    difficulty=timing_target.difficulty,
                    song_duration_ms=prepared.normalized.duration_ms,
                )
            )
    family_router_plan = plan_recoveries(
        tuple(family_requests),
        available_generation_ms=(
            inference_budget.remaining_work_ms
            if inference_budget.remaining_work_ms is not None
            else 0
        ),
        available_calls=inference_budget.remaining,
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
                block_unresolved=False,
                model_retry_target=(
                    selected_family_request.key_mode,
                    selected_family_request.difficulty,
                ),
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

    selections = _complete_playtest_selections(
        selections,
        prepared=prepared,
        authority=authority,
        onset_analysis=onset_analysis,
        run_dir=run_dir,
        base_seed=seed,
        authority_epoch=authority_epoch,
    )
    selections = _apply_chart_specific_coverage_recovery(
        selections,
        prepared=prepared,
        authority=authority,
        onset_analysis=onset_analysis,
    )
    selections, intro_start_contract, intro_contract_review = _apply_intro_start_contract(
        selections,
        song_context,
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
    _persist_selection_candidate_payloads(selections, run_dir=run_dir)
    final_song_selector_mode = "V2" if prepared.difficulty_selector_mode == "V2" else "SHADOW_V2"
    song_boundary = (
        build_song_boundary_contract(
            onset_analysis.activity,
            prepared.normalized.duration_ms,
            enforcement_mode=prepared.boundary_policy_mode,
            terminal_silence=onset_analysis.terminal_silence,
        )
        if onset_analysis.activity is not None
        else None
    )
    selections, song_selection_shadow = _compare_song_selection(
        selections,
        prepared=prepared,
        authority=authority,
        run_dir=run_dir,
        intro_contract=intro_start_contract,
        boundary=song_boundary,
        mode=final_song_selector_mode,
    )
    difficulty_family_compiler_shadow: tuple[
        DifficultyFamilyCompilerDecision, ...
    ] = ()
    unresolved_compiler_key_modes: set[int] = set()
    bounded_difficulty_recovery_admitted = False
    if prepared.difficulty_family_resolution_enabled:
        # Relabelling four already-safe, unique payloads is lossless and much
        # cheaper than deleting rows or invoking the model.  The final pass
        # below runs again after any bounded recovery and is the reported one.
        selections, _pre_resolution_assignments = _apply_safe_family_assignment(
            selections,
            run_dir=run_dir,
            intro_contract=intro_start_contract,
            boundary=song_boundary,
            song_context=song_context,
        )
        difficulty_family_compiler_shadow = _compile_difficulty_family_shadows(
            selections,
            prepared=prepared,
            authority=authority,
            onset_analysis=onset_analysis,
            run_dir=run_dir,
            enforce=True,
        )
        # Compiler proposals are only candidates.  Re-run the safety selector
        # before deciding whether an extra model call is actually necessary.
        selections, post_compiler_assignments = _apply_safe_family_assignment(
            selections,
            run_dir=run_dir,
            intro_contract=intro_start_contract,
            boundary=song_boundary,
            song_context=song_context,
        )
        unresolved_compiler_key_modes = _unresolved_family_key_modes(
            post_compiler_assignments
        )
    # Spend the optional research-only call against the assignment that the
    # enforced V2 selector actually chose.  Planning earlier can target a
    # transient fallback that V2 already replaces and leave the real final
    # inversion without a challenger.
    if (
        (
            prepared.difficulty_shadow_challenger_enabled
            or bool(unresolved_compiler_key_modes)
        )
        and model_inference_disabled_by is None
    ):
        shadow_scope = (
            [
                selection
                for selection in selections
                if next(iter(selection[0].values())).key_mode
                in unresolved_compiler_key_modes
            ]
            if unresolved_compiler_key_modes
            else selections
        )
        shadow_slots = difficulty_shadow_slots(shadow_scope)
        difficulty_shadow_target = choose_difficulty_shadow_target(shadow_slots)
        if difficulty_shadow_target is None:
            target_reason = difficulty_shadow_target_failure_reason(shadow_slots)
            for states, _assignment, _review in selections:
                for state in states.values():
                    state.attempt_evidence.append(
                        {
                            "reason": target_reason,
                            "calibrationState": (
                                CURRENT_DIFFICULTY_FAMILY_CALIBRATION_STATE
                            ),
                            "mutatesSelection": False,
                        }
                    )
        if difficulty_shadow_target is not None:
            target_states, target_assignment, _target_review = next(
                selection
                for selection in selections
                if next(iter(selection[0].values())).key_mode
                == difficulty_shadow_target.key_mode
            )
            target_state = target_states[difficulty_shadow_target.difficulty]
            partial_decision = plan_difficulty_shadow_partial_repair(
                difficulty_shadow_target,
                target_state.candidates.playtest_candidates,
                authority.bpm_events,
                duration_ms=prepared.normalized.duration_ms,
            )
            if partial_decision.plan is not None:
                required_evidence_decision = (
                    build_required_gameplay_evidence_for_shadow(
                        selections,
                        source=partial_decision.plan.source,
                        authority=authority,
                        onset_analysis=onset_analysis,
                    )
                )
                target_state.attempt_evidence.append(
                    {
                        "reason": "REQUIRED_GAMEPLAY_EVIDENCE_DECISION",
                        "decision": required_evidence_decision.to_report(),
                        "mutatesSelection": False,
                    }
                )
                partial_decision = bind_required_gameplay_interval(
                    partial_decision,
                    evidence=required_evidence_decision.evidence,
                    bpm_events=authority.bpm_events,
                    duration_ms=prepared.normalized.duration_ms,
                    mode=RequiredGameplayIntervalMode.SHADOW_ENFORCE,
                )
            target_state.attempt_evidence.append(
                {
                    "reason": "DIFFICULTY_SHADOW_PARTIAL_PLAN",
                    "difficultyShadowTarget": {
                        "easierDifficulty": (
                            difficulty_shadow_target.easier_difficulty
                        ),
                        "harderDifficulty": difficulty_shadow_target.difficulty,
                        "easierRating": difficulty_shadow_target.easier_rating,
                        "harderRating": difficulty_shadow_target.harder_rating,
                        "ratingDeficit": difficulty_shadow_target.rating_deficit,
                        "minimumRating": difficulty_shadow_target.minimum_rating,
                        "maximumRating": difficulty_shadow_target.maximum_rating,
                    },
                    "decision": partial_decision.to_report(),
                    "mutatesSelection": False,
                }
            )
            full_map_decision = None
            if partial_decision.plan is None:
                full_map_decision = plan_difficulty_shadow_full_map(
                    difficulty_shadow_target,
                    target_assignment[difficulty_shadow_target.difficulty],
                )
                target_state.attempt_evidence.append(
                    {
                        "reason": "DIFFICULTY_SHADOW_FULL_MAP_PLAN",
                        "decision": full_map_decision.to_report(),
                        "mutatesSelection": False,
                    }
                )
            if partial_decision.plan is not None or (
                full_map_decision is not None and full_map_decision.plan is not None
            ):
                partial_plan = partial_decision.plan
                full_map_plan = (
                    full_map_decision.plan if full_map_decision is not None else None
                )
                planned_work_ms = (
                    partial_plan.window.end_ms - partial_plan.window.start_ms
                    if partial_plan is not None
                    else prepared.normalized.duration_ms
                )
                difficulty_shadow_request = difficulty_shadow_recovery_request(
                    key_mode=difficulty_shadow_target.key_mode,
                    difficulty=difficulty_shadow_target.difficulty,
                    window_ms=planned_work_ms,
                )
                recovery_plan = plan_recoveries(
                    (difficulty_shadow_request,),
                    available_generation_ms=(
                        inference_budget.remaining_work_ms
                        if inference_budget.remaining_work_ms is not None
                        else 0
                    ),
                    available_calls=inference_budget.remaining,
                )
                target_state.attempt_evidence.append(
                    {
                        "reason": "RECOVERY_ROUTER_DECISION",
                        "requestId": difficulty_shadow_request.request_id,
                        "plan": recovery_plan.to_report(),
                    }
                )
                if recovery_plan.selected:
                    execute_shadow = (
                        execute_difficulty_shadow_challenger
                        if partial_plan is not None
                        else execute_difficulty_shadow_full_map_challenger
                    )
                    execution_plan = (
                        partial_plan if partial_plan is not None else full_map_plan
                    )
                    assert execution_plan is not None
                    challenger = execute_shadow(
                        target_state,
                        execution_plan,
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
                    if challenger is not None:
                        persist_candidate_payload(
                            run_dir=run_dir,
                            osu_text=challenger.osu_text,
                        )
                        selected_current = target_assignment[
                            difficulty_shadow_target.difficulty
                        ]
                        if selected_current is None:
                            raise AssertionError(
                                "difficulty shadow target must have a final selection"
                            )
                        replacement_veto = _decide_candidate_replacement(
                            _candidate_quality_snapshot(selected_current),
                            _candidate_quality_snapshot(challenger),
                            stage=(
                                "DIFFICULTY_BOUNDED_RECOVERY"
                                if difficulty_shadow_target.key_mode
                                in unresolved_compiler_key_modes
                                else "DIFFICULTY_SHADOW_REPORT_ONLY"
                            ),
                            objective_improved=True,
                        )
                        unique_against_family = all(
                            candidate is None
                            or target_difficulty
                            == difficulty_shadow_target.difficulty
                            or candidate.osu_text != challenger.osu_text
                            for target_difficulty, candidate in target_assignment.items()
                        )
                        production_authority = (
                            difficulty_shadow_target.key_mode
                            in unresolved_compiler_key_modes
                            and replacement_veto.accepted
                            and unique_against_family
                        )
                        if production_authority:
                            target_state.candidates.admit(challenger)
                            target_assignment[
                                difficulty_shadow_target.difficulty
                            ] = challenger
                            bounded_difficulty_recovery_admitted = True
                        else:
                            target_state.candidates.add_shadow(challenger)
                        target_state.attempt_evidence.append(
                            {
                                "reason": (
                                    "DIFFICULTY_BOUNDED_RECOVERY_ADMITTED"
                                    if production_authority
                                    else "DIFFICULTY_SHADOW_FALLBACK_VETO_REPORTED"
                                ),
                                "replacementPolicyAccepted": replacement_veto.accepted,
                                "uniqueAgainstFamily": unique_against_family,
                                "productionAuthority": production_authority,
                                "decision": replacement_veto.to_report(),
                                "mutatesSelection": production_authority,
                            }
                        )
                else:
                    target_state.attempt_evidence.append(
                        {
                            "reason": (
                                "BUDGET_EXHAUSTED_AFTER_HIGHER_PRIORITY_RECOVERY"
                            ),
                            "requestId": difficulty_shadow_request.request_id,
                            "router": recovery_plan.to_report(),
                            "budget": inference_budget.to_report(),
                            "mutatesSelection": False,
                        }
                    )
    if bounded_difficulty_recovery_admitted:
        post_recovery_decisions = _compile_difficulty_family_shadows(
            selections,
            prepared=prepared,
            authority=authority,
            onset_analysis=onset_analysis,
            run_dir=run_dir,
            enforce=True,
        )
        post_by_key = {
            decision.key_mode: decision for decision in post_recovery_decisions
        }
        difficulty_family_compiler_shadow = tuple(
            post_by_key.get(decision.key_mode, decision)
            if decision.key_mode in unresolved_compiler_key_modes
            else decision
            for decision in difficulty_family_compiler_shadow
        )
    # This is the sole publication-authoritative family selection.  It runs
    # after every deterministic proposal and optional bounded challenger has
    # entered the repositories.
    selections, safe_family_assignments = _apply_safe_family_assignment(
        selections,
        run_dir=run_dir,
        intro_contract=intro_start_contract,
        boundary=song_boundary,
        song_context=song_context,
        post_resolution_ordering=(
            not prepared.difficulty_family_resolution_enabled
        ),
    )
    if prepared.difficulty_family_resolution_enabled:
        final_unresolved_key_modes = _unresolved_family_key_modes(
            safe_family_assignments
        )
        supplied_fallback_key_modes = (
            _ensure_coherent_safe_fallback_family_candidates(
                selections,
                unresolved_key_modes=final_unresolved_key_modes,
                prepared=prepared,
                authority=authority,
                onset_analysis=onset_analysis,
                run_dir=run_dir,
                base_seed=seed,
                authority_epoch=authority_epoch,
            )
            if final_unresolved_key_modes
            else set()
        )
        if supplied_fallback_key_modes:
            selections, safe_family_assignments = _apply_safe_family_assignment(
                selections,
                run_dir=run_dir,
                intro_contract=intro_start_contract,
                boundary=song_boundary,
                song_context=song_context,
            )
        # Candidate generation, repair, compiler proposals, bounded inference,
        # and coherent fallback supply are now complete.  Freeze that payload
        # set and map its measured difficulty order to the four public labels.
        selections, safe_family_assignments = _apply_safe_family_assignment(
            selections,
            run_dir=run_dir,
            intro_contract=intro_start_contract,
            boundary=song_boundary,
            song_context=song_context,
            post_resolution_ordering=True,
        )
    # Freeze report evidence only after every production-authoritative relabel,
    # bounded challenger, and deterministic compiler mutation is complete.
    # The shadow proposal still points to immutable candidate IDs preserved in
    # the repositories, while currentAssignment now matches publication.
    song_selection_evidence_v3 = _build_song_selection_evidence_v3(
        selections,
        prepared=prepared,
        authority=authority,
        run_dir=run_dir,
        song_context=song_context,
        boundary=song_boundary,
    )
    song_selection_shadow_v3 = evaluate_shadow_v3_proposal(
        song_selection_evidence_v3,
        proposal=tuple(sorted(song_selection_shadow.shadow_assignment.items())),
    )
    # The enforced V2 selector may choose a different already-validated
    # candidate.  Reports and the exact intro contract must describe the
    # published assignment, not the pre-selector assignment.
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
    intro_phrase_family_reviews = _intro_phrase_family_reviews(
        selections,
        song_context=song_context,
        run_dir=run_dir,
    )
    timing_family_reviews = _timing_family_reviews(selections)
    if (
        not prepared.difficulty_family_resolution_enabled
        and prepared.difficulty_family_compiler_shadow_enabled
    ):
        difficulty_family_compiler_shadow = _compile_difficulty_family_shadows(
            selections,
            prepared=prepared,
            authority=authority,
            onset_analysis=onset_analysis,
            run_dir=run_dir,
        )
    for states, _assignment, order_review in selections:
        if order_review is None or order_review.status != "RETRY":
            continue
        evidence = {
            "reason": "DIFFICULTY_ORDER_UNRESOLVED_NO_CALIBRATED_RETRY",
            "calibrationState": CURRENT_DIFFICULTY_FAMILY_CALIBRATION_STATE,
            "difficultyOrder": order_review.to_report(),
        }
        for state in states.values():
            state.attempt_evidence.append(dict(evidence))
    _verify_selection_candidate_payloads(selections, run_dir=run_dir)
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
                        note.time_ms + (note.duration_ms if note.kind == "HOLD" else 0)
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
        song_selection_evidence_v3=song_selection_evidence_v3,
        song_selection_shadow_v3=song_selection_shadow_v3,
        safe_family_assignments=safe_family_assignments,
        difficulty_family_compiler_shadow=difficulty_family_compiler_shadow,
        intro_start_contract=intro_start_contract,
        intro_contract_review=intro_contract_review,
        intro_phrase_family_reviews=intro_phrase_family_reviews,
        timing_family_reviews=timing_family_reviews,
        outro_family_review=outro_family_review,
        additional_inference_calls=inference_budget.used,
        additional_inference_work_ms=inference_budget.used_work_ms,
        additional_inference_work_limit_ms=(inference_budget.work_limit_ms or 0),
        diagnostic_raw_candidates=publication.diagnostic_raw_candidates,
    )
