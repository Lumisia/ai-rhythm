"""Plan and execute the one bounded, localized coverage repair.

This module owns the PARTIAL_REMAP lifecycle.  Song-level orchestration decides
whether its request wins the shared recovery budget; it no longer implements
the recovery's file mutation, inference, validation, or evidence protocol.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path

from chart_worker.analysis.onset import OnsetAnalysis
from chart_worker.errors import WorkerError
from chart_worker.generation.candidate_state import Candidate, VariantState
from chart_worker.generation.generation_control import (
    MAX_VARIANT_ATTEMPTS,
    RecoveryKind,
)
from chart_worker.generation.inference_execution import (
    error_report_json,
    record_candidate_event,
    record_gate_event,
    run_inference_with_journal,
)
from chart_worker.generation.mapperatorinator import ChartGenerator, GeneratedChart
from chart_worker.generation.partial_remap import (
    PartialRemapWindow,
    build_partial_remap_window,
)
from chart_worker.generation.recovery_router import (
    RecoveryRequest,
    partial_remap_recovery_request,
)
from chart_worker.schema.types import DIFFICULTIES, KEY_MODES
from chart_worker.stages.types import PreparedAudio, SongTimingAuthority
from chart_worker.validation.generated_chart import (
    GeneratedChartValidationError,
    validate_generated_chart,
)
from chart_worker.validation.quality_gate import (
    ChartAcceptance,
    GateAction,
)
from chart_worker.validation.timing_authority import (
    TimingAuthorityValidationError,
    validate_timing_identity,
)

_VARIANT_COUNT = len(KEY_MODES) * len(DIFFICULTIES)

EvaluateCandidate = Callable[..., ChartAcceptance]
SerializeCandidate = Callable[..., str]
IntroAnchorCovered = Callable[[GeneratedChart, SongTimingAuthority], bool | None]
ShouldRetry = Callable[[Exception], bool]


@dataclass(frozen=True, slots=True)
class PartialRepairPlan:
    state: VariantState
    source: Candidate
    window: PartialRemapWindow
    request: RecoveryRequest


def plan_partial_repair(
    state: VariantState,
    *,
    authority: SongTimingAuthority,
    duration_ms: int,
) -> PartialRepairPlan | None:
    """Choose the shortest viable gap without consuming the recovery claim."""

    if (
        state.recovery.was_attempted(RecoveryKind.PARTIAL_REMAP)
        or not state.candidates.partial_sources
    ):
        return None

    planned_sources: list[tuple[int, Candidate, PartialRemapWindow]] = []
    for source in state.candidates.partial_sources:
        window = build_partial_remap_window(
            source.generated.notes,
            source.acceptance.timing.coverage_gaps,
            authority.bpm_events,
            duration_ms=duration_ms,
        )
        if window is not None:
            planned_sources.append((window.end_ms - window.start_ms, source, window))
    if not planned_sources:
        return None
    window_ms, source, window = min(
        planned_sources,
        key=lambda item: (item[0], item[1].attempt, item[1].seed),
    )
    return PartialRepairPlan(
        state=state,
        source=source,
        window=window,
        request=partial_remap_recovery_request(
            key_mode=state.key_mode,
            difficulty=state.difficulty,
            window_ms=window_ms,
        ),
    )


def execute_partial_repair(
    plan: PartialRepairPlan,
    *,
    prepared: PreparedAudio,
    authority: SongTimingAuthority,
    onset_analysis: OnsetAnalysis,
    run_dir: Path,
    generator: ChartGenerator,
    base_seed: int,
    authority_epoch: int,
    evaluate_candidate: EvaluateCandidate,
    serialize_candidate: SerializeCandidate,
    intro_anchor_covered: IntroAnchorCovered,
    should_retry: ShouldRetry,
) -> Candidate | None:
    """Execute one already-budgeted repair and return only an admitted candidate."""

    state = plan.state
    if not state.recovery.claim(RecoveryKind.PARTIAL_REMAP):
        return None
    source = plan.source
    window = plan.window

    reference_path = (
        run_dir
        / "raw"
        / "work"
        / f"epoch-{authority_epoch}"
        / "partial-references"
        / f"{state.key_mode}k-{state.difficulty.lower()}-attempt-{source.attempt}.osu"
    )
    reference_path.parent.mkdir(parents=True, exist_ok=True)
    reference_path.write_text(source.osu_text, encoding="utf-8")
    partial_seed = base_seed + state.flat_index + MAX_VARIANT_ATTEMPTS * _VARIANT_COUNT
    request = replace(
        source.request,
        timing_reference_path=reference_path,
        seed=partial_seed,
        partial_start_ms=window.start_ms,
        partial_end_ms=window.end_ms,
        add_to_beatmap=True,
    )
    workdir = (
        run_dir
        / "raw"
        / "work"
        / f"epoch-{authority_epoch}"
        / f"{state.key_mode}k-{state.difficulty.lower()}"
        / "partial-remap"
    )
    partial_attempt = MAX_VARIANT_ATTEMPTS + 1
    inference_completed = False
    try:
        generated = run_inference_with_journal(
            state,
            generator=generator,
            request=request,
            workdir=workdir,
            run_dir=run_dir,
            prepared=prepared,
            authority=authority,
            authority_epoch=authority_epoch,
            attempt=partial_attempt,
            seed=partial_seed,
            purpose="PARTIAL_REMAP",
        )
        inference_completed = True
        validate_timing_identity(generated.bpm_events, authority.bpm_events)
        validate_generated_chart(
            generated,
            key_mode=state.key_mode,
            duration_ms=prepared.normalized.duration_ms,
            max_note_start_ms=request.max_note_start_ms,
        )
        acceptance = evaluate_candidate(
            generated,
            authority,
            onset_analysis,
            requested_key_mode=state.key_mode,
            requested_difficulty=state.difficulty,
            duration_ms=prepared.normalized.duration_ms,
            boundary_policy_mode=prepared.boundary_policy_mode,
        )
        record_gate_event(
            state,
            authority_epoch=authority_epoch,
            attempt=partial_attempt,
            seed=partial_seed,
            purpose="PARTIAL_REMAP",
            acceptance=acceptance,
        )
        if acceptance.action is GateAction.RETRY_MAP:
            evidence: dict[str, object] = {
                "seed": partial_seed,
                "workdir": workdir.relative_to(run_dir).as_posix(),
                "reason": "PARTIAL_REMAP_REJECTED",
                "partialWindow": {
                    "startMs": window.start_ms,
                    "endMs": window.end_ms,
                },
                "gateReport": acceptance.to_report(),
            }
            if generated.resnap_diagnostics.status != "UNOBSERVED":
                evidence["resnapDiagnostics"] = (
                    generated.resnap_diagnostics.to_report()
                )
            state.attempt_evidence.append(evidence)
            state.attempt_errors.append(
                json.dumps(
                    evidence,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
            )
            record_candidate_event(
                state,
                admitted=False,
                authority_epoch=authority_epoch,
                attempt=partial_attempt,
                seed=partial_seed,
                purpose="PARTIAL_REMAP",
                reason="QUALITY_GATE_RETRY",
                acceptance=acceptance,
            )
            return None
        osu_text = serialize_candidate(
            generated,
            authority=authority,
            prepared=prepared,
            key_mode=state.key_mode,
        )
        record_candidate_event(
            state,
            admitted=True,
            authority_epoch=authority_epoch,
            attempt=partial_attempt,
            seed=partial_seed,
            purpose="PARTIAL_REMAP",
            reason="ACTIVE_COVERAGE_GAP",
            acceptance=acceptance,
        )
        return Candidate(
            request=request,
            generated=generated,
            acceptance=acceptance,
            osu_text=osu_text,
            workdir=workdir,
            attempt=partial_attempt,
            seed=partial_seed,
            provenance="PARTIAL_REMAP",
            recovery_reason="ACTIVE_COVERAGE_GAP",
            intro_anchor_covered=intro_anchor_covered(generated, authority),
        )
    except (
        GeneratedChartValidationError,
        TimingAuthorityValidationError,
        WorkerError,
    ) as error:
        if inference_completed:
            record_candidate_event(
                state,
                admitted=False,
                authority_epoch=authority_epoch,
                attempt=partial_attempt,
                seed=partial_seed,
                purpose="PARTIAL_REMAP",
                reason="VALIDATION_ERROR",
            )
        if not should_retry(error):
            raise
        state.attempt_errors.append(error_report_json(error))
        return None
