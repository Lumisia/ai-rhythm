"""Intro recovery planning and bounded retry execution."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, replace
from math import ceil
from pathlib import Path

from chart_worker.analysis.intro_anchor import (
    GRID_SUPPORT_WINDOW_MS,
    IntroAnchorEvidence,
)
from chart_worker.analysis.onset import OnsetAnalysis
from chart_worker.errors import WorkerError
from chart_worker.generation.candidate_state import Candidate, VariantState
from chart_worker.generation.generation_control import (
    AdditionalInferenceBudget,
    RecoveryKind,
)
from chart_worker.generation.inference_execution import (
    error_report_json,
    record_candidate_event,
    record_gate_event,
    run_inference_with_journal,
)
from chart_worker.generation.mapperatorinator import ChartGenerator, GeneratedChart
from chart_worker.generation.osu_parser import OsuBpmEvent
from chart_worker.schema.note import Chart
from chart_worker.schema.types import DIFFICULTIES, KEY_MODES
from chart_worker.stages.types import PreparedAudio, SongTimingAuthority
from chart_worker.validation.generated_chart import (
    GeneratedChartValidationError,
    validate_generated_chart,
)
from chart_worker.validation.quality_gate import ChartAcceptance, GateAction
from chart_worker.validation.timing_authority import (
    TimingAuthorityValidationError,
    validate_timing_identity,
)

INTRO_CONTEXT_BEATS = 4
_VARIANT_COUNT = len(KEY_MODES) * len(DIFFICULTIES)

EvaluateCandidate = Callable[..., ChartAcceptance]
SerializeCandidate = Callable[..., str]
IntroAnchorCovered = Callable[[GeneratedChart, SongTimingAuthority], bool | None]


@dataclass(frozen=True, slots=True)
class IntroRecoveryPlan:
    anchor_ms: int
    anchor_grid_ms: int
    partial_start_ms: int
    partial_end_ms: int
    conditioned_bpm_events: tuple[OsuBpmEvent, ...]

    def to_report(self) -> dict[str, object]:
        return {
            "anchorMs": self.anchor_ms,
            "anchorGridMs": self.anchor_grid_ms,
            "partialStartMs": self.partial_start_ms,
            "partialEndMs": self.partial_end_ms,
            "conditionedTimingEventCount": len(self.conditioned_bpm_events),
        }


def build_intro_recovery_plan(
    evidence: IntroAnchorEvidence,
    bpm_events: tuple[OsuBpmEvent, ...],
    *,
    duration_ms: int,
) -> IntroRecoveryPlan | None:
    """Build one bounded MAP retry; never mutate the published authority."""
    if evidence.status != "CONFIRMED":
        return None
    if evidence.anchor_ms is None or evidence.anchor_grid_ms is None:
        return None
    if not bpm_events:
        raise ValueError("intro recovery requires timing events")
    if duration_ms <= 0:
        raise ValueError("duration_ms must be positive")
    if evidence.anchor_ms >= bpm_events[0].time_ms:
        return None

    beat_ms = 60_000.0 / bpm_events[0].bpm
    partial_end_ms = min(
        duration_ms,
        ceil(
            max(
                bpm_events[0].time_ms + 2 * beat_ms,
                evidence.anchor_ms + INTRO_CONTEXT_BEATS * beat_ms,
            )
        ),
    )
    conditioned_events = (
        OsuBpmEvent(evidence.anchor_ms, bpm_events[0].bpm),
        *bpm_events,
    )
    return IntroRecoveryPlan(
        anchor_ms=evidence.anchor_ms,
        anchor_grid_ms=evidence.anchor_grid_ms,
        partial_start_ms=0,
        partial_end_ms=partial_end_ms,
        conditioned_bpm_events=conditioned_events,
    )


def intro_anchor_distance_ms(
    notes: Chart, evidence: IntroAnchorEvidence
) -> int | None:
    if evidence.anchor_grid_ms is None or not notes:
        return None
    return min(abs(note.time_ms - evidence.anchor_grid_ms) for note in notes)


def covers_intro_anchor(notes: Chart, evidence: IntroAnchorEvidence) -> bool:
    distance = intro_anchor_distance_ms(notes, evidence)
    return distance is not None and distance <= GRID_SUPPORT_WINDOW_MS


def execute_intro_retry(
    state: VariantState,
    source: Candidate,
    *,
    prepared: PreparedAudio,
    authority: SongTimingAuthority,
    onset_analysis: OnsetAnalysis,
    run_dir: Path,
    generator: ChartGenerator,
    base_seed: int,
    authority_epoch: int,
    inference_budget: AdditionalInferenceBudget,
    evaluate_candidate: EvaluateCandidate,
    serialize_candidate: SerializeCandidate,
    intro_anchor_covered: IntroAnchorCovered,
    recovery_reason: str = "INTRO_START_CONTRACT",
    evidence_prefix: str = "INTRO_CONTRACT",
    workdir_name: str = "intro-contract-retry",
) -> Candidate | None:
    """Generate one full-map alternate seed for an unresolved intro policy."""

    if state.recovery.was_attempted(RecoveryKind.INTRO):
        return None
    if not inference_budget.consume():
        state.attempt_evidence.append(
            {
                "reason": f"{evidence_prefix}_RETRY_BUDGET_EXHAUSTED",
                "budgetLimit": inference_budget.limit,
                "budgetUsed": inference_budget.used,
            }
        )
        return None
    if not state.recovery.claim(RecoveryKind.INTRO):
        return None
    attempt = state.budget.next_attempt
    retry_seed = base_seed + state.flat_index + (attempt - 1) * _VARIANT_COUNT
    state.budget.reserve_additional_attempt(seed=retry_seed)
    request = replace(
        source.request,
        timing_reference_path=authority.reference_path,
        seed=retry_seed,
        partial_start_ms=None,
        partial_end_ms=None,
        add_to_beatmap=False,
    )
    workdir = (
        run_dir
        / "raw"
        / "work"
        / f"epoch-{authority_epoch}"
        / f"{state.key_mode}k-{state.difficulty.lower()}"
        / workdir_name
    )
    purpose = f"{evidence_prefix}_RETRY"
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
            attempt=attempt,
            seed=retry_seed,
            purpose=purpose,
        )
        inference_completed = True
        state.budget.record_quality_attempt()
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
            attempt=attempt,
            seed=retry_seed,
            purpose=purpose,
            acceptance=acceptance,
        )
        evidence = {
            "seed": retry_seed,
            "workdir": workdir.relative_to(run_dir).as_posix(),
            "reason": (
                f"{evidence_prefix}_RETRY_ACCEPTED"
                if acceptance.action is not GateAction.RETRY_MAP
                else f"{evidence_prefix}_RETRY_REJECTED"
            ),
            "gateReport": acceptance.to_report(),
        }
        state.attempt_evidence.append(evidence)
        if acceptance.action is GateAction.RETRY_MAP:
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
                attempt=attempt,
                seed=retry_seed,
                purpose=purpose,
                reason="QUALITY_GATE_RETRY",
                acceptance=acceptance,
            )
            return None
        validate_timing_identity(generated.bpm_events, authority.bpm_events)
        validate_generated_chart(
            generated,
            key_mode=state.key_mode,
            duration_ms=prepared.normalized.duration_ms,
            max_note_start_ms=request.max_note_start_ms,
        )
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
            attempt=attempt,
            seed=retry_seed,
            purpose=purpose,
            reason=recovery_reason,
            acceptance=acceptance,
        )
        return Candidate(
            request=request,
            generated=generated,
            acceptance=acceptance,
            osu_text=osu_text,
            workdir=workdir,
            attempt=attempt,
            seed=retry_seed,
            provenance="RETRY",
            recovery_reason=recovery_reason,
            intro_anchor_covered=intro_anchor_covered(generated, authority),
        )
    except (
        GeneratedChartValidationError,
        TimingAuthorityValidationError,
        WorkerError,
        OSError,
        RuntimeError,
        ValueError,
    ) as error:
        if inference_completed:
            record_candidate_event(
                state,
                admitted=False,
                authority_epoch=authority_epoch,
                attempt=attempt,
                seed=retry_seed,
                purpose=purpose,
                reason="VALIDATION_ERROR",
            )
        state.attempt_errors.append(error_report_json(error))
        return None
