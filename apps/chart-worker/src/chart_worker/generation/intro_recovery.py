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
from chart_worker.generation.required_gameplay_interval import (
    RequiredGameplayIntervalV1,
    advance_tempo_map_beats,
    tempo_map_addresses,
)
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
MAX_INTRO_RECOVERY_FRACTION = 0.80
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


def intro_phrase_recovery_end_ms(
    second_row_ms: int | None,
    bpm_events: tuple[OsuBpmEvent, ...],
    *,
    duration_ms: int,
) -> int | None:
    """Return a phrase-bounded end for an isolated leading-gap remap."""

    if second_row_ms is None:
        return None
    if type(second_row_ms) is not int or second_row_ms < 0:
        raise ValueError("second_row_ms must be a non-negative exact integer or None")
    if type(duration_ms) is not int or duration_ms <= 0:
        raise ValueError("duration_ms must be a positive exact integer")
    if not bpm_events or second_row_ms >= duration_ms:
        return None
    end_ms = min(
        duration_ms,
        advance_tempo_map_beats(
            second_row_ms,
            float(INTRO_CONTEXT_BEATS),
            bpm_events,
        ),
    )
    if end_ms / duration_ms > MAX_INTRO_RECOVERY_FRACTION:
        return None
    return end_ms


def intro_region_recovery_end_ms(
    region_end_ms: int | None,
    bpm_events: tuple[OsuBpmEvent, ...],
    *,
    duration_ms: int,
) -> int | None:
    """Bound an intro repair by audio evidence plus local tempo context.

    Unlike the legacy phrase repair, this window is independent of the
    defective chart's next row.  A chart that starts a minute late therefore
    cannot turn an intro-only repair into a near-full-song inference.
    """

    # A timing authority whose first event follows the confirmed intro does
    # not define how four beats should advance from that region.  Decline the
    # optional recovery window instead of inventing a pre-authority tempo.
    # Keep malformed inputs on the normal validation path below so they still
    # fail explicitly rather than being mistaken for unavailable evidence.
    if (
        type(region_end_ms) is int
        and region_end_ms >= 0
        and not tempo_map_addresses(region_end_ms, bpm_events)
    ):
        return None

    return intro_phrase_recovery_end_ms(
        region_end_ms,
        bpm_events,
        duration_ms=duration_ms,
    )


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
    partial_end_ms: int | None = None,
    required_gameplay_interval: RequiredGameplayIntervalV1 | None = None,
) -> Candidate | None:
    """Generate one phrase-bounded alternate for an unresolved intro policy."""

    if state.full_length_retry_blocked_by is not None:
        state.attempt_evidence.append(
            {
                "reason": f"{evidence_prefix}_RETRY_SUPPRESSED_BY_TAIL_EXHAUSTION",
                "blockedBy": dict(state.full_length_retry_blocked_by),
            }
        )
        return None
    if state.recovery.was_attempted(RecoveryKind.INTRO):
        return None
    if (
        type(partial_end_ms) is not int
        or partial_end_ms <= 0
        or partial_end_ms > prepared.normalized.duration_ms
    ):
        state.attempt_evidence.append(
            {"reason": f"{evidence_prefix}_LOCAL_WINDOW_UNAVAILABLE"}
        )
        return None
    if not inference_budget.consume(partial_end_ms):
        state.attempt_evidence.append(
            {
                "reason": f"{evidence_prefix}_RETRY_BUDGET_EXHAUSTED",
                "budgetLimit": inference_budget.limit,
                "budgetUsed": inference_budget.used,
                "budget": inference_budget.to_report(),
            }
        )
        return None
    if not state.recovery.claim(RecoveryKind.INTRO):
        return None
    attempt = state.budget.next_attempt
    retry_seed = base_seed + state.flat_index + (attempt - 1) * _VARIANT_COUNT
    state.budget.reserve_additional_attempt(seed=retry_seed)
    reference_path = (
        run_dir
        / "raw"
        / "work"
        / f"epoch-{authority_epoch}"
        / "intro-references"
        / (
            f"{state.key_mode}k-{state.difficulty.lower()}-"
            f"attempt-{source.attempt}.osu"
        )
    )
    request = replace(
        source.request,
        timing_reference_path=reference_path,
        seed=retry_seed,
        partial_start_ms=0,
        partial_end_ms=partial_end_ms,
        add_to_beatmap=True,
        required_gameplay_interval=required_gameplay_interval,
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
        reference_path.parent.mkdir(parents=True, exist_ok=True)
        reference_path.write_text(source.osu_text, encoding="utf-8")
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
            "partialWindow": {"startMs": 0, "endMs": partial_end_ms},
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
            provenance="INTRO_RECOVERY",
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
