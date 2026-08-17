"""Generated-candidate data and per-variant mutable state.

This module owns candidate lifecycle state, but not generation sequencing,
family selection, recovery policy, or publication.  Keeping those policies out
of the data container prevents rejected and admitted candidates from being
silently mixed while the S2 stage is decomposed incrementally.
"""

from dataclasses import dataclass, field
from pathlib import Path

from chart_worker.errors import WorkerError
from chart_worker.generation.attempt_journal import AttemptJournal
from chart_worker.generation.candidate_repository import CandidateRepository
from chart_worker.generation.generation_control import (
    MAX_CRASH_ATTEMPTS,
    MAX_TOTAL_ATTEMPTS,
    MAX_VARIANT_ATTEMPTS,
    AttemptBudgetState,
    RecoveryRouterState,
)
from chart_worker.generation.mapperatorinator import GeneratedChart
from chart_worker.generation.params import GenerationRequest
from chart_worker.stages.timing_feedback import MapTimingFailureSignature
from chart_worker.stages.types import GenerationProvenance
from chart_worker.validation.candidate_replacement import (
    CandidateQualitySnapshot,
    decide_candidate_replacement,
)
from chart_worker.validation.quality_gate import (
    ChartAcceptance,
    GateAction,
    GateAxis,
)


@dataclass(frozen=True, slots=True)
class Candidate:
    request: GenerationRequest
    generated: GeneratedChart
    acceptance: ChartAcceptance
    osu_text: str
    workdir: Path
    attempt: int
    seed: int
    provenance: GenerationProvenance
    recovery_reason: str | None = None
    intro_anchor_covered: bool | None = None
    coverage_repair_gap_count: int = 0


@dataclass(slots=True)
class VariantState:
    key_mode: int
    difficulty: str
    flat_index: int
    journal: AttemptJournal | None = None
    budget: AttemptBudgetState = field(
        default_factory=lambda: AttemptBudgetState(
            max_quality_attempts=MAX_VARIANT_ATTEMPTS,
            max_crash_attempts=MAX_CRASH_ATTEMPTS,
            max_total_attempts=MAX_TOTAL_ATTEMPTS,
        )
    )
    candidates: CandidateRepository[Candidate] = field(
        default_factory=CandidateRepository
    )
    recovery: RecoveryRouterState = field(default_factory=RecoveryRouterState)
    attempt_errors: list[str] = field(default_factory=list)
    attempt_evidence: list[dict[str, object]] = field(default_factory=list)
    timing_failures: list[MapTimingFailureSignature] = field(default_factory=list)
    partial_attempts: list[int] = field(default_factory=list)
    partial_seeds: list[int] = field(default_factory=list)
    full_length_retry_blocked_by: dict[str, object] | None = None
    publication_block_reason: str | None = None
    exhausted_error: WorkerError | None = None

    @property
    def budget_left(self) -> bool:
        return self.full_length_retry_blocked_by is None and self.budget.can_attempt


def candidate_quality_snapshot(candidate: Candidate) -> CandidateQualitySnapshot:
    acceptance = candidate.acceptance
    retry_axes = tuple(
        sorted(
            {
                decision.axis.value
                for decision in acceptance.decisions
                if decision.action is GateAction.RETRY_MAP
            }
        )
    )
    review_axes = tuple(
        sorted(
            {
                decision.axis.value
                for decision in acceptance.decisions
                if decision.action is GateAction.REVIEW
            }
        )
    )
    return CandidateQualitySnapshot(
        provenance=candidate.provenance,
        overall_action=acceptance.action,
        retry_axes=retry_axes,
        review_axes=review_axes,
        structure_pass=(
            acceptance.decision(GateAxis.STRUCTURE).action is GateAction.PASS
        ),
        timing_identity_pass=(
            acceptance.decision(GateAxis.TIMING_IDENTITY).action is GateAction.PASS
        ),
        song_bounds_action=acceptance.decision(GateAxis.SONG_BOUNDS).action,
    )


def candidate_replacement_allowed(
    state: VariantState,
    current: Candidate,
    challenger: Candidate,
    *,
    stage: str,
    objective_improved: bool,
) -> bool:
    decision = decide_candidate_replacement(
        candidate_quality_snapshot(current),
        candidate_quality_snapshot(challenger),
        stage=stage,
        objective_improved=objective_improved,
    )
    state.attempt_evidence.append(
        {
            "reason": (
                "CANDIDATE_REPLACEMENT_POLICY_ACCEPTED"
                if decision.accepted
                else "CANDIDATE_REPLACEMENT_POLICY_REJECTED"
            ),
            "sourceSeed": current.seed,
            "challengerSeed": challenger.seed,
            "decision": decision.to_report(),
        }
    )
    return decision.accepted


def candidate_evidence(
    candidate: Candidate,
    *,
    reason: str,
    run_dir: Path,
) -> dict[str, object]:
    """Return the stable report projection for one generated candidate."""

    return {
        "seed": candidate.seed,
        "attempt": candidate.attempt,
        "workdir": candidate.workdir.relative_to(run_dir).as_posix(),
        "reason": reason,
        "gateReport": candidate.acceptance.to_report(),
        "serializationValidated": True,
    }
