"""Pure targeting policy for one research-only difficulty challenger.

The module does not call a model, mutate selections, or admit candidates.  It
only identifies the strongest evidence-backed candidate-supply failure that is
eligible to compete for the song-wide advisory inference budget.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from dataclasses import dataclass, replace
from hashlib import sha256
from pathlib import Path
from typing import Literal

from chart_worker.analysis.intro_anchor import GRID_SUPPORT_WINDOW_MS
from chart_worker.analysis.onset import OnsetAnalysis
from chart_worker.errors import ErrorCode, WorkerError
from chart_worker.generation.candidate_state import (
    Candidate,
    VariantState,
    candidate_quality_snapshot,
)
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
from chart_worker.generation.params import DESCRIPTORS, REQUESTED_STAR, GenerationRequest
from chart_worker.generation.partial_remap import (
    PartialRemapWindow,
    expand_partial_remap_window,
    partial_suffix_signature,
)
from chart_worker.generation.required_gameplay_interval import (
    RequiredGameplayEvidenceV1,
    RequiredGameplayFamilySlotV1,
    RequiredGameplayIntervalMode,
    RequiredGameplayIntervalV1,
    advance_tempo_map_beats,
    plan_required_gameplay_interval,
)
from chart_worker.schema.types import DIFFICULTIES, KEY_MODES
from chart_worker.stages.types import PreparedAudio, SongTimingAuthority
from chart_worker.validation.candidate_replacement import decide_candidate_replacement
from chart_worker.validation.difficulty_order import MIN_ADJACENT_RATING_GAP
from chart_worker.validation.family_evidence_v3 import (
    CandidateSafetyEvidenceV3,
    GapIntervalEvidence,
    build_intro_selection_evidence,
    compare_gap_evidence,
)
from chart_worker.validation.final_difficulty_family import (
    CURRENT_DIFFICULTY_FAMILY_CALIBRATION_STATE,
)
from chart_worker.validation.generated_chart import (
    GeneratedChartValidationError,
    validate_generated_chart,
)
from chart_worker.validation.quality_gate import ChartAcceptance, GateAction, GateAxis
from chart_worker.validation.song_family_selector import (
    MATCHED_F1_EPSILON,
    MATCHED_PRECISION_EPSILON,
)
from chart_worker.validation.timing_authority import (
    TimingAuthorityValidationError,
    validate_timing_identity,
)

Selection = tuple[
    dict[str, VariantState],
    dict[str, Candidate | None],
    object | None,
]
EvaluateCandidate = Callable[..., ChartAcceptance]
SerializeCandidate = Callable[..., str]
IntroAnchorCovered = Callable[[GeneratedChart, SongTimingAuthority], bool | None]
_VARIANT_COUNT = len(KEY_MODES) * len(DIFFICULTIES)
_MATCHED_F1_EPSILON = 0.005


@dataclass(frozen=True, slots=True)
class DifficultyShadowSlot:
    key_mode: int
    difficulty: str
    project_rating: float
    selected_provenance: str
    has_existing_safe_resolution: bool = False

    def __post_init__(self) -> None:
        if type(self.key_mode) is not int or self.key_mode not in KEY_MODES:
            raise ValueError("key_mode is unsupported")
        if type(self.difficulty) is not str or self.difficulty not in DIFFICULTIES:
            raise ValueError("difficulty is unsupported")
        if type(self.project_rating) is not float or not math.isfinite(
            self.project_rating
        ):
            raise TypeError("project_rating must be a finite exact float")
        if type(self.selected_provenance) is not str or not self.selected_provenance:
            raise TypeError("selected_provenance must be a non-empty exact string")
        if type(self.has_existing_safe_resolution) is not bool:
            raise TypeError("has_existing_safe_resolution must be an exact boolean")


@dataclass(frozen=True, slots=True)
class DifficultyShadowTarget:
    key_mode: int
    easier_difficulty: str
    difficulty: str
    easier_rating: float
    harder_rating: float
    rating_deficit: float
    minimum_rating: float | None = None
    maximum_rating: float | None = None

    def __post_init__(self) -> None:
        if type(self.key_mode) is not int or self.key_mode not in KEY_MODES:
            raise ValueError("key_mode is unsupported")
        if self.easier_difficulty not in DIFFICULTIES:
            raise ValueError("easier_difficulty is unsupported")
        if self.difficulty not in DIFFICULTIES:
            raise ValueError("difficulty is unsupported")
        if (
            DIFFICULTIES.index(self.difficulty)
            != DIFFICULTIES.index(self.easier_difficulty) + 1
        ):
            raise ValueError("difficulty target must be the adjacent harder label")
        for field_name in ("easier_rating", "harder_rating", "rating_deficit"):
            value = getattr(self, field_name)
            if type(value) is not float or not math.isfinite(value):
                raise TypeError(f"{field_name} must be a finite exact float")
        if self.rating_deficit <= 0:
            raise ValueError("rating_deficit must be positive")
        minimum_rating = self.minimum_rating
        if minimum_rating is None:
            minimum_rating = self.easier_rating + MIN_ADJACENT_RATING_GAP
            object.__setattr__(self, "minimum_rating", minimum_rating)
        if type(minimum_rating) is not float or not math.isfinite(minimum_rating):
            raise TypeError("minimum_rating must be a finite exact float")
        maximum_rating = self.maximum_rating
        if maximum_rating is not None:
            if type(maximum_rating) is not float or not math.isfinite(maximum_rating):
                raise TypeError("maximum_rating must be a finite exact float or None")
            if maximum_rating < minimum_rating:
                raise ValueError("target rating interval is infeasible")


@dataclass(frozen=True, slots=True)
class DifficultyShadowPartialPlan:
    target: DifficultyShadowTarget
    source: Candidate
    window: PartialRemapWindow
    required_gameplay_interval: RequiredGameplayIntervalV1 | None = None


@dataclass(frozen=True, slots=True)
class DifficultyShadowFullMapPlan:
    target: DifficultyShadowTarget
    source: Candidate
    requested_star: float
    calibration_state: Literal["PILOT_ONLY"]

    def __post_init__(self) -> None:
        if type(self.requested_star) is not float or not math.isfinite(
            self.requested_star
        ):
            raise TypeError("requested_star must be a finite exact float")
        if self.requested_star <= 0:
            raise ValueError("requested_star must be positive")
        if self.calibration_state != CURRENT_DIFFICULTY_FAMILY_CALIBRATION_STATE:
            raise ValueError("full-map shadow requires the current pilot-only state")


@dataclass(frozen=True, slots=True)
class DifficultyShadowFullMapDecision:
    reason: str
    plan: DifficultyShadowFullMapPlan | None

    def to_report(self) -> dict[str, object]:
        report: dict[str, object] = {
            "reason": self.reason,
            "calibrationState": CURRENT_DIFFICULTY_FAMILY_CALIBRATION_STATE,
            "mutatesSelection": False,
        }
        if self.plan is not None:
            report["requestedStar"] = self.plan.requested_star
            report["source"] = {
                "attempt": self.plan.source.attempt,
                "seed": self.plan.source.seed,
                "provenance": self.plan.source.provenance,
                "sha256": sha256(
                    self.plan.source.osu_text.encode("utf-8")
                ).hexdigest(),
            }
        return report


@dataclass(frozen=True, slots=True)
class DifficultyShadowPartialDecision:
    reason: str
    plan: DifficultyShadowPartialPlan | None
    considered_candidate_count: int

    def to_report(self) -> dict[str, object]:
        report: dict[str, object] = {
            "reason": self.reason,
            "consideredCandidateCount": self.considered_candidate_count,
        }
        if self.plan is not None:
            report["window"] = {
                "startMs": self.plan.window.start_ms,
                "endMs": self.plan.window.end_ms,
            }
            report["source"] = {
                "attempt": self.plan.source.attempt,
                "seed": self.plan.source.seed,
                "provenance": self.plan.source.provenance,
                "sha256": sha256(
                    self.plan.source.osu_text.encode("utf-8")
                ).hexdigest(),
            }
            interval = self.plan.required_gameplay_interval
            report["requiredGameplayInterval"] = (
                {
                    "startMs": interval.start_ms,
                    "endMs": interval.end_ms,
                    "minimumCompleteGroups": interval.minimum_complete_groups,
                    "allowedGroupTypes": [
                        item.value for item in interval.allowed_group_types
                    ],
                    "evidenceClass": interval.evidence_class.value,
                    "evidenceDigest": interval.evidence_digest,
                    "mode": interval.mode.value,
                }
                if interval is not None
                else None
            )
        return report


@dataclass(frozen=True, slots=True)
class DifficultyShadowRequiredEvidenceDecision:
    evidence: RequiredGameplayEvidenceV1 | None
    reason: str

    def __post_init__(self) -> None:
        if self.evidence is not None and type(self.evidence) is not RequiredGameplayEvidenceV1:
            raise TypeError("evidence must be RequiredGameplayEvidenceV1 or None")
        if type(self.reason) is not str or not self.reason:
            raise TypeError("reason must be a non-empty exact string")

    def to_report(self) -> dict[str, object]:
        return {
            "reason": self.reason,
            "eligible": self.evidence is not None,
            "evidenceDigest": (
                self.evidence.anchor_evidence_digest
                if self.evidence is not None
                else None
            ),
        }


def _partial_decision(
    reason: str,
    *,
    considered_candidate_count: int,
) -> DifficultyShadowPartialDecision:
    return DifficultyShadowPartialDecision(
        reason=reason,
        plan=None,
        considered_candidate_count=considered_candidate_count,
    )


def _has_single_leading_gap(candidate: Candidate) -> bool:
    gaps = candidate.acceptance.timing.coverage_gaps
    if len(gaps) != 1:
        return False
    gap = gaps[0]
    return gap.start_ms == 0 and gap.position == "LEADING"


def _candidate_rank(candidate: Candidate) -> tuple[object, ...]:
    profile = candidate.acceptance.profile
    vector = profile.difficulty_vector_v2 if profile is not None else None
    ordering_score = vector.ordering_score if vector is not None else -math.inf
    matched_f1 = candidate.acceptance.timing.overall.matched_f1_50
    action_rank = {
        GateAction.PASS: 0,
        GateAction.REVIEW: 1,
        GateAction.RETRY_MAP: 2,
    }[candidate.acceptance.action]
    gap = candidate.acceptance.timing.coverage_gaps[0]
    return (
        action_rank,
        -(matched_f1 if matched_f1 is not None else -1.0),
        gap.end_ms - gap.start_ms,
        -ordering_score,
        candidate.attempt,
        candidate.seed,
        sha256(candidate.osu_text.encode("utf-8")).hexdigest(),
    )


def _family_representative(
    state: VariantState,
    selected: Candidate | None,
) -> Candidate | None:
    if (
        selected is not None
        and selected.provenance != "SAFE_FALLBACK"
        and _hard_safe(selected)
    ):
        return selected
    candidates = tuple(
        candidate
        for candidate in state.candidates.playtest_candidates
        if candidate.provenance != "SAFE_FALLBACK" and _hard_safe(candidate)
    )
    if not candidates:
        return None
    action_rank = {
        GateAction.PASS: 0,
        GateAction.REVIEW: 1,
        GateAction.RETRY_MAP: 2,
    }
    return min(
        candidates,
        key=lambda candidate: (
            action_rank[candidate.acceptance.action],
            candidate.provenance == "RAW_UNVERIFIED",
            candidate.attempt,
            candidate.seed,
            sha256(candidate.osu_text.encode("utf-8")).hexdigest(),
        ),
    )


def build_required_gameplay_evidence_for_shadow(
    selections: list[Selection],
    *,
    source: Candidate,
    authority: SongTimingAuthority,
    onset_analysis: OnsetAnalysis,
) -> DifficultyShadowRequiredEvidenceDecision:
    """Project only identity-free, already-observed evidence for one source."""

    leading = authority.leading_coverage
    if leading is None:
        return DifficultyShadowRequiredEvidenceDecision(
            None,
            "LEADING_COVERAGE_EVIDENCE_UNAVAILABLE",
        )
    anchor = leading.intro_anchor
    if (
        anchor.anchor_ms is None
        or anchor.anchor_grid_ms is None
        or anchor.aggregate_percentile_rank is None
    ):
        return DifficultyShadowRequiredEvidenceDecision(
            None,
            "INTRO_ANCHOR_EVIDENCE_UNAVAILABLE",
        )
    activity = onset_analysis.activity
    if activity is None:
        return DifficultyShadowRequiredEvidenceDecision(
            None,
            "ACTIVE_AUDIO_EVIDENCE_UNAVAILABLE",
        )
    active_onsets = tuple(sorted(set(activity.active_onset_ms)))
    if not active_onsets:
        return DifficultyShadowRequiredEvidenceDecision(
            None,
            "ACTIVE_AUDIO_EVIDENCE_UNAVAILABLE",
        )
    distinct_rows = sorted({note.time_ms for note in source.generated.notes})
    if len(distinct_rows) < 2:
        return DifficultyShadowRequiredEvidenceDecision(
            None,
            "REFERENCE_ROWS_UNAVAILABLE",
        )

    family_slots: list[RequiredGameplayFamilySlotV1] = []
    repeated_refusals = 0
    for states, assignment, _review in selections:
        key_mode = next(iter(states.values())).key_mode
        for difficulty in DIFFICULTIES:
            state = states[difficulty]
            representative = _family_representative(
                state,
                assignment[difficulty],
            )
            if representative is not None:
                family_slots.append(
                    RequiredGameplayFamilySlotV1(
                        key_mode=key_mode,
                        difficulty=difficulty,
                        supports_anchor=representative.intro_anchor_covered is True,
                    )
                )
            if (key_mode, difficulty) == (
                source.request.key_mode,
                source.request.difficulty,
            ):
                repeated_refusals += sum(
                    candidate.recovery_reason
                    == "DIFFICULTY_SHADOW_PARTIAL_CHALLENGER"
                    and candidate.intro_anchor_covered is False
                    for candidate in state.candidates.shadow_candidates
                )

    anchor_digest = build_intro_selection_evidence(
        anchor,
        active_onset_ms=active_onsets,
        votes=(),
    ).audio_evidence_digest
    first_row_ms = distinct_rows[0]
    reference_first_row_supported = any(
        abs(first_row_ms - onset_ms) <= GRID_SUPPORT_WINDOW_MS
        for onset_ms in active_onsets
    )
    gaps = source.acceptance.timing.coverage_gaps
    local_audio_supported = (
        len(gaps) == 1
        and gaps[0].start_ms == 0
        and gaps[0].position == "LEADING"
        and gaps[0].active_onset_count > 0
        and gaps[0].active_frame_ratio > 0.0
        and any(
            abs(anchor.anchor_ms - onset_ms) <= GRID_SUPPORT_WINDOW_MS
            for onset_ms in active_onsets
        )
    )
    try:
        evidence = RequiredGameplayEvidenceV1(
            anchor_status=anchor.status,
            anchor_ms=anchor.anchor_ms,
            anchor_grid_ms=anchor.anchor_grid_ms,
            aggregate_rank=float(anchor.aggregate_percentile_rank),
            prominent_band_count=anchor.prominent_band_count,
            pulse_support_count=anchor.pulse_continuation_matches,
            family_slots=tuple(family_slots),
            local_audio_supported=local_audio_supported,
            reference_first_row_supported=reference_first_row_supported,
            repeated_high_confidence_refusal=repeated_refusals >= 2,
            timing_authority_valid=bool(authority.bpm_events),
            timing_authority_digest=authority.sha256,
            anchor_evidence_digest=anchor_digest,
        )
    except (TypeError, ValueError):
        return DifficultyShadowRequiredEvidenceDecision(
            None,
            "REQUIRED_GAMEPLAY_EVIDENCE_INVALID",
        )
    return DifficultyShadowRequiredEvidenceDecision(
        evidence,
        "REQUIRED_GAMEPLAY_EVIDENCE_READY",
    )


def plan_difficulty_shadow_partial_repair(
    target: DifficultyShadowTarget,
    candidates: tuple[Candidate, ...],
    bpm_events: tuple[OsuBpmEvent, ...],
    *,
    duration_ms: int,
) -> DifficultyShadowPartialDecision:
    """Choose one hard-safe, difficult candidate with only a proven intro gap."""

    if type(duration_ms) is not int or duration_ms <= 0:
        raise ValueError("duration_ms must be a positive exact integer")
    scoped = tuple(
        candidate
        for candidate in candidates
        if candidate.request.key_mode == target.key_mode
        and candidate.request.difficulty == target.difficulty
    )
    considered_count = len(scoped)
    cohort = tuple(
        candidate
        for candidate in scoped
        if candidate.provenance not in {"RAW_UNVERIFIED", "SAFE_FALLBACK"}
    )
    if not cohort:
        return _partial_decision(
            "NO_NEAR_SOLUTION_CANDIDATE",
            considered_candidate_count=considered_count,
        )

    cohort = tuple(candidate for candidate in cohort if _hard_safe(candidate))
    if not cohort:
        return _partial_decision(
            "CANDIDATE_NOT_HARD_SAFE",
            considered_candidate_count=considered_count,
        )

    minimum_rating = target.minimum_rating
    assert minimum_rating is not None
    cohort = tuple(
        candidate
        for candidate in cohort
        if candidate.acceptance.profile is not None
        and candidate.acceptance.profile.difficulty.project_rating >= minimum_rating
    )
    if not cohort:
        return _partial_decision(
            "CANDIDATE_DIFFICULTY_INSUFFICIENT",
            considered_candidate_count=considered_count,
        )

    cohort = tuple(
        candidate for candidate in cohort if _has_single_leading_gap(candidate)
    )
    if not cohort:
        return _partial_decision(
            "GAP_NOT_SINGLE_LEADING",
            considered_candidate_count=considered_count,
        )

    cohort = tuple(
        candidate for candidate in cohort if candidate.intro_anchor_covered is False
    )
    if not cohort:
        return _partial_decision(
            "INTRO_EVIDENCE_UNAVAILABLE",
            considered_candidate_count=considered_count,
        )

    local_windows: list[tuple[Candidate, int]] = []
    local_evidence_count = 0
    for candidate in cohort:
        distinct_rows = sorted({note.time_ms for note in candidate.generated.notes})
        if len(distinct_rows) < 2 or not bpm_events:
            continue
        local_evidence_count += 1
        end_ms = advance_tempo_map_beats(
            distinct_rows[1],
            4.0,
            bpm_events,
        )
        if end_ms <= duration_ms:
            local_windows.append((candidate, end_ms))
    if not local_windows:
        return _partial_decision(
            (
                "WINDOW_TOO_LARGE"
                if local_evidence_count > 0
                else "LOCAL_WINDOW_UNAVAILABLE"
            ),
            considered_candidate_count=considered_count,
        )

    plans: list[DifficultyShadowPartialPlan] = []
    for candidate, end_ms in local_windows:
        window = expand_partial_remap_window(
            candidate.generated.notes,
            start_ms=0,
            end_ms=end_ms,
            duration_ms=duration_ms,
        )
        if window is None or window.end_ms >= duration_ms:
            continue
        plans.append(DifficultyShadowPartialPlan(target, candidate, window))
    if not plans:
        return _partial_decision(
            "WINDOW_TOO_LARGE",
            considered_candidate_count=considered_count,
        )

    selected = min(plans, key=lambda plan: _candidate_rank(plan.source))
    return DifficultyShadowPartialDecision(
        reason="PARTIAL_NEAR_SOLUTION_SELECTED",
        plan=selected,
        considered_candidate_count=considered_count,
    )


def plan_difficulty_shadow_full_map(
    target: DifficultyShadowTarget,
    source: Candidate | None,
) -> DifficultyShadowFullMapDecision:
    """Plan one uncalibrated full-map research sample for a relative deficit."""

    if source is None:
        return DifficultyShadowFullMapDecision("SOURCE_UNAVAILABLE", None)
    if source.request.key_mode != target.key_mode:
        return DifficultyShadowFullMapDecision("SOURCE_IDENTITY_MISMATCH", None)
    if not _hard_safe(source):
        return DifficultyShadowFullMapDecision("SOURCE_NOT_HARD_SAFE", None)
    requested_step = (
        REQUESTED_STAR[target.difficulty]
        - REQUESTED_STAR[target.easier_difficulty]
    )
    if requested_step <= 0:
        return DifficultyShadowFullMapDecision("REQUESTED_STAR_STEP_UNAVAILABLE", None)
    # A relabelled source is provenance, not the target prompt.  Starting from
    # the source label's request value previously turned an EXPERT->NORMAL
    # repair into another EXPERT-strength request.  Keep the bounded step, but
    # anchor it to the target label's declared generation scale.
    requested_star = float(REQUESTED_STAR[target.difficulty] + requested_step)
    return DifficultyShadowFullMapDecision(
        "FULL_MAP_SHADOW_PLANNED",
        DifficultyShadowFullMapPlan(
            target=target,
            source=source,
            requested_star=requested_star,
            calibration_state=CURRENT_DIFFICULTY_FAMILY_CALIBRATION_STATE,
        ),
    )


def build_difficulty_shadow_full_map_request(
    plan: DifficultyShadowFullMapPlan,
    *,
    timing_reference_path: Path,
    seed: int,
) -> GenerationRequest:
    """Build an immutable target-labelled request from source provenance."""

    return replace(
        plan.source.request,
        key_mode=plan.target.key_mode,
        difficulty=plan.target.difficulty,
        timing_reference_path=timing_reference_path,
        seed=seed,
        descriptors=DESCRIPTORS[plan.target.difficulty],
        requested_star=plan.requested_star,
        partial_start_ms=None,
        partial_end_ms=None,
        add_to_beatmap=False,
        required_gameplay_interval=None,
    )


def bind_required_gameplay_interval(
    decision: DifficultyShadowPartialDecision,
    *,
    evidence: RequiredGameplayEvidenceV1 | None,
    bpm_events: tuple[OsuBpmEvent, ...],
    duration_ms: int,
    mode: RequiredGameplayIntervalMode,
) -> DifficultyShadowPartialDecision:
    """Bind one evidence-derived interval without changing an ineligible plan."""

    if type(decision) is not DifficultyShadowPartialDecision:
        raise TypeError("decision must be DifficultyShadowPartialDecision")
    if type(mode) is not RequiredGameplayIntervalMode:
        raise TypeError("mode must be RequiredGameplayIntervalMode")
    plan = decision.plan
    if plan is None:
        return decision
    if plan.required_gameplay_interval is not None:
        return decision
    if evidence is None:
        return _partial_decision(
            "REQUIRED_GAMEPLAY_EVIDENCE_UNAVAILABLE",
            considered_candidate_count=decision.considered_candidate_count,
        )

    distinct_rows = sorted({note.time_ms for note in plan.source.generated.notes})
    if len(distinct_rows) < 2:
        return _partial_decision(
            "REQUIRED_GAMEPLAY_SECOND_ROW_UNAVAILABLE",
            considered_candidate_count=decision.considered_candidate_count,
        )
    interval_decision = plan_required_gameplay_interval(
        evidence,
        partial_window=plan.window,
        bpm_events=bpm_events,
        second_distinct_row_ms=distinct_rows[1],
        duration_ms=duration_ms,
        mode=mode,
    )
    if interval_decision.interval is None:
        return _partial_decision(
            interval_decision.reason,
            considered_candidate_count=decision.considered_candidate_count,
        )
    return DifficultyShadowPartialDecision(
        reason=interval_decision.reason,
        plan=replace(
            plan,
            required_gameplay_interval=interval_decision.interval,
        ),
        considered_candidate_count=decision.considered_candidate_count,
    )


def difficulty_shadow_candidate_failure_reason(
    plan: DifficultyShadowPartialPlan,
    challenger: Candidate,
) -> str | None:
    """Return the first failed post-inference contract for a partial challenger."""

    if any(
        note.kind == "HOLD"
        and note.time_ms <= plan.window.end_ms
        < note.time_ms + (note.duration_ms or 0)
        for notes in (plan.source.generated.notes, challenger.generated.notes)
        for note in notes
    ):
        return "SUFFIX_SEMANTIC_MUTATION"
    if partial_suffix_signature(
        plan.source.generated.notes,
        end_ms=plan.window.end_ms,
    ) != partial_suffix_signature(
        challenger.generated.notes,
        end_ms=plan.window.end_ms,
    ):
        return "SUFFIX_SEMANTIC_MUTATION"
    if not _hard_safe(challenger):
        return "HARD_SAFETY_REJECTED"
    if challenger.intro_anchor_covered is not True:
        return "INTRO_OR_GAP_UNRESOLVED"
    if challenger.acceptance.timing.coverage_gaps:
        return "INTRO_OR_GAP_UNRESOLVED"

    profile = challenger.acceptance.profile
    if profile is None:
        return "QUALITY_EVIDENCE_UNAVAILABLE"
    minimum_rating = plan.target.minimum_rating
    assert minimum_rating is not None
    if profile.difficulty.project_rating < minimum_rating:
        return "DIFFICULTY_LOST"
    maximum_rating = plan.target.maximum_rating
    if (
        maximum_rating is not None
        and profile.difficulty.project_rating > maximum_rating
    ):
        return "NEXT_DIFFICULTY_ORDER_LOST"

    source_metrics = plan.source.acceptance.timing.overall
    challenger_metrics = challenger.acceptance.timing.overall
    values = (
        source_metrics.matched_f1_50,
        source_metrics.matched_precision_50,
        challenger_metrics.matched_f1_50,
        challenger_metrics.matched_precision_50,
    )
    if any(value is None or not math.isfinite(value) for value in values):
        return "QUALITY_EVIDENCE_UNAVAILABLE"
    if (
        challenger_metrics.matched_f1_50
        < source_metrics.matched_f1_50 - MATCHED_F1_EPSILON
        or challenger_metrics.matched_precision_50
        < source_metrics.matched_precision_50 - MATCHED_PRECISION_EPSILON
    ):
        return "QUALITY_NONREGRESSION_FAILED"
    return None


def difficulty_shadow_full_map_candidate_failure_reason(
    plan: DifficultyShadowFullMapPlan,
    challenger: Candidate,
) -> str | None:
    """Return the first failed non-regression contract for a full-map sample."""

    if not _hard_safe(challenger):
        return "HARD_SAFETY_REJECTED"
    if (
        plan.source.intro_anchor_covered is True
        and challenger.intro_anchor_covered is not True
    ):
        return "INTRO_REGRESSION"
    try:
        gap_comparison = compare_gap_evidence(
            _gap_safety(plan.source, candidate_id="full-map-source"),
            _gap_safety(challenger, candidate_id="full-map-challenger"),
        )
    except (TypeError, ValueError):
        return "GAP_EVIDENCE_UNAVAILABLE"
    if gap_comparison.status != "NON_REGRESSION":
        return "ACTIVE_GAP_REGRESSION"

    profile = challenger.acceptance.profile
    if profile is None:
        return "QUALITY_EVIDENCE_UNAVAILABLE"
    rating = profile.difficulty.project_rating
    minimum_rating = plan.target.minimum_rating
    assert minimum_rating is not None
    if rating < minimum_rating:
        return "DIFFICULTY_LOST"
    maximum_rating = plan.target.maximum_rating
    if maximum_rating is not None and rating > maximum_rating:
        return "NEXT_DIFFICULTY_ORDER_LOST"

    source_metrics = plan.source.acceptance.timing.overall
    challenger_metrics = challenger.acceptance.timing.overall
    values = (
        source_metrics.matched_f1_50,
        source_metrics.matched_precision_50,
        challenger_metrics.matched_f1_50,
        challenger_metrics.matched_precision_50,
    )
    if any(value is None or not math.isfinite(value) for value in values):
        return "QUALITY_EVIDENCE_UNAVAILABLE"
    if (
        challenger_metrics.matched_f1_50
        < source_metrics.matched_f1_50 - MATCHED_F1_EPSILON
        or challenger_metrics.matched_precision_50
        < source_metrics.matched_precision_50 - MATCHED_PRECISION_EPSILON
    ):
        return "QUALITY_NONREGRESSION_FAILED"
    return None


_REQUIRED_GAMEPLAY_FAILURE_REASONS = frozenset(
    {
        "REQUIRED_GAMEPLAY_INTERVAL_NOT_ADDRESSABLE",
        "REQUIRED_GAMEPLAY_INTERVAL_TOKEN_BUDGET_EXHAUSTED",
        "REQUIRED_GAMEPLAY_INTERVAL_NO_LEGAL_GROUP",
        "REQUIRED_GAMEPLAY_INTERVAL_UNSATISFIED_AT_CUT",
        "REQUIRED_GAMEPLAY_INTERVAL_ACCOUNTING_MISMATCH",
    }
)


def difficulty_shadow_inference_failure_reason(
    error: Exception,
    *,
    inference_completed: bool,
) -> str:
    """Keep a validated runtime reason instead of collapsing it to inference failed."""

    if (
        isinstance(error, WorkerError)
        and error.code is ErrorCode.MANIA_REQUIRED_GAMEPLAY_FAILED
    ):
        reason = error.context.get("reason")
        if reason in _REQUIRED_GAMEPLAY_FAILURE_REASONS:
            assert isinstance(reason, str)
            return reason
    if not inference_completed:
        return "INFERENCE_FAILED"
    if isinstance(error, TimingAuthorityValidationError):
        return "TIMING_IDENTITY_CHANGED"
    return "CANONICAL_VALIDATION_FAILED"


def choose_difficulty_shadow_target(
    slots: tuple[DifficultyShadowSlot, ...],
) -> DifficultyShadowTarget | None:
    """Choose at most one feasible adjacent-label separation deficit.

    This is a project-rating SHADOW screener, not calibrated human truth.  It
    treats the pre-existing family review gap as a relative interval and never
    compares non-adjacent labels across missing evidence.
    """

    identities = [(slot.key_mode, slot.difficulty) for slot in slots]
    if len(identities) != len(set(identities)):
        raise ValueError("duplicate difficulty shadow slot")

    targets: list[DifficultyShadowTarget] = []
    for key_mode in sorted({slot.key_mode for slot in slots}):
        by_difficulty = {
            slot.difficulty: slot for slot in slots if slot.key_mode == key_mode
        }
        for harder_index in range(1, len(DIFFICULTIES)):
            easier_difficulty = DIFFICULTIES[harder_index - 1]
            harder_difficulty = DIFFICULTIES[harder_index]
            easier = by_difficulty.get(easier_difficulty)
            harder = by_difficulty.get(harder_difficulty)
            if easier is None or harder is None:
                continue
            minimum_rating = easier.project_rating + MIN_ADJACENT_RATING_GAP
            deficit = minimum_rating - harder.project_rating
            if deficit <= 0:
                continue
            if harder.has_existing_safe_resolution:
                continue
            maximum_rating: float | None = None
            if harder_index + 1 < len(DIFFICULTIES):
                next_slot = by_difficulty.get(DIFFICULTIES[harder_index + 1])
                if next_slot is not None:
                    maximum_rating = (
                        next_slot.project_rating - MIN_ADJACENT_RATING_GAP
                    )
                    if maximum_rating < minimum_rating:
                        continue
            targets.append(
                DifficultyShadowTarget(
                    key_mode=key_mode,
                    easier_difficulty=easier.difficulty,
                    difficulty=harder.difficulty,
                    easier_rating=easier.project_rating,
                    harder_rating=harder.project_rating,
                    rating_deficit=deficit,
                    minimum_rating=minimum_rating,
                    maximum_rating=maximum_rating,
                )
            )

    if not targets:
        return None
    return min(
        targets,
        key=lambda target: (
            -target.rating_deficit,
            target.key_mode,
            DIFFICULTIES.index(target.difficulty),
        ),
    )


def difficulty_shadow_target_failure_reason(
    slots: tuple[DifficultyShadowSlot, ...],
) -> str | None:
    """Explain why the generic SHADOW policy has no executable target."""

    if choose_difficulty_shadow_target(slots) is not None:
        return None
    if not slots:
        return "INCOMPLETE_FAMILY_EVIDENCE"

    deficits = 0
    existing_resolutions = 0
    infeasible_intervals = 0
    incomplete = False
    for key_mode in sorted({slot.key_mode for slot in slots}):
        by_difficulty = {
            slot.difficulty: slot for slot in slots if slot.key_mode == key_mode
        }
        if set(by_difficulty) != set(DIFFICULTIES):
            incomplete = True
        for harder_index in range(1, len(DIFFICULTIES)):
            easier = by_difficulty.get(DIFFICULTIES[harder_index - 1])
            harder = by_difficulty.get(DIFFICULTIES[harder_index])
            if easier is None or harder is None:
                continue
            minimum_rating = easier.project_rating + MIN_ADJACENT_RATING_GAP
            if harder.project_rating >= minimum_rating:
                continue
            deficits += 1
            if harder.has_existing_safe_resolution:
                existing_resolutions += 1
                continue
            if harder_index + 1 < len(DIFFICULTIES):
                next_slot = by_difficulty.get(DIFFICULTIES[harder_index + 1])
                if (
                    next_slot is not None
                    and next_slot.project_rating - MIN_ADJACENT_RATING_GAP
                    < minimum_rating
                ):
                    infeasible_intervals += 1

    if deficits == 0:
        return (
            "INCOMPLETE_FAMILY_EVIDENCE"
            if incomplete
            else "NO_RELATIVE_DEFICIT"
        )
    if existing_resolutions == deficits:
        return "EXISTING_SAFE_RESOLUTION"
    if infeasible_intervals > 0:
        return "NO_FEASIBLE_RELATIVE_INTERVAL"
    return "NO_FEASIBLE_DIFFICULTY_SHADOW_TARGET"


def _hard_safe(candidate: Candidate) -> bool:
    return all(
        candidate.acceptance.decision(axis).action is GateAction.PASS
        for axis in (
            GateAxis.STRUCTURE,
            GateAxis.TIMING_IDENTITY,
            GateAxis.SONG_BOUNDS,
        )
    )


def _gap_safety(candidate: Candidate, *, candidate_id: str) -> CandidateSafetyEvidenceV3:
    gaps = tuple(
        GapIntervalEvidence(
            start_ms=gap.start_ms,
            end_ms=gap.end_ms,
            position=gap.position,
            active_onset_count=gap.active_onset_count,
            active_frame_ratio=float(gap.active_frame_ratio),
            opportunity_kind=(
                gap.opportunity.kind.value if gap.opportunity is not None else "UNKNOWN"
            ),
            local_audio_evidence_digest=None,
        )
        for gap in sorted(
            candidate.acceptance.timing.coverage_gaps,
            key=lambda item: (item.start_ms, item.end_ms),
        )
    )
    return CandidateSafetyEvidenceV3(
        candidate_id=candidate_id,
        structure_safe=(
            candidate.acceptance.decision(GateAxis.STRUCTURE).action
            is GateAction.PASS
        ),
        timing_identity_safe=(
            candidate.acceptance.decision(GateAxis.TIMING_IDENTITY).action
            is GateAction.PASS
        ),
        song_bounds_safe=(
            candidate.acceptance.decision(GateAxis.SONG_BOUNDS).action
            is GateAction.PASS
        ),
        serialization_safe=True,
        publication_tier="PLAYTEST_ONLY",
        model_backed=candidate.provenance != "SAFE_FALLBACK",
        recovery_trust_rank=0,
        active_gaps=gaps,
    )


def existing_candidate_resolves_fallback(
    current: Candidate,
    *,
    easier_rating: float,
    candidates: tuple[Candidate, ...],
) -> bool:
    """Return true only for a demonstrably non-regressing existing resolution.

    The historical name is retained for report compatibility; the check is
    provenance-agnostic and applies to PRIMARY and recovery selections alike.
    """

    current_f1 = current.acceptance.timing.overall.matched_f1_50
    for challenger in candidates:
        if challenger is current or challenger.provenance in {
            "RAW_UNVERIFIED",
            "SAFE_FALLBACK",
        }:
            continue
        profile = challenger.acceptance.profile
        if (
            profile is None
            or profile.difficulty.project_rating
            < easier_rating + MIN_ADJACENT_RATING_GAP
        ):
            continue
        if not _hard_safe(challenger):
            continue
        replacement = decide_candidate_replacement(
            candidate_quality_snapshot(current),
            candidate_quality_snapshot(challenger),
            stage="DIFFICULTY_SHADOW_EXISTING_CANDIDATE",
            objective_improved=True,
        )
        if not replacement.accepted:
            continue
        if current.intro_anchor_covered is True and challenger.intro_anchor_covered is not True:
            continue
        challenger_f1 = challenger.acceptance.timing.overall.matched_f1_50
        if current_f1 is not None and (
            challenger_f1 is None or challenger_f1 < current_f1 - _MATCHED_F1_EPSILON
        ):
            continue
        try:
            gap_comparison = compare_gap_evidence(
                _gap_safety(current, candidate_id="current"),
                _gap_safety(challenger, candidate_id="challenger"),
            )
        except (TypeError, ValueError):
            # Optional research inference must not make production fail when a
            # legacy diagnostic cannot be projected canonically.
            continue
        if gap_comparison.status == "NON_REGRESSION":
            return True
    return False


def difficulty_shadow_slots(selections: list[Selection]) -> tuple[DifficultyShadowSlot, ...]:
    slots: list[DifficultyShadowSlot] = []
    for states, assignment, _review in selections:
        key_mode = next(iter(states.values())).key_mode
        selected_ratings = {
            difficulty: selected.acceptance.profile.difficulty.project_rating
            for difficulty in DIFFICULTIES
            if (selected := assignment[difficulty]) is not None
            and selected.acceptance.profile is not None
        }
        for difficulty in DIFFICULTIES:
            selected = assignment[difficulty]
            if selected is None or selected.acceptance.profile is None:
                continue
            difficulty_index = DIFFICULTIES.index(difficulty)
            adjacent_easier_rating = (
                selected_ratings.get(DIFFICULTIES[difficulty_index - 1])
                if difficulty_index > 0
                else None
            )
            selected_rating = selected.acceptance.profile.difficulty.project_rating
            slots.append(
                DifficultyShadowSlot(
                    key_mode=key_mode,
                    difficulty=difficulty,
                    project_rating=float(selected_rating),
                    selected_provenance=selected.provenance,
                    has_existing_safe_resolution=(
                        existing_candidate_resolves_fallback(
                            selected,
                            easier_rating=adjacent_easier_rating,
                            candidates=states[difficulty].candidates.playtest_candidates,
                        )
                        if adjacent_easier_rating is not None
                        else False
                    ),
                )
            )
    return tuple(slots)


def execute_difficulty_shadow_challenger(
    state: VariantState,
    plan: DifficultyShadowPartialPlan,
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
) -> Candidate | None:
    """Generate one bounded research candidate without changing selection."""

    source = plan.source
    window = plan.window
    blocked_reason = (
        state.full_length_retry_blocked_by.get("reason")
        if state.full_length_retry_blocked_by is not None
        else None
    )
    if blocked_reason not in {None, "HARD_SAFE_RAW_AVAILABLE"}:
        state.attempt_evidence.append(
            {
                "reason": "DIFFICULTY_SHADOW_SUPPRESSED_BY_FULL_LENGTH_BLOCK",
                "blockedBy": dict(state.full_length_retry_blocked_by or {}),
            }
        )
        return None


    if state.recovery.was_attempted(RecoveryKind.DIFFICULTY_SHADOW):
        return None
    workdir = (
        run_dir
        / "raw"
        / "work"
        / f"epoch-{authority_epoch}"
        / f"{state.key_mode}k-{state.difficulty.lower()}"
        / "difficulty-shadow-challenger"
    )
    reference_dir = workdir.parent / "difficulty-shadow-reference"
    reference_path = reference_dir / "near-solution-reference.osu"
    try:
        if not source.osu_text:
            raise ValueError("difficulty shadow source has no serialized beatmap")
        reference_dir.mkdir(parents=True, exist_ok=True)
        reference_path.write_text(source.osu_text, encoding="utf-8", newline="\n")
    except (OSError, ValueError) as error:
        state.attempt_errors.append(error_report_json(error))
        state.attempt_evidence.append(
            {
                "reason": "REFERENCE_PREPARATION_FAILED",
                "errorType": type(error).__name__,
                "message": str(error),
                "mutatesSelection": False,
            }
        )
        return None
    window_ms = window.end_ms - window.start_ms
    if not inference_budget.consume(window_ms):
        state.attempt_evidence.append(
            {
                "reason": "DIFFICULTY_SHADOW_BUDGET_EXHAUSTED",
                "budget": inference_budget.to_report(),
                "plannedWindowMs": window_ms,
                "mutatesSelection": False,
            }
        )
        return None
    if not state.recovery.claim(RecoveryKind.DIFFICULTY_SHADOW):
        return None

    attempt = state.budget.next_attempt
    retry_seed = base_seed + state.flat_index + (attempt - 1) * _VARIANT_COUNT
    state.budget.reserve_additional_attempt(seed=retry_seed)
    request = replace(
        source.request,
        timing_reference_path=reference_path,
        seed=retry_seed,
        partial_start_ms=window.start_ms,
        partial_end_ms=window.end_ms,
        add_to_beatmap=True,
        required_gameplay_interval=plan.required_gameplay_interval,
    )
    source_sha256 = sha256(source.osu_text.encode("utf-8")).hexdigest()
    suffix_signature = partial_suffix_signature(
        source.generated.notes,
        end_ms=window.end_ms,
    )
    suffix_digest = sha256(
        json.dumps(suffix_signature, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
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
            purpose="DIFFICULTY_SHADOW_CHALLENGER",
        )
        inference_completed = True
        state.budget.record_quality_attempt()
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
            attempt=attempt,
            seed=retry_seed,
            purpose="DIFFICULTY_SHADOW_CHALLENGER",
            acceptance=acceptance,
        )
        osu_text = serialize_candidate(
            generated,
            authority=authority,
            prepared=prepared,
            key_mode=state.key_mode,
        )
        challenger = Candidate(
            request=request,
            generated=generated,
            acceptance=acceptance,
            osu_text=osu_text,
            workdir=workdir,
            attempt=attempt,
            seed=retry_seed,
            provenance="RETRY",
            recovery_reason="DIFFICULTY_SHADOW_PARTIAL_CHALLENGER",
            intro_anchor_covered=intro_anchor_covered(generated, authority),
        )
        evidence = {
            "reason": "DIFFICULTY_SHADOW_CANDIDATE_EVALUATED",
            "seed": retry_seed,
            "workdir": workdir.relative_to(run_dir).as_posix(),
            "target": {
                "keyMode": plan.target.key_mode,
                "easierDifficulty": plan.target.easier_difficulty,
                "difficulty": plan.target.difficulty,
                "easierRating": plan.target.easier_rating,
            },
            "source": {
                "attempt": source.attempt,
                "seed": source.seed,
                "provenance": source.provenance,
                "sha256": source_sha256,
                "referencePath": reference_path.relative_to(run_dir).as_posix(),
            },
            "partialWindow": {
                "startMs": window.start_ms,
                "endMs": window.end_ms,
            },
            "suffixSignatureSha256": suffix_digest,
            "gateReport": acceptance.to_report(),
            "mutatesSelection": False,
        }
        state.attempt_evidence.append(evidence)
        failure_reason = difficulty_shadow_candidate_failure_reason(plan, challenger)
        if failure_reason is not None:
            state.attempt_evidence.append(
                {
                    **evidence,
                    "reason": failure_reason,
                }
            )
            record_candidate_event(
                state,
                admitted=False,
                authority_epoch=authority_epoch,
                attempt=attempt,
                seed=retry_seed,
                purpose="DIFFICULTY_SHADOW_CHALLENGER",
                reason=failure_reason,
                acceptance=acceptance,
            )
            return None
        state.attempt_evidence.append(
            {
                **evidence,
                "reason": "SHADOW_CANDIDATE_PRESERVED",
            }
        )
        record_candidate_event(
            state,
            admitted=False,
            authority_epoch=authority_epoch,
            attempt=attempt,
            seed=retry_seed,
            purpose="DIFFICULTY_SHADOW_CHALLENGER",
            reason="SHADOW_CANDIDATE_PRESERVED",
            acceptance=acceptance,
        )
        return challenger
    except (
        GeneratedChartValidationError,
        TimingAuthorityValidationError,
        WorkerError,
        OSError,
        RuntimeError,
        ValueError,
    ) as error:
        failure_reason = difficulty_shadow_inference_failure_reason(
            error,
            inference_completed=inference_completed,
        )
        if inference_completed:
            record_candidate_event(
                state,
                admitted=False,
                authority_epoch=authority_epoch,
                attempt=attempt,
                seed=retry_seed,
                purpose="DIFFICULTY_SHADOW_CHALLENGER",
                reason=failure_reason,
            )
        state.attempt_errors.append(error_report_json(error))
        state.attempt_evidence.append(
            {
                "reason": failure_reason,
                "seed": retry_seed,
                "errorType": type(error).__name__,
                "message": str(error),
                "mutatesSelection": False,
            }
        )
        return None


def execute_difficulty_shadow_full_map_challenger(
    state: VariantState,
    plan: DifficultyShadowFullMapPlan,
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
) -> Candidate | None:
    """Generate one full-map research sample without publication authority."""

    source = plan.source
    blocked_reason = (
        state.full_length_retry_blocked_by.get("reason")
        if state.full_length_retry_blocked_by is not None
        else None
    )
    if blocked_reason not in {None, "HARD_SAFE_RAW_AVAILABLE"}:
        state.attempt_evidence.append(
            {
                "reason": "DIFFICULTY_SHADOW_SUPPRESSED_BY_FULL_LENGTH_BLOCK",
                "blockedBy": dict(state.full_length_retry_blocked_by or {}),
                "mode": "FULL_MAP",
                "mutatesSelection": False,
            }
        )
        return None
    if state.recovery.was_attempted(RecoveryKind.DIFFICULTY_SHADOW):
        return None

    workdir = (
        run_dir
        / "raw"
        / "work"
        / f"epoch-{authority_epoch}"
        / f"{state.key_mode}k-{state.difficulty.lower()}"
        / "difficulty-shadow-full-map"
    )
    work_ms = prepared.normalized.duration_ms
    if not inference_budget.consume(work_ms):
        state.attempt_evidence.append(
            {
                "reason": "DIFFICULTY_SHADOW_BUDGET_EXHAUSTED",
                "budget": inference_budget.to_report(),
                "plannedWindowMs": work_ms,
                "mode": "FULL_MAP",
                "mutatesSelection": False,
            }
        )
        return None
    if not state.recovery.claim(RecoveryKind.DIFFICULTY_SHADOW):
        return None

    attempt = state.budget.next_attempt
    retry_seed = base_seed + state.flat_index + (attempt - 1) * _VARIANT_COUNT
    state.budget.reserve_additional_attempt(seed=retry_seed)
    request = build_difficulty_shadow_full_map_request(
        plan,
        timing_reference_path=authority.reference_path,
        seed=retry_seed,
    )
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
            purpose="DIFFICULTY_SHADOW_FULL_MAP_CHALLENGER",
        )
        inference_completed = True
        state.budget.record_quality_attempt()
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
            attempt=attempt,
            seed=retry_seed,
            purpose="DIFFICULTY_SHADOW_FULL_MAP_CHALLENGER",
            acceptance=acceptance,
        )
        osu_text = serialize_candidate(
            generated,
            authority=authority,
            prepared=prepared,
            key_mode=state.key_mode,
        )
        challenger = Candidate(
            request=request,
            generated=generated,
            acceptance=acceptance,
            osu_text=osu_text,
            workdir=workdir,
            attempt=attempt,
            seed=retry_seed,
            provenance="RETRY",
            recovery_reason="DIFFICULTY_SHADOW_FULL_MAP_CHALLENGER",
            intro_anchor_covered=intro_anchor_covered(generated, authority),
        )
        evidence = {
            "reason": "DIFFICULTY_SHADOW_FULL_MAP_CANDIDATE_EVALUATED",
            "seed": retry_seed,
            "workdir": workdir.relative_to(run_dir).as_posix(),
            "calibrationState": plan.calibration_state,
            "requestedStarBefore": source.request.requested_star,
            "requestedStarAfter": plan.requested_star,
            "originDifficulty": source.request.difficulty,
            "targetDifficulty": plan.target.difficulty,
            "target": {
                "keyMode": plan.target.key_mode,
                "easierDifficulty": plan.target.easier_difficulty,
                "difficulty": plan.target.difficulty,
                "minimumRating": plan.target.minimum_rating,
                "maximumRating": plan.target.maximum_rating,
            },
            "gateReport": acceptance.to_report(),
            "mutatesSelection": False,
        }
        state.attempt_evidence.append(evidence)
        failure_reason = difficulty_shadow_full_map_candidate_failure_reason(
            plan,
            challenger,
        )
        if failure_reason is not None:
            state.attempt_evidence.append({**evidence, "reason": failure_reason})
            record_candidate_event(
                state,
                admitted=False,
                authority_epoch=authority_epoch,
                attempt=attempt,
                seed=retry_seed,
                purpose="DIFFICULTY_SHADOW_FULL_MAP_CHALLENGER",
                reason=failure_reason,
                acceptance=acceptance,
            )
            return None
        state.attempt_evidence.append(
            {**evidence, "reason": "SHADOW_CANDIDATE_PRESERVED"}
        )
        record_candidate_event(
            state,
            admitted=False,
            authority_epoch=authority_epoch,
            attempt=attempt,
            seed=retry_seed,
            purpose="DIFFICULTY_SHADOW_FULL_MAP_CHALLENGER",
            reason="SHADOW_CANDIDATE_PRESERVED",
            acceptance=acceptance,
        )
        return challenger
    except (
        GeneratedChartValidationError,
        TimingAuthorityValidationError,
        WorkerError,
        OSError,
        RuntimeError,
        ValueError,
    ) as error:
        failure_reason = difficulty_shadow_inference_failure_reason(
            error,
            inference_completed=inference_completed,
        )
        if inference_completed:
            record_candidate_event(
                state,
                admitted=False,
                authority_epoch=authority_epoch,
                attempt=attempt,
                seed=retry_seed,
                purpose="DIFFICULTY_SHADOW_FULL_MAP_CHALLENGER",
                reason=failure_reason,
            )
        state.attempt_errors.append(error_report_json(error))
        state.attempt_evidence.append(
            {
                "reason": failure_reason,
                "seed": retry_seed,
                "errorType": type(error).__name__,
                "message": str(error),
                "mode": "FULL_MAP",
                "mutatesSelection": False,
            }
        )
        return None
