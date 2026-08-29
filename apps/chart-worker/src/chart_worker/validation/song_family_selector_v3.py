"""Report-only V3 proposal gate for difficulty-family replacements.

The module never mutates the selected assignment.  It answers whether a proposed
assignment would be eligible once a hash-bound, in-domain calibration is present.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from itertools import pairwise, product
from typing import Literal

from chart_worker.schema.types import DIFFICULTIES
from chart_worker.validation.difficulty_hazard_v1 import (
    AdjacentDifficultyHazardV1,
    CandidateDifficultyEvidenceV1,
    assess_assignment_difficulty_hazards,
    difficulty_evidence_corpus_sha256,
)
from chart_worker.validation.family_evidence_v3 import (
    CandidateFamilyEvidenceV3,
    SongSelectionEvidenceV3,
    compare_gap_evidence,
)

CalibrationStateV3 = Literal["IN_DOMAIN", "UNKNOWN"]
AssignmentV3 = tuple[tuple[str, str | None], ...]

_PUBLICATION_RANK = {
    "PRODUCTION_CANDIDATE": 0,
    "PLAYTEST_ONLY": 1,
    "DIAGNOSTIC_ONLY": 2,
}
_MATCHED_F1_EPSILON = 0.005
_MAX_FAMILY_COMBINATIONS = 4_096


def _exact_string(value: object, *, name: str) -> str:
    if type(value) is not str or not value:
        raise TypeError(f"{name} must be a non-empty exact string")
    return value


def _sha256(value: object, *, name: str) -> str:
    digest = _exact_string(value, name=name)
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return digest


@dataclass(frozen=True, slots=True)
class CalibrationPredictionV3:
    candidate_id: str
    state: CalibrationStateV3
    score: float | None
    calibration_sha256: str

    def __post_init__(self) -> None:
        _exact_string(self.candidate_id, name="candidate_id")
        if self.state not in {"IN_DOMAIN", "UNKNOWN"}:
            raise ValueError("calibration state is unsupported")
        _sha256(self.calibration_sha256, name="calibration_sha256")
        if self.state == "IN_DOMAIN":
            if type(self.score) is not float or not math.isfinite(self.score):
                raise ValueError("in-domain score must be a finite exact float")
        elif self.score is not None:
            raise ValueError("UNKNOWN calibration cannot carry a score")

    def to_report(self) -> dict[str, object]:
        return {
            "candidateId": self.candidate_id,
            "state": self.state,
            "score": self.score,
            "calibrationSha256": self.calibration_sha256,
        }


@dataclass(frozen=True, slots=True)
class ShadowV3ProposalEvaluation:
    selected_assignment: AssignmentV3
    proposed_assignment: AssignmentV3
    shadow_assignment: AssignmentV3
    proposal_eligible: bool
    blockers: tuple[str, ...]
    current_inversions: tuple[tuple[str, str], ...]
    proposed_inversions: tuple[tuple[str, str], ...]
    resolved_inversions: tuple[tuple[str, str], ...]
    created_inversions: tuple[tuple[str, str], ...]
    calibration_sha256: str | None
    mutates_selection: Literal[False] = False
    mode: Literal["SHADOW_V3"] = "SHADOW_V3"

    def to_report(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "selectedAssignment": dict(self.selected_assignment),
            "proposedAssignment": dict(self.proposed_assignment),
            "shadowAssignment": dict(self.shadow_assignment),
            "proposalEligible": self.proposal_eligible,
            "blockers": list(self.blockers),
            "currentInversions": [list(pair) for pair in self.current_inversions],
            "proposedInversions": [list(pair) for pair in self.proposed_inversions],
            "resolvedInversions": [list(pair) for pair in self.resolved_inversions],
            "createdInversions": [list(pair) for pair in self.created_inversions],
            "calibrationSha256": self.calibration_sha256,
            "mutatesSelection": self.mutates_selection,
        }


@dataclass(frozen=True, slots=True)
class ShadowHazardProposalEvaluationV1:
    """Report-only two-axis proposal result; production selection is immutable."""

    selected_assignment: AssignmentV3
    proposed_assignment: AssignmentV3
    shadow_assignment: AssignmentV3
    proposal_eligible: bool
    blockers: tuple[str, ...]
    current_at_risk: tuple[tuple[str, str], ...]
    proposed_at_risk: tuple[tuple[str, str], ...]
    current_unknown: tuple[tuple[str, str], ...]
    proposed_unknown: tuple[tuple[str, str], ...]
    resolved_at_risk: tuple[tuple[str, str], ...]
    created_at_risk: tuple[tuple[str, str], ...]
    difficulty_evidence_sha256: str
    mutates_selection: Literal[False] = False
    mode: Literal["SHADOW_V3_HAZARD"] = "SHADOW_V3_HAZARD"

    def to_report(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "selectedAssignment": dict(self.selected_assignment),
            "proposedAssignment": dict(self.proposed_assignment),
            "shadowAssignment": dict(self.shadow_assignment),
            "proposalEligible": self.proposal_eligible,
            "blockers": list(self.blockers),
            "currentAtRisk": [list(pair) for pair in self.current_at_risk],
            "proposedAtRisk": [list(pair) for pair in self.proposed_at_risk],
            "currentUnknown": [list(pair) for pair in self.current_unknown],
            "proposedUnknown": [list(pair) for pair in self.proposed_unknown],
            "resolvedAtRisk": [list(pair) for pair in self.resolved_at_risk],
            "createdAtRisk": [list(pair) for pair in self.created_at_risk],
            "difficultyEvidenceSha256": self.difficulty_evidence_sha256,
            "mutatesSelection": self.mutates_selection,
        }


def _normalize_assignment(
    value: AssignmentV3,
    *,
    expected_slots: set[str],
    name: str,
) -> AssignmentV3:
    if type(value) is not tuple:
        raise TypeError(f"{name} must be a tuple")
    normalized = []
    for item in value:
        if type(item) is not tuple or len(item) != 2:
            raise TypeError(f"{name} entries must be pairs")
        slot, candidate_id = item
        _exact_string(slot, name=f"{name} slot")
        if candidate_id is not None:
            _exact_string(candidate_id, name=f"{name} candidate")
        normalized.append((slot, candidate_id))
    result = tuple(sorted(normalized))
    if result != value or len({slot for slot, _candidate in result}) != len(result):
        raise ValueError(f"{name} must be sorted with unique slots")
    if {slot for slot, _candidate in result} != expected_slots:
        raise ValueError(f"{name} must contain exactly the current slots")
    return result


def _inversions(
    assignment: AssignmentV3,
    predictions: dict[str, CalibrationPredictionV3],
) -> tuple[tuple[str, str], ...]:
    by_slot = dict(assignment)
    keys = sorted({slot.split(":", 1)[0] for slot in by_slot})
    inversions = []
    for key in keys:
        ordered_slots = tuple(
            f"{key}:{difficulty}" for difficulty in DIFFICULTIES if f"{key}:{difficulty}" in by_slot
        )
        for easier_slot, harder_slot in pairwise(ordered_slots):
            easier_id = by_slot[easier_slot]
            harder_id = by_slot[harder_slot]
            if easier_id is None or harder_id is None:
                continue
            easier_score = predictions[easier_id].score
            harder_score = predictions[harder_id].score
            if easier_score is None or harder_score is None:
                raise AssertionError("UNKNOWN predictions must be rejected before scoring")
            if harder_score <= easier_score:
                inversions.append((easier_slot, harder_slot))
    return tuple(inversions)


def _candidate_slots(candidate: CandidateFamilyEvidenceV3) -> tuple[str, ...]:
    return tuple(
        f"{candidate.key_mode}K:{difficulty}"
        for difficulty in candidate.target_difficulties
    )


def _proposal_safety_blockers(
    evidence: SongSelectionEvidenceV3,
    proposed: AssignmentV3,
) -> tuple[str, ...]:
    """Return quality/safety regressions independently of difficulty scoring."""
    candidates = {candidate.candidate_id: candidate for candidate in evidence.candidates}
    current_by_slot = dict(evidence.current_assignment)
    blockers: list[str] = []
    assigned_ids = [
        candidate_id
        for _slot, candidate_id in proposed
        if candidate_id is not None
    ]
    if len(set(assigned_ids)) != len(assigned_ids):
        blockers.append("DUPLICATE_CANDIDATE_ASSIGNMENT")
    for slot, challenger_id in proposed:
        current_id = current_by_slot[slot]
        if challenger_id == current_id:
            continue
        if current_id is None or challenger_id is None:
            blockers.append(f"SLOT_COMPLETENESS_CHANGED:{slot}")
            continue
        base = candidates[current_id]
        challenger = candidates[challenger_id]
        if not challenger.safety.hard_safe:
            blockers.append(f"HARD_SAFETY_REGRESSION:{slot}")
        gap_comparison = compare_gap_evidence(base.safety, challenger.safety)
        if gap_comparison.status != "NON_REGRESSION":
            blockers.append(f"ACTIVE_GAP_REGRESSION:{slot}")
        if (
            _PUBLICATION_RANK[challenger.safety.publication_tier]
            > _PUBLICATION_RANK[base.safety.publication_tier]
        ):
            blockers.append(f"PUBLICATION_TIER_REGRESSION:{slot}")
        if base.first_row_audio_supported and not challenger.first_row_audio_supported:
            blockers.append(f"INTRO_EVIDENCE_REGRESSION:{slot}")
        reference_ms = evidence.intro_selection.reference_first_row_ms
        if (
            evidence.intro_selection.reference_state == "CONFIRMED_AUDIO"
            and reference_ms is not None
            and base.first_row_ms == reference_ms
            and challenger.first_row_ms != reference_ms
        ):
            blockers.append(f"INTRO_EVIDENCE_REGRESSION:{slot}")
        if base.matched_f1_50 is not None and (
            challenger.matched_f1_50 is None
            or challenger.matched_f1_50 < base.matched_f1_50 - _MATCHED_F1_EPSILON
        ):
            blockers.append(f"TIMING_F1_REGRESSION:{slot}")
        if challenger.review_rank > base.review_rank:
            blockers.append(f"QUALITY_REVIEW_REGRESSION:{slot}")
    return tuple(dict.fromkeys(blockers))


def evaluate_shadow_v3_proposal(
    evidence: SongSelectionEvidenceV3,
    *,
    proposal: AssignmentV3,
    predictions: tuple[CalibrationPredictionV3, ...] = (),
) -> ShadowV3ProposalEvaluation:
    """Evaluate a proposal while always returning the current production assignment."""
    if not isinstance(evidence, SongSelectionEvidenceV3):
        raise TypeError("evidence must be SongSelectionEvidenceV3")
    current = evidence.current_assignment
    expected_slots = {slot for slot, _candidate_id in current}
    proposed = _normalize_assignment(
        proposal,
        expected_slots=expected_slots,
        name="proposal",
    )
    candidates = {candidate.candidate_id: candidate for candidate in evidence.candidates}
    current_by_slot = dict(current)
    for slot, candidate_id in proposed:
        if candidate_id is None:
            continue
        candidate = candidates.get(candidate_id)
        if candidate is None:
            raise ValueError("proposal references an unknown candidate")
        if (
            slot not in _candidate_slots(candidate)
            and current_by_slot.get(slot) != candidate_id
        ):
            raise ValueError(f"candidate {candidate_id} does not belong to slot {slot}")

    if not predictions:
        return ShadowV3ProposalEvaluation(
            selected_assignment=current,
            proposed_assignment=proposed,
            shadow_assignment=current,
            proposal_eligible=False,
            blockers=("CALIBRATION_UNAVAILABLE",),
            current_inversions=(),
            proposed_inversions=(),
            resolved_inversions=(),
            created_inversions=(),
            calibration_sha256=None,
        )

    by_prediction = {prediction.candidate_id: prediction for prediction in predictions}
    if len(by_prediction) != len(predictions):
        raise ValueError("calibration predictions must have unique candidate identities")
    calibration_hashes = {prediction.calibration_sha256 for prediction in predictions}
    if len(calibration_hashes) != 1:
        raise ValueError("calibration predictions must use one activation hash")
    calibration_sha256 = next(iter(calibration_hashes))
    assigned_ids = {
        candidate_id
        for assignment in (current, proposed)
        for _slot, candidate_id in assignment
        if candidate_id is not None
    }
    for candidate_id in sorted(assigned_ids):
        prediction = by_prediction.get(candidate_id)
        if prediction is None or prediction.state == "UNKNOWN":
            return ShadowV3ProposalEvaluation(
                selected_assignment=current,
                proposed_assignment=proposed,
                shadow_assignment=current,
                proposal_eligible=False,
                blockers=(f"CALIBRATION_UNKNOWN:{candidate_id}",),
                current_inversions=(),
                proposed_inversions=(),
                resolved_inversions=(),
                created_inversions=(),
                calibration_sha256=calibration_sha256,
            )

    blockers = list(_proposal_safety_blockers(evidence, proposed))

    current_inversions = _inversions(current, by_prediction)
    proposed_inversions = _inversions(proposed, by_prediction)
    resolved = tuple(pair for pair in current_inversions if pair not in proposed_inversions)
    created = tuple(pair for pair in proposed_inversions if pair not in current_inversions)
    if created:
        blockers.append("NEW_DIFFICULTY_INVERSION")
    if proposed != current and (
        not resolved or len(proposed_inversions) >= len(current_inversions)
    ):
        blockers.append("NO_CALIBRATED_DIFFICULTY_IMPROVEMENT")

    unique_blockers = tuple(dict.fromkeys(blockers))
    eligible = not unique_blockers
    return ShadowV3ProposalEvaluation(
        selected_assignment=current,
        proposed_assignment=proposed,
        shadow_assignment=proposed if eligible else current,
        proposal_eligible=eligible,
        blockers=unique_blockers,
        current_inversions=current_inversions,
        proposed_inversions=proposed_inversions,
        resolved_inversions=resolved,
        created_inversions=created,
        calibration_sha256=calibration_sha256,
    )


def propose_shadow_v3_assignment(
    evidence: SongSelectionEvidenceV3,
    *,
    predictions: tuple[CalibrationPredictionV3, ...] = (),
) -> ShadowV3ProposalEvaluation:
    """Search existing candidates per key; never changes the production assignment."""
    if not isinstance(evidence, SongSelectionEvidenceV3):
        raise TypeError("evidence must be SongSelectionEvidenceV3")
    current = evidence.current_assignment
    baseline = evaluate_shadow_v3_proposal(
        evidence,
        proposal=current,
        predictions=predictions,
    )
    if not predictions or any(
        blocker.startswith("CALIBRATION_UNKNOWN:") for blocker in baseline.blockers
    ):
        return baseline
    if not baseline.current_inversions:
        return replace(
            baseline,
            proposal_eligible=False,
            blockers=("NO_CALIBRATED_DIFFICULTY_IMPROVEMENT",),
        )

    by_prediction = {prediction.candidate_id: prediction for prediction in predictions}
    if len(by_prediction) != len(predictions):
        raise ValueError("calibration predictions must have unique candidate identities")
    by_slot: dict[str, list[str]] = {}
    for candidate in evidence.candidates:
        prediction = by_prediction.get(candidate.candidate_id)
        if prediction is None or prediction.state != "IN_DOMAIN":
            continue
        for slot in _candidate_slots(candidate):
            by_slot.setdefault(slot, []).append(candidate.candidate_id)
    current_by_slot = dict(current)
    best: ShadowV3ProposalEvaluation | None = None
    best_rank: tuple[object, ...] | None = None
    for key in sorted({slot.split(":", 1)[0] for slot in current_by_slot}):
        slots = tuple(
            f"{key}:{difficulty}"
            for difficulty in DIFFICULTIES
            if f"{key}:{difficulty}" in current_by_slot
        )
        option_groups = []
        for slot in slots:
            options = set(by_slot.get(slot, ()))
            current_id = current_by_slot[slot]
            if current_id is not None:
                options.add(current_id)
            option_groups.append(tuple(sorted(options)))
        combination_count = math.prod(len(options) for options in option_groups)
        if combination_count > _MAX_FAMILY_COMBINATIONS:
            return replace(
                baseline,
                proposal_eligible=False,
                blockers=("FAMILY_SEARCH_LIMIT_EXCEEDED",),
            )
        for chosen_ids in product(*option_groups):
            proposed_map = dict(current_by_slot)
            proposed_map.update(zip(slots, chosen_ids))
            proposal = tuple(sorted(proposed_map.items()))
            if proposal == current:
                continue
            evaluation = evaluate_shadow_v3_proposal(
                evidence,
                proposal=proposal,
                predictions=predictions,
            )
            if not evaluation.proposal_eligible:
                continue
            changes = sum(
                candidate_id != current_by_slot[slot]
                for slot, candidate_id in evaluation.shadow_assignment
            )
            rank = (
                len(evaluation.proposed_inversions),
                -len(evaluation.resolved_inversions),
                changes,
                evaluation.shadow_assignment,
            )
            if best_rank is None or rank < best_rank:
                best = evaluation
                best_rank = rank
    if best is not None:
        return best
    return replace(
        baseline,
        proposal_eligible=False,
        blockers=("NO_SAFE_CALIBRATED_PROPOSAL",),
    )


def _validate_difficulty_evidence_bindings(
    evidence: SongSelectionEvidenceV3,
    difficulty_evidence: tuple[CandidateDifficultyEvidenceV1, ...],
) -> str:
    corpus_sha256 = difficulty_evidence_corpus_sha256(difficulty_evidence)
    candidates = {candidate.candidate_id: candidate for candidate in evidence.candidates}
    for metric in difficulty_evidence:
        candidate = candidates.get(metric.candidate_id)
        if candidate is None:
            raise ValueError("difficulty evidence references an unknown candidate")
        if candidate.candidate_payload_sha256 != metric.candidate_payload_sha256:
            raise ValueError(
                f"difficulty evidence payload differs for candidate {metric.candidate_id}"
            )
    return corpus_sha256


def _hazard_pairs(
    hazards: tuple[AdjacentDifficultyHazardV1, ...],
    *,
    status: str,
) -> tuple[tuple[str, str], ...]:
    return tuple(hazard.pair for hazard in hazards if hazard.status == status)


def evaluate_shadow_v3_hazard_proposal(
    evidence: SongSelectionEvidenceV3,
    *,
    proposal: AssignmentV3,
    difficulty_evidence: tuple[CandidateDifficultyEvidenceV1, ...],
) -> ShadowHazardProposalEvaluationV1:
    """Evaluate official/project difficulty jointly without mutating production."""
    if not isinstance(evidence, SongSelectionEvidenceV3):
        raise TypeError("evidence must be SongSelectionEvidenceV3")
    current = evidence.current_assignment
    expected_slots = {slot for slot, _candidate_id in current}
    proposed = _normalize_assignment(
        proposal,
        expected_slots=expected_slots,
        name="proposal",
    )
    candidates = {candidate.candidate_id: candidate for candidate in evidence.candidates}
    current_by_slot = dict(current)
    for slot, candidate_id in proposed:
        if candidate_id is None:
            continue
        candidate = candidates.get(candidate_id)
        if candidate is None:
            raise ValueError("proposal references an unknown candidate")
        if (
            slot not in _candidate_slots(candidate)
            and current_by_slot.get(slot) != candidate_id
        ):
            raise ValueError(f"candidate {candidate_id} does not belong to slot {slot}")

    corpus_sha256 = _validate_difficulty_evidence_bindings(evidence, difficulty_evidence)
    current_hazards = assess_assignment_difficulty_hazards(
        current,
        evidence=difficulty_evidence,
    )
    proposed_hazards = assess_assignment_difficulty_hazards(
        proposed,
        evidence=difficulty_evidence,
    )
    current_at_risk = _hazard_pairs(current_hazards, status="AT_RISK")
    proposed_at_risk = _hazard_pairs(proposed_hazards, status="AT_RISK")
    current_unknown = _hazard_pairs(current_hazards, status="UNKNOWN")
    proposed_unknown = _hazard_pairs(proposed_hazards, status="UNKNOWN")
    resolved = tuple(pair for pair in current_at_risk if pair not in proposed_at_risk)
    created = tuple(pair for pair in proposed_at_risk if pair not in current_at_risk)

    blockers = list(_proposal_safety_blockers(evidence, proposed))
    for pair in (*current_unknown, *proposed_unknown):
        blockers.append(f"DIFFICULTY_EVIDENCE_UNKNOWN:{pair[0]}->{pair[1]}")
    if created:
        blockers.append("NEW_TWO_AXIS_DIFFICULTY_RISK")
    if proposed == current or not resolved or len(proposed_at_risk) >= len(current_at_risk):
        blockers.append("NO_TWO_AXIS_DIFFICULTY_IMPROVEMENT")
    unique_blockers = tuple(dict.fromkeys(blockers))
    eligible = not unique_blockers
    return ShadowHazardProposalEvaluationV1(
        selected_assignment=current,
        proposed_assignment=proposed,
        shadow_assignment=proposed if eligible else current,
        proposal_eligible=eligible,
        blockers=unique_blockers,
        current_at_risk=current_at_risk,
        proposed_at_risk=proposed_at_risk,
        current_unknown=current_unknown,
        proposed_unknown=proposed_unknown,
        resolved_at_risk=resolved,
        created_at_risk=created,
        difficulty_evidence_sha256=corpus_sha256,
    )


def propose_shadow_v3_hazard_assignment(
    evidence: SongSelectionEvidenceV3,
    *,
    difficulty_evidence: tuple[CandidateDifficultyEvidenceV1, ...],
) -> ShadowHazardProposalEvaluationV1:
    """Search existing candidates with two-axis screening and safety vetoes."""
    if not isinstance(evidence, SongSelectionEvidenceV3):
        raise TypeError("evidence must be SongSelectionEvidenceV3")
    current = evidence.current_assignment
    baseline = evaluate_shadow_v3_hazard_proposal(
        evidence,
        proposal=current,
        difficulty_evidence=difficulty_evidence,
    )
    if baseline.current_unknown:
        return baseline
    if not baseline.current_at_risk:
        return replace(
            baseline,
            proposal_eligible=False,
            blockers=("NO_TWO_AXIS_DIFFICULTY_IMPROVEMENT",),
        )

    metric_ids = {metric.candidate_id for metric in difficulty_evidence}
    by_slot: dict[str, list[str]] = {}
    for candidate in evidence.candidates:
        if candidate.candidate_id not in metric_ids:
            continue
        for slot in _candidate_slots(candidate):
            by_slot.setdefault(slot, []).append(candidate.candidate_id)
    current_by_slot = dict(current)
    best: ShadowHazardProposalEvaluationV1 | None = None
    best_rank: tuple[object, ...] | None = None
    for key in sorted({slot.split(":", 1)[0] for slot in current_by_slot}):
        slots = tuple(
            f"{key}:{difficulty}"
            for difficulty in DIFFICULTIES
            if f"{key}:{difficulty}" in current_by_slot
        )
        option_groups = []
        for slot in slots:
            options = set(by_slot.get(slot, ()))
            current_id = current_by_slot[slot]
            if current_id is not None:
                options.add(current_id)
            option_groups.append(tuple(sorted(options)))
        combination_count = math.prod(len(options) for options in option_groups)
        if combination_count > _MAX_FAMILY_COMBINATIONS:
            return replace(
                baseline,
                proposal_eligible=False,
                blockers=("FAMILY_SEARCH_LIMIT_EXCEEDED",),
            )
        for chosen_ids in product(*option_groups):
            proposed_map = dict(current_by_slot)
            proposed_map.update(zip(slots, chosen_ids))
            proposal = tuple(sorted(proposed_map.items()))
            if proposal == current:
                continue
            evaluation = evaluate_shadow_v3_hazard_proposal(
                evidence,
                proposal=proposal,
                difficulty_evidence=difficulty_evidence,
            )
            if not evaluation.proposal_eligible:
                continue
            changes = sum(
                candidate_id != current_by_slot[slot]
                for slot, candidate_id in evaluation.shadow_assignment
            )
            rank = (
                len(evaluation.proposed_at_risk),
                -len(evaluation.resolved_at_risk),
                changes,
                evaluation.shadow_assignment,
            )
            if best_rank is None or rank < best_rank:
                best = evaluation
                best_rank = rank
    if best is not None:
        return best
    return replace(
        baseline,
        proposal_eligible=False,
        blockers=("NO_SAFE_TWO_AXIS_PROPOSAL",),
    )
