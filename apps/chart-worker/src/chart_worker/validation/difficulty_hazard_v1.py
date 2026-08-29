"""Transparent two-axis difficulty screening for report-only family selection."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from itertools import pairwise
from typing import Literal

from chart_worker.schema.types import DIFFICULTIES
from chart_worker.validation.mania_star_evidence import ManiaStarEvidenceV1

DifficultyHazardStatusV1 = Literal["NO_OBSERVED_RISK", "AT_RISK", "UNKNOWN"]
AssignmentV1 = tuple[tuple[str, str | None], ...]


def _exact_string(value: object, *, name: str) -> str:
    if type(value) is not str or not value:
        raise TypeError(f"{name} must be a non-empty exact string")
    return value


def _sha256(value: object, *, name: str) -> str:
    digest = _exact_string(value, name=name)
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return digest


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True, slots=True)
class CandidateDifficultyEvidenceV1:
    candidate_id: str
    candidate_payload_sha256: str
    official_star: ManiaStarEvidenceV1
    project_rating: float
    project_rating_evidence_sha256: str

    def __post_init__(self) -> None:
        _exact_string(self.candidate_id, name="candidate_id")
        payload = _sha256(
            self.candidate_payload_sha256,
            name="candidate_payload_sha256",
        )
        if not isinstance(self.official_star, ManiaStarEvidenceV1):
            raise TypeError("official_star must be ManiaStarEvidenceV1")
        if not self.official_star.authorizes_calibration_feature:
            raise ValueError("official star must come from verified pinned tool execution")
        if self.official_star.input_osu_sha256 != payload:
            raise ValueError("official star input differs from candidate payload")
        if type(self.project_rating) is not float or not math.isfinite(self.project_rating):
            raise ValueError("project_rating must be a finite exact float")
        _sha256(
            self.project_rating_evidence_sha256,
            name="project_rating_evidence_sha256",
        )

    def to_report(self) -> dict[str, object]:
        return {
            "version": "candidate-difficulty-evidence-v1",
            "candidateId": self.candidate_id,
            "candidatePayloadSha256": self.candidate_payload_sha256,
            "officialStar": self.official_star.to_report(),
            "projectRating": self.project_rating,
            "projectRatingEvidenceSha256": self.project_rating_evidence_sha256,
        }

    def stable_sha256(self) -> str:
        return _canonical_sha256(self.to_report())


@dataclass(frozen=True, slots=True)
class AdjacentDifficultyHazardV1:
    easier_slot: str
    harder_slot: str
    status: DifficultyHazardStatusV1
    official_delta: float | None
    project_delta: float | None
    reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        _exact_string(self.easier_slot, name="easier_slot")
        _exact_string(self.harder_slot, name="harder_slot")
        if self.status not in {"NO_OBSERVED_RISK", "AT_RISK", "UNKNOWN"}:
            raise ValueError("difficulty hazard status is unsupported")
        for value, name in (
            (self.official_delta, "official_delta"),
            (self.project_delta, "project_delta"),
        ):
            if value is not None and (type(value) is not float or not math.isfinite(value)):
                raise ValueError(f"{name} must be a finite exact float or None")
        if type(self.reasons) is not tuple or any(
            type(reason) is not str or not reason for reason in self.reasons
        ):
            raise TypeError("reasons must be a tuple of non-empty exact strings")
        if self.status == "UNKNOWN" and (
            self.official_delta is not None or self.project_delta is not None
        ):
            raise ValueError("UNKNOWN hazard cannot carry partial deltas")
        if self.status == "NO_OBSERVED_RISK" and self.reasons:
            raise ValueError("no-risk hazard cannot carry reasons")

    @property
    def pair(self) -> tuple[str, str]:
        return (self.easier_slot, self.harder_slot)

    def to_report(self) -> dict[str, object]:
        return {
            "version": "adjacent-difficulty-hazard-v1",
            "easierSlot": self.easier_slot,
            "harderSlot": self.harder_slot,
            "status": self.status,
            "officialDelta": self.official_delta,
            "projectDelta": self.project_delta,
            "reasons": list(self.reasons),
        }


def difficulty_evidence_corpus_sha256(
    evidence: tuple[CandidateDifficultyEvidenceV1, ...],
) -> str:
    if type(evidence) is not tuple or any(
        not isinstance(item, CandidateDifficultyEvidenceV1) for item in evidence
    ):
        raise TypeError("evidence must be a tuple of CandidateDifficultyEvidenceV1")
    candidate_ids = tuple(item.candidate_id for item in evidence)
    if len(set(candidate_ids)) != len(candidate_ids):
        raise ValueError("difficulty evidence candidate identities must be unique")
    reports = [item.to_report() for item in sorted(evidence, key=lambda item: item.candidate_id)]
    return _canonical_sha256(reports)


def _normalize_assignment(assignment: AssignmentV1) -> AssignmentV1:
    if type(assignment) is not tuple:
        raise TypeError("assignment must be a tuple")
    normalized = []
    for item in assignment:
        if type(item) is not tuple or len(item) != 2:
            raise TypeError("assignment entries must be exact pairs")
        slot, candidate_id = item
        _exact_string(slot, name="assignment slot")
        if candidate_id is not None:
            _exact_string(candidate_id, name="assignment candidate")
        normalized.append((slot, candidate_id))
    result = tuple(sorted(normalized))
    if result != assignment or len({slot for slot, _candidate in result}) != len(result):
        raise ValueError("assignment must be sorted with unique slots")
    return result


def assess_assignment_difficulty_hazards(
    assignment: AssignmentV1,
    *,
    evidence: tuple[CandidateDifficultyEvidenceV1, ...],
) -> tuple[AdjacentDifficultyHazardV1, ...]:
    """Screen adjacent slots; missing evidence is explicit UNKNOWN."""
    normalized = _normalize_assignment(assignment)
    difficulty_evidence_corpus_sha256(evidence)
    by_candidate = {item.candidate_id: item for item in evidence}
    by_slot = dict(normalized)
    keys = sorted({slot.split(":", 1)[0] for slot in by_slot})
    hazards = []
    for key in keys:
        ordered_slots = tuple(f"{key}:{difficulty}" for difficulty in DIFFICULTIES)
        for easier_slot, harder_slot in pairwise(ordered_slots):
            if easier_slot not in by_slot or harder_slot not in by_slot:
                continue
            easier_id = by_slot[easier_slot]
            harder_id = by_slot[harder_slot]
            missing = next(
                (
                    candidate_id
                    for candidate_id in (easier_id, harder_id)
                    if candidate_id is None or candidate_id not in by_candidate
                ),
                None,
            )
            if missing is not None or easier_id is None or harder_id is None:
                label = "NONE" if missing is None else missing
                hazards.append(
                    AdjacentDifficultyHazardV1(
                        easier_slot=easier_slot,
                        harder_slot=harder_slot,
                        status="UNKNOWN",
                        official_delta=None,
                        project_delta=None,
                        reasons=(f"MISSING_CANDIDATE_EVIDENCE:{label}",),
                    )
                )
                continue
            easier = by_candidate[easier_id]
            harder = by_candidate[harder_id]
            official_delta = float(
                harder.official_star.star_rating - easier.official_star.star_rating
            )
            project_delta = float(harder.project_rating - easier.project_rating)
            reasons = []
            if official_delta <= 0.0:
                reasons.append("OFFICIAL_NOT_STRICTLY_INCREASING")
            if project_delta <= 0.0:
                reasons.append("PROJECT_NOT_STRICTLY_INCREASING")
            hazards.append(
                AdjacentDifficultyHazardV1(
                    easier_slot=easier_slot,
                    harder_slot=harder_slot,
                    status="AT_RISK" if reasons else "NO_OBSERVED_RISK",
                    official_delta=official_delta,
                    project_delta=project_delta,
                    reasons=tuple(reasons),
                )
            )
    return tuple(hazards)
