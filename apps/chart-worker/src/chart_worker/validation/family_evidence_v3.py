"""Immutable evidence used by the report-only V3 family selector.

This module deliberately contains no candidate-selection or chart-mutation authority.
It normalizes the evidence that later selectors are allowed to compare.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Literal

from chart_worker.analysis.intro_anchor import IntroAnchorEvidence
from chart_worker.schema.types import DIFFICULTIES, KEY_MODES

GAP_INTERVAL_EVIDENCE_VERSION = "gap-interval-evidence-v1"
CANDIDATE_SAFETY_EVIDENCE_VERSION = "candidate-safety-evidence-v3"
INTRO_SELECTION_EVIDENCE_VERSION = "intro-selection-evidence-v3"
SONG_SELECTION_EVIDENCE_VERSION = "song-selection-evidence-v3"

GapPosition = Literal["LEADING", "POST_FIRST", "MIDDLE", "TRAILING"]
OpportunityKind = Literal[
    "ATTACK_REQUIRED",
    "SUSTAIN_COVERED",
    "MUSICAL_REST_OR_SIMPLIFICATION",
    "UNCERTAIN",
    "SUSTAIN_REPRESENTABLE",
    "INSUFFICIENT_EVIDENCE",
    "UNKNOWN",
]
PublicationTier = Literal[
    "PRODUCTION_CANDIDATE",
    "PLAYTEST_ONLY",
    "DIAGNOSTIC_ONLY",
]
GapComparisonStatus = Literal["NON_REGRESSION", "REGRESSION", "INCOMPARABLE"]
IntroReferenceState = Literal[
    "CONFIRMED_AUDIO",
    "CROSS_SLOT_CONSENSUS",
    "UNCERTAIN",
]
CandidateRole = Literal["PLAYTEST_POOL", "SHADOW_CHALLENGER"]

_GAP_POSITIONS = {"LEADING", "POST_FIRST", "MIDDLE", "TRAILING"}
_OPPORTUNITY_KINDS = {
    "ATTACK_REQUIRED",
    "SUSTAIN_COVERED",
    "MUSICAL_REST_OR_SIMPLIFICATION",
    "UNCERTAIN",
    "SUSTAIN_REPRESENTABLE",
    "INSUFFICIENT_EVIDENCE",
    "UNKNOWN",
}
_PUBLICATION_TIERS = {
    "PRODUCTION_CANDIDATE",
    "PLAYTEST_ONLY",
    "DIAGNOSTIC_ONLY",
}
_OPPORTUNITY_SEVERITY = {
    "UNKNOWN": 0,
    "UNCERTAIN": 0,
    "INSUFFICIENT_EVIDENCE": 0,
    "MUSICAL_REST_OR_SIMPLIFICATION": 0,
    "SUSTAIN_COVERED": 1,
    "SUSTAIN_REPRESENTABLE": 1,
    "ATTACK_REQUIRED": 2,
}


def _exact_non_negative_int(value: object, *, name: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be an exact integer")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value


def _exact_string(value: object, *, name: str) -> str:
    if type(value) is not str or not value:
        raise TypeError(f"{name} must be a non-empty exact string")
    return value


def _exact_probability(value: object, *, name: str) -> float:
    if type(value) is not float:
        raise TypeError(f"{name} must be an exact float")
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be finite and between 0 and 1")
    return value


def _optional_sha256(value: object, *, name: str) -> str | None:
    if value is None:
        return None
    digest = _exact_string(value, name=name)
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return digest


def _required_sha256(value: object, *, name: str) -> str:
    digest = _optional_sha256(value, name=name)
    if digest is None:
        raise TypeError(f"{name} must be a lowercase SHA-256 digest")
    return digest


def _canonical_sha256(report: dict[str, object]) -> str:
    payload = _canonical_json(report).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _canonical_json(report: object) -> str:
    return json.dumps(
        report,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _exact_mapping(
    value: object,
    *,
    name: str,
    keys: frozenset[str],
) -> dict[str, object]:
    if type(value) is not dict or any(type(key) is not str for key in value):
        raise TypeError(f"{name} must be an exact string-keyed object")
    actual = frozenset(value)
    if actual != keys:
        missing = sorted(keys - actual)
        extra = sorted(actual - keys)
        raise ValueError(f"{name} keys differ: missing={missing}, extra={extra}")
    return value


def _exact_list(value: object, *, name: str) -> list[object]:
    if type(value) is not list:
        raise TypeError(f"{name} must be an exact list")
    return value


def _require_projection(
    value: dict[str, object], projected: dict[str, object], *, name: str
) -> None:
    if _canonical_json(value) != _canonical_json(projected):
        raise ValueError(f"{name} contains inconsistent derived or noncanonical values")


@dataclass(frozen=True, slots=True)
class GapIntervalEvidence:
    start_ms: int
    end_ms: int
    position: GapPosition
    active_onset_count: int
    active_frame_ratio: float
    opportunity_kind: OpportunityKind
    local_audio_evidence_digest: str | None
    version: Literal["gap-interval-evidence-v1"] = GAP_INTERVAL_EVIDENCE_VERSION

    def __post_init__(self) -> None:
        if self.version != GAP_INTERVAL_EVIDENCE_VERSION:
            raise ValueError("unsupported gap interval evidence version")
        start_ms = _exact_non_negative_int(self.start_ms, name="start_ms")
        end_ms = _exact_non_negative_int(self.end_ms, name="end_ms")
        if end_ms <= start_ms:
            raise ValueError("end_ms must follow start_ms")
        if type(self.position) is not str or self.position not in _GAP_POSITIONS:
            raise ValueError("position is unsupported")
        _exact_non_negative_int(
            self.active_onset_count,
            name="active_onset_count",
        )
        _exact_probability(self.active_frame_ratio, name="active_frame_ratio")
        if (
            type(self.opportunity_kind) is not str
            or self.opportunity_kind not in _OPPORTUNITY_KINDS
        ):
            raise ValueError("opportunity_kind is unsupported")
        _optional_sha256(
            self.local_audio_evidence_digest,
            name="local_audio_evidence_digest",
        )

    @property
    def duration_ms(self) -> int:
        return self.end_ms - self.start_ms

    def to_report(self) -> dict[str, object]:
        return {
            "version": self.version,
            "startMs": self.start_ms,
            "endMs": self.end_ms,
            "durationMs": self.duration_ms,
            "position": self.position,
            "activeOnsetCount": self.active_onset_count,
            "activeFrameRatio": self.active_frame_ratio,
            "opportunityKind": self.opportunity_kind,
            "localAudioEvidenceDigest": self.local_audio_evidence_digest,
        }


@dataclass(frozen=True, slots=True)
class CandidateSafetyEvidenceV3:
    candidate_id: str
    structure_safe: bool
    timing_identity_safe: bool
    song_bounds_safe: bool
    serialization_safe: bool
    publication_tier: PublicationTier
    model_backed: bool
    recovery_trust_rank: int
    active_gaps: tuple[GapIntervalEvidence, ...]
    version: Literal["candidate-safety-evidence-v3"] = CANDIDATE_SAFETY_EVIDENCE_VERSION

    def __post_init__(self) -> None:
        if self.version != CANDIDATE_SAFETY_EVIDENCE_VERSION:
            raise ValueError("unsupported candidate safety evidence version")
        _exact_string(self.candidate_id, name="candidate_id")
        for name, value in (
            ("structure_safe", self.structure_safe),
            ("timing_identity_safe", self.timing_identity_safe),
            ("song_bounds_safe", self.song_bounds_safe),
            ("serialization_safe", self.serialization_safe),
            ("model_backed", self.model_backed),
        ):
            if type(value) is not bool:
                raise TypeError(f"{name} must be an exact boolean")
        if (
            type(self.publication_tier) is not str
            or self.publication_tier not in _PUBLICATION_TIERS
        ):
            raise ValueError("publication_tier is unsupported")
        _exact_non_negative_int(
            self.recovery_trust_rank,
            name="recovery_trust_rank",
        )
        if type(self.active_gaps) is not tuple or any(
            not isinstance(item, GapIntervalEvidence) for item in self.active_gaps
        ):
            raise TypeError("active_gaps must be a tuple of GapIntervalEvidence")
        expected = tuple(
            sorted(
                self.active_gaps,
                key=lambda item: (item.start_ms, item.end_ms, item.position),
            )
        )
        if expected != self.active_gaps or any(
            left.end_ms > right.start_ms
            for left, right in zip(self.active_gaps, self.active_gaps[1:])
        ):
            raise ValueError("active_gaps must be sorted and disjoint")

    @property
    def hard_safe(self) -> bool:
        return (
            self.structure_safe
            and self.timing_identity_safe
            and self.song_bounds_safe
            and self.serialization_safe
        )

    @property
    def total_active_gap_ms(self) -> int:
        return sum(item.duration_ms for item in self.active_gaps)

    @property
    def max_active_gap_ms(self) -> int:
        return max((item.duration_ms for item in self.active_gaps), default=0)

    def to_report(self) -> dict[str, object]:
        return {
            "version": self.version,
            "candidateId": self.candidate_id,
            "structureSafe": self.structure_safe,
            "timingIdentitySafe": self.timing_identity_safe,
            "songBoundsSafe": self.song_bounds_safe,
            "serializationSafe": self.serialization_safe,
            "hardSafe": self.hard_safe,
            "publicationTier": self.publication_tier,
            "modelBacked": self.model_backed,
            "recoveryTrustRank": self.recovery_trust_rank,
            "activeGapCount": len(self.active_gaps),
            "totalActiveGapMs": self.total_active_gap_ms,
            "maxActiveGapMs": self.max_active_gap_ms,
            "activeGaps": [item.to_report() for item in self.active_gaps],
        }

    def stable_sha256(self) -> str:
        return _canonical_sha256(self.to_report())


@dataclass(frozen=True, slots=True)
class GapEvidenceComparison:
    status: GapComparisonStatus
    reasons: tuple[str, ...]

    def to_report(self) -> dict[str, object]:
        return {"status": self.status, "reasons": list(self.reasons)}


def _count_position(
    candidate: CandidateSafetyEvidenceV3,
    position: GapPosition,
) -> int:
    return sum(item.position == position for item in candidate.active_gaps)


def compare_gap_evidence(
    current: CandidateSafetyEvidenceV3,
    challenger: CandidateSafetyEvidenceV3,
) -> GapEvidenceComparison:
    """Compare active gaps conservatively; shifted gaps are not assumed equivalent."""
    if not isinstance(current, CandidateSafetyEvidenceV3) or not isinstance(
        challenger,
        CandidateSafetyEvidenceV3,
    ):
        raise TypeError("gap comparison requires CandidateSafetyEvidenceV3 values")

    reasons: list[str] = []
    if challenger.total_active_gap_ms > current.total_active_gap_ms:
        reasons.append("TOTAL_DURATION_INCREASED")
    if challenger.max_active_gap_ms > current.max_active_gap_ms:
        reasons.append("MAX_DURATION_INCREASED")
    if _count_position(challenger, "LEADING") > _count_position(current, "LEADING"):
        reasons.append("LEADING_GAP_COUNT_INCREASED")
    if _count_position(challenger, "POST_FIRST") > _count_position(
        current,
        "POST_FIRST",
    ):
        reasons.append("POST_FIRST_GAP_COUNT_INCREASED")

    unmatched = False
    for candidate_gap in challenger.active_gaps:
        containing = tuple(
            base_gap
            for base_gap in current.active_gaps
            if base_gap.position == candidate_gap.position
            and base_gap.start_ms <= candidate_gap.start_ms
            and candidate_gap.end_ms <= base_gap.end_ms
        )
        if not containing:
            unmatched = True
            continue
        base_gap = min(containing, key=lambda item: item.duration_ms)
        if (
            _OPPORTUNITY_SEVERITY[candidate_gap.opportunity_kind]
            > _OPPORTUNITY_SEVERITY[base_gap.opportunity_kind]
        ):
            reasons.append("OPPORTUNITY_SEVERITY_INCREASED")
        if candidate_gap.active_onset_count > base_gap.active_onset_count:
            reasons.append("ACTIVE_ONSET_COUNT_INCREASED")
        if candidate_gap.active_frame_ratio > base_gap.active_frame_ratio:
            reasons.append("ACTIVE_FRAME_RATIO_INCREASED")

    if reasons:
        return GapEvidenceComparison("REGRESSION", tuple(dict.fromkeys(reasons)))
    if unmatched:
        return GapEvidenceComparison("INCOMPARABLE", ("UNMATCHED_GAP_INTERVAL",))
    return GapEvidenceComparison("NON_REGRESSION", ())


@dataclass(frozen=True, slots=True)
class IntroCandidateVoteV3:
    slot: str
    candidate_id: str
    first_row_ms: int | None

    def __post_init__(self) -> None:
        _exact_string(self.slot, name="slot")
        _exact_string(self.candidate_id, name="candidate_id")
        if self.first_row_ms is not None:
            _exact_non_negative_int(self.first_row_ms, name="first_row_ms")


@dataclass(frozen=True, slots=True)
class IntroSelectionEvidenceV3:
    reference_state: IntroReferenceState
    reference_first_row_ms: int | None
    audio_anchor_ms: int | None
    audio_anchor_grid_ms: int | None
    audio_evidence_digest: str
    consensus_support_count: int
    contributing_slots: tuple[str, ...]
    abstaining_slots: tuple[str, ...]
    authorizes_first_row_change: bool
    version: Literal["intro-selection-evidence-v3"] = INTRO_SELECTION_EVIDENCE_VERSION

    def __post_init__(self) -> None:
        if self.version != INTRO_SELECTION_EVIDENCE_VERSION:
            raise ValueError("unsupported intro selection evidence version")
        if self.reference_state not in {
            "CONFIRMED_AUDIO",
            "CROSS_SLOT_CONSENSUS",
            "UNCERTAIN",
        }:
            raise ValueError("reference_state is unsupported")
        for name, value in (
            ("reference_first_row_ms", self.reference_first_row_ms),
            ("audio_anchor_ms", self.audio_anchor_ms),
            ("audio_anchor_grid_ms", self.audio_anchor_grid_ms),
        ):
            if value is not None:
                _exact_non_negative_int(value, name=name)
        _required_sha256(self.audio_evidence_digest, name="audio_evidence_digest")
        _exact_non_negative_int(
            self.consensus_support_count,
            name="consensus_support_count",
        )
        for name, values in (
            ("contributing_slots", self.contributing_slots),
            ("abstaining_slots", self.abstaining_slots),
        ):
            if type(values) is not tuple or any(
                type(value) is not str or not value for value in values
            ):
                raise TypeError(f"{name} must be a tuple of non-empty exact strings")
            if values != tuple(sorted(set(values))):
                raise ValueError(f"{name} must be sorted and unique")
        if type(self.authorizes_first_row_change) is not bool:
            raise TypeError("authorizes_first_row_change must be an exact boolean")
        if self.reference_state == "CONFIRMED_AUDIO":
            if self.reference_first_row_ms is None or self.audio_anchor_grid_ms is None:
                raise ValueError("confirmed audio evidence requires an anchor grid")
            if not self.authorizes_first_row_change:
                raise ValueError("confirmed audio evidence must authorize its reference")
        elif self.authorizes_first_row_change:
            raise ValueError("non-audio evidence cannot authorize a first-row change")
        if self.reference_state == "CROSS_SLOT_CONSENSUS" and (
            self.reference_first_row_ms is None or self.consensus_support_count < 2
        ):
            raise ValueError("cross-slot consensus requires at least two slots")
        if self.reference_state == "UNCERTAIN" and self.reference_first_row_ms is not None:
            raise ValueError("uncertain intro evidence cannot claim a reference")

    def to_report(self) -> dict[str, object]:
        return {
            "version": self.version,
            "referenceState": self.reference_state,
            "referenceFirstRowMs": self.reference_first_row_ms,
            "audioAnchorMs": self.audio_anchor_ms,
            "audioAnchorGridMs": self.audio_anchor_grid_ms,
            "audioEvidenceDigest": self.audio_evidence_digest,
            "consensusSupportCount": self.consensus_support_count,
            "contributingSlots": list(self.contributing_slots),
            "abstainingSlots": list(self.abstaining_slots),
            "authorizesFirstRowChange": self.authorizes_first_row_change,
        }

    def stable_sha256(self) -> str:
        return _canonical_sha256(self.to_report())


def _validate_intro_anchor(anchor: IntroAnchorEvidence) -> None:
    if not isinstance(anchor, IntroAnchorEvidence):
        raise TypeError("anchor must be IntroAnchorEvidence")
    if anchor.status not in {"CONFIRMED", "UNCERTAIN", "NON_RHYTHMIC"}:
        raise ValueError("intro anchor status is unsupported")
    for name, value in (
        ("anchor_ms", anchor.anchor_ms),
        ("anchor_grid_ms", anchor.anchor_grid_ms),
        ("grid_distance_ms", anchor.grid_distance_ms),
        ("prominent_band_count", anchor.prominent_band_count),
        ("pulse_continuation_matches", anchor.pulse_continuation_matches),
        ("pulse_continuation_opportunities", anchor.pulse_continuation_opportunities),
    ):
        if value is not None:
            _exact_non_negative_int(value, name=name)
    rank = anchor.aggregate_percentile_rank
    if rank is not None and (
        type(rank) is not float or not math.isfinite(rank) or not 0.0 <= rank <= 1.0
    ):
        raise ValueError("aggregate_percentile_rank must be a finite float in [0, 1]")
    if anchor.status == "CONFIRMED":
        if anchor.anchor_ms is None or anchor.anchor_grid_ms is None:
            raise ValueError("confirmed intro anchor requires timestamps")
        if anchor.grid_distance_ms != abs(anchor.anchor_ms - anchor.anchor_grid_ms):
            raise ValueError("intro anchor grid distance is inconsistent")


def build_intro_selection_evidence(
    anchor: IntroAnchorEvidence,
    *,
    active_onset_ms: tuple[int, ...],
    votes: tuple[IntroCandidateVoteV3, ...],
) -> IntroSelectionEvidenceV3:
    """Build one immutable reference without giving model consensus mutation authority."""
    _validate_intro_anchor(anchor)
    if type(active_onset_ms) is not tuple or active_onset_ms != tuple(sorted(set(active_onset_ms))):
        raise ValueError("active_onset_ms must be a sorted unique tuple")
    for value in active_onset_ms:
        _exact_non_negative_int(value, name="active_onset_ms item")
    if type(votes) is not tuple or any(
        not isinstance(item, IntroCandidateVoteV3) for item in votes
    ):
        raise TypeError("votes must be a tuple of IntroCandidateVoteV3")

    audio_report: dict[str, object] = {
        "introAnchor": anchor.to_report(),
        "activeOnsetMs": list(active_onset_ms),
    }
    audio_digest = _canonical_sha256(audio_report)
    if anchor.status == "CONFIRMED":
        return IntroSelectionEvidenceV3(
            reference_state="CONFIRMED_AUDIO",
            reference_first_row_ms=anchor.anchor_grid_ms,
            audio_anchor_ms=anchor.anchor_ms,
            audio_anchor_grid_ms=anchor.anchor_grid_ms,
            audio_evidence_digest=audio_digest,
            consensus_support_count=0,
            contributing_slots=(),
            abstaining_slots=(),
            authorizes_first_row_change=True,
        )

    by_slot: dict[str, set[int]] = defaultdict(set)
    seen_slots: set[str] = set()
    for vote in votes:
        seen_slots.add(vote.slot)
        if vote.first_row_ms is not None:
            by_slot[vote.slot].add(vote.first_row_ms)
    abstaining = {slot for slot in seen_slots if len(by_slot.get(slot, set())) != 1}
    slot_votes = {slot: next(iter(times)) for slot, times in by_slot.items() if len(times) == 1}
    counts = Counter(slot_votes.values())
    if counts:
        best_count = max(counts.values())
        best_times = tuple(
            sorted(time_ms for time_ms, count in counts.items() if count == best_count)
        )
    else:
        best_count = 0
        best_times = ()
    if best_count >= 2 and len(best_times) == 1:
        reference = best_times[0]
        contributing = tuple(
            sorted(slot for slot, time_ms in slot_votes.items() if time_ms == reference)
        )
        return IntroSelectionEvidenceV3(
            reference_state="CROSS_SLOT_CONSENSUS",
            reference_first_row_ms=reference,
            audio_anchor_ms=anchor.anchor_ms,
            audio_anchor_grid_ms=anchor.anchor_grid_ms,
            audio_evidence_digest=audio_digest,
            consensus_support_count=best_count,
            contributing_slots=contributing,
            abstaining_slots=tuple(sorted(abstaining)),
            authorizes_first_row_change=False,
        )

    return IntroSelectionEvidenceV3(
        reference_state="UNCERTAIN",
        reference_first_row_ms=None,
        audio_anchor_ms=anchor.anchor_ms,
        audio_anchor_grid_ms=anchor.anchor_grid_ms,
        audio_evidence_digest=audio_digest,
        consensus_support_count=best_count,
        contributing_slots=(),
        abstaining_slots=tuple(sorted(abstaining)),
        authorizes_first_row_change=False,
    )


@dataclass(frozen=True, slots=True)
class CandidateFamilyEvidenceV3:
    candidate_id: str
    key_mode: int
    difficulty: str
    provenance: str
    candidate_payload_ref: str
    candidate_payload_sha256: str
    safety: CandidateSafetyEvidenceV3
    first_row_ms: int | None
    first_row_audio_supported: bool
    first_row_grid_distance_ms: int | None
    intro_reference_state: IntroReferenceState
    matched_f1_50: float | None = None
    matched_precision_50: float | None = None
    review_rank: int = 0
    candidate_role: CandidateRole | None = "PLAYTEST_POOL"
    eligible_target_difficulties: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _exact_string(self.candidate_id, name="candidate_id")
        if type(self.key_mode) is not int or self.key_mode not in KEY_MODES:
            raise ValueError("key_mode is unsupported")
        if type(self.difficulty) is not str or self.difficulty not in DIFFICULTIES:
            raise ValueError("difficulty is unsupported")
        _exact_string(self.provenance, name="provenance")
        payload_ref = _exact_string(
            self.candidate_payload_ref,
            name="candidate_payload_ref",
        )
        path = PurePosixPath(payload_ref)
        if path.is_absolute() or path.as_posix() != payload_ref or ".." in path.parts:
            raise ValueError("candidate_payload_ref must be a normalized relative path")
        _required_sha256(
            self.candidate_payload_sha256,
            name="candidate_payload_sha256",
        )
        if not isinstance(self.safety, CandidateSafetyEvidenceV3):
            raise TypeError("safety must be CandidateSafetyEvidenceV3")
        if self.safety.candidate_id != self.candidate_id:
            raise ValueError("candidate identity differs from safety evidence")
        if self.first_row_ms is not None:
            _exact_non_negative_int(self.first_row_ms, name="first_row_ms")
        if type(self.first_row_audio_supported) is not bool:
            raise TypeError("first_row_audio_supported must be an exact boolean")
        if self.first_row_grid_distance_ms is not None:
            _exact_non_negative_int(
                self.first_row_grid_distance_ms,
                name="first_row_grid_distance_ms",
            )
        if self.first_row_ms is None and (
            self.first_row_audio_supported or self.first_row_grid_distance_ms is not None
        ):
            raise ValueError("missing first row cannot carry support evidence")
        if self.intro_reference_state not in {
            "CONFIRMED_AUDIO",
            "CROSS_SLOT_CONSENSUS",
            "UNCERTAIN",
        }:
            raise ValueError("intro_reference_state is unsupported")
        for name, value in (
            ("matched_f1_50", self.matched_f1_50),
            ("matched_precision_50", self.matched_precision_50),
        ):
            if value is not None:
                _exact_probability(value, name=name)
        _exact_non_negative_int(self.review_rank, name="review_rank")
        if self.candidate_role not in {
            None,
            "PLAYTEST_POOL",
            "SHADOW_CHALLENGER",
        }:
            raise ValueError("candidate_role is unsupported")
        if type(self.eligible_target_difficulties) is not tuple:
            raise TypeError("eligible_target_difficulties must be a tuple")
        expected_targets = tuple(
            difficulty
            for difficulty in DIFFICULTIES
            if difficulty in self.eligible_target_difficulties
        )
        if (
            expected_targets != self.eligible_target_difficulties
            or len(set(self.eligible_target_difficulties))
            != len(self.eligible_target_difficulties)
        ):
            raise ValueError(
                "eligible_target_difficulties must be unique and difficulty ordered"
            )

    @property
    def target_difficulties(self) -> tuple[str, ...]:
        """Explicit relabel targets, or the legacy source-only target."""

        return self.eligible_target_difficulties or (self.difficulty,)

    def to_report(self) -> dict[str, object]:
        report = {
            "candidateId": self.candidate_id,
            "keyMode": self.key_mode,
            "difficulty": self.difficulty,
            "provenance": self.provenance,
            "candidatePayloadRef": self.candidate_payload_ref,
            "candidatePayloadSha256": self.candidate_payload_sha256,
            "safety": self.safety.to_report(),
            "firstRowMs": self.first_row_ms,
            "firstRowAudioSupported": self.first_row_audio_supported,
            "firstRowGridDistanceMs": self.first_row_grid_distance_ms,
            "introReferenceState": self.intro_reference_state,
            "matchedF150": self.matched_f1_50,
            "matchedPrecision50": self.matched_precision_50,
            "reviewRank": self.review_rank,
        }
        if self.candidate_role is not None:
            report["candidateRole"] = self.candidate_role
        if self.eligible_target_difficulties:
            report["eligibleTargetDifficulties"] = list(
                self.eligible_target_difficulties
            )
        return report


@dataclass(frozen=True, slots=True)
class SongSelectionEvidenceV3:
    context_id: str
    intro_selection: IntroSelectionEvidenceV3
    candidates: tuple[CandidateFamilyEvidenceV3, ...]
    current_assignment: tuple[tuple[str, str | None], ...]
    version: Literal["song-selection-evidence-v3"] = SONG_SELECTION_EVIDENCE_VERSION

    def __post_init__(self) -> None:
        if self.version != SONG_SELECTION_EVIDENCE_VERSION:
            raise ValueError("unsupported song selection evidence version")
        _exact_string(self.context_id, name="context_id")
        if not isinstance(self.intro_selection, IntroSelectionEvidenceV3):
            raise TypeError("intro_selection must be IntroSelectionEvidenceV3")
        if type(self.candidates) is not tuple or any(
            not isinstance(item, CandidateFamilyEvidenceV3) for item in self.candidates
        ):
            raise TypeError("candidates must be a tuple of CandidateFamilyEvidenceV3")
        expected_candidates = tuple(sorted(self.candidates, key=lambda item: item.candidate_id))
        candidate_ids = tuple(item.candidate_id for item in self.candidates)
        if self.candidates != expected_candidates or len(set(candidate_ids)) != len(candidate_ids):
            raise ValueError("candidates must be sorted with unique identities")
        if type(self.current_assignment) is not tuple:
            raise TypeError("current_assignment must be a tuple")
        for slot, candidate_id in self.current_assignment:
            _exact_string(slot, name="current_assignment slot")
            if candidate_id is not None:
                _exact_string(candidate_id, name="current_assignment candidate")
                if candidate_id not in candidate_ids:
                    raise ValueError("current_assignment references an unknown candidate")
        if self.current_assignment != tuple(
            sorted(set(self.current_assignment), key=lambda item: item[0])
        ) or len({slot for slot, _candidate_id in self.current_assignment}) != len(
            self.current_assignment
        ):
            raise ValueError("current_assignment slots must be sorted and unique")

    def to_report(self) -> dict[str, object]:
        return {
            "version": self.version,
            "contextId": self.context_id,
            "introSelection": self.intro_selection.to_report(),
            "introSelectionSha256": self.intro_selection.stable_sha256(),
            "currentAssignment": dict(self.current_assignment),
            "candidates": [item.to_report() for item in self.candidates],
            "mutatesSelection": False,
            "additionalModelCalls": 0,
        }

    def stable_sha256(self) -> str:
        return _canonical_sha256(self.to_report())


def _parse_gap_interval(value: object) -> GapIntervalEvidence:
    report = _exact_mapping(
        value,
        name="gap interval",
        keys=frozenset(
            {
                "version",
                "startMs",
                "endMs",
                "durationMs",
                "position",
                "activeOnsetCount",
                "activeFrameRatio",
                "opportunityKind",
                "localAudioEvidenceDigest",
            }
        ),
    )
    parsed = GapIntervalEvidence(
        version=report["version"],
        start_ms=report["startMs"],
        end_ms=report["endMs"],
        position=report["position"],
        active_onset_count=report["activeOnsetCount"],
        active_frame_ratio=report["activeFrameRatio"],
        opportunity_kind=report["opportunityKind"],
        local_audio_evidence_digest=report["localAudioEvidenceDigest"],
    )
    _require_projection(report, parsed.to_report(), name="gap interval")
    return parsed


def _parse_candidate_safety(value: object) -> CandidateSafetyEvidenceV3:
    report = _exact_mapping(
        value,
        name="candidate safety",
        keys=frozenset(
            {
                "version",
                "candidateId",
                "structureSafe",
                "timingIdentitySafe",
                "songBoundsSafe",
                "serializationSafe",
                "hardSafe",
                "publicationTier",
                "modelBacked",
                "recoveryTrustRank",
                "activeGapCount",
                "totalActiveGapMs",
                "maxActiveGapMs",
                "activeGaps",
            }
        ),
    )
    gaps = tuple(
        _parse_gap_interval(item) for item in _exact_list(report["activeGaps"], name="activeGaps")
    )
    parsed = CandidateSafetyEvidenceV3(
        version=report["version"],
        candidate_id=report["candidateId"],
        structure_safe=report["structureSafe"],
        timing_identity_safe=report["timingIdentitySafe"],
        song_bounds_safe=report["songBoundsSafe"],
        serialization_safe=report["serializationSafe"],
        publication_tier=report["publicationTier"],
        model_backed=report["modelBacked"],
        recovery_trust_rank=report["recoveryTrustRank"],
        active_gaps=gaps,
    )
    _require_projection(report, parsed.to_report(), name="candidate safety")
    return parsed


def _parse_intro_selection(value: object) -> IntroSelectionEvidenceV3:
    report = _exact_mapping(
        value,
        name="intro selection",
        keys=frozenset(
            {
                "version",
                "referenceState",
                "referenceFirstRowMs",
                "audioAnchorMs",
                "audioAnchorGridMs",
                "audioEvidenceDigest",
                "consensusSupportCount",
                "contributingSlots",
                "abstainingSlots",
                "authorizesFirstRowChange",
            }
        ),
    )
    parsed = IntroSelectionEvidenceV3(
        version=report["version"],
        reference_state=report["referenceState"],
        reference_first_row_ms=report["referenceFirstRowMs"],
        audio_anchor_ms=report["audioAnchorMs"],
        audio_anchor_grid_ms=report["audioAnchorGridMs"],
        audio_evidence_digest=report["audioEvidenceDigest"],
        consensus_support_count=report["consensusSupportCount"],
        contributing_slots=tuple(
            _exact_list(report["contributingSlots"], name="contributingSlots")
        ),
        abstaining_slots=tuple(_exact_list(report["abstainingSlots"], name="abstainingSlots")),
        authorizes_first_row_change=report["authorizesFirstRowChange"],
    )
    _require_projection(report, parsed.to_report(), name="intro selection")
    return parsed


def _parse_candidate_family(value: object) -> CandidateFamilyEvidenceV3:
    base_keys = frozenset(
        {
            "candidateId",
            "keyMode",
            "difficulty",
            "provenance",
            "candidatePayloadRef",
            "candidatePayloadSha256",
            "safety",
            "firstRowMs",
            "firstRowAudioSupported",
            "firstRowGridDistanceMs",
            "introReferenceState",
            "matchedF150",
            "matchedPrecision50",
            "reviewRank",
        }
    )
    has_candidate_role = type(value) is dict and "candidateRole" in value
    has_eligible_targets = (
        type(value) is dict and "eligibleTargetDifficulties" in value
    )
    report = _exact_mapping(
        value,
        name="candidate family evidence",
        keys=(
            base_keys
            | ({"candidateRole"} if has_candidate_role else set())
            | ({"eligibleTargetDifficulties"} if has_eligible_targets else set())
        ),
    )
    parsed = CandidateFamilyEvidenceV3(
        candidate_id=report["candidateId"],
        key_mode=report["keyMode"],
        difficulty=report["difficulty"],
        provenance=report["provenance"],
        candidate_payload_ref=report["candidatePayloadRef"],
        candidate_payload_sha256=report["candidatePayloadSha256"],
        safety=_parse_candidate_safety(report["safety"]),
        first_row_ms=report["firstRowMs"],
        first_row_audio_supported=report["firstRowAudioSupported"],
        first_row_grid_distance_ms=report["firstRowGridDistanceMs"],
        intro_reference_state=report["introReferenceState"],
        matched_f1_50=report["matchedF150"],
        matched_precision_50=report["matchedPrecision50"],
        review_rank=report["reviewRank"],
        candidate_role=(report["candidateRole"] if has_candidate_role else None),
        eligible_target_difficulties=(
            tuple(
                _exact_list(
                    report["eligibleTargetDifficulties"],
                    name="eligibleTargetDifficulties",
                )
            )
            if has_eligible_targets
            else ()
        ),
    )
    _require_projection(report, parsed.to_report(), name="candidate family evidence")
    return parsed


def parse_song_selection_evidence_v3(value: object) -> SongSelectionEvidenceV3:
    """Strictly reconstruct a report-only evidence object from archived JSON."""
    report = _exact_mapping(
        value,
        name="song selection evidence",
        keys=frozenset(
            {
                "version",
                "contextId",
                "introSelection",
                "introSelectionSha256",
                "currentAssignment",
                "candidates",
                "mutatesSelection",
                "additionalModelCalls",
            }
        ),
    )
    assignment_report = report["currentAssignment"]
    if type(assignment_report) is not dict or any(
        type(slot) is not str
        or not slot
        or (candidate_id is not None and (type(candidate_id) is not str or not candidate_id))
        for slot, candidate_id in assignment_report.items()
    ):
        raise TypeError("currentAssignment must map exact strings to exact strings or null")
    intro = _parse_intro_selection(report["introSelection"])
    if report["introSelectionSha256"] != intro.stable_sha256():
        raise ValueError("intro selection digest differs from its payload")
    parsed = SongSelectionEvidenceV3(
        version=report["version"],
        context_id=report["contextId"],
        intro_selection=intro,
        candidates=tuple(
            _parse_candidate_family(item)
            for item in _exact_list(report["candidates"], name="candidates")
        ),
        current_assignment=tuple(sorted(assignment_report.items())),
    )
    _require_projection(report, parsed.to_report(), name="song selection evidence")
    return parsed
