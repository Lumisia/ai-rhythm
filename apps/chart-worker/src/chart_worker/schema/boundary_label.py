"""사람이 표시한 곡 끝 경계를 보존하는 boundary-label v1/v2 계약."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any, Literal, Self
from uuid import UUID

from pydantic import Field, StringConstraints, model_validator

from chart_worker.schema.chart import CamelModel, Sha256
from chart_worker.schema.playtest_run import SafeRelativePath

NonEmptyReviewerId = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=80),
]
NonEmptyGroupId = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=120),
]
OptionalReason = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=500),
]
Millisecond = Annotated[int, Field(strict=True, ge=0)]

BoundaryPolicyState = Literal["EXPERIMENTAL", "PROVISIONAL", "CALIBRATED", "FROZEN"]
BoundaryPolicyConfidence = Literal["HIGH", "MEDIUM", "LOW", "UNKNOWN"]
BoundaryEnforcementMode = Literal[
    "SHADOW",
    "EXPERIMENTAL_ENFORCED",
    "HIGH_CONFIDENCE_ENFORCED",
]
BoundaryEvidenceAvailability = Literal["AVAILABLE", "UNAVAILABLE"]
BoundaryGroupRelation = Literal["EXACT_RECORDING", "RELATED_VERSION", "UNKNOWN"]
BoundaryVerdict = Literal[
    "TOO_EARLY",
    "ACCEPTABLE",
    "TOO_LATE",
    "UNCERTAIN",
    "NOT_AVAILABLE",
]
BoundaryTailCharacter = Literal[
    "MUSIC",
    "FADE_OR_REVERB",
    "NOISE",
    "ENCODING_TAIL",
    "SILENCE",
    "MIXED_OR_UNCERTAIN",
]
HumanLabelConfidence = Literal["HIGH", "MEDIUM", "LOW"]


class BoundaryRunIdentity(CamelModel):
    run_id: UUID
    title: str = Field(min_length=1)
    song_version_id: UUID
    game_audio_asset_id: UUID


class BoundaryAudioIdentity(CamelModel):
    sha256: Sha256
    duration_ms: Millisecond = Field(gt=0)


class BoundaryGenerationReportRef(CamelModel):
    path: SafeRelativePath
    sha256: Sha256


class BoundaryLabelGroup(CamelModel):
    group_id: NonEmptyGroupId
    relation: BoundaryGroupRelation
    confirmed: bool


class BoundaryAutomaticEvidence(CamelModel):
    availability: BoundaryEvidenceAvailability
    unavailable_reason: OptionalReason | None = None
    evaluation_version: str | None = Field(default=None, min_length=1)
    policy_state: BoundaryPolicyState | None = None
    policy_confidence: BoundaryPolicyConfidence | None = None
    enforcement_mode: BoundaryEnforcementMode | None = None
    observation_sha256: Sha256 | None = None
    last_detected_onset_ms: Millisecond | None = None
    last_active_rms_end_ms: Millisecond | None = None
    last_evidence_ms: Millisecond | None = None
    provisional_max_note_start_ms: Millisecond | None = None
    provisional_release_end_ms: Millisecond | None = None
    effective_max_note_start_ms: Millisecond | None = None
    effective_release_end_ms: Millisecond | None = None

    @model_validator(mode="after")
    def _check_availability(self) -> Self:
        evidence_values = (
            self.evaluation_version,
            self.policy_state,
            self.policy_confidence,
            self.enforcement_mode,
            self.observation_sha256,
            self.last_detected_onset_ms,
            self.last_active_rms_end_ms,
            self.last_evidence_ms,
            self.provisional_max_note_start_ms,
            self.provisional_release_end_ms,
            self.effective_max_note_start_ms,
            self.effective_release_end_ms,
        )
        if self.availability == "UNAVAILABLE":
            if self.unavailable_reason is None:
                raise ValueError("unavailable automatic evidence requires a reason")
            if any(value is not None for value in evidence_values):
                raise ValueError("unavailable automatic evidence cannot contain evidence values")
            return self

        if self.unavailable_reason is not None:
            raise ValueError("available automatic evidence cannot contain an unavailable reason")
        required = (
            self.evaluation_version,
            self.policy_state,
            self.policy_confidence,
            self.enforcement_mode,
            self.observation_sha256,
            self.provisional_max_note_start_ms,
            self.provisional_release_end_ms,
            self.effective_max_note_start_ms,
            self.effective_release_end_ms,
        )
        if any(value is None for value in required):
            raise ValueError("available automatic evidence is missing required values")
        return self


class TimeUncertaintyInterval(CamelModel):
    earliest_ms: Millisecond
    latest_ms: Millisecond

    @model_validator(mode="after")
    def _check_order(self) -> Self:
        if self.earliest_ms > self.latest_ms:
            raise ValueError("earliestMs must not exceed latestMs")
        return self


class BoundaryHumanAnnotation(CamelModel):
    last_meaningful_attack: TimeUncertaintyInterval
    last_acceptable_release: TimeUncertaintyInterval
    provisional_boundary_verdict: BoundaryVerdict
    tail_characters: list[BoundaryTailCharacter] = Field(min_length=1)
    confidence: HumanLabelConfidence
    comment: str = Field(default="", max_length=4000)

    @model_validator(mode="after")
    def _check_tail_characters(self) -> Self:
        if len(self.tail_characters) != len(set(self.tail_characters)):
            raise ValueError("tail characters must be unique")
        if "MIXED_OR_UNCERTAIN" in self.tail_characters and len(self.tail_characters) != 1:
            raise ValueError("mixed or uncertain tail cannot be combined with another tail")
        return self


class BoundaryHumanAnnotationV2(CamelModel):
    """사람이 구분해 들은 새 타점·주요 콘텐츠·허용 release 경계."""

    last_playable_attack: TimeUncertaintyInterval
    primary_content_end: TimeUncertaintyInterval
    acceptable_release_end: TimeUncertaintyInterval
    provisional_boundary_verdict: BoundaryVerdict
    tail_characters: list[BoundaryTailCharacter] = Field(min_length=1)
    confidence: HumanLabelConfidence
    comment: str = Field(default="", max_length=4000)

    @model_validator(mode="after")
    def _check_tail_characters(self) -> Self:
        if len(self.tail_characters) != len(set(self.tail_characters)):
            raise ValueError("tail characters must be unique")
        if "MIXED_OR_UNCERTAIN" in self.tail_characters and len(self.tail_characters) != 1:
            raise ValueError("mixed or uncertain tail cannot be combined with another tail")
        return self


class BoundaryLabelV1(CamelModel):
    version: Literal[1] = 1
    label_id: UUID
    created_at: datetime
    reviewer_id: NonEmptyReviewerId
    run: BoundaryRunIdentity
    audio: BoundaryAudioIdentity
    generation_report: BoundaryGenerationReportRef
    group: BoundaryLabelGroup
    automatic_evidence: BoundaryAutomaticEvidence
    annotation: BoundaryHumanAnnotation

    @model_validator(mode="after")
    def _check_contract(self) -> Self:
        attack = self.annotation.last_meaningful_attack
        release = self.annotation.last_acceptable_release
        duration_ms = self.audio.duration_ms
        if attack.latest_ms > duration_ms or release.latest_ms > duration_ms:
            raise ValueError("human boundary interval exceeds audio duration")
        if attack.earliest_ms > release.latest_ms:
            raise ValueError("last meaningful attack must not begin after the final release")

        verdict = self.annotation.provisional_boundary_verdict
        if self.automatic_evidence.availability == "AVAILABLE":
            if verdict == "NOT_AVAILABLE":
                raise ValueError("available automatic evidence requires a comparable verdict")
        elif verdict != "NOT_AVAILABLE":
            raise ValueError("unavailable automatic evidence requires NOT_AVAILABLE verdict")
        return self


class BoundaryLabelV2(CamelModel):
    """세 가지 사람 경계를 혼동하지 않는 boundary-label-v2 계약."""

    version: Literal[2] = 2
    label_id: UUID
    created_at: datetime
    reviewer_id: NonEmptyReviewerId
    run: BoundaryRunIdentity
    audio: BoundaryAudioIdentity
    generation_report: BoundaryGenerationReportRef
    group: BoundaryLabelGroup
    automatic_evidence: BoundaryAutomaticEvidence
    annotation: BoundaryHumanAnnotationV2

    @model_validator(mode="after")
    def _check_contract(self) -> Self:
        attack = self.annotation.last_playable_attack
        content_end = self.annotation.primary_content_end
        release = self.annotation.acceptable_release_end
        duration_ms = self.audio.duration_ms
        if any(
            interval.latest_ms > duration_ms
            for interval in (attack, content_end, release)
        ):
            raise ValueError("human boundary interval exceeds audio duration")
        if attack.earliest_ms > content_end.latest_ms:
            raise ValueError("last playable attack must not begin after primary content end")
        if content_end.earliest_ms > release.latest_ms:
            raise ValueError("primary content end must not begin after acceptable release end")

        verdict = self.annotation.provisional_boundary_verdict
        if self.automatic_evidence.availability == "AVAILABLE":
            if verdict == "NOT_AVAILABLE":
                raise ValueError("available automatic evidence requires a comparable verdict")
        elif verdict != "NOT_AVAILABLE":
            raise ValueError("unavailable automatic evidence requires NOT_AVAILABLE verdict")
        return self


def boundary_label_json_schema() -> dict[str, Any]:
    return BoundaryLabelV1.model_json_schema(by_alias=True)


def boundary_label_v2_json_schema() -> dict[str, Any]:
    return BoundaryLabelV2.model_json_schema(by_alias=True)
