"""로컬 플레이테스터 실행 디렉터리의 진입 계약."""

from datetime import datetime
from pathlib import PurePosixPath
from typing import Annotated, Literal, Self
from uuid import UUID

from pydantic import AfterValidator, Field, field_validator, model_validator

from chart_worker.schema.chart import CamelModel, Sha256
from chart_worker.schema.types import DIFFICULTIES, KEY_MODES, Difficulty
from chart_worker.validation.outcome_status import (
    CompletenessStatus,
    ExecutionStatus,
    FailureCategory,
    OutcomeStatus,
    QualityStatus,
)
from chart_worker.validation.publication_policy import (
    PUBLICATION_POLICY_VERSION,
    PublicationDecisionName,
    PublicationReasonCode,
    PublicationStrictBlocker,
    decide_publication,
)


def safe_relative_path(value: str) -> str:
    path = PurePosixPath(value.replace("\\", "/"))
    if (
        not value
        or path == PurePosixPath(".")
        or path.is_absolute()
        or ".." in path.parts
        or ":" in value
    ):
        raise ValueError("path must be a safe relative path")
    return path.as_posix()


SafeRelativePath = Annotated[str, AfterValidator(safe_relative_path)]


class AudioFileRef(CamelModel):
    path: SafeRelativePath
    sha256: Sha256


class RunChartRef(AudioFileRef):
    key_mode: Literal[4, 6, 7]
    difficulty: Difficulty


class CoverageSummary(CamelModel):
    first_note_time_ms: int | None
    max_gap_ms: int = Field(ge=0)
    attack_required_gap_count: int = Field(ge=0)
    attack_required_gap_total_ms: int = Field(ge=0)
    repaired_gap_count: int = Field(ge=0)


class RunChartRefV2(RunChartRef):
    """A chart reference whose trust tier cannot be hidden from consumers."""

    provenance: Literal[
        "PRIMARY",
        "RETRY",
        "PARTIAL_REMAP",
        "INTRO_RECOVERY",
        "INTRO_ALIGNED",
        "COVERAGE_REPAIR",
        "RAW_UNVERIFIED",
        "SAFE_FALLBACK",
    ] = "PRIMARY"
    family_assignment_kind: Literal[
        "ORIGINAL",
        "REASSIGNED",
        "EMERGENCY_DUPLICATE",
    ] = "ORIGINAL"
    source_difficulty: Difficulty | None = None
    family_resolution_state: Literal[
        "RESOLVED",
        "NARROW_REVIEW",
        "UNRESOLVED",
    ] = "RESOLVED"
    family_resolution_reasons: list[str] = Field(default_factory=list)
    production_eligible: bool = True
    distribution_tier: Literal["PRODUCTION_CANDIDATE", "PLAYTEST_ONLY"] = (
        "PRODUCTION_CANDIDATE"
    )
    playability_tier: Literal[
        "MODEL_PLAYABLE",
        "RECOVERY_PLAYABLE",
        "DIAGNOSTIC_ONLY",
    ] | None = None
    coverage_summary: CoverageSummary | None = None

    @model_validator(mode="after")
    def _check_distribution_tier(self) -> Self:
        provenance_fallback = self.provenance in {
            "COVERAGE_REPAIR",
            "RAW_UNVERIFIED",
            "SAFE_FALLBACK",
        }
        family_adapted = self.family_assignment_kind != "ORIGINAL"
        if family_adapted and self.source_difficulty is None:
            raise ValueError("adapted family assignment requires source difficulty")
        if (
            self.family_assignment_kind == "ORIGINAL"
            and self.source_difficulty is not None
            and self.source_difficulty != self.difficulty
        ):
            raise ValueError("original family assignment source difficulty must match target")
        is_fallback = provenance_fallback or family_adapted
        family_unresolved = self.family_resolution_state != "RESOLVED"
        if family_unresolved and not self.family_resolution_reasons:
            raise ValueError("unresolved family state requires reason codes")
        if not family_unresolved and self.family_resolution_reasons:
            raise ValueError("resolved family state cannot carry unresolved reasons")
        if self.family_resolution_reasons != sorted(
            set(self.family_resolution_reasons)
        ):
            raise ValueError("family resolution reasons must be sorted and unique")
        if is_fallback and (
            self.production_eligible or self.distribution_tier != "PLAYTEST_ONLY"
        ):
            raise ValueError(
                "fallback provenance or family assignment must be PLAYTEST_ONLY "
                "and not production eligible"
            )
        if family_unresolved and (
            self.production_eligible
            or self.distribution_tier != "PLAYTEST_ONLY"
        ):
            raise ValueError("unresolved family must be playtest-only")
        if not is_fallback and not family_unresolved and (
            not self.production_eligible
            or self.distribution_tier != "PRODUCTION_CANDIDATE"
        ):
            raise ValueError(
                "model-backed provenance must remain a production candidate"
            )
        if (self.playability_tier is None) != (self.coverage_summary is None):
            raise ValueError(
                "playability tier and coverage summary must be present together"
            )
        if self.playability_tier is None:
            return self
        if not is_fallback and self.playability_tier != "MODEL_PLAYABLE":
            raise ValueError(
                "model-backed provenance must be MODEL_PLAYABLE"
            )
        if self.provenance == "RAW_UNVERIFIED" and (
            self.playability_tier != "DIAGNOSTIC_ONLY"
        ):
            raise ValueError("RAW_UNVERIFIED must be DIAGNOSTIC_ONLY")
        if (
            self.provenance in {"COVERAGE_REPAIR", "SAFE_FALLBACK"}
            or family_adapted
        ) and (
            self.playability_tier == "MODEL_PLAYABLE"
        ):
            raise ValueError("recovery provenance or family assignment cannot be MODEL_PLAYABLE")
        return self


class MissingChartRef(CamelModel):
    """이번 실행에서 발행하지 못한 조합. 프론트가 목록에서 빠지는 근거다."""

    key_mode: Literal[4, 6, 7]
    difficulty: Difficulty
    reason: str = Field(min_length=1)


class RunAudioRefs(CamelModel):
    game: AudioFileRef
    no_drums: AudioFileRef | None = None
    keys: AudioFileRef | None = None


def _validate_run_fields(
    *,
    charts: list[RunChartRef],
    missing_charts: list[MissingChartRef],
    audio: RunAudioRefs,
    keysound_manifest_path: str | None,
) -> None:
    combinations = [(chart.key_mode, chart.difficulty) for chart in charts]
    if len(combinations) != len(set(combinations)):
        raise ValueError("duplicate chart combination")
    expected = {(key_mode, difficulty) for key_mode in KEY_MODES for difficulty in DIFFICULTIES}
    unexpected = set(combinations) - expected
    if unexpected:
        raise ValueError("run contains an unsupported chart combination")

    declared_missing = {(entry.key_mode, entry.difficulty) for entry in missing_charts}
    if declared_missing & set(combinations):
        raise ValueError("a chart cannot be both published and missing")
    if declared_missing | set(combinations) != expected:
        raise ValueError("published and missing charts must together cover all 12 combinations")

    keysound_parts = (
        audio.no_drums is not None,
        audio.keys is not None,
        keysound_manifest_path is not None,
    )
    if any(keysound_parts) and not all(keysound_parts):
        raise ValueError("keysound references must all be present or absent")


class PlaytestRunManifest(CamelModel):
    version: Literal[1] = 1
    run_id: UUID
    title: str = Field(min_length=1)
    generated_at: datetime
    worker_version: str = Field(min_length=1)
    audio: RunAudioRefs
    charts: list[RunChartRef] = Field(min_length=1)
    missing_charts: list[MissingChartRef] = Field(default_factory=list)
    keysound_manifest_path: SafeRelativePath | None = None
    generation_report_path: SafeRelativePath

    @model_validator(mode="after")
    def _check_run(self) -> Self:
        _validate_run_fields(
            charts=self.charts,
            missing_charts=self.missing_charts,
            audio=self.audio,
            keysound_manifest_path=self.keysound_manifest_path,
        )
        return self


class ReportFileRef(AudioFileRef):
    pass


class OutcomeStatusSnapshot(CamelModel):
    execution: ExecutionStatus
    completeness: CompletenessStatus
    quality: QualityStatus
    failure_category: FailureCategory
    publishable_strict: bool

    def to_domain(self) -> OutcomeStatus:
        return OutcomeStatus(
            execution=self.execution,
            completeness=self.completeness,
            quality=self.quality,
            failure_category=self.failure_category,
            publishable_strict=self.publishable_strict,
        )


class PublicationDecisionSnapshot(CamelModel):
    policy_version: Literal["PUBLICATION_POLICY_V2"] = PUBLICATION_POLICY_VERSION
    decision: PublicationDecisionName
    reason_codes: list[PublicationReasonCode]

    @model_validator(mode="after")
    def _check_reason_codes(self) -> Self:
        if self.reason_codes != sorted(set(self.reason_codes)):
            raise ValueError("publication reason codes must be sorted and unique")
        return self


class PlaytestRunManifestV2(CamelModel):
    version: Literal[2]
    run_id: UUID
    title: str = Field(min_length=1)
    generated_at: datetime
    worker_version: str = Field(min_length=1)
    audio: RunAudioRefs
    charts: list[RunChartRefV2] = Field(min_length=1)
    missing_charts: list[MissingChartRef]
    keysound_manifest_path: SafeRelativePath | None = None
    generation_report: ReportFileRef
    outcome: OutcomeStatusSnapshot
    strict_blockers: list[PublicationStrictBlocker]
    publication: PublicationDecisionSnapshot

    @field_validator("charts", mode="before")
    @classmethod
    def _upgrade_v1_chart_refs(cls, value: object) -> object:
        """Keep the in-process V1->V2 migration API backward compatible."""
        if not isinstance(value, list):
            return value
        return [
            chart.model_dump()
            if isinstance(chart, RunChartRef)
            and not isinstance(chart, RunChartRefV2)
            else chart
            for chart in value
        ]

    @model_validator(mode="after")
    def _check_run(self) -> Self:
        _validate_run_fields(
            charts=self.charts,
            missing_charts=self.missing_charts,
            audio=self.audio,
            keysound_manifest_path=self.keysound_manifest_path,
        )
        if self.strict_blockers != sorted(set(self.strict_blockers)):
            raise ValueError("publication strict blockers must be sorted and unique")
        expected = decide_publication(
            outcome=self.outcome.to_domain(),
            published_slots=len(self.charts),
            expected_slots=len(KEY_MODES) * len(DIFFICULTIES),
            strict_blockers=tuple(self.strict_blockers),
        )
        if self.publication.model_dump(by_alias=True) != expected.to_report():
            raise ValueError("publication decision disagrees with run outcome")
        return self
