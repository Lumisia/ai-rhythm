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


class RunChartRefV2(RunChartRef):
    """A chart reference whose trust tier cannot be hidden from consumers."""

    provenance: Literal[
        "PRIMARY",
        "RETRY",
        "PARTIAL_REMAP",
        "INTRO_RECOVERY",
        "INTRO_ALIGNED",
        "RAW_UNVERIFIED",
        "SAFE_FALLBACK",
    ] = "PRIMARY"
    production_eligible: bool = True
    distribution_tier: Literal["PRODUCTION_CANDIDATE", "PLAYTEST_ONLY"] = (
        "PRODUCTION_CANDIDATE"
    )

    @model_validator(mode="after")
    def _check_distribution_tier(self) -> Self:
        is_fallback = self.provenance in {"RAW_UNVERIFIED", "SAFE_FALLBACK"}
        if is_fallback and (
            self.production_eligible or self.distribution_tier != "PLAYTEST_ONLY"
        ):
            raise ValueError(
                "fallback provenance must be PLAYTEST_ONLY and not production eligible"
            )
        if not is_fallback and (
            not self.production_eligible
            or self.distribution_tier != "PRODUCTION_CANDIDATE"
        ):
            raise ValueError(
                "model-backed provenance must remain a production candidate"
            )
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
    version: Literal[2] = 2
    run_id: UUID
    title: str = Field(min_length=1)
    generated_at: datetime
    worker_version: str = Field(min_length=1)
    audio: RunAudioRefs
    charts: list[RunChartRefV2] = Field(min_length=1)
    missing_charts: list[MissingChartRef] = Field(default_factory=list)
    keysound_manifest_path: SafeRelativePath | None = None
    generation_report: ReportFileRef
    outcome: OutcomeStatusSnapshot
    strict_blockers: list[PublicationStrictBlocker] = Field(default_factory=list)
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
