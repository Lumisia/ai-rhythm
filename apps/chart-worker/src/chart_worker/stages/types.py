"""Immutable contracts between the direct generation stages."""

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from chart_worker.analysis.grid_alignment import TempoCandidateMetrics
from chart_worker.audio.normalize import NormalizedAudio
from chart_worker.generation.generation_control import AdditionalInferenceBudget  # noqa: F401
from chart_worker.generation.mapperatorinator import GeneratedChart
from chart_worker.generation.osu_parser import OsuBpmEvent
from chart_worker.schema.chart import ChartDocument
from chart_worker.validation.leading_timing_coverage import LeadingTimingCoverage
from chart_worker.validation.timing_review import TimingAuthorityReview

if TYPE_CHECKING:
    from chart_worker.generation.diagnostic_fallback import DiagnosticRawCandidate
    from chart_worker.validation.difficulty_order import DifficultyOrderReview
    from chart_worker.validation.difficulty_selector import DifficultySelectionComparison
    from chart_worker.validation.intro_phrase_family import IntroPhraseFamilyReview
    from chart_worker.validation.intro_start_contract import (
        IntroContractReview,
        IntroStartContract,
    )
    from chart_worker.validation.local_timing_review import LocalTimingAuthorityReview
    from chart_worker.validation.outro_family_review import OutroFamilyReview
    from chart_worker.validation.quality_gate import ChartAcceptance
    from chart_worker.validation.recovery_preflight import RecoveryPreflight
    from chart_worker.validation.song_family_selector import SongSelectionComparison
    from chart_worker.validation.timing_candidate_selector import TimingCandidateSelection
    from chart_worker.validation.timing_family_review import TimingFamilyReview
    from chart_worker.validation.timing_integrity import TimingIntegrityAssessment

GenerationProvenance = Literal[
    "PRIMARY",
    "RETRY",
    "PARTIAL_REMAP",
    "INTRO_RECOVERY",
    "INTRO_ALIGNED",
    "COVERAGE_REPAIR",
    "RAW_UNVERIFIED",
    "SAFE_FALLBACK",
]
"""Playtest provenance contract.

RAW_UNVERIFIED is a model result that passed STRUCTURE, TIMING_IDENTITY, and
SONG_BOUNDS but failed a softer musical-quality axis. COVERAGE_REPAIR preserves
that source and adds TAP rows only inside proven active gaps. SAFE_FALLBACK is
a deterministic whole-chart recovery. Every recovery provenance forces
PLAYTEST_ONLY/REVIEW.
"""


@dataclass(frozen=True, slots=True)
class PreparedAudio:
    normalized: NormalizedAudio
    difficulty_selector_mode: Literal["CURRENT", "SHADOW_V2", "V2"] = "SHADOW_V2"
    boundary_policy_mode: Literal[
        "SHADOW",
        "EXPERIMENTAL_ENFORCED",
        "HIGH_CONFIDENCE_ENFORCED",
    ] = "HIGH_CONFIDENCE_ENFORCED"
    beat_this_enabled: bool = False
    beat_this_checkpoint: str = "final0"
    beat_this_device: Literal["cpu", "cuda"] = "cpu"
    beat_this_float16: bool = False


@dataclass(frozen=True, slots=True)
class SongTimingAuthority:
    reference_path: Path
    sha256: str
    audio_sha256: str
    bpm_events: tuple[OsuBpmEvent, ...]
    generator_name: str
    seed: int | None
    mode: Literal["STANDARD", "SUPER_TIMING", "BEAT_THIS_FALLBACK"]
    attempt_count: int
    tempo_metrics: TempoCandidateMetrics | None = None
    review: TimingAuthorityReview | None = None
    leading_coverage: LeadingTimingCoverage | None = None
    local_review: "LocalTimingAuthorityReview | None" = None
    recovery_preflight: "RecoveryPreflight | None" = None
    candidate_selection: "TimingCandidateSelection | None" = None
    timing_integrity: "TimingIntegrityAssessment | None" = None


@dataclass(frozen=True, slots=True)
class GeneratedVariant:
    key_mode: int
    difficulty: str
    requested_star: float
    raw_osu_path: Path
    generated: GeneratedChart
    acceptance: "ChartAcceptance"
    cfg_scale: float = 1.0
    attempt: int = 1
    attempt_errors: tuple[str, ...] = ()
    attempt_evidence: tuple[dict[str, object], ...] = ()
    timing_authority_sha256: str = ""
    candidate_count: int = 1
    generation_attempt_count: int = 1
    selected_seed: int | None = None
    difficulty_order: "DifficultyOrderReview | None" = None
    provenance: GenerationProvenance = "PRIMARY"
    recovery_reason: str | None = None
    coverage_repair_gap_count: int = 0
    """Why this is not PRIMARY, including any playtest-only fallback policy."""


@dataclass(frozen=True, slots=True)
class MissingVariant:
    """발행 가능한 후보가 없어 이번 실행에서 빠진 조합.

    한 조합의 실패가 같은 키 모드의 다른 조합을 끌어내리지 않도록,
    실패는 조합 단위로 격리하고 곡 상태만 PARTIAL 로 낮춘다.
    """

    key_mode: int
    difficulty: str
    reason: str
    attempt_errors: tuple[str, ...] = ()
    attempt_evidence: tuple[dict[str, object], ...] = ()

    def to_report(self) -> dict[str, object]:
        return {
            "keyMode": self.key_mode,
            "difficulty": self.difficulty,
            "reason": self.reason,
            "attemptErrors": list(self.attempt_errors),
            "attemptEvidence": list(self.attempt_evidence),
        }


@dataclass(frozen=True, slots=True)
class GenerationOutcome:
    """생성 단계의 전체 결과. 성공 변형과 빠진 조합을 함께 보고한다."""

    variants: tuple[GeneratedVariant, ...]
    missing: tuple[MissingVariant, ...] = ()
    difficulty_selection_shadows: tuple["DifficultySelectionComparison", ...] = ()
    song_selection_shadow: "SongSelectionComparison | None" = None
    intro_start_contract: "IntroStartContract | None" = None
    intro_contract_review: "IntroContractReview | None" = None
    intro_phrase_family_reviews: tuple["IntroPhraseFamilyReview", ...] = ()
    timing_family_reviews: tuple["TimingFamilyReview", ...] = ()
    outro_family_review: "OutroFamilyReview | None" = None
    additional_inference_calls: int = 0
    additional_inference_work_ms: int = 0
    additional_inference_work_limit_ms: int = 0
    diagnostic_raw_candidates: tuple["DiagnosticRawCandidate", ...] = ()


@dataclass(frozen=True, slots=True)
class ExportedVariant:
    document: ChartDocument
    path: Path
    sha256: str
