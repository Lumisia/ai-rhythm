"""Song-wide exact first-row contract derived from model and audio evidence."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

from chart_worker.analysis.intro_anchor import GRID_SUPPORT_WINDOW_MS
from chart_worker.analysis.song_context import SongAnalysisContext
from chart_worker.schema.note import Chart, NoteEvent
from chart_worker.validation.intro_region_contract import (
    IntroRegionContract,
    build_intro_region_contract,
    review_intro_region_candidate,
)

INTRO_START_CONTRACT_V2 = "intro-start-contract-v2"
INTRO_START_CONTRACT_V3 = "intro-start-contract-v3"
IntroStartResolution = Literal["RESOLVED", "AMBIGUOUS", "UNAVAILABLE"]


@dataclass(frozen=True, slots=True)
class IntroCandidateView:
    key_mode: int
    difficulty: str
    first_row_ms: int | None
    seed: int | None
    raw_supported: bool
    audio_supported: bool

    def to_report(self) -> dict[str, object]:
        return {
            "keyMode": self.key_mode,
            "difficulty": self.difficulty,
            "firstRowMs": self.first_row_ms,
            "seed": self.seed,
            "rawSupported": self.raw_supported,
            "audioSupported": self.audio_supported,
        }


@dataclass(frozen=True, slots=True)
class IntroCandidateCluster:
    first_row_ms: int
    support_count: int
    audio_support_count: int
    candidates: tuple[IntroCandidateView, ...]

    @property
    def audio_supported(self) -> bool:
        return self.audio_support_count > 0

    def to_report(self) -> dict[str, object]:
        return {
            "firstRowMs": self.first_row_ms,
            "supportCount": self.support_count,
            "audioSupportCount": self.audio_support_count,
            "audioSupported": self.audio_supported,
            "candidates": [candidate.to_report() for candidate in self.candidates],
        }


@dataclass(frozen=True, slots=True)
class IntroStartContract:
    version: Literal["intro-start-contract-v2", "intro-start-contract-v3"]
    canonical_first_row_ms: int | None
    candidate_support_count: int
    raw_supported: bool
    audio_supported: bool
    grid_distance_ms: int | None
    candidates: tuple[IntroCandidateView, ...]
    intro_region: IntroRegionContract | None = None
    resolution: IntroStartResolution = "RESOLVED"
    conflict_clusters: tuple[IntroCandidateCluster, ...] = ()

    def to_report(self) -> dict[str, object]:
        return {
            "version": self.version,
            "canonicalFirstRowMs": self.canonical_first_row_ms,
            "candidateSupportCount": self.candidate_support_count,
            "rawSupported": self.raw_supported,
            "audioSupported": self.audio_supported,
            "gridDistanceMs": self.grid_distance_ms,
            "candidates": [candidate.to_report() for candidate in self.candidates],
            "resolution": self.resolution,
            "conflictClusters": [cluster.to_report() for cluster in self.conflict_clusters],
            "introRegion": (
                self.intro_region.to_report()
                if self.intro_region is not None
                else None
            ),
        }


@dataclass(frozen=True, slots=True)
class IntroContractReview:
    status: Literal["PASS", "REVIEW"]
    mismatches: tuple[tuple[int, str, int | None], ...]
    corrected_count: int = 0
    correction_reasons: tuple[str, ...] = ()
    exact_mismatches: tuple[tuple[int, str, int | None], ...] = ()
    reasons: tuple[str, ...] = ()

    def to_report(self) -> dict[str, object]:
        return {
            "status": self.status,
            "mismatches": [
                {
                    "keyMode": key_mode,
                    "difficulty": difficulty,
                    "firstRowMs": first_row_ms,
                }
                for key_mode, difficulty, first_row_ms in self.mismatches
            ],
            "correctedCount": self.corrected_count,
            "correctionReasons": list(self.correction_reasons),
            "exactMismatches": [
                {
                    "keyMode": key_mode,
                    "difficulty": difficulty,
                    "firstRowMs": first_row_ms,
                }
                for key_mode, difficulty, first_row_ms in self.exact_mismatches
            ],
            "reasons": list(self.reasons),
        }


@dataclass(frozen=True, slots=True)
class IntroAlignmentEvidence:
    raw_supported: bool
    audio_supported: bool


@dataclass(frozen=True, slots=True)
class IntroAlignmentResult:
    notes: Chart
    status: Literal["ALIGNED", "UNCHANGED"]
    reason: str


def _has_onset(context: SongAnalysisContext, time_ms: int) -> bool:
    return any(
        abs(onset_ms - time_ms) <= GRID_SUPPORT_WINDOW_MS
        for onset_ms in context.onset_analysis.onset_ms
    )


def _audio_rank(context: SongAnalysisContext, time_ms: int) -> int:
    anchor = context.intro_anchor
    if (
        anchor.status == "CONFIRMED"
        and anchor.anchor_ms is not None
        and abs(anchor.anchor_ms - time_ms) <= GRID_SUPPORT_WINDOW_MS
    ):
        return 2
    return int(_has_onset(context, time_ms))


def _grid_distance_ms(context: SongAnalysisContext, time_ms: int) -> int:
    event = context.tempo_map.at(time_ms)
    half_beat_ms = 30_000.0 / event.bpm
    step = round((time_ms - event.time_ms) / half_beat_ms)
    grid_ms = max(0, round(event.time_ms + step * half_beat_ms))
    return abs(time_ms - grid_ms)


def _candidate_clusters(
    candidates: tuple[IntroCandidateView, ...],
) -> tuple[IntroCandidateCluster, ...]:
    present = sorted(
        (candidate for candidate in candidates if candidate.first_row_ms is not None),
        key=lambda candidate: (
            candidate.first_row_ms,
            candidate.key_mode,
            candidate.difficulty,
            candidate.seed if candidate.seed is not None else -1,
        ),
    )
    grouped: list[list[IntroCandidateView]] = []
    for candidate in present:
        first_row_ms = candidate.first_row_ms
        assert first_row_ms is not None
        if (
            not grouped
            or first_row_ms - (grouped[-1][0].first_row_ms or 0)
            > GRID_SUPPORT_WINDOW_MS
        ):
            grouped.append([candidate])
        else:
            grouped[-1].append(candidate)
    return tuple(
        IntroCandidateCluster(
            first_row_ms=group[0].first_row_ms or 0,
            support_count=len(group),
            audio_support_count=sum(candidate.audio_supported for candidate in group),
            candidates=tuple(group),
        )
        for group in grouped
    )


def build_intro_start_contract(
    song_context: SongAnalysisContext,
    candidates: tuple[IntroCandidateView, ...],
) -> IntroStartContract:
    """Choose one evidence-backed timestamp without changing any chart."""
    intro_region = build_intro_region_contract(song_context)
    proposal_times = {
        candidate.first_row_ms
        for candidate in candidates
        if candidate.first_row_ms is not None and candidate.raw_supported
    }
    anchor = song_context.intro_anchor
    if anchor.status == "CONFIRMED" and anchor.anchor_grid_ms is not None:
        proposal_times.add(anchor.anchor_grid_ms)
    if not proposal_times:
        return IntroStartContract(
            version=INTRO_START_CONTRACT_V3,
            canonical_first_row_ms=None,
            candidate_support_count=0,
            raw_supported=False,
            audio_supported=False,
            grid_distance_ms=None,
            candidates=candidates,
            intro_region=intro_region,
            resolution="UNAVAILABLE",
        )

    clusters = _candidate_clusters(candidates)
    if intro_region.status == "UNKNOWN" and len(clusters) >= 2:
        support_cluster = min(
            clusters,
            key=lambda cluster: (-cluster.support_count, cluster.first_row_ms),
        )
        audio_cluster = min(
            clusters,
            key=lambda cluster: (
                -cluster.audio_support_count,
                -cluster.support_count,
                cluster.first_row_ms,
            ),
        )
        if audio_cluster.audio_supported and audio_cluster != support_cluster:
            return IntroStartContract(
                version=INTRO_START_CONTRACT_V3,
                canonical_first_row_ms=None,
                candidate_support_count=0,
                raw_supported=False,
                audio_supported=False,
                grid_distance_ms=None,
                candidates=candidates,
                intro_region=intro_region,
                resolution="AMBIGUOUS",
                conflict_clusters=clusters,
            )

    def proposal_evidence(time_ms: int) -> tuple[int, int, int]:
        support = sum(
            candidate.first_row_ms is not None
            and abs(candidate.first_row_ms - time_ms) <= GRID_SUPPORT_WINDOW_MS
            for candidate in candidates
        )
        audio_rank = max(
            _audio_rank(song_context, time_ms),
            int(
                any(
                    candidate.first_row_ms == time_ms and candidate.audio_supported
                    for candidate in candidates
                )
            ),
        )
        return support, audio_rank, _grid_distance_ms(song_context, time_ms)

    present = tuple(
        candidate for candidate in candidates if candidate.first_row_ms is not None
    )
    unanimous_times = {candidate.first_row_ms for candidate in present}
    unanimous_cross_key_audio_row = (
        len(present) >= 4
        and len({candidate.key_mode for candidate in present}) >= 2
        and len(unanimous_times) == 1
        and all(candidate.raw_supported and candidate.audio_supported for candidate in present)
    )
    if unanimous_cross_key_audio_row:
        canonical = next(iter(unanimous_times))
        assert canonical is not None
    else:
        canonical = min(
            proposal_times,
            key=lambda time_ms: (
                -proposal_evidence(time_ms)[1],
                -proposal_evidence(time_ms)[0],
                proposal_evidence(time_ms)[2],
                time_ms,
            ),
        )
    support_count, audio_rank, grid_distance = proposal_evidence(canonical)
    raw_supported = any(
        candidate.raw_supported
        and candidate.first_row_ms is not None
        and abs(candidate.first_row_ms - canonical) <= GRID_SUPPORT_WINDOW_MS
        for candidate in candidates
    )
    return IntroStartContract(
        version=INTRO_START_CONTRACT_V3,
        canonical_first_row_ms=canonical,
        candidate_support_count=support_count,
        raw_supported=raw_supported,
        audio_supported=audio_rank > 0,
        grid_distance_ms=grid_distance,
        candidates=candidates,
        intro_region=intro_region,
    )


def validate_exact_first_row(
    contract: IntroStartContract,
    selected: tuple[IntroCandidateView, ...],
    *,
    corrected_count: int = 0,
    correction_reasons: tuple[str, ...] = (),
) -> IntroContractReview:
    if contract.version == INTRO_START_CONTRACT_V3 and contract.resolution == "AMBIGUOUS":
        return IntroContractReview(
            status="REVIEW",
            mismatches=(),
            corrected_count=corrected_count,
            correction_reasons=correction_reasons,
            exact_mismatches=(),
            reasons=("AMBIGUOUS_INTRO_START_EVIDENCE",),
        )
    canonical = contract.canonical_first_row_ms
    exact_mismatches = tuple(
        (candidate.key_mode, candidate.difficulty, candidate.first_row_ms)
        for candidate in selected
        if canonical is None or candidate.first_row_ms != canonical
    )
    if (
        contract.version == INTRO_START_CONTRACT_V3
        and contract.intro_region is not None
        and contract.intro_region.status == "CONFIRMED"
    ):
        mismatches = tuple(
            (candidate.key_mode, candidate.difficulty, candidate.first_row_ms)
            for candidate in selected
            if review_intro_region_candidate(
                contract.intro_region,
                first_row_ms=candidate.first_row_ms,
            ).status
            == "DEFECT"
        )
    else:
        mismatches = exact_mismatches
    return IntroContractReview(
        status="REVIEW" if mismatches else "PASS",
        mismatches=mismatches,
        corrected_count=corrected_count,
        correction_reasons=correction_reasons,
        exact_mismatches=exact_mismatches,
    )


def _same_lane_next_time(notes: Chart, first: NoteEvent) -> int | None:
    return min(
        (
            note.time_ms
            for note in notes
            if note.lane == first.lane and note.time_ms > first.time_ms
        ),
        default=None,
    )


def align_first_row(
    notes: Chart,
    contract: IntroStartContract,
    evidence: IntroAlignmentEvidence,
) -> IntroAlignmentResult:
    """Move only the first row when the common timestamp is structurally safe."""
    canonical = contract.canonical_first_row_ms
    if canonical is None or not notes:
        return IntroAlignmentResult(notes, "UNCHANGED", "NO_CANONICAL_FIRST_ROW")
    if not evidence.raw_supported and not evidence.audio_supported:
        return IntroAlignmentResult(notes, "UNCHANGED", "CANONICAL_TIME_UNSUPPORTED")
    first_time = min(note.time_ms for note in notes)
    if first_time == canonical:
        return IntroAlignmentResult(notes, "UNCHANGED", "ALREADY_EXACT")
    later_times = sorted({note.time_ms for note in notes if note.time_ms > first_time})
    if canonical < 0 or (later_times and canonical >= later_times[0]):
        return IntroAlignmentResult(notes, "UNCHANGED", "WOULD_CROSS_SECOND_ROW")

    first_row = [note for note in notes if note.time_ms == first_time]
    for note in first_row:
        next_time = _same_lane_next_time(notes, note)
        new_end_ms = canonical + (note.duration_ms or 0)
        if next_time is not None and new_end_ms > next_time:
            return IntroAlignmentResult(notes, "UNCHANGED", "WOULD_OVERLAP_SAME_LANE")
    first_ids = {id(note) for note in first_row}
    aligned = [
        replace(note, time_ms=canonical) if id(note) in first_ids else note
        for note in notes
    ]
    aligned.sort(key=lambda note: (note.time_ms, note.lane))
    return IntroAlignmentResult(aligned, "ALIGNED", "FIRST_ROW_ALIGNED")
