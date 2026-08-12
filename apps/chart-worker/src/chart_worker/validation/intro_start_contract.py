"""Song-wide exact first-row contract derived from model and audio evidence."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

from chart_worker.analysis.intro_anchor import GRID_SUPPORT_WINDOW_MS
from chart_worker.analysis.song_context import SongAnalysisContext
from chart_worker.schema.note import Chart, NoteEvent


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
class IntroStartContract:
    version: Literal["intro-start-contract-v2"]
    canonical_first_row_ms: int | None
    candidate_support_count: int
    raw_supported: bool
    audio_supported: bool
    grid_distance_ms: int | None
    candidates: tuple[IntroCandidateView, ...]

    def to_report(self) -> dict[str, object]:
        return {
            "version": self.version,
            "canonicalFirstRowMs": self.canonical_first_row_ms,
            "candidateSupportCount": self.candidate_support_count,
            "rawSupported": self.raw_supported,
            "audioSupported": self.audio_supported,
            "gridDistanceMs": self.grid_distance_ms,
            "candidates": [candidate.to_report() for candidate in self.candidates],
        }


@dataclass(frozen=True, slots=True)
class IntroContractReview:
    status: Literal["PASS", "REVIEW"]
    mismatches: tuple[tuple[int, str, int | None], ...]
    corrected_count: int = 0
    correction_reasons: tuple[str, ...] = ()

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


def build_intro_start_contract(
    song_context: SongAnalysisContext,
    candidates: tuple[IntroCandidateView, ...],
) -> IntroStartContract:
    """Choose one evidence-backed timestamp without changing any chart."""
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
            version="intro-start-contract-v2",
            canonical_first_row_ms=None,
            candidate_support_count=0,
            raw_supported=False,
            audio_supported=False,
            grid_distance_ms=None,
            candidates=candidates,
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
        version="intro-start-contract-v2",
        canonical_first_row_ms=canonical,
        candidate_support_count=support_count,
        raw_supported=raw_supported,
        audio_supported=audio_rank > 0,
        grid_distance_ms=grid_distance,
        candidates=candidates,
    )


def validate_exact_first_row(
    contract: IntroStartContract,
    selected: tuple[IntroCandidateView, ...],
    *,
    corrected_count: int = 0,
    correction_reasons: tuple[str, ...] = (),
) -> IntroContractReview:
    canonical = contract.canonical_first_row_ms
    mismatches = tuple(
        (candidate.key_mode, candidate.difficulty, candidate.first_row_ms)
        for candidate in selected
        if canonical is None or candidate.first_row_ms != canonical
    )
    return IntroContractReview(
        status="REVIEW" if mismatches else "PASS",
        mismatches=mismatches,
        corrected_count=corrected_count,
        correction_reasons=correction_reasons,
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
