from dataclasses import replace

import numpy as np

from chart_worker.analysis.intro_anchor import IntroAnchorEvidence
from chart_worker.analysis.onset import OnsetAnalysis
from chart_worker.analysis.song_context import LocalTempoMap, SongAnalysisContext
from chart_worker.generation.osu_parser import OsuBpmEvent
from chart_worker.schema.note import NoteEvent
from chart_worker.validation.intro_start_contract import (
    IntroAlignmentEvidence,
    IntroCandidateView,
    align_first_row,
    build_intro_start_contract,
    validate_exact_first_row,
)


def context(
    *,
    first_timing_ms: int = 500,
    onsets: tuple[int, ...] = (100, 500, 1_000),
) -> SongAnalysisContext:
    analysis = OnsetAnalysis(
        sample_rate_hz=1_000,
        hop_length=10,
        strength=np.ones(201),
        band_strength=np.ones((3, 201)),
        onset_ms=onsets,
        n_fft=10,
    )
    return SongAnalysisContext(
        duration_ms=2_000,
        tempo_map=LocalTempoMap((OsuBpmEvent(first_timing_ms, 120.0),)),
        onset_analysis=analysis,
        intro_anchor=IntroAnchorEvidence(
            status="CONFIRMED",
            anchor_ms=100,
            anchor_grid_ms=0,
            grid_distance_ms=100,
            aggregate_percentile_rank=0.99,
            prominent_band_count=3,
            pulse_continuation_matches=3,
            pulse_continuation_opportunities=4,
        ),
    )


def candidate(
    key_mode: int,
    difficulty: str,
    first_row_ms: int | None,
    *,
    seed: int = 1,
    audio_supported: bool | None = None,
) -> IntroCandidateView:
    return IntroCandidateView(
        key_mode=key_mode,
        difficulty=difficulty,
        first_row_ms=first_row_ms,
        seed=seed,
        raw_supported=first_row_ms is not None,
        audio_supported=(
            first_row_ms in {100, 500}
            if audio_supported is None
            else audio_supported
        ),
    )


def test_contract_rejects_near_but_not_exact_first_rows():
    candidates = tuple(
        candidate(4, f"D{index}", 500 if index < 11 else 501, seed=index)
        for index in range(12)
    )
    contract = build_intro_start_contract(context(), candidates)

    review = validate_exact_first_row(contract, candidates)

    assert contract.canonical_first_row_ms == 500
    assert review.status == "REVIEW"
    assert review.mismatches == ((4, "D11", 501),)


def test_audio_supported_time_before_first_timing_point_can_be_canonical():
    candidates = (
        candidate(4, "EASY", 100),
        candidate(6, "EASY", 100),
        candidate(7, "EASY", 500),
    )

    contract = build_intro_start_contract(
        context(first_timing_ms=500, onsets=(100, 500)),
        candidates,
    )

    assert contract.canonical_first_row_ms == 100
    assert contract.audio_supported is True


def test_confirmed_audio_anchor_outweighs_a_raw_first_row_majority():
    song_context = replace(
        context(first_timing_ms=500, onsets=(500,)),
        intro_anchor=IntroAnchorEvidence(
            status="CONFIRMED",
            anchor_ms=500,
            anchor_grid_ms=500,
            grid_distance_ms=0,
            aggregate_percentile_rank=0.99,
            prominent_band_count=3,
            pulse_continuation_matches=3,
            pulse_continuation_opportunities=4,
        ),
    )
    candidates = tuple(candidate(4, f"D{index}", 1_000) for index in range(12))

    contract = build_intro_start_contract(song_context, candidates)

    assert contract.canonical_first_row_ms == 500
    assert contract.raw_supported is False
    assert contract.audio_supported is True


def test_unanimous_cross_key_audio_backed_model_row_outweighs_audio_anchor():
    """One detector anchor must not create 11 false mismatches against raw consensus."""
    song_context = replace(
        context(first_timing_ms=500, onsets=(500, 685)),
        intro_anchor=IntroAnchorEvidence(
            status="CONFIRMED",
            anchor_ms=500,
            anchor_grid_ms=500,
            grid_distance_ms=0,
            aggregate_percentile_rank=0.99,
            prominent_band_count=3,
            pulse_continuation_matches=3,
            pulse_continuation_opportunities=4,
        ),
    )
    candidates = tuple(
        candidate(key_mode, difficulty, 685, audio_supported=True)
        for key_mode in (4, 6, 7)
        for difficulty in ("EASY", "NORMAL", "HARD", "EXPERT")
        if not (key_mode == 7 and difficulty == "EASY")
    )

    contract = build_intro_start_contract(song_context, candidates)
    review = validate_exact_first_row(contract, candidates)

    assert contract.canonical_first_row_ms == 685
    assert contract.candidate_support_count == 11
    assert contract.raw_supported is True
    assert contract.audio_supported is True
    assert review.status == "PASS"


def test_split_candidate_rows_do_not_override_a_confirmed_anchor():
    song_context = replace(
        context(first_timing_ms=500, onsets=(500, 685, 750)),
        intro_anchor=IntroAnchorEvidence(
            status="CONFIRMED",
            anchor_ms=500,
            anchor_grid_ms=500,
            grid_distance_ms=0,
            aggregate_percentile_rank=0.99,
            prominent_band_count=3,
            pulse_continuation_matches=3,
            pulse_continuation_opportunities=4,
        ),
    )
    candidates = tuple(
        candidate(
            key_mode,
            difficulty,
            685 if index < 6 else 750,
            audio_supported=True,
        )
        for index, (key_mode, difficulty) in enumerate(
            (key_mode, difficulty)
            for key_mode in (4, 6, 7)
            for difficulty in ("EASY", "NORMAL", "HARD", "EXPERT")
        )
    )

    contract = build_intro_start_contract(song_context, candidates)

    assert contract.canonical_first_row_ms == 500


def test_first_row_alignment_changes_no_later_rows():
    notes = [NoteEvent(501, 0), NoteEvent(700, 1), NoteEvent(900, 2)]
    contract = build_intro_start_contract(
        context(),
        (candidate(4, "EASY", 500), candidate(6, "EASY", 500)),
    )

    result = align_first_row(
        notes,
        contract,
        IntroAlignmentEvidence(raw_supported=True, audio_supported=True),
    )

    assert result.status == "ALIGNED"
    assert result.notes[0] == NoteEvent(500, 0)
    assert result.notes[1:] == notes[1:]


def test_contract_never_aligns_without_raw_or_audio_support():
    notes = [NoteEvent(501, 0), NoteEvent(700, 1)]
    contract = build_intro_start_contract(
        context(),
        (candidate(4, "EASY", 500), candidate(6, "EASY", 500)),
    )

    result = align_first_row(
        notes,
        contract,
        IntroAlignmentEvidence(raw_supported=False, audio_supported=False),
    )

    assert result.status == "UNCHANGED"
    assert result.reason == "CANONICAL_TIME_UNSUPPORTED"
    assert result.notes == notes


def test_alignment_refuses_to_cross_the_second_row():
    notes = [NoteEvent(100, 0), NoteEvent(200, 1)]
    contract = build_intro_start_contract(
        context(),
        (candidate(4, "EASY", 500), candidate(6, "EASY", 500)),
    )

    result = align_first_row(
        notes,
        contract,
        IntroAlignmentEvidence(raw_supported=True, audio_supported=True),
    )

    assert result.status == "UNCHANGED"
    assert result.reason == "WOULD_CROSS_SECOND_ROW"
    assert result.notes == notes
