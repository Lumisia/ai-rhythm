from dataclasses import replace

import numpy as np

from chart_worker.analysis.intro_anchor import IntroAnchorEvidence
from chart_worker.analysis.onset import OnsetAnalysis
from chart_worker.analysis.song_context import LocalTempoMap, SongAnalysisContext
from chart_worker.generation.osu_parser import OsuBpmEvent
from chart_worker.schema.note import NoteEvent
from chart_worker.validation.intro_region_contract import IntroRegionContract
from chart_worker.validation.intro_start_contract import (
    IntroAlignmentEvidence,
    IntroCandidateView,
    IntroStartContract,
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


def test_confirmed_region_accepts_distinct_first_rows_but_keeps_exact_observation():
    song_context = replace(
        context(first_timing_ms=442, onsets=(21, 224, 459, 683, 896)),
        intro_anchor=IntroAnchorEvidence(
            status="CONFIRMED",
            anchor_ms=21,
            anchor_grid_ms=0,
            grid_distance_ms=21,
            aggregate_percentile_rank=0.78,
            prominent_band_count=1,
            pulse_continuation_matches=4,
            pulse_continuation_opportunities=4,
            supported_pulse_ms=(0, 231, 462, 692, 923),
        ),
    )
    candidates = tuple(
        candidate(key_mode, difficulty, first_row_ms, audio_supported=True)
        for (key_mode, difficulty), first_row_ms in zip(
            (
                (key_mode, difficulty)
                for key_mode in (4, 6, 7)
                for difficulty in ("EASY", "NORMAL", "HARD", "EXPERT")
            ),
            (903, 211, 211, 95, 903, 903, 95, 95, 903, 672, 442, 95),
            strict=True,
        )
    )

    contract = build_intro_start_contract(song_context, candidates)
    review = validate_exact_first_row(contract, candidates)

    assert contract.version == "intro-start-contract-v3"
    assert contract.intro_region is not None
    assert contract.intro_region.allowed_first_row_ms == (0, 993)
    assert review.status == "PASS"
    assert review.mismatches == ()
    assert len(review.exact_mismatches) > 0


def test_legacy_v2_pass_keeps_exact_first_row_semantics():
    candidates = (
        candidate(4, "EASY", 100, audio_supported=True),
        candidate(4, "NORMAL", 500, audio_supported=True),
    )
    legacy_contract = IntroStartContract(
        version="intro-start-contract-v2",
        canonical_first_row_ms=500,
        candidate_support_count=1,
        raw_supported=True,
        audio_supported=True,
        grid_distance_ms=0,
        candidates=candidates,
        intro_region=IntroRegionContract(
            version="intro-region-contract-v1",
            status="CONFIRMED",
            allowed_first_row_ms=(0, 993),
            leading_silence_end_ms=None,
            anchor_ms=100,
            anchor_grid_ms=0,
            supported_pulse_ms=(0, 231, 462, 692, 923),
            quantization_tolerance_ms=70,
            reasons=("AUDIO_SUPPORTED_INTRO_REGION",),
        ),
    )

    review = validate_exact_first_row(legacy_contract, candidates)

    assert review.status == "REVIEW"
    assert review.mismatches == ((4, "EASY", 100),)
    assert review.exact_mismatches == review.mismatches


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


def test_unknown_audio_singleton_conflict_is_ambiguous():
    song_context = replace(
        context(first_timing_ms=0, onsets=(15_206,)),
        duration_ms=30_000,
        intro_anchor=IntroAnchorEvidence(
            status="UNCERTAIN",
            anchor_ms=15_206,
            anchor_grid_ms=15_000,
            grid_distance_ms=206,
            aggregate_percentile_rank=0.99,
            prominent_band_count=3,
            pulse_continuation_matches=0,
            pulse_continuation_opportunities=4,
        ),
    )
    identities = tuple(
        (key_mode, difficulty)
        for key_mode in (4, 6, 7)
        for difficulty in ("EASY", "NORMAL", "HARD", "EXPERT")
    )
    candidates = tuple(
        candidate(
            key_mode,
            difficulty,
            15_206 if (key_mode, difficulty) == (7, "EASY") else 5,
            seed=index,
            audio_supported=(key_mode, difficulty) == (7, "EASY"),
        )
        for index, (key_mode, difficulty) in enumerate(identities)
    )

    contract = build_intro_start_contract(song_context, candidates)
    review = validate_exact_first_row(contract, candidates)

    assert contract.resolution == "AMBIGUOUS"
    assert contract.canonical_first_row_ms is None
    assert [cluster.support_count for cluster in contract.conflict_clusters] == [11, 1]
    assert [cluster.audio_supported for cluster in contract.conflict_clusters] == [False, True]
    assert review.status == "REVIEW"
    assert review.mismatches == ()
    assert review.exact_mismatches == ()
    assert review.reasons == ("AMBIGUOUS_INTRO_START_EVIDENCE",)
    assert review.correction_reasons == ()


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
