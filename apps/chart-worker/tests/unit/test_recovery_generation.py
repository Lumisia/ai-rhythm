from collections import Counter
from dataclasses import replace
from itertools import pairwise
from pathlib import Path

import numpy as np
import pytest

from chart_worker.analysis.activity import AudioActivity
from chart_worker.analysis.onset import OnsetAnalysis
from chart_worker.generation.osu_parser import OsuBpmEvent
from chart_worker.generation.params import GenerationRequest
from chart_worker.generation.recovery import (
    RECOVERY_BUDGETS,
    build_recovery_chart,
    plan_recovery_rows,
    select_recovery_plan,
)
from chart_worker.rating.project_rating import measure_rating
from chart_worker.schema.types import DIFFICULTIES
from chart_worker.stages.types import SongTimingAuthority
from chart_worker.validation.generated_chart import validate_generated_chart
from chart_worker.validation.recovery_preflight import (
    RecoveryPreflightAction,
    review_recovery_preflight,
)

DURATION_MS = 20_000


def authority(*events: tuple[int, float]) -> SongTimingAuthority:
    bpm_events = events or ((0, 120.0),)
    return SongTimingAuthority(
        reference_path=Path("timing.osu"),
        sha256="timing",
        audio_sha256="audio",
        bpm_events=tuple(
            OsuBpmEvent(time_ms=time_ms, bpm=bpm)
            for time_ms, bpm in bpm_events
        ),
        generator_name="test",
        seed=0,
        mode="STANDARD",
        attempt_count=1,
    )


def request(
    *,
    key_mode: int = 4,
    difficulty: str = "NORMAL",
    duration_ms: int = DURATION_MS,
    generation_end_ms: int | None = None,
    last_attack_ms: int | None = None,
    max_note_start_ms: int | None = None,
) -> GenerationRequest:
    return GenerationRequest(
        audio_path=Path("audio.flac"),
        timing_reference_path=Path("timing.osu"),
        key_mode=key_mode,
        difficulty=difficulty,
        duration_ms=duration_ms,
        seed=7,
        generation_end_ms=generation_end_ms,
        last_attack_ms=last_attack_ms,
        max_note_start_ms=max_note_start_ms,
    )


def onsets(*, sustained: bool = True) -> OnsetAnalysis:
    frame_ms = 100
    frame_count = DURATION_MS // frame_ms
    strength = np.zeros(frame_count, dtype=np.float64)
    onset_ms = tuple(range(0, DURATION_MS, 125))
    for time_ms in onset_ms:
        strength[min(frame_count - 1, round(time_ms / frame_ms))] = 1.0
    activity = AudioActivity(
        frame_ms=frame_ms,
        rms_db=np.full(frame_count, -12.0 if sustained else -80.0),
        floor_db=-40.0,
        # Quiet mixes may still contain locally prominent attacks.  Keep those
        # attacks available while the low RMS prevents sustain-only HOLDs.
        active_onset_ms=onset_ms,
    )
    return OnsetAnalysis(
        sample_rate_hz=1_000,
        hop_length=100,
        strength=strength,
        band_strength=np.vstack((strength, strength, strength)),
        onset_ms=onset_ms,
        activity=activity,
    )


def corpus_onsets(duration_ms: int, *, onset_step_ms: int) -> OnsetAnalysis:
    frame_ms = 100
    frame_count = max(1, duration_ms // frame_ms)
    onset_ms = tuple(range(0, duration_ms, onset_step_ms))
    strength = np.zeros(frame_count, dtype=np.float64)
    for time_ms in onset_ms:
        strength[min(frame_count - 1, round(time_ms / frame_ms))] = 1.0
    activity = AudioActivity(
        frame_ms=frame_ms,
        rms_db=np.full(frame_count, -12.0),
        floor_db=-40.0,
        active_onset_ms=onset_ms,
    )
    return OnsetAnalysis(
        sample_rate_hz=1_000,
        hop_length=100,
        strength=strength,
        band_strength=np.vstack((strength, strength, strength)),
        onset_ms=onset_ms,
        activity=activity,
    )


def test_recovery_stops_rows_when_audio_evidence_ends_before_file_duration():
    active_end_ms = 12_000
    frame_ms = 100
    frame_count = DURATION_MS // frame_ms
    onset_ms = tuple(range(0, active_end_ms, 250))
    strength = np.zeros(frame_count, dtype=np.float64)
    for time_ms in onset_ms:
        strength[round(time_ms / frame_ms)] = 1.0
    rms_db = np.full(frame_count, -80.0)
    rms_db[: active_end_ms // frame_ms] = -12.0
    analysis = OnsetAnalysis(
        sample_rate_hz=1_000,
        hop_length=100,
        strength=strength,
        band_strength=np.vstack((strength, strength, strength)),
        onset_ms=onset_ms,
        activity=AudioActivity(
            frame_ms=frame_ms,
            rms_db=rms_db,
            floor_db=-40.0,
            active_onset_ms=onset_ms,
        ),
    )

    chart = build_recovery_chart(
        request(difficulty="EXPERT"),
        authority((0, 120.0)),
        analysis,
    )

    assert chart.notes
    assert max(note.time_ms for note in chart.notes) < active_end_ms


def test_recovery_keeps_rows_for_sustained_audio_without_detected_onsets():
    frame_ms = 100
    frame_count = DURATION_MS // frame_ms
    strength = np.zeros(frame_count, dtype=np.float64)
    analysis = OnsetAnalysis(
        sample_rate_hz=1_000,
        hop_length=100,
        strength=strength,
        band_strength=np.vstack((strength, strength, strength)),
        onset_ms=(),
        activity=AudioActivity(
            frame_ms=frame_ms,
            rms_db=np.full(frame_count, -12.0),
            floor_db=-40.0,
            active_onset_ms=(),
        ),
    )

    chart = build_recovery_chart(
        request(difficulty="EASY"),
        authority((0, 120.0)),
        analysis,
    )

    assert chart.notes
    assert min(note.time_ms for note in chart.notes) == 0
    assert max(note.time_ms for note in chart.notes) >= DURATION_MS - 500


def test_recovery_uses_every_tempo_segment_and_audio_bounds():
    chart = build_recovery_chart(
        request(),
        authority((0, 120.0), (10_000, 180.0)),
        onsets(),
    )

    validate_generated_chart(chart, key_mode=4, duration_ms=DURATION_MS)
    assert any(note.time_ms < 10_000 for note in chart.notes)
    assert any(note.time_ms >= 10_000 for note in chart.notes)
    assert chart.bpm_events == authority((0, 120.0), (10_000, 180.0)).bpm_events


def test_recovery_never_starts_notes_after_the_canonical_music_boundary():
    generation_request = request(
        generation_end_ms=13_000,
        last_attack_ms=11_500,
        max_note_start_ms=12_000,
    )

    chart = build_recovery_chart(
        generation_request,
        authority((0, 90.0), (10_000, 210.0)),
        onsets(),
    )

    validate_generated_chart(
        chart,
        key_mode=4,
        duration_ms=DURATION_MS,
        max_note_start_ms=12_000,
        max_hold_end_ms=13_000,
    )
    assert chart.notes
    assert max(note.time_ms for note in chart.notes) <= 12_000


@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_recovery_row_plan_preserves_existing_output(difficulty: str):
    generation_request = request(difficulty=difficulty)
    timing_authority = authority((0, 120.0), (10_000, 180.0))
    analysis = onsets()

    original = build_recovery_chart(generation_request, timing_authority, analysis)
    plan = plan_recovery_rows(generation_request, timing_authority, analysis)
    planned = build_recovery_chart(
        generation_request,
        timing_authority,
        analysis,
        plan=plan,
    )

    assert plan.rows == tuple(sorted({note.time_ms for note in original.notes}))
    assert planned == original


@pytest.mark.parametrize("key_mode", [4, 6, 7])
def test_recovery_balances_all_requested_lanes(key_mode: int):
    chart = build_recovery_chart(request(key_mode=key_mode), authority(), onsets())
    counts = Counter(note.lane for note in chart.notes)

    assert set(counts) == set(range(key_mode))
    assert max(counts.values()) - min(counts.values()) <= 1


def test_recovery_difficulty_burden_is_strictly_monotonic():
    charts = [
        build_recovery_chart(request(difficulty=difficulty), authority(), onsets())
        for difficulty in DIFFICULTIES
    ]
    ratings = [measure_rating(chart.notes, DURATION_MS).rating for chart in charts]

    assert all(left < right for left, right in pairwise(ratings))


def test_recovery_adds_hold_only_with_sustained_activity():
    dry = build_recovery_chart(
        request(difficulty="EXPERT"),
        authority(),
        onsets(sustained=False),
    )
    sustained = build_recovery_chart(
        request(difficulty="EXPERT"),
        authority(),
        onsets(sustained=True),
    )

    assert not any(note.kind == "HOLD" for note in dry.notes)
    assert any(note.kind == "HOLD" for note in sustained.notes)


def test_healthy_recovery_grid_keeps_the_difficulty_default():
    generation_request = request(difficulty="NORMAL")
    timing_authority = authority()
    analysis = onsets()
    preflight = review_recovery_preflight(
        timing_authority,
        analysis,
        duration_ms=DURATION_MS,
    )
    timing_authority = replace(timing_authority, recovery_preflight=preflight)

    plan = select_recovery_plan(generation_request, timing_authority, analysis)

    assert plan.subdivisions == RECOVERY_BUDGETS["NORMAL"].subdivisions
    assert plan.selection_reason == "DEFAULT"


def test_damaged_default_uses_the_nearest_preflight_viable_divisor():
    generation_request = request(difficulty="EASY")
    timing_authority = authority((0, 5.0))
    analysis = onsets()
    preflight = review_recovery_preflight(
        timing_authority,
        analysis,
        duration_ms=DURATION_MS,
    )
    easy = preflight.for_difficulty("EASY")
    assert easy.action is RecoveryPreflightAction.REVIEW
    assert easy.viable_divisors
    timing_authority = replace(timing_authority, recovery_preflight=preflight)

    plan = select_recovery_plan(generation_request, timing_authority, analysis)

    expected = min(
        easy.viable_divisors,
        key=lambda divisor: (
            abs(divisor - RECOVERY_BUDGETS["EASY"].subdivisions),
            divisor,
        ),
    )
    assert plan.subdivisions == expected
    assert plan.subdivisions != RECOVERY_BUDGETS["EASY"].subdivisions
    assert plan.selection_reason == "PREFLIGHT_ALTERNATE"
    assert plan.viable_divisors == easy.viable_divisors


def test_no_viable_alternate_keeps_default_for_final_gate_decision():
    generation_request = request(difficulty="EASY")
    timing_authority = authority((0, 0.25))
    analysis = onsets()
    preflight = review_recovery_preflight(
        timing_authority,
        analysis,
        duration_ms=DURATION_MS,
    )
    easy = preflight.for_difficulty("EASY")
    assert easy.action is RecoveryPreflightAction.DAMAGED
    timing_authority = replace(timing_authority, recovery_preflight=preflight)

    plan = select_recovery_plan(generation_request, timing_authority, analysis)

    assert plan.subdivisions == RECOVERY_BUDGETS["EASY"].subdivisions
    assert plan.selection_reason == "NO_VIABLE_ALTERNATE"
    assert plan.viable_divisors == ()


@pytest.mark.parametrize(
    ("duration_ms", "tempo_events", "onset_step_ms"),
    (
        (30_000, ((0, 60.0),), 500),
        (180_000, ((0, 120.0),), 250),
        (600_000, ((0, 240.0),), 125),
        (
            180_000,
            ((0, 90.0), (60_000, 180.0), (120_000, 72.0)),
            250,
        ),
    ),
)
def test_recovery_is_structurally_stable_across_duration_tempo_and_keycount(
    duration_ms: int,
    tempo_events: tuple[tuple[int, float], ...],
    onset_step_ms: int,
):
    timing_authority = authority(*tempo_events)
    analysis = corpus_onsets(duration_ms, onset_step_ms=onset_step_ms)

    for key_mode in (4, 6, 7):
        for difficulty in DIFFICULTIES:
            generation_request = request(
                key_mode=key_mode,
                difficulty=difficulty,
                duration_ms=duration_ms,
            )
            first = build_recovery_chart(
                generation_request,
                timing_authority,
                analysis,
            )
            second = build_recovery_chart(
                generation_request,
                timing_authority,
                analysis,
            )

            validate_generated_chart(
                first,
                key_mode=key_mode,
                duration_ms=duration_ms,
            )
            assert first == second
            assert {note.lane for note in first.notes} == set(range(key_mode))
