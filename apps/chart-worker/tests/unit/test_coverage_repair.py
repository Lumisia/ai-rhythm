from pathlib import Path

import numpy as np
import pytest

from chart_worker.analysis.activity import AudioActivity
from chart_worker.analysis.onset import OnsetAnalysis
from chart_worker.generation.coverage_repair import build_coverage_repair_chart
from chart_worker.generation.mapperatorinator import GeneratedChart
from chart_worker.generation.osu_parser import OsuBpmEvent
from chart_worker.generation.params import GenerationRequest
from chart_worker.schema.note import NoteEvent
from chart_worker.stages.types import SongTimingAuthority
from chart_worker.validation.quality_gate import (
    GateAction,
    GateAxis,
    evaluate_chart_candidate,
)


def _authority(*, bpm: float = 120.0) -> SongTimingAuthority:
    return SongTimingAuthority(
        reference_path=Path("timing.osu"),
        sha256="timing",
        audio_sha256="audio",
        bpm_events=(OsuBpmEvent(time_ms=0, bpm=bpm),),
        generator_name="test",
        seed=0,
        mode="STANDARD",
        attempt_count=1,
    )


def _request(
    *,
    duration_ms: int = 40_000,
    key_mode: int = 4,
    difficulty: str = "EASY",
) -> GenerationRequest:
    return GenerationRequest(
        audio_path=Path("audio.flac"),
        timing_reference_path=Path("timing.osu"),
        key_mode=key_mode,
        difficulty=difficulty,
        duration_ms=duration_ms,
        seed=7,
    )


def _analysis(duration_ms: int, *, onset_step_ms: int = 500) -> OnsetAnalysis:
    frame_ms = 100
    frame_count = duration_ms // frame_ms + 1
    onset_ms = tuple(range(250, duration_ms, onset_step_ms))
    strength = np.zeros(frame_count, dtype=np.float64)
    for time_ms in onset_ms:
        strength[round(time_ms / frame_ms)] = 1.0
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


def _raw_chart(
    authority: SongTimingAuthority,
    *,
    key_mode: int,
    rows: tuple[int, ...],
) -> GeneratedChart:
    return GeneratedChart(
        notes=[
            NoteEvent(time_ms=time_ms, lane=index % key_mode)
            for index, time_ms in enumerate(rows)
        ],
        key_mode=key_mode,
        osu_text="",
        generator_name="raw-model",
        seed=7,
        bpm_events=authority.bpm_events,
    )


def _acceptance(
    request: GenerationRequest,
    chart: GeneratedChart,
    authority: SongTimingAuthority,
    analysis: OnsetAnalysis,
):
    return evaluate_chart_candidate(
        chart,
        authority,
        analysis,
        requested_key_mode=request.key_mode,
        requested_difficulty=request.difficulty,
        duration_ms=request.duration_ms,
    )


def test_repair_preserves_source_and_fills_one_bounded_active_gap():
    request = _request()
    authority = _authority()
    analysis = _analysis(request.duration_ms, onset_step_ms=1_000)
    active_rows = analysis.activity.active_onset_ms  # type: ignore[union-attr]
    gap_start_ms = 12_000
    gap_end_ms = 20_000
    source = _raw_chart(
        authority,
        key_mode=request.key_mode,
        rows=tuple(
            sorted(
                {
                    gap_start_ms,
                    gap_end_ms,
                    *(row for row in active_rows if not gap_start_ms < row < gap_end_ms),
                }
            )
        ),
    )
    acceptance = _acceptance(request, source, authority, analysis)
    assert acceptance.decision(GateAxis.COVERAGE).action is GateAction.RETRY_MAP
    assert len(acceptance.timing.coverage_gaps) == 1
    assert acceptance.timing.coverage_gaps[0].position == "MIDDLE"

    repaired, plan = build_coverage_repair_chart(
        request,
        source,
        acceptance,
        authority,
        analysis,
    )

    assert all(note in repaired.notes for note in source.notes)
    assert plan.repaired_gap_count == 1
    assert plan.inserted_notes
    assert all(note.kind == "TAP" for note in plan.inserted_notes)
    active_onsets = set(analysis.activity.active_onset_ms)  # type: ignore[union-attr]
    assert all(
        any(abs(note.time_ms - onset_ms) <= 70 for onset_ms in active_onsets)
        for note in plan.inserted_notes
    )
    gaps = acceptance.timing.coverage_gaps
    assert all(
        any(gap.start_ms < note.time_ms < gap.end_ms for gap in gaps)
        for note in plan.inserted_notes
    )
    assert repaired.bpm_events == source.bpm_events


def test_repair_rejects_large_tap_synthesis_so_partial_remap_can_take_over():
    request = _request()
    authority = _authority()
    analysis = _analysis(request.duration_ms)
    source = _raw_chart(
        authority,
        key_mode=request.key_mode,
        rows=(10_000, 20_000, 30_000),
    )
    acceptance = _acceptance(request, source, authority, analysis)

    with pytest.raises(ValueError, match="coverage repair exceeds"):
        build_coverage_repair_chart(
            request,
            source,
            acceptance,
            authority,
            analysis,
        )


def test_repair_rejects_a_candidate_without_attack_required_gaps():
    request = _request(duration_ms=20_000)
    authority = _authority()
    analysis = _analysis(request.duration_ms)
    source = _raw_chart(
        authority,
        key_mode=request.key_mode,
        rows=tuple(range(0, request.duration_ms, 500)),
    )
    acceptance = _acceptance(request, source, authority, analysis)
    assert acceptance.decision(GateAxis.COVERAGE).action is not GateAction.RETRY_MAP

    with pytest.raises(ValueError, match="attack-required coverage gap"):
        build_coverage_repair_chart(request, source, acceptance, authority, analysis)


def test_repair_never_inserts_a_tap_inside_an_open_hold_on_the_same_lane():
    request = _request()
    authority = _authority()
    analysis = _analysis(request.duration_ms, onset_step_ms=1_000)
    active_rows = analysis.activity.active_onset_ms  # type: ignore[union-attr]
    gap_start_ms = 12_000
    gap_end_ms = 20_000
    source = GeneratedChart(
        notes=[
            NoteEvent(time_ms=10_000, lane=0, kind="HOLD", duration_ms=20_000),
            *(
                NoteEvent(time_ms=row, lane=1 + index % 3)
                for index, row in enumerate(
                    sorted(
                        {
                            gap_start_ms,
                            gap_end_ms,
                            *(
                                row
                                for row in active_rows
                                if not gap_start_ms < row < gap_end_ms
                            ),
                        }
                    )
                )
            ),
        ],
        key_mode=4,
        osu_text="",
        generator_name="raw-model-with-hold",
        seed=7,
        bpm_events=authority.bpm_events,
    )
    acceptance = _acceptance(request, source, authority, analysis)
    assert acceptance.decision(GateAxis.COVERAGE).action is GateAction.RETRY_MAP

    repaired, plan = build_coverage_repair_chart(
        request,
        source,
        acceptance,
        authority,
        analysis,
    )

    assert plan.inserted_notes
    assert all(note.lane != 0 for note in plan.inserted_notes)
    reparsed = _acceptance(request, repaired, authority, analysis)
    assert reparsed.decision(GateAxis.STRUCTURE).action is GateAction.PASS


@pytest.mark.parametrize(
    ("duration_ms", "bpm", "key_mode", "difficulty", "onset_step_ms"),
    (
        (30_000, 60.0, 4, "EASY", 1_000),
        (180_000, 120.0, 6, "NORMAL", 500),
        (600_000, 240.0, 7, "EXPERT", 250),
    ),
)
def test_repair_is_deterministic_across_duration_bpm_keycount_and_difficulty(
    duration_ms: int,
    bpm: float,
    key_mode: int,
    difficulty: str,
    onset_step_ms: int,
):
    request = _request(
        duration_ms=duration_ms,
        key_mode=key_mode,
        difficulty=difficulty,
    )
    authority = _authority(bpm=bpm)
    analysis = _analysis(duration_ms, onset_step_ms=onset_step_ms)
    active_rows = analysis.activity.active_onset_ms  # type: ignore[union-attr]
    gap_start_ms = duration_ms // 2 - 4_000
    gap_end_ms = gap_start_ms + 8_000
    source = _raw_chart(
        authority,
        key_mode=key_mode,
        rows=tuple(
            sorted(
                {
                    gap_start_ms,
                    gap_end_ms,
                    *(
                        row
                        for row in active_rows
                        if not gap_start_ms < row < gap_end_ms
                    ),
                }
            )
        ),
    )
    acceptance = _acceptance(request, source, authority, analysis)

    first = build_coverage_repair_chart(
        request, source, acceptance, authority, analysis
    )
    second = build_coverage_repair_chart(
        request, source, acceptance, authority, analysis
    )

    assert first == second
    repaired, plan = first
    assert all(0 <= note.lane < key_mode for note in repaired.notes)
    assert all(0 <= note.time_ms < duration_ms for note in plan.inserted_notes)
