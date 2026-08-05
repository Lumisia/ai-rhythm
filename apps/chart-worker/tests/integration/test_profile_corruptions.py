from pathlib import Path

import numpy as np

from chart_worker.analysis.chart_profile import build_chart_quality_profile
from chart_worker.analysis.onset import OnsetAnalysis
from chart_worker.generation.mapperatorinator import GeneratedChart
from chart_worker.generation.osu_parser import OsuBpmEvent
from chart_worker.schema.note import NoteEvent
from chart_worker.stages.types import SongTimingAuthority
from chart_worker.validation.difficulty_order import review_difficulty_order
from chart_worker.validation.profile_review import review_profile
from chart_worker.validation.quality_gate import (
    GateAction,
    GateAxis,
    evaluate_chart_candidate,
)

DURATION_MS = 60_000
BEAT_MS = 500.0


def _tap(time_ms: int, lane: int) -> NoteEvent:
    return NoteEvent(time_ms=time_ms, lane=lane)


def _hold(time_ms: int, lane: int, duration_ms: int) -> NoteEvent:
    return NoteEvent(time_ms=time_ms, lane=lane, kind="HOLD", duration_ms=duration_ms)


def _profile(notes: list[NoteEvent]):
    return build_chart_quality_profile(
        notes,
        key_mode=4,
        duration_ms=DURATION_MS,
        beat_ms=BEAT_MS,
        activity=None,
    )


def _pattern_decision(notes: list[NoteEvent]):
    profile = _profile(notes)
    return review_profile(profile, key_mode=4, difficulty="HARD")[0]


def _section_notes(sequences: tuple[tuple[int, ...], ...]) -> list[NoteEvent]:
    return [
        _tap(section * 15_000 + index * 250, lane)
        for section, sequence in enumerate(sequences)
        for index, lane in enumerate(sequence)
    ]


def _balanced_sequence() -> tuple[int, ...]:
    return tuple(index % 4 for index in range(60))


def _lane_sequence(*, moved: int) -> tuple[int, ...]:
    sequence = list(_balanced_sequence())
    replaced = 0
    for index, lane in enumerate(sequence):
        if lane == 3 and replaced < moved:
            sequence[index] = 0
            replaced += 1
    return tuple(sequence)


def _base4_block(value: int) -> tuple[int, int, int, int]:
    return tuple((value // (4**power)) % 4 for power in (3, 2, 1, 0))


def _block_sequence(repeat_count: int) -> tuple[int, ...]:
    repeated = (0, 1, 2, 3) * repeat_count
    remaining_blocks = 15 - repeat_count
    tail = tuple(
        lane
        for block_index in range(32, 32 + remaining_blocks)
        for lane in _base4_block(block_index)
    )
    return repeated + tail


def test_single_section_hold_concentration_is_review_without_note_mutation():
    notes = [
        _hold(0, 0, 250),
        _tap(5_000, 3),
        _tap(10_000, 3),
        _hold(15_000, 0, 500),
        _tap(20_000, 3),
        _tap(25_000, 3),
        _hold(30_000, 0, 10_000),
        _hold(30_000, 1, 10_000),
        _hold(30_000, 2, 10_000),
        _tap(35_000, 3),
        _tap(40_000, 3),
        _hold(45_000, 0, 750),
        _tap(50_000, 3),
        _tap(55_000, 3),
    ]
    before = tuple(notes)

    decision = _pattern_decision(notes)

    assert decision.action is GateAction.REVIEW
    assert "HOLD_SECTION_OUTLIER" in decision.reasons
    assert tuple(notes) == before


def test_same_lane_hold_overlap_remains_a_structural_retry():
    notes = [_hold(1_000, 0, 2_000), _hold(2_000, 0, 500)]
    authority = SongTimingAuthority(
        reference_path=Path("timing-reference.osu"),
        sha256="reference",
        audio_sha256="audio",
        bpm_events=(OsuBpmEvent(0, 120.0),),
        generator_name="fixture",
        seed=0,
        mode="STANDARD",
        attempt_count=1,
    )
    analysis = OnsetAnalysis(
        sample_rate_hz=1_000,
        hop_length=100,
        strength=np.zeros(51),
        band_strength=np.zeros((3, 51)),
        onset_ms=(1_000, 2_000),
    )
    chart = GeneratedChart(
        notes=notes,
        key_mode=4,
        osu_text="",
        generator_name="fixture",
        seed=0,
        bpm_events=authority.bpm_events,
    )

    acceptance = evaluate_chart_candidate(
        chart,
        authority,
        analysis,
        requested_key_mode=4,
        requested_difficulty="HARD",
        duration_ms=5_000,
    )

    assert acceptance.action is GateAction.RETRY_MAP
    assert acceptance.decision(GateAxis.STRUCTURE).reasons == ("STRUCTURE_INVALID",)
    assert acceptance.profile is None


def test_one_lane_jack_concentration_is_review():
    notes = _section_notes(
        (
            _lane_sequence(moved=0),
            _lane_sequence(moved=1),
            (0,) * 60,
            _lane_sequence(moved=2),
        )
    )

    decision = _pattern_decision(notes)

    assert decision.action is GateAction.REVIEW
    assert "LANE_IMBALANCE_SECTION_OUTLIER" in decision.reasons


def test_four_row_loop_collapse_is_review():
    notes = _section_notes(
        (
            _block_sequence(1),
            _block_sequence(2),
            _block_sequence(15),
            _block_sequence(3),
        )
    )

    profile = _profile(notes)
    decision = review_profile(profile, key_mode=4, difficulty="HARD")[0]

    assert profile.pattern.section_longest_row_ngram_repeats[2] == 15
    assert decision.action is GateAction.REVIEW
    assert "ROW_LOOP_SECTION_OUTLIER" in decision.reasons


def test_relative_order_finds_hard_expert_inversion_from_measured_charts():
    steps = {"EASY": 1_000, "NORMAL": 500, "HARD": 125, "EXPERT": 250}
    profiles = {
        difficulty: _profile(
            [
                _tap(time_ms, index % 4)
                for index, time_ms in enumerate(range(0, DURATION_MS, step_ms))
            ]
        ).difficulty
        for difficulty, step_ms in steps.items()
    }

    review = review_difficulty_order(profiles)

    assert review.status == "RETRY"
    assert review.inverted_pairs == (("HARD", "EXPERT"),)
    assert review.retry_difficulties == frozenset({"HARD", "EXPERT"})


def test_unchanged_balanced_control_remains_pass():
    notes = _section_notes((_balanced_sequence(),) * 4)

    decision = _pattern_decision(notes)

    assert decision.action is GateAction.PASS
