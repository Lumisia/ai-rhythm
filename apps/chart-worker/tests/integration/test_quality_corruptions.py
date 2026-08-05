import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from chart_worker.analysis.activity import AudioActivity
from chart_worker.analysis.onset import OnsetAnalysis
from chart_worker.generation.mapperatorinator import GeneratedChart
from chart_worker.generation.osu_parser import OsuBpmEvent
from chart_worker.schema.note import NoteEvent
from chart_worker.stages.types import SongTimingAuthority
from chart_worker.validation.quality_gate import GateAction, GateAxis, evaluate_chart_candidate

DURATION_MS = 60_000
CONTROL_ONSETS = (
    1_000,
    2_000,
    3_000,
    4_000,
    5_000,
    6_000,
    7_000,
    8_000,
    9_000,
    10_000,
    11_000,
    12_000,
    13_000,
    14_000,
    15_000,
    16_000,
    17_000,
    18_000,
    19_000,
    20_000,
    21_000,
    22_000,
    23_000,
    24_000,
    25_000,
    26_000,
    27_000,
    28_000,
    29_000,
    30_000,
    31_000,
    32_000,
    33_000,
    34_000,
    35_000,
    36_000,
    37_000,
    38_000,
    39_000,
    40_000,
    41_000,
    42_000,
    43_000,
    44_000,
    45_000,
    46_000,
    47_000,
    48_000,
    49_000,
    50_000,
    51_000,
    52_000,
    53_000,
    54_000,
    55_000,
    56_000,
    57_000,
    58_000,
    59_000,
)
AUTHORITY_EVENTS = (OsuBpmEvent(time_ms=0, bpm=120.0),)


def _authority() -> SongTimingAuthority:
    return SongTimingAuthority(
        reference_path=Path("timing-reference.osu"),
        sha256="literal-timing-reference",
        audio_sha256="literal-canonical-audio",
        bpm_events=AUTHORITY_EVENTS,
        generator_name="literal-test-authority",
        seed=41,
        mode="STANDARD",
        attempt_count=1,
    )


def _analysis() -> OnsetAnalysis:
    return OnsetAnalysis(
        sample_rate_hz=1_000,
        hop_length=1_000,
        strength=np.ones(61),
        band_strength=np.ones((3, 61)),
        onset_ms=CONTROL_ONSETS,
        activity=AudioActivity(
            frame_ms=1_000.0,
            rms_db=np.full(60, -10.0),
            floor_db=-20.0,
            active_onset_ms=CONTROL_ONSETS,
        ),
    )


def _control_notes() -> list[NoteEvent]:
    return [
        NoteEvent(
            time_ms=time_ms + offset_ms,
            lane=index % 4,
            kind="HOLD" if time_ms == 50_000 and offset_ms == 0 else "TAP",
            duration_ms=250 if time_ms == 50_000 and offset_ms == 0 else None,
        )
        for index, (time_ms, offset_ms) in enumerate(
            (time_ms, offset_ms)
            for time_ms in CONTROL_ONSETS
            for offset_ms in (-10, 0, 10)
        )
    ]


def _chart(
    notes: list[NoteEvent],
    *,
    bpm_events: tuple[OsuBpmEvent, ...] = AUTHORITY_EVENTS,
) -> GeneratedChart:
    return GeneratedChart(
        notes=notes,
        key_mode=4,
        osu_text="literal controlled-corruption chart",
        generator_name="literal-test-map",
        seed=73,
        bpm_events=bpm_events,
    )


def _note_projection(notes: list[NoteEvent]) -> tuple[tuple[object, ...], ...]:
    return tuple(
        (
            note.time_ms,
            note.lane,
            note.kind,
            note.duration_ms,
            note.onset_strength,
            note.band,
            note.is_downbeat,
            note.beat_fraction,
            note.section,
            note.origin_lane,
        )
        for note in notes
    )


def _note_bytes(notes: list[NoteEvent]) -> bytes:
    return json.dumps(
        _note_projection(notes), ensure_ascii=True, separators=(",", ":")
    ).encode("utf-8")


def _candidate(case: str) -> GeneratedChart:
    notes = _control_notes()
    if case == "leading-gap":
        notes = [note for note in notes if note.time_ms >= 12_000]
    elif case == "middle-gap":
        notes = [
            note for note in notes if not 25_000 <= note.time_ms < 35_000
        ]
    elif case == "global-shift":
        notes = [replace(note, time_ms=note.time_ms + 60) for note in notes]
    elif case == "section-shift":
        notes = [
            replace(note, time_ms=note.time_ms + 100)
            if 30_000 <= note.time_ms < 45_000
            else note
            for note in notes
        ]
    elif case == "timing-identity":
        return _chart(notes, bpm_events=(OsuBpmEvent(time_ms=0, bpm=121.0),))
    elif case != "control":
        raise ValueError(f"unknown corruption case: {case}")
    return _chart(notes)


PASS = ("PASS", ())


@pytest.mark.parametrize(
    ("case", "expected_action", "expected_decisions"),
    [
        pytest.param(
            "leading-gap",
            "RETRY_MAP",
            {
                "STRUCTURE": PASS,
                "TIMING_IDENTITY": PASS,
                "TIMING_ALIGNMENT": PASS,
                "COVERAGE": ("RETRY_MAP", ("ACTIVE_LEADING_GAP",)),
                "PATTERN": PASS,
            },
            id="front-12-seconds-deleted",
        ),
        pytest.param(
            "middle-gap",
            "RETRY_MAP",
            {
                "STRUCTURE": PASS,
                "TIMING_IDENTITY": PASS,
                "TIMING_ALIGNMENT": PASS,
                "COVERAGE": ("RETRY_MAP", ("ACTIVE_MIDDLE_GAP",)),
                "PATTERN": PASS,
            },
            id="middle-10-seconds-deleted",
        ),
        pytest.param(
            "global-shift",
            "RETRY_MAP",
            {
                "STRUCTURE": PASS,
                "TIMING_IDENTITY": PASS,
                "TIMING_ALIGNMENT": (
                    "RETRY_MAP",
                    ("OVERALL_TIMING_MISALIGNED",),
                ),
                "COVERAGE": PASS,
                "PATTERN": ("PASS", ("INSUFFICIENT_PROFILE_VARIATION",)),
            },
            id="global-plus-60ms",
        ),
        pytest.param(
            "section-shift",
            "RETRY_MAP",
            {
                "STRUCTURE": PASS,
                "TIMING_IDENTITY": PASS,
                "TIMING_ALIGNMENT": (
                    "RETRY_MAP",
                    ("SECTION_TIMING_MISALIGNED",),
                ),
                "COVERAGE": PASS,
                "PATTERN": PASS,
            },
            id="one-section-plus-100ms",
        ),
        pytest.param(
            "timing-identity",
            "RETRY_MAP",
            {
                "STRUCTURE": PASS,
                "TIMING_IDENTITY": (
                    "RETRY_MAP",
                    ("TIMING_REFERENCE_MISMATCH",),
                ),
                "TIMING_ALIGNMENT": PASS,
                "COVERAGE": PASS,
                "PATTERN": PASS,
            },
            id="bpm-identity-corrupted",
        ),
        pytest.param(
            "control",
            "PASS",
            {
                "STRUCTURE": PASS,
                "TIMING_IDENTITY": PASS,
                "TIMING_ALIGNMENT": PASS,
                "COVERAGE": PASS,
                "PATTERN": PASS,
            },
            id="unchanged-control",
        ),
    ],
)
def test_controlled_corruption_activates_only_its_independent_axis(
    case: str,
    expected_action: str,
    expected_decisions: dict[str, tuple[str, tuple[str, ...]]],
):
    chart = _candidate(case)
    note_values_before = _note_projection(chart.notes)
    note_bytes_before = _note_bytes(chart.notes)

    acceptance = evaluate_chart_candidate(
        chart,
        _authority(),
        _analysis(),
        requested_key_mode=4,
        requested_difficulty="NORMAL",
        duration_ms=DURATION_MS,
    )

    assert acceptance.action is GateAction(expected_action)
    assert {
        axis.value: (
            acceptance.decision(axis).action.value,
            acceptance.decision(axis).reasons,
        )
        for axis in GateAxis
    } == expected_decisions
    assert _note_projection(chart.notes) == note_values_before
    assert _note_bytes(chart.notes) == note_bytes_before

    report = json.loads(json.dumps(acceptance.to_report()))
    assert set(report["decisions"]) == {
        "STRUCTURE",
        "TIMING_IDENTITY",
        "TIMING_ALIGNMENT",
        "COVERAGE",
        "PATTERN",
    }
    assert report["action"] == expected_action
    assert {
        axis: (decision["action"], tuple(decision["reasons"]))
        for axis, decision in report["decisions"].items()
    } == expected_decisions
