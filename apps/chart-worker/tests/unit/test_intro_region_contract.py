from __future__ import annotations

import numpy as np
import pytest

from chart_worker.analysis.activity import AudioActivity
from chart_worker.analysis.intro_anchor import IntroAnchorEvidence
from chart_worker.analysis.leading_silence import (
    DEFAULT_THRESHOLDS_DB,
    LeadingSilenceObservation,
    LeadingThresholdCandidate,
)
from chart_worker.analysis.onset import OnsetAnalysis
from chart_worker.analysis.song_context import LocalTempoMap, SongAnalysisContext
from chart_worker.generation.osu_parser import OsuBpmEvent
from chart_worker.validation.intro_region_contract import (
    build_intro_region_contract,
    review_intro_region_candidate,
)


def _context(
    *,
    anchor: IntroAnchorEvidence,
    leading: LeadingSilenceObservation | None = None,
    duration_ms: int = 30_000,
) -> SongAnalysisContext:
    analysis = OnsetAnalysis(
        sample_rate_hz=1_000,
        hop_length=10,
        strength=np.ones(duration_ms // 10 + 1),
        band_strength=np.ones((3, duration_ms // 10 + 1)),
        onset_ms=(21, 224, 459, 683, 896),
        n_fft=10,
        activity=AudioActivity(
            frame_ms=10.0,
            rms_db=np.full(duration_ms // 10 + 1, -20.0),
            floor_db=-40.0,
            active_onset_ms=(21, 896),
        ),
        leading_silence=leading,
    )
    return SongAnalysisContext(
        duration_ms=duration_ms,
        tempo_map=LocalTempoMap((OsuBpmEvent(442, 130.0),)),
        onset_analysis=analysis,
        intro_anchor=anchor,
    )


def _anchor(**overrides: object) -> IntroAnchorEvidence:
    values: dict[str, object] = {
        "status": "CONFIRMED",
        "anchor_ms": 21,
        "anchor_grid_ms": 0,
        "grid_distance_ms": 21,
        "aggregate_percentile_rank": 0.78,
        "prominent_band_count": 1,
        "pulse_continuation_matches": 4,
        "pulse_continuation_opportunities": 4,
        "supported_pulse_ms": (0, 231, 462, 692, 923),
    }
    values.update(overrides)
    return IntroAnchorEvidence(**values)  # type: ignore[arg-type]


def _leading(end_ms: int, *, first_onset_ms: int) -> LeadingSilenceObservation:
    return LeadingSilenceObservation(
        version="leading-silence-observation-v1",
        duration_ms=30_000,
        frame_ms=20,
        channel_count=2,
        candidates=tuple(
            LeadingThresholdCandidate(rms_db, peak_db, end_ms, end_ms)
            for rms_db, peak_db in DEFAULT_THRESHOLDS_DB
        ),
        candidate_spread_ms=0,
        first_onset_ms=first_onset_ms,
    )


@pytest.mark.parametrize("first_row_ms", [95, 211, 442, 672, 903])
def test_distinct_audio_supported_starts_share_one_intro_region(
    first_row_ms: int,
) -> None:
    contract = build_intro_region_contract(_context(anchor=_anchor()))

    review = review_intro_region_candidate(contract, first_row_ms=first_row_ms)

    assert contract.status == "CONFIRMED"
    assert contract.allowed_first_row_ms == (0, 993)
    assert review.status == "PASS"
    assert review.reason == "WITHIN_CONFIRMED_INTRO_REGION"


def test_v9_late_start_is_a_confirmed_defect() -> None:
    contract = build_intro_region_contract(_context(anchor=_anchor()))

    review = review_intro_region_candidate(contract, first_row_ms=20_748)

    assert review.status == "DEFECT"
    assert review.reason == "CONFIRMED_INTRO_REGION_MISSED"
    assert review.lateness_ms == 19_755


def test_confirmed_leading_silence_is_only_a_lower_bound() -> None:
    anchor = _anchor(
        anchor_ms=5_020,
        anchor_grid_ms=5_000,
        grid_distance_ms=20,
        supported_pulse_ms=(5_000, 5_250, 5_500),
    )
    contract = build_intro_region_contract(
        _context(anchor=anchor, leading=_leading(5_000, first_onset_ms=5_020))
    )

    early = review_intro_region_candidate(contract, first_row_ms=4_900)
    valid = review_intro_region_candidate(contract, first_row_ms=5_000)

    assert contract.allowed_first_row_ms == (5_000, 5_570)
    assert early.status == "DEFECT"
    assert early.reason == "NOTE_INSIDE_CONFIRMED_LEADING_SILENCE"
    assert valid.status == "PASS"


def test_uncertain_anchor_never_authorizes_candidate_mutation() -> None:
    contract = build_intro_region_contract(
        _context(anchor=_anchor(status="UNCERTAIN", supported_pulse_ms=()))
    )

    review = review_intro_region_candidate(contract, first_row_ms=20_748)

    assert contract.status == "UNKNOWN"
    assert contract.allowed_first_row_ms is None
    assert review.status == "UNKNOWN"
    assert review.reason == "INTRO_REGION_EVIDENCE_UNAVAILABLE"
