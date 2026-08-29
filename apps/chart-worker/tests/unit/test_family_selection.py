from pathlib import Path
from types import SimpleNamespace

import numpy as np

from chart_worker.analysis.activity import AudioActivity, SongBoundaryContract
from chart_worker.analysis.intro_anchor import IntroAnchorEvidence
from chart_worker.analysis.onset import OnsetAnalysis
from chart_worker.analysis.song_context import SongAnalysisContext
from chart_worker.generation.family_selection import (
    _safe_family_view,
    apply_safe_family_assignment,
    family_score,
)
from chart_worker.generation.osu_parser import OsuBpmEvent
from chart_worker.validation.intro_region_contract import IntroRegionContract
from chart_worker.validation.intro_start_contract import IntroStartContract
from chart_worker.validation.quality_gate import GateAction, GateAxis


def _candidate(
    *, provenance: str = "PRIMARY", intro_anchor_covered: bool | None = True
) -> SimpleNamespace:
    return SimpleNamespace(
        provenance=provenance,
        intro_anchor_covered=intro_anchor_covered,
        attempt=1,
        seed=7,
    )


def test_family_score_keeps_coverage_ahead_of_raw_candidate_penalty() -> None:
    complete_with_raw = (
        _candidate(),
        _candidate(),
        _candidate(),
        _candidate(provenance="RAW_UNVERIFIED"),
    )
    missing_one = (_candidate(), _candidate(), _candidate(), None)

    assert family_score(complete_with_raw, None) < family_score(missing_one, None)


def test_family_score_penalizes_confirmed_intro_miss_after_provenance() -> None:
    covered = (_candidate(),)
    missed = (_candidate(intro_anchor_covered=False),)

    assert family_score(covered, None) < family_score(missed, None)


class _Acceptance:
    def __init__(self, score: float) -> None:
        self.action = GateAction.PASS
        self.profile = SimpleNamespace(
            difficulty=SimpleNamespace(project_rating=score),
            difficulty_vector_v2=SimpleNamespace(ordering_score=score),
        )
        self.timing = SimpleNamespace(
            coverage_gaps=(),
            overall=SimpleNamespace(matched_f1_50=0.8),
        )

    def decision(self, _axis: GateAxis) -> SimpleNamespace:
        return SimpleNamespace(action=GateAction.PASS)


def _full_candidate(difficulty: str, score: float, first_row_ms: int):
    return SimpleNamespace(
        request=SimpleNamespace(key_mode=4, difficulty=difficulty),
        generated=SimpleNamespace(
            notes=[
                SimpleNamespace(
                    time_ms=first_row_ms,
                    kind="TAP",
                    duration_ms=None,
                ),
                SimpleNamespace(time_ms=50_000, kind="TAP", duration_ms=None),
            ]
        ),
        acceptance=_Acceptance(score),
        osu_text=f"payload-{difficulty}-{first_row_ms}",
        attempt=1,
        seed=int(score * 10),
        provenance="PRIMARY",
    )


def _boundary() -> SongBoundaryContract:
    return SongBoundaryContract(
        version="song-boundary-contract-v2",
        last_attack_ms=50_000,
        max_note_start_ms=50_070,
        release_end_ms=51_000,
        generation_end_ms=51_000,
        required_coverage_end_ms=50_000,
        quantization_tolerance_ms=70,
        hold_completion_guard_ms=10_000,
        policy_reason="TEST_TERMINAL_CONSENSUS",
        policy_applied=True,
        enforcement_mode="HIGH_CONFIDENCE_ENFORCED",
        effective_source="TERMINAL_SILENCE_CONSENSUS",
    )


def test_safe_family_adapter_preserves_unique_payload_when_intro_is_late() -> None:
    candidates = {
        "EASY": _full_candidate("EASY", 1.0, 1_000),
        "NORMAL": _full_candidate("NORMAL", 2.0, 1_000),
        "HARD": _full_candidate("HARD", 3.0, 1_000),
        "EXPERT": _full_candidate("EXPERT", 4.0, 20_000),
    }
    states = {
        difficulty: SimpleNamespace(
            key_mode=4,
            candidates=SimpleNamespace(playtest_candidates=(candidate,)),
            attempt_evidence=[],
        )
        for difficulty, candidate in candidates.items()
    }
    intro_contract = IntroStartContract(
        version="intro-start-contract-v2",
        canonical_first_row_ms=1_000,
        candidate_support_count=3,
        raw_supported=True,
        audio_supported=True,
        grid_distance_ms=0,
        candidates=(),
    )

    updated, decisions = apply_safe_family_assignment(
        [(states, dict(candidates), None)],
        run_dir=Path("."),
        intro_contract=intro_contract,
        boundary=_boundary(),
    )

    selected = updated[0][1]
    assert selected["EXPERT"] is candidates["EXPERT"]
    assert decisions[0].emergency_duplicate_slots == ()
    assert decisions[0].selected_score.intro_violations == 1
    assert states["EXPERT"].attempt_evidence == []


def test_safe_family_adapter_applies_final_order_without_mutating_payloads() -> None:
    candidates = {
        "EASY": _full_candidate("EASY", 2.65, 20_000),
        "NORMAL": _full_candidate("NORMAL", 1.94, 1_000),
        "HARD": _full_candidate("HARD", 3.68, 1_000),
        "EXPERT": _full_candidate("EXPERT", 4.96, 1_000),
    }
    states = {
        difficulty: SimpleNamespace(
            key_mode=4,
            candidates=SimpleNamespace(playtest_candidates=(candidate,)),
            attempt_evidence=[],
        )
        for difficulty, candidate in candidates.items()
    }
    intro_contract = IntroStartContract(
        version="intro-start-contract-v2",
        canonical_first_row_ms=1_000,
        candidate_support_count=3,
        raw_supported=True,
        audio_supported=True,
        grid_distance_ms=0,
        candidates=(),
    )
    payloads_before = {id(candidate): candidate.osu_text for candidate in candidates.values()}

    updated, decisions = apply_safe_family_assignment(
        [(states, dict(candidates), None)],
        run_dir=Path("."),
        intro_contract=intro_contract,
        boundary=_boundary(),
        post_resolution_ordering=True,
    )

    selected = updated[0][1]
    assert selected["EASY"] is candidates["NORMAL"]
    assert selected["NORMAL"] is candidates["EASY"]
    assert selected["HARD"] is candidates["HARD"]
    assert selected["EXPERT"] is candidates["EXPERT"]
    assert {id(candidate): candidate.osu_text for candidate in candidates.values()} == payloads_before
    assert decisions[0].post_resolution_ordering_status == "ORDERED"
    assert decisions[0].additional_model_calls == 0
    assert any(
        item["reason"] == "POST_RESOLUTION_DIFFICULTY_ORDERING_APPLIED"
        and item["sourceDifficulty"] == "NORMAL"
        for item in states["EASY"].attempt_evidence
    )


def test_safe_family_view_preserves_continuous_intro_and_tail_risk() -> None:
    candidate = _full_candidate("EASY", 1.0, 7_000)
    activity = AudioActivity(
        frame_ms=1_000.0,
        rms_db=np.array([-10.0] * 55 + [-80.0] * 5),
        floor_db=-40.0,
        active_onset_ms=(51_000, 53_000),
    )
    context = SongAnalysisContext.build(
        SimpleNamespace(bpm_events=(OsuBpmEvent(0, 120.0),)),
        OnsetAnalysis(
            sample_rate_hz=1_000,
            hop_length=1_000,
            strength=np.ones(60),
            band_strength=np.ones((3, 60)),
            onset_ms=(51_000, 53_000),
            activity=activity,
        ),
        duration_ms=60_000,
        intro_anchor=IntroAnchorEvidence(
            status="CONFIRMED",
            anchor_ms=1_000,
            anchor_grid_ms=1_000,
            grid_distance_ms=0,
            aggregate_percentile_rank=0.99,
            prominent_band_count=3,
            pulse_continuation_matches=3,
            pulse_continuation_opportunities=4,
            supported_pulse_ms=(1_000, 2_000, 3_000),
        ),
    )
    intro_contract = IntroStartContract(
        version="intro-start-contract-v2",
        canonical_first_row_ms=1_000,
        candidate_support_count=4,
        raw_supported=True,
        audio_supported=True,
        grid_distance_ms=0,
        candidates=(),
        intro_region=IntroRegionContract(
            version="intro-region-contract-v1",
            status="CONFIRMED",
            allowed_first_row_ms=(1_000, 5_000),
            leading_silence_end_ms=500,
            anchor_ms=1_000,
            anchor_grid_ms=1_000,
            supported_pulse_ms=(1_000, 2_000, 3_000),
            quantization_tolerance_ms=70,
            reasons=("FIXTURE",),
        ),
    )

    view = _safe_family_view(
        candidate,
        candidate_id="fixture",
        source_difficulty="EASY",
        intro_contract=intro_contract,
        boundary=SongBoundaryContract(
            version="song-boundary-contract-v2",
            last_attack_ms=54_000,
            max_note_start_ms=54_070,
            release_end_ms=56_000,
            generation_end_ms=56_000,
            required_coverage_end_ms=54_000,
            quantization_tolerance_ms=70,
            hold_completion_guard_ms=10_000,
            policy_reason="FIXTURE_PROVISIONAL",
            policy_applied=False,
            effective_source="FULL_DURATION_BASELINE",
        ),
        song_context=context,
    )

    assert view.intro_distance_ms == 2_000
    assert view.tail_coverage_deficit_ms == 5_000
    assert view.tail_active_onset_count == 2
    assert view.terminal_overflow_ms == 0
    assert view.terminal_overflow_confidence == "PROVISIONAL"


def test_intro_evidence_cannot_authorize_duplicate_sibling_payload() -> None:
    candidates = {
        "EASY": _full_candidate("EASY", 1.0, 903),
        "NORMAL": _full_candidate("NORMAL", 2.0, 211),
        "HARD": _full_candidate("HARD", 3.0, 211),
        "EXPERT": _full_candidate("EXPERT", 4.0, 20_748),
    }
    states = {
        difficulty: SimpleNamespace(
            key_mode=4,
            candidates=SimpleNamespace(playtest_candidates=(candidate,)),
            attempt_evidence=[],
        )
        for difficulty, candidate in candidates.items()
    }
    intro_contract = IntroStartContract(
        version="intro-start-contract-v2",
        canonical_first_row_ms=903,
        candidate_support_count=4,
        raw_supported=True,
        audio_supported=True,
        grid_distance_ms=1,
        candidates=(),
    )
    context = SongAnalysisContext.build(
        SimpleNamespace(bpm_events=(OsuBpmEvent(0, 150.0),)),
        OnsetAnalysis(
            sample_rate_hz=48_000,
            hop_length=512,
            strength=np.zeros(1),
            band_strength=np.zeros((3, 1)),
            onset_ms=(211, 903, 20_748),
        ),
        duration_ms=60_000,
        intro_anchor=IntroAnchorEvidence(
            status="CONFIRMED",
            anchor_ms=211,
            anchor_grid_ms=211,
            grid_distance_ms=0,
            aggregate_percentile_rank=0.99,
            prominent_band_count=3,
            pulse_continuation_matches=3,
            pulse_continuation_opportunities=4,
        ),
    )

    updated, decisions = apply_safe_family_assignment(
        [(states, dict(candidates), None)],
        run_dir=Path("."),
        intro_contract=intro_contract,
        boundary=_boundary(),
        song_context=context,
    )

    selected = updated[0][1]
    assert selected["NORMAL"] is candidates["NORMAL"]
    assert selected["HARD"] is candidates["HARD"]
    assert selected["EXPERT"] is candidates["EXPERT"]
    assert decisions[0].emergency_duplicate_slots == ()
