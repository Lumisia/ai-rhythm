import dataclasses
import json
from copy import deepcopy

import pytest

from chart_worker.analysis.intro_anchor import IntroAnchorEvidence
from chart_worker.validation.family_evidence_v3 import (
    CandidateFamilyEvidenceV3,
    CandidateSafetyEvidenceV3,
    GapIntervalEvidence,
    IntroCandidateVoteV3,
    SongSelectionEvidenceV3,
    build_intro_selection_evidence,
    compare_gap_evidence,
    parse_song_selection_evidence_v3,
)


def gap(
    start_ms: int,
    end_ms: int,
    *,
    position: str = "MIDDLE",
    opportunity_kind: str = "ATTACK_REQUIRED",
) -> GapIntervalEvidence:
    return GapIntervalEvidence(
        start_ms=start_ms,
        end_ms=end_ms,
        position=position,
        active_onset_count=10,
        active_frame_ratio=0.75,
        opportunity_kind=opportunity_kind,
        local_audio_evidence_digest="a" * 64,
    )


def candidate(*gaps: GapIntervalEvidence) -> CandidateSafetyEvidenceV3:
    return CandidateSafetyEvidenceV3(
        candidate_id="candidate-1",
        structure_safe=True,
        timing_identity_safe=True,
        song_bounds_safe=True,
        serialization_safe=True,
        publication_tier="PRODUCTION_CANDIDATE",
        model_backed=True,
        recovery_trust_rank=0,
        active_gaps=tuple(gaps),
    )


def anchor(
    status: str,
    *,
    anchor_ms: int | None = None,
    anchor_grid_ms: int | None = None,
) -> IntroAnchorEvidence:
    return IntroAnchorEvidence(
        status=status,
        anchor_ms=anchor_ms,
        anchor_grid_ms=anchor_grid_ms,
        grid_distance_ms=(
            None if anchor_ms is None or anchor_grid_ms is None else abs(anchor_ms - anchor_grid_ms)
        ),
        aggregate_percentile_rank=0.95 if anchor_ms is not None else None,
        prominent_band_count=2 if anchor_ms is not None else 0,
        pulse_continuation_matches=3 if anchor_ms is not None else 0,
        pulse_continuation_opportunities=4 if anchor_ms is not None else 0,
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("start_ms", True),
        ("end_ms", 2.5),
        ("active_onset_count", False),
        ("active_frame_ratio", 1),
        ("position", "UNKNOWN"),
        ("opportunity_kind", "MAYBE"),
        ("local_audio_evidence_digest", "bad"),
    ],
)
def test_gap_interval_rejects_noncanonical_values(field, value):
    values = {
        "start_ms": 1_000,
        "end_ms": 9_000,
        "position": "MIDDLE",
        "active_onset_count": 8,
        "active_frame_ratio": 0.5,
        "opportunity_kind": "ATTACK_REQUIRED",
        "local_audio_evidence_digest": "a" * 64,
    }
    values[field] = value

    with pytest.raises((TypeError, ValueError)):
        GapIntervalEvidence(**values)


def test_candidate_gap_intervals_must_be_sorted_and_disjoint():
    with pytest.raises(ValueError, match="sorted and disjoint"):
        candidate(gap(10_000, 20_000), gap(5_000, 12_000))


def test_equal_gap_count_with_longer_gap_is_a_regression():
    current = candidate(gap(10_000, 18_000))
    challenger = candidate(gap(9_000, 29_000))

    comparison = compare_gap_evidence(current, challenger)

    assert comparison.status == "REGRESSION"
    assert "TOTAL_DURATION_INCREASED" in comparison.reasons
    assert "MAX_DURATION_INCREASED" in comparison.reasons


def test_gap_that_shrinks_inside_same_interval_is_non_regressing():
    current = candidate(gap(10_000, 20_000))
    challenger = candidate(gap(12_000, 18_000))

    comparison = compare_gap_evidence(current, challenger)

    assert comparison.status == "NON_REGRESSION"
    assert comparison.reasons == ()


def test_shifted_gap_is_incomparable_even_when_shorter():
    current = candidate(gap(10_000, 20_000))
    challenger = candidate(gap(30_000, 35_000))

    comparison = compare_gap_evidence(current, challenger)

    assert comparison.status == "INCOMPARABLE"
    assert comparison.reasons == ("UNMATCHED_GAP_INTERVAL",)


def test_new_leading_gap_is_not_hidden_by_equal_count():
    current = candidate(gap(10_000, 18_000, position="MIDDLE"))
    challenger = candidate(gap(0, 8_000, position="LEADING"))

    comparison = compare_gap_evidence(current, challenger)

    assert comparison.status == "REGRESSION"
    assert "LEADING_GAP_COUNT_INCREASED" in comparison.reasons


def test_confirmed_audio_reference_is_independent_of_candidate_votes():
    evidence_a = build_intro_selection_evidence(
        anchor("CONFIRMED", anchor_ms=1_030, anchor_grid_ms=1_000),
        active_onset_ms=(1_030, 2_000),
        votes=(IntroCandidateVoteV3("4K:EASY", "a", 5_000),),
    )
    evidence_b = build_intro_selection_evidence(
        anchor("CONFIRMED", anchor_ms=1_030, anchor_grid_ms=1_000),
        active_onset_ms=(1_030, 2_000),
        votes=(
            IntroCandidateVoteV3("4K:EASY", "b", 7_000),
            IntroCandidateVoteV3("6K:HARD", "c", 9_000),
        ),
    )

    assert evidence_a.reference_state == "CONFIRMED_AUDIO"
    assert evidence_a.reference_first_row_ms == 1_000
    assert evidence_a.audio_evidence_digest == evidence_b.audio_evidence_digest
    assert evidence_a.authorizes_first_row_change is True


def test_duplicate_attempts_in_one_slot_count_as_one_consensus_vote():
    evidence = build_intro_selection_evidence(
        anchor("NON_RHYTHMIC"),
        active_onset_ms=(),
        votes=(
            IntroCandidateVoteV3("4K:EASY", "attempt-1", 2_000),
            IntroCandidateVoteV3("4K:EASY", "attempt-2", 2_000),
            IntroCandidateVoteV3("6K:EASY", "attempt-1", 2_000),
        ),
    )

    assert evidence.reference_state == "CROSS_SLOT_CONSENSUS"
    assert evidence.reference_first_row_ms == 2_000
    assert evidence.consensus_support_count == 2
    assert evidence.authorizes_first_row_change is False


def test_conflicting_attempts_make_that_slot_abstain():
    evidence = build_intro_selection_evidence(
        anchor("UNCERTAIN", anchor_ms=980, anchor_grid_ms=1_000),
        active_onset_ms=(980,),
        votes=(
            IntroCandidateVoteV3("4K:EASY", "attempt-1", 1_000),
            IntroCandidateVoteV3("4K:EASY", "attempt-2", 2_000),
            IntroCandidateVoteV3("6K:EASY", "attempt-1", 2_000),
            IntroCandidateVoteV3("7K:EASY", "attempt-1", 2_000),
        ),
    )

    assert evidence.reference_state == "CROSS_SLOT_CONSENSUS"
    assert evidence.reference_first_row_ms == 2_000
    assert evidence.consensus_support_count == 2
    assert evidence.abstaining_slots == ("4K:EASY",)
    assert evidence.authorizes_first_row_change is False


def test_intro_and_gap_evidence_is_frozen_and_json_canonical():
    evidence = build_intro_selection_evidence(
        anchor("CONFIRMED", anchor_ms=1_030, anchor_grid_ms=1_000),
        active_onset_ms=(1_030,),
        votes=(),
    )
    safety = candidate(gap(10_000, 18_000))

    with pytest.raises(dataclasses.FrozenInstanceError):
        evidence.reference_first_row_ms = 5_000
    with pytest.raises(dataclasses.FrozenInstanceError):
        safety.active_gaps = ()

    json.dumps(evidence.to_report(), allow_nan=False)
    json.dumps(safety.to_report(), allow_nan=False)
    assert len(evidence.stable_sha256()) == 64
    assert len(safety.stable_sha256()) == 64


def test_song_evidence_binds_candidate_payload_and_keeps_selection_report_only():
    intro = build_intro_selection_evidence(
        anchor("CONFIRMED", anchor_ms=1_030, anchor_grid_ms=1_000),
        active_onset_ms=(1_030,),
        votes=(),
    )
    record = CandidateFamilyEvidenceV3(
        candidate_id="candidate-1",
        key_mode=4,
        difficulty="EASY",
        provenance="PRIMARY",
        candidate_payload_ref="candidate-payloads/a.osu",
        candidate_payload_sha256="b" * 64,
        safety=candidate(),
        first_row_ms=1_000,
        first_row_audio_supported=True,
        first_row_grid_distance_ms=0,
        intro_reference_state=intro.reference_state,
    )
    report = SongSelectionEvidenceV3(
        context_id="context-1",
        intro_selection=intro,
        candidates=(record,),
        current_assignment=(("4K:EASY", "candidate-1"),),
    )

    projected = report.to_report()
    assert projected["mutatesSelection"] is False
    assert projected["additionalModelCalls"] == 0
    assert projected["candidates"][0]["candidatePayloadSha256"] == "b" * 64
    assert projected["candidates"][0]["candidateRole"] == "PLAYTEST_POOL"
    assert projected["introSelectionSha256"] == intro.stable_sha256()
    assert len(report.stable_sha256()) == 64
    json.dumps(projected, allow_nan=False)


def test_candidate_record_rejects_mismatched_safety_identity():
    with pytest.raises(ValueError, match="candidate identity"):
        CandidateFamilyEvidenceV3(
            candidate_id="different",
            key_mode=4,
            difficulty="EASY",
            provenance="PRIMARY",
            candidate_payload_ref="candidate-payloads/a.osu",
            candidate_payload_sha256="b" * 64,
            safety=candidate(),
            first_row_ms=1_000,
            first_row_audio_supported=True,
            first_row_grid_distance_ms=0,
            intro_reference_state="CONFIRMED_AUDIO",
        )


def _song_evidence_report() -> dict[str, object]:
    intro = build_intro_selection_evidence(
        anchor("CONFIRMED", anchor_ms=1_030, anchor_grid_ms=1_000),
        active_onset_ms=(1_030,),
        votes=(),
    )
    record = CandidateFamilyEvidenceV3(
        candidate_id="candidate-1",
        key_mode=4,
        difficulty="EASY",
        provenance="PRIMARY",
        candidate_payload_ref="raw/candidates/sha256/" + "b" * 64 + ".osu",
        candidate_payload_sha256="b" * 64,
        safety=candidate(),
        first_row_ms=1_000,
        first_row_audio_supported=True,
        first_row_grid_distance_ms=0,
        intro_reference_state=intro.reference_state,
        matched_f1_50=0.8,
        matched_precision_50=0.9,
    )
    return SongSelectionEvidenceV3(
        context_id="context-1",
        intro_selection=intro,
        candidates=(record,),
        current_assignment=(("4K:EASY", "candidate-1"),),
    ).to_report()


def test_song_evidence_strict_parser_round_trips_canonical_report():
    report = _song_evidence_report()

    parsed = parse_song_selection_evidence_v3(report)

    assert parsed.to_report() == report


def test_song_evidence_strict_parser_preserves_legacy_candidate_without_role():
    report = _song_evidence_report()
    report["candidates"][0].pop("candidateRole")

    parsed = parse_song_selection_evidence_v3(report)

    assert parsed.candidates[0].candidate_role is None
    assert parsed.to_report() == report


def test_candidate_role_distinguishes_shadow_challenger():
    report = _song_evidence_report()
    report["candidates"][0]["candidateRole"] = "SHADOW_CHALLENGER"

    parsed = parse_song_selection_evidence_v3(report)

    assert parsed.candidates[0].candidate_role == "SHADOW_CHALLENGER"
    assert parsed.to_report() == report


@pytest.mark.parametrize(
    "mutate",
    [
        lambda value: value.update({"unexpected": True}),
        lambda value: value.pop("contextId"),
        lambda value: value.update({"mutatesSelection": True}),
        lambda value: value["introSelection"].update({"referenceFirstRowMs": True}),
        lambda value: value["candidates"][0]["safety"].update({"hardSafe": False}),
        lambda value: value["candidates"][0]["safety"]["activeGaps"].append({}),
        lambda value: value["candidates"][0].update({"candidateRole": "PUBLICATION"}),
    ],
)
def test_song_evidence_strict_parser_rejects_malformed_or_inconsistent_report(mutate):
    report = deepcopy(_song_evidence_report())
    mutate(report)

    with pytest.raises((TypeError, ValueError)):
        parse_song_selection_evidence_v3(report)
