import hashlib
import json
from itertools import permutations

import pytest

from chart_worker.validation.song_family_selector import (
    CandidateSnapshot,
    ProtectedMetrics,
    TimingSectionSnapshot,
    compare_song_families,
    replay_song_selection_report,
)


def candidate(
    key_mode: int,
    difficulty: str,
    suffix: str,
    *,
    context_id: str = "ctx",
    first_row_ms: int = 1_000,
    precision: float = 0.9,
    recall: float | None = None,
    f1: float | None = None,
    row_count: int = 100,
    onset_count: int = 100,
    matched_count: int | None = None,
    v2_score: float | None = 1.0,
    review_rank: int = 0,
    hard_eligible: bool = True,
    section_rows: int = 32,
    attempt: int = 1,
) -> CandidateSnapshot:
    if matched_count is None:
        matched_count = round(row_count * precision)
    if recall is None:
        recall = round(matched_count / onset_count, 6)
    if f1 is None:
        f1 = round(
            2 * precision * recall / (precision + recall),
            6,
        )
    payload_ref = f"payload/{key_mode}/{difficulty}/{suffix}"
    return CandidateSnapshot(
        candidate_id=f"{key_mode}k:{difficulty}:{suffix}",
        context_id=context_id,
        key_mode=key_mode,
        difficulty=difficulty,
        attempt=attempt,
        seed=int(suffix) if suffix.isdigit() else 1,
        provenance="PRIMARY",
        hard_eligible=hard_eligible,
        axis_actions=(("STRUCTURE", "PASS"),),
        protected_metrics=ProtectedMetrics(
            row_count=row_count,
            onset_count=onset_count,
            matched_count_50=matched_count,
            matched_precision_50=precision,
            matched_recall_50=recall,
            matched_f1_50=f1,
            active_gap_count=0,
            hold_integrity_violations=0,
            review_rank=review_rank,
        ),
        difficulty_ordering_score=v2_score,
        first_row_ms=first_row_ms,
        timing_sections=(TimingSectionSnapshot(section_rows, precision),) * 3,
        candidate_payload_ref=payload_ref,
        candidate_payload_sha256=hashlib.sha256(payload_ref.encode("utf-8")).hexdigest(),
    )


def complete_pools(*, bad_expert: bool = False):
    pools = {}
    current = {}
    for key_mode in (4, 6, 7):
        for index, difficulty in enumerate(("EASY", "NORMAL", "HARD", "EXPERT")):
            current_candidate = candidate(
                key_mode,
                difficulty,
                "1",
                precision=(0.40 if bad_expert and key_mode == 4 and difficulty == "EXPERT" else 0.9),
                v2_score=float(index + 1),
                section_rows=(
                    64
                    if bad_expert and key_mode == 4 and difficulty == "EXPERT"
                    else 32
                ),
            )
            options = [current_candidate]
            if key_mode == 4 and difficulty == "EXPERT":
                options.append(
                    candidate(
                        key_mode,
                        difficulty,
                        "2",
                        precision=0.88,
                        v2_score=float(index + 1),
                    )
                )
            pools[(key_mode, difficulty)] = tuple(options)
            current[(key_mode, difficulty)] = current_candidate.candidate_id
    return pools, current


def test_shadow_finds_song_wide_cross_key_improvement_without_changing_current():
    pools, current = complete_pools(bad_expert=True)

    selected, comparison = compare_song_families(
        pools,
        current,
        canonical_first_row_ms=1_000,
        mode="SHADOW_V2",
    )

    assert selected == current
    assert comparison.shadow_assignment["4K:EXPERT"].endswith(":2")
    assert comparison.current_score.cross_key_outliers == 1
    assert comparison.shadow_score.cross_key_outliers == 0


def test_selection_is_invariant_to_candidate_pool_permutation():
    pools, current = complete_pools(bad_expert=True)
    slot = (4, "EXPERT")
    shadows = set()
    for ordering in permutations(pools[slot]):
        permuted = dict(pools)
        permuted[slot] = tuple(ordering)
        _, comparison = compare_song_families(
            permuted,
            current,
            canonical_first_row_ms=1_000,
            mode="SHADOW_V2",
        )
        shadows.add(tuple(sorted(comparison.shadow_assignment.items())))

    assert len(shadows) == 1


def test_context_mismatch_is_rejected_instead_of_comparing_incomparable_candidates():
    pools, current = complete_pools()
    pools[(7, "NORMAL")] = (
        candidate(7, "NORMAL", "9", context_id="other", v2_score=2.0),
    )

    with pytest.raises(ValueError, match="evaluation contexts"):
        compare_song_families(
            pools,
            current,
            canonical_first_row_ms=1_000,
            mode="SHADOW_V2",
        )


def test_hard_ineligible_candidate_is_pruned_even_when_its_objective_is_better():
    pools, current = complete_pools()
    invalid = candidate(
        4,
        "EXPERT",
        "0",
        precision=1.0,
        v2_score=4.0,
        hard_eligible=False,
    )
    pools[(4, "EXPERT")] = (invalid, *pools[(4, "EXPERT")])

    _, comparison = compare_song_families(
        pools,
        current,
        canonical_first_row_ms=1_000,
        mode="SHADOW_V2",
    )

    assert comparison.shadow_assignment["4K:EXPERT"] != invalid.candidate_id
    assert invalid.candidate_id in comparison.pruned_candidate_ids


def test_slot_pruning_preserves_candidate_with_distinct_intro_row():
    pools, current = complete_pools()
    canonical = candidate(
        4,
        "EXPERT",
        "3",
        first_row_ms=1_000,
        precision=0.80,
        v2_score=4.0,
        attempt=2,
    )
    noncanonical = candidate(
        4,
        "EXPERT",
        "4",
        first_row_ms=1_200,
        precision=0.95,
        v2_score=4.0,
    )
    pools[(4, "EXPERT")] = (canonical, noncanonical)
    current[(4, "EXPERT")] = noncanonical.candidate_id

    _, comparison = compare_song_families(
        pools,
        current,
        canonical_first_row_ms=1_000,
        mode="SHADOW_V2",
    )

    assert canonical.candidate_id not in comparison.pruned_candidate_ids
    assert comparison.shadow_assignment["4K:EXPERT"] == canonical.candidate_id


def test_slot_pruning_preserves_candidate_with_distinct_difficulty_score():
    pools, current = complete_pools()
    ordered = candidate(
        4,
        "EXPERT",
        "3",
        precision=0.80,
        v2_score=4.0,
        attempt=2,
    )
    too_close = candidate(
        4,
        "EXPERT",
        "4",
        precision=0.95,
        v2_score=3.05,
    )
    pools[(4, "EXPERT")] = (ordered, too_close)
    current[(4, "EXPERT")] = too_close.candidate_id

    _, comparison = compare_song_families(
        pools,
        current,
        canonical_first_row_ms=1_000,
        mode="SHADOW_V2",
    )

    assert ordered.candidate_id not in comparison.pruned_candidate_ids
    assert comparison.shadow_assignment["4K:EXPERT"] == ordered.candidate_id


def test_v2_does_not_fix_difficulty_order_by_degrading_timing_precision():
    pools, current = complete_pools()
    current_expert = candidate(
        4,
        "EXPERT",
        "3",
        precision=0.95,
        v2_score=3.05,
    )
    ordered_but_worse = candidate(
        4,
        "EXPERT",
        "4",
        precision=0.70,
        v2_score=4.0,
    )
    pools[(4, "EXPERT")] = (current_expert, ordered_but_worse)
    current[(4, "EXPERT")] = current_expert.candidate_id

    selected, comparison = compare_song_families(
        pools,
        current,
        canonical_first_row_ms=1_000,
        mode="V2",
    )

    assert selected[(4, "EXPERT")] == current_expert.candidate_id
    assert comparison.shadow_assignment["4K:EXPERT"] == current_expert.candidate_id


def test_v2_hard_ineligible_current_does_not_bypass_timing_regression_guard():
    pools, current = complete_pools()
    current_expert = candidate(
        4,
        "EXPERT",
        "3",
        precision=0.95,
        recall=0.95,
        f1=0.95,
        v2_score=3.05,
        hard_eligible=False,
    )
    hard_eligible_but_worse = candidate(
        4,
        "EXPERT",
        "4",
        precision=0.70,
        recall=0.70,
        f1=0.70,
        v2_score=4.0,
    )
    pools[(4, "EXPERT")] = (current_expert, hard_eligible_but_worse)
    current[(4, "EXPERT")] = current_expert.candidate_id

    selected, comparison = compare_song_families(
        pools,
        current,
        canonical_first_row_ms=1_000,
        mode="V2",
    )

    assert selected[(4, "EXPERT")] == current_expert.candidate_id
    assert hard_eligible_but_worse.candidate_id in comparison.pruned_candidate_ids


def test_v2_replaces_sparse_fallback_when_f1_is_near_tied_and_recall_recovers():
    pools, current = complete_pools()
    easy = candidate(4, "EASY", "7", v2_score=84.6202)
    sparse_fallback = candidate(
        4,
        "NORMAL",
        "8",
        precision=0.778626,
        recall=0.561468,
        f1=0.652452,
        row_count=393,
        onset_count=545,
        matched_count=306,
        v2_score=44.89936,
        review_rank=0,
        hard_eligible=False,
    )
    coverage_repair = candidate(
        4,
        "NORMAL",
        "9",
        precision=0.534063,
        recall=0.805505,
        f1=0.642282,
        row_count=822,
        onset_count=545,
        matched_count=439,
        v2_score=118.385436,
        review_rank=1,
        hard_eligible=True,
    )
    hard = candidate(4, "HARD", "10", v2_score=130.0)
    expert = candidate(4, "EXPERT", "11", v2_score=165.0)
    for difficulty, item in (
        ("EASY", easy),
        ("NORMAL", sparse_fallback),
        ("HARD", hard),
        ("EXPERT", expert),
    ):
        pools[(4, difficulty)] = (item,)
        current[(4, difficulty)] = item.candidate_id
    pools[(4, "NORMAL")] = (sparse_fallback, coverage_repair)

    selected, comparison = compare_song_families(
        pools,
        current,
        canonical_first_row_ms=1_000,
        mode="V2",
    )

    assert selected[(4, "NORMAL")] == coverage_repair.candidate_id
    assert comparison.current_score.difficulty_violations == 1
    assert comparison.shadow_score.difficulty_violations == 0
    assert comparison.shadow_score.worst_matched_f1_50 is not None


def test_v2_keeps_fallback_when_ordered_challenger_has_severe_f1_regression():
    pools, current = complete_pools()
    sparse_fallback = candidate(
        4,
        "EXPERT",
        "8",
        precision=0.779443,
        recall=0.66789,
        f1=0.719368,
        row_count=467,
        onset_count=545,
        matched_count=364,
        v2_score=3.05,
        hard_eligible=False,
    )
    timing_regression = candidate(
        4,
        "EXPERT",
        "9",
        precision=0.444327,
        recall=0.768807,
        f1=0.563172,
        row_count=943,
        onset_count=545,
        matched_count=419,
        v2_score=4.0,
        review_rank=1,
        hard_eligible=True,
    )
    pools[(4, "EXPERT")] = (sparse_fallback, timing_regression)
    current[(4, "EXPERT")] = sparse_fallback.candidate_id

    selected, comparison = compare_song_families(
        pools,
        current,
        canonical_first_row_ms=1_000,
        mode="V2",
    )

    assert selected[(4, "EXPERT")] == sparse_fallback.candidate_id
    assert timing_regression.candidate_id in comparison.pruned_candidate_ids


def test_v2_keeps_fallback_when_near_f1_challenger_loses_recall_and_matches():
    pools, current = complete_pools()
    sparse_fallback = candidate(
        4,
        "NORMAL",
        "8",
        precision=0.70,
        recall=0.70,
        f1=0.70,
        row_count=100,
        onset_count=100,
        matched_count=70,
        v2_score=0.0,
        hard_eligible=False,
    )
    lower_recall = candidate(
        4,
        "NORMAL",
        "9",
        precision=0.90,
        recall=0.65,
        f1=0.754839,
        row_count=65,
        onset_count=100,
        matched_count=65,
        v2_score=1.5,
    )
    pools[(4, "NORMAL")] = (sparse_fallback, lower_recall)
    current[(4, "NORMAL")] = sparse_fallback.candidate_id

    selected, comparison = compare_song_families(
        pools,
        current,
        canonical_first_row_ms=1_000,
        mode="V2",
    )

    assert selected[(4, "NORMAL")] == sparse_fallback.candidate_id
    assert lower_recall.candidate_id in comparison.pruned_candidate_ids


def test_v2_can_replace_one_fallback_while_retaining_another_current_fallback():
    pools, current = complete_pools()
    easy = candidate(4, "EASY", "7", v2_score=84.6202)
    normal_fallback = candidate(
        4,
        "NORMAL",
        "8",
        precision=0.778626,
        recall=0.561468,
        f1=0.652452,
        row_count=393,
        onset_count=545,
        matched_count=306,
        v2_score=44.89936,
        hard_eligible=False,
    )
    normal_repair = candidate(
        4,
        "NORMAL",
        "9",
        precision=0.534063,
        recall=0.805505,
        f1=0.642282,
        row_count=822,
        onset_count=545,
        matched_count=439,
        v2_score=118.385436,
        review_rank=1,
    )
    hard = candidate(4, "HARD", "10", v2_score=130.0)
    expert_fallback = candidate(
        4,
        "EXPERT",
        "11",
        precision=0.779443,
        recall=0.66789,
        f1=0.719368,
        row_count=467,
        onset_count=545,
        matched_count=364,
        v2_score=70.077782,
        hard_eligible=False,
    )
    expert_regression = candidate(
        4,
        "EXPERT",
        "12",
        precision=0.444327,
        recall=0.768807,
        f1=0.563172,
        row_count=943,
        onset_count=545,
        matched_count=419,
        v2_score=163.527976,
        review_rank=1,
    )
    for difficulty, item in (
        ("EASY", easy),
        ("NORMAL", normal_fallback),
        ("HARD", hard),
        ("EXPERT", expert_fallback),
    ):
        pools[(4, difficulty)] = (item,)
        current[(4, difficulty)] = item.candidate_id
    pools[(4, "NORMAL")] = (normal_fallback, normal_repair)
    pools[(4, "EXPERT")] = (expert_fallback, expert_regression)

    selected, comparison = compare_song_families(
        pools,
        current,
        canonical_first_row_ms=1_000,
        mode="V2",
    )

    assert selected[(4, "NORMAL")] == normal_repair.candidate_id
    assert selected[(4, "EXPERT")] == expert_fallback.candidate_id
    assert comparison.shadow_score.hard_violations == 1
    assert (
        comparison.shadow_score.difficulty_deficit
        < comparison.current_score.difficulty_deficit
    )


def test_v2_prefers_smaller_difficulty_deficit_when_violation_count_is_equal():
    pools, current = complete_pools()
    current_normal = candidate(4, "NORMAL", "8", v2_score=0.0)
    smaller_deficit = candidate(4, "NORMAL", "9", v2_score=0.8)
    pools[(4, "NORMAL")] = (current_normal, smaller_deficit)
    current[(4, "NORMAL")] = current_normal.candidate_id

    selected, comparison = compare_song_families(
        pools,
        current,
        canonical_first_row_ms=1_000,
        mode="V2",
    )

    assert comparison.current_score.difficulty_violations == 1
    assert comparison.shadow_score.difficulty_violations == 1
    assert selected[(4, "NORMAL")] == smaller_deficit.candidate_id
    assert comparison.reason == "DIFFICULTY_SEVERITY"


def test_v2_does_not_prefer_missing_difficulty_evidence_to_measured_inversion():
    pools, current = complete_pools()
    measured = candidate(4, "NORMAL", "8", v2_score=0.0)
    unknown = candidate(4, "NORMAL", "9", v2_score=None)
    pools[(4, "NORMAL")] = (measured, unknown)
    current[(4, "NORMAL")] = measured.candidate_id

    selected, comparison = compare_song_families(
        pools,
        current,
        canonical_first_row_ms=1_000,
        mode="V2",
    )

    assert comparison.current_score.difficulty_violations == 1
    assert selected[(4, "NORMAL")] == measured.candidate_id
    assert comparison.shadow_score.difficulty_unscored_pairs == 0


def test_selection_reason_identifies_f1_axis_without_zip_truncation():
    pools, current = complete_pools()
    lower_f1 = candidate(
        4,
        "EXPERT",
        "8",
        precision=0.9,
        recall=0.45,
        f1=0.6,
        row_count=100,
        onset_count=200,
        matched_count=90,
        v2_score=4.0,
    )
    higher_f1 = candidate(
        4,
        "EXPERT",
        "9",
        precision=0.9,
        recall=0.6,
        f1=0.72,
        row_count=100,
        onset_count=150,
        matched_count=90,
        v2_score=4.0,
    )
    pools[(4, "EXPERT")] = (lower_f1, higher_f1)
    current[(4, "EXPERT")] = lower_f1.candidate_id

    selected, comparison = compare_song_families(
        pools,
        current,
        canonical_first_row_ms=1_000,
        mode="V2",
    )

    assert selected[(4, "EXPERT")] == higher_f1.candidate_id
    assert comparison.reason == "WORST_MATCHED_F1"


def test_v2_hard_ineligible_current_allows_monotonic_hard_eligible_replacement():
    pools, current = complete_pools()
    current_expert = candidate(
        4,
        "EXPERT",
        "3",
        precision=0.90,
        v2_score=4.0,
        hard_eligible=False,
    )
    hard_eligible = candidate(
        4,
        "EXPERT",
        "4",
        precision=0.90,
        v2_score=4.0,
    )
    pools[(4, "EXPERT")] = (current_expert, hard_eligible)
    current[(4, "EXPERT")] = current_expert.candidate_id

    selected, comparison = compare_song_families(
        pools,
        current,
        canonical_first_row_ms=1_000,
        mode="V2",
    )

    assert selected[(4, "EXPERT")] == hard_eligible.candidate_id
    assert comparison.shadow_assignment["4K:EXPERT"] == hard_eligible.candidate_id


def test_missing_slot_tie_break_never_compares_none_with_string():
    pools, current = complete_pools()
    pools[(4, "EXPERT")] = ()
    current[(4, "EXPERT")] = None

    _, comparison = compare_song_families(
        pools,
        current,
        canonical_first_row_ms=1_000,
        mode="SHADOW_V2",
    )

    assert comparison.shadow_assignment["4K:EXPERT"] is None


def test_all_empty_pools_are_reported_without_context_error():
    pools = {
        (key_mode, difficulty): ()
        for key_mode in (4, 6, 7)
        for difficulty in ("EASY", "NORMAL", "HARD", "EXPERT")
    }
    current = {slot: None for slot in pools}

    selected, comparison = compare_song_families(
        pools,
        current,
        canonical_first_row_ms=None,
        mode="SHADOW_V2",
    )

    assert selected == current
    assert comparison.context_id == "EMPTY"
    assert comparison.current_score.missing_slots == 12


def test_report_contains_hashed_complete_ledger_that_replays_exact_selection():
    pools, current = complete_pools(bad_expert=True)

    selected, comparison = compare_song_families(
        pools,
        current,
        canonical_first_row_ms=1_000,
        mode="V2",
    )
    report = json.loads(json.dumps(comparison.to_report()))

    replayed, replay_comparison = replay_song_selection_report(report, mode="V2")

    assert report["replayInput"]["version"] == "song-selection-replay-v2"
    assert len(report["replayInput"]["candidates"]) == 13
    expert = next(
        item
        for item in report["replayInput"]["candidates"]
        if item["candidateId"] == "4k:EXPERT:1"
    )
    assert expert["timingSections"] == [
        {"rowCount": 64, "matchedPrecision50": 0.4},
        {"rowCount": 64, "matchedPrecision50": 0.4},
        {"rowCount": 64, "matchedPrecision50": 0.4},
    ]
    assert expert["protectedMetrics"] == {
        "rowCount": 100,
        "onsetCount": 100,
        "matchedCount50": 40,
        "matchedPrecision50": 0.4,
        "matchedRecall50": 0.4,
        "matchedF150": 0.4,
        "activeGapCount": 0,
        "holdIntegrityViolations": 0,
        "reviewRank": 0,
    }
    assert len(expert["candidatePayloadSha256"]) == 64
    assert len(report["replayInputSha256"]) == 64
    assert replayed == selected
    assert replay_comparison.shadow_assignment == comparison.shadow_assignment
    assert replay_comparison.shadow_score == comparison.shadow_score
    assert replay_comparison.replay_input_sha256 == report["replayInputSha256"]


def test_replay_rejects_tampered_candidate_evidence_before_selection():
    pools, current = complete_pools(bad_expert=True)
    _, comparison = compare_song_families(
        pools,
        current,
        canonical_first_row_ms=1_000,
        mode="V2",
    )
    report = json.loads(json.dumps(comparison.to_report()))
    report["replayInput"]["candidates"][0]["timingSections"][0][
        "matchedPrecision50"
    ] = 1.0

    with pytest.raises(ValueError, match="replay input digest"):
        replay_song_selection_report(report, mode="V2")


def test_replay_rejects_internally_inconsistent_matched_evidence():
    pools, current = complete_pools()
    _, comparison = compare_song_families(
        pools,
        current,
        canonical_first_row_ms=1_000,
        mode="V2",
    )
    report = json.loads(json.dumps(comparison.to_report()))
    metrics = report["replayInput"]["candidates"][0]["protectedMetrics"]
    metrics["matchedRecall50"] = 0.1
    serialized = json.dumps(
        report["replayInput"],
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    report["replayInputSha256"] = hashlib.sha256(serialized).hexdigest()

    with pytest.raises(ValueError, match="matchedRecall50"):
        replay_song_selection_report(report, mode="V2")


@pytest.mark.parametrize(
    "payload_sha256",
    [
        "",
        "A" * 64,
        "g" * 64,
        "0" * 63,
        True,
    ],
)
def test_replay_rejects_malformed_payload_hash_even_with_recomputed_ledger_digest(
    payload_sha256,
):
    pools, current = complete_pools()
    _, comparison = compare_song_families(
        pools,
        current,
        canonical_first_row_ms=1_000,
        mode="V2",
    )
    report = json.loads(json.dumps(comparison.to_report()))
    report["replayInput"]["candidates"][0][
        "candidatePayloadSha256"
    ] = payload_sha256
    serialized = json.dumps(
        report["replayInput"],
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    report["replayInputSha256"] = hashlib.sha256(serialized).hexdigest()

    with pytest.raises(ValueError, match="candidatePayloadSha256"):
        replay_song_selection_report(report, mode="V2")
