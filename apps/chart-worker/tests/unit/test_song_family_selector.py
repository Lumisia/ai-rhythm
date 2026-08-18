from itertools import permutations

import pytest

from chart_worker.validation.song_family_selector import (
    CandidateSnapshot,
    ProtectedMetrics,
    TimingSectionSnapshot,
    compare_song_families,
)


def candidate(
    key_mode: int,
    difficulty: str,
    suffix: str,
    *,
    context_id: str = "ctx",
    first_row_ms: int = 1_000,
    precision: float = 0.9,
    v2_score: float = 1.0,
    review_rank: int = 0,
    hard_eligible: bool = True,
    section_rows: int = 32,
    attempt: int = 1,
) -> CandidateSnapshot:
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
            matched_precision_50=precision,
            active_gap_count=0,
            hold_integrity_violations=0,
            review_rank=review_rank,
        ),
        difficulty_ordering_score=v2_score,
        first_row_ms=first_row_ms,
        timing_sections=(TimingSectionSnapshot(section_rows, precision),) * 3,
        candidate_payload_ref=f"payload/{key_mode}/{difficulty}/{suffix}",
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
