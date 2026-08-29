from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from chart_worker.analysis.timing_diagnostics import TimingCoverageGap
from chart_worker.errors import ErrorCode, WorkerError
from chart_worker.generation.candidate_state import Candidate
from chart_worker.generation.difficulty_shadow_challenger import (
    DifficultyShadowFullMapPlan,
    DifficultyShadowPartialPlan,
    DifficultyShadowSlot,
    DifficultyShadowTarget,
    bind_required_gameplay_interval,
    build_difficulty_shadow_full_map_request,
    choose_difficulty_shadow_target,
    difficulty_shadow_candidate_failure_reason,
    difficulty_shadow_full_map_candidate_failure_reason,
    difficulty_shadow_inference_failure_reason,
    difficulty_shadow_target_failure_reason,
    existing_candidate_resolves_fallback,
    plan_difficulty_shadow_full_map,
    plan_difficulty_shadow_partial_repair,
)
from chart_worker.generation.mapperatorinator import GeneratedChart
from chart_worker.generation.osu_parser import OsuBpmEvent
from chart_worker.generation.params import DESCRIPTORS, GenerationRequest
from chart_worker.generation.partial_remap import PartialRemapWindow
from chart_worker.generation.required_gameplay_interval import (
    RequiredGameplayEvidenceV1,
    RequiredGameplayFamilySlotV1,
    RequiredGameplayIntervalMode,
)
from chart_worker.schema.note import NoteEvent
from chart_worker.validation.quality_gate import GateAction, GateAxis


class FakeAcceptance:
    def __init__(
        self,
        *,
        rating: float,
        gaps: tuple[TimingCoverageGap, ...],
        matched_f1: float = 0.7,
        matched_precision: float = 0.8,
        hard_safe: bool = True,
        action: GateAction = GateAction.REVIEW,
        ordering_score: float | None = None,
    ) -> None:
        self.action = action
        self.profile = SimpleNamespace(
            difficulty=SimpleNamespace(project_rating=rating),
            difficulty_vector_v2=SimpleNamespace(
                ordering_score=rating if ordering_score is None else ordering_score
            ),
        )
        self.timing = SimpleNamespace(
            coverage_gaps=gaps,
            overall=SimpleNamespace(
                matched_f1_50=matched_f1,
                matched_precision_50=matched_precision,
            ),
        )
        self._hard_safe = hard_safe
        self.decisions = tuple(
            SimpleNamespace(axis=axis, action=self.decision(axis).action)
            for axis in GateAxis
        )

    def decision(self, axis: GateAxis) -> SimpleNamespace:
        action = (
            GateAction.PASS
            if self._hard_safe
            or axis not in {GateAxis.STRUCTURE, GateAxis.TIMING_IDENTITY, GateAxis.SONG_BOUNDS}
            else GateAction.RETRY_MAP
        )
        return SimpleNamespace(action=action)


def active_gap(
    start_ms: int,
    end_ms: int,
    position: str = "LEADING",
) -> TimingCoverageGap:
    return TimingCoverageGap(
        start_ms=start_ms,
        end_ms=end_ms,
        onset_count=20,
        active_onset_count=20,
        active_frame_ratio=1.0,
        position=position,  # type: ignore[arg-type]
    )


def candidate(
    *,
    provenance: str = "COVERAGE_REPAIR",
    rating: float = 4.96,
    first_rows: tuple[int, ...] = (12_697, 14_041),
    gaps: tuple[TimingCoverageGap, ...] | None = None,
    intro_anchor_covered: bool | None = False,
    matched_f1: float = 0.563172,
    matched_precision: float = 0.444327,
    hard_safe: bool = True,
    attempt: int = 2,
    seed: int = 13,
) -> Candidate:
    generated = GeneratedChart(
        notes=[NoteEvent(row, index % 4) for index, row in enumerate(first_rows)],
        key_mode=4,
        osu_text="fixture",
        generator_name="difficulty-shadow-plan-fixture",
        seed=seed,
        bpm_events=(OsuBpmEvent(0, 67.0),),
    )
    return Candidate(
        request=GenerationRequest(
            audio_path=Path("audio.wav"),
            timing_reference_path=Path("timing.osu"),
            key_mode=4,
            difficulty="EXPERT",
            seed=seed,
            duration_ms=323_453,
        ),
        generated=generated,
        acceptance=FakeAcceptance(
            rating=rating,
            gaps=(active_gap(0, 12_697),) if gaps is None else gaps,
            matched_f1=matched_f1,
            matched_precision=matched_precision,
            hard_safe=hard_safe,
        ),  # type: ignore[arg-type]
        osu_text="fixture",
        workdir=Path("work"),
        attempt=attempt,
        seed=seed,
        provenance=provenance,  # type: ignore[arg-type]
        intro_anchor_covered=intro_anchor_covered,
    )


def slot(
    key_mode: int,
    difficulty: str,
    rating: float,
    *,
    provenance: str = "PRIMARY",
    has_existing_safe_resolution: bool = False,
) -> DifficultyShadowSlot:
    return DifficultyShadowSlot(
        key_mode=key_mode,
        difficulty=difficulty,
        project_rating=rating,
        selected_provenance=provenance,
        has_existing_safe_resolution=has_existing_safe_resolution,
    )


def required_gameplay_evidence() -> RequiredGameplayEvidenceV1:
    return RequiredGameplayEvidenceV1(
        anchor_status="CONFIRMED",
        anchor_ms=500,
        anchor_grid_ms=500,
        aggregate_rank=0.95,
        prominent_band_count=2,
        pulse_support_count=2,
        family_slots=(
            RequiredGameplayFamilySlotV1(4, "HARD", False),
            RequiredGameplayFamilySlotV1(4, "EXPERT", True),
            RequiredGameplayFamilySlotV1(6, "HARD", True),
            RequiredGameplayFamilySlotV1(7, "EXPERT", True),
        ),
        local_audio_supported=True,
        reference_first_row_supported=True,
        repeated_high_confidence_refusal=False,
        timing_authority_valid=True,
        timing_authority_digest="a" * 64,
        anchor_evidence_digest="b" * 64,
    )


def test_strong_fallback_inversion_targets_the_harder_slot():
    target = choose_difficulty_shadow_target(
        (
            slot(4, "EASY", 1.0),
            slot(4, "NORMAL", 2.0),
            slot(4, "HARD", 3.68),
            slot(4, "EXPERT", 1.94, provenance="SAFE_FALLBACK"),
        )
    )

    assert target is not None
    assert target.key_mode == 4
    assert target.easier_difficulty == "HARD"
    assert target.difficulty == "EXPERT"
    assert target.rating_deficit == pytest.approx(2.04)
    assert target.minimum_rating == pytest.approx(3.98)
    assert target.maximum_rating is None


def test_separated_family_does_not_spend_inference():
    assert (
        choose_difficulty_shadow_target(
            (
                slot(4, "HARD", 3.68),
                slot(4, "EXPERT", 3.98, provenance="SAFE_FALLBACK"),
            )
        )
        is None
    )


def test_primary_to_primary_deficit_is_not_silently_ignored():
    target = choose_difficulty_shadow_target(
        (slot(4, "HARD", 3.68), slot(4, "EXPERT", 1.94))
    )

    assert target is not None
    assert target.difficulty == "EXPERT"


def test_existing_safe_resolution_prevents_redundant_inference():
    assert (
        choose_difficulty_shadow_target(
            (
                slot(4, "HARD", 3.68),
                slot(
                    4,
                    "EXPERT",
                    1.94,
                    provenance="SAFE_FALLBACK",
                    has_existing_safe_resolution=True,
                ),
            )
        )
        is None
    )


def test_small_inversion_is_a_separation_deficit_in_shadow_mode():
    target = choose_difficulty_shadow_target(
        (
            slot(6, "HARD", 5.39),
            slot(6, "EXPERT", 5.36, provenance="SAFE_FALLBACK"),
        )
    )

    assert target is not None
    assert target.rating_deficit == pytest.approx(0.33)


@pytest.mark.parametrize("key_mode", (4, 6, 7))
@pytest.mark.parametrize(
    ("easier", "harder"),
    (("EASY", "NORMAL"), ("NORMAL", "HARD"), ("HARD", "EXPERT")),
)
def test_every_key_and_adjacent_label_pair_uses_the_same_policy(
    key_mode: int,
    easier: str,
    harder: str,
) -> None:
    ratings = {
        "EASY": 1.0,
        "NORMAL": 2.0,
        "HARD": 3.0,
        "EXPERT": 4.0,
    }
    ratings[harder] = ratings[easier]

    target = choose_difficulty_shadow_target(
        tuple(slot(key_mode, difficulty, ratings[difficulty]) for difficulty in ratings)
    )

    assert target is not None
    assert (target.key_mode, target.easier_difficulty, target.difficulty) == (
        key_mode,
        easier,
        harder,
    )


def test_missing_intermediate_label_is_incomplete_not_an_artificial_adjacent_pair():
    slots = (slot(4, "EASY", 3.0), slot(4, "HARD", 1.0))

    assert choose_difficulty_shadow_target(slots) is None
    assert (
        difficulty_shadow_target_failure_reason(slots)
        == "INCOMPLETE_FAMILY_EVIDENCE"
    )


def test_clean_complete_family_has_a_typed_zero_inference_reason() -> None:
    slots = tuple(
        slot(7, difficulty, rating)
        for difficulty, rating in zip(
            ("EASY", "NORMAL", "HARD", "EXPERT"),
            (1.0, 2.0, 3.0, 4.0),
            strict=True,
        )
    )

    assert choose_difficulty_shadow_target(slots) is None
    assert difficulty_shadow_target_failure_reason(slots) == "NO_RELATIVE_DEFICIT"


def test_infeasible_middle_interval_is_skipped_for_a_feasible_later_target():
    target = choose_difficulty_shadow_target(
        (
            slot(4, "EASY", 3.0),
            slot(4, "NORMAL", 1.0),
            slot(4, "HARD", 1.1),
        )
    )

    assert target is not None
    assert target.difficulty == "HARD"


@pytest.mark.parametrize(
    ("easier", "harder", "source_star", "expected_star"),
    (
        ("EASY", "NORMAL", 1.5, 2.0),
        ("NORMAL", "HARD", 2.0, 2.5),
        ("HARD", "EXPERT", 3.0, 4.0),
    ),
)
def test_full_map_shadow_uses_one_predeclared_label_step_not_a_fitted_score(
    easier: str,
    harder: str,
    source_star: float,
    expected_star: float,
) -> None:
    source = candidate()
    source = replace(
        source,
        request=replace(
            source.request,
            difficulty=harder,
            requested_star=source_star,
        ),
    )
    target = DifficultyShadowTarget(
        4,
        easier,
        harder,
        3.0,
        2.0,
        1.3,
    )

    decision = plan_difficulty_shadow_full_map(target, source)

    assert decision.reason == "FULL_MAP_SHADOW_PLANNED"
    assert decision.plan is not None
    assert decision.plan.source is source
    assert decision.plan.requested_star == pytest.approx(expected_star)
    assert decision.plan.calibration_state == "PILOT_ONLY"


def test_full_map_shadow_fails_closed_when_selected_source_key_is_wrong() -> None:
    target = DifficultyShadowTarget(6, "NORMAL", "HARD", 3.0, 2.0, 1.3)

    decision = plan_difficulty_shadow_full_map(target, candidate())

    assert decision.reason == "SOURCE_IDENTITY_MISMATCH"
    assert decision.plan is None


def test_full_map_shadow_accepts_a_safe_cross_label_source() -> None:
    source = candidate()
    target = DifficultyShadowTarget(4, "NORMAL", "HARD", 3.0, 2.0, 1.3)

    decision = plan_difficulty_shadow_full_map(target, source)

    assert decision.reason == "FULL_MAP_SHADOW_PLANNED"
    assert decision.plan is not None
    assert decision.plan.source is source
    assert source.request.difficulty == "EXPERT"
    assert decision.plan.requested_star == pytest.approx(2.5)


def test_full_map_request_uses_target_identity_without_mutating_source() -> None:
    source = candidate()
    target = DifficultyShadowTarget(4, "NORMAL", "HARD", 3.0, 2.0, 1.3)
    decision = plan_difficulty_shadow_full_map(target, source)
    assert decision.plan is not None

    request = build_difficulty_shadow_full_map_request(
        decision.plan,
        timing_reference_path=Path("canonical-timing.osu"),
        seed=101,
    )

    assert request.key_mode == 4
    assert request.difficulty == "HARD"
    assert request.requested_star == pytest.approx(2.5)
    assert request.descriptors == DESCRIPTORS["HARD"]
    assert request.timing_reference_path == Path("canonical-timing.osu")
    assert request.seed == 101
    assert source.request.difficulty == "EXPERT"
    assert source.request.descriptors == DESCRIPTORS["EXPERT"]


def test_full_map_shadow_rejects_a_candidate_that_crosses_the_next_label() -> None:
    source = candidate(rating=2.0, intro_anchor_covered=True)
    target = DifficultyShadowTarget(
        4,
        "NORMAL",
        "HARD",
        3.0,
        2.0,
        1.3,
        minimum_rating=3.3,
        maximum_rating=3.7,
    )
    source = replace(source, request=replace(source.request, difficulty="HARD"))
    plan = DifficultyShadowFullMapPlan(
        target=target,
        source=source,
        requested_star=2.5,
        calibration_state="PILOT_ONLY",
    )
    challenger = replace(
        source,
        acceptance=FakeAcceptance(rating=3.8, gaps=()),  # type: ignore[arg-type]
        intro_anchor_covered=True,
    )

    assert (
        difficulty_shadow_full_map_candidate_failure_reason(plan, challenger)
        == "NEXT_DIFFICULTY_ORDER_LOST"
    )


def test_largest_deficit_wins_independently_of_input_order():
    values = (
        slot(7, "HARD", 6.0),
        slot(7, "EXPERT", 4.0, provenance="SAFE_FALLBACK"),
        slot(4, "HARD", 5.0),
        slot(4, "EXPERT", 4.0, provenance="SAFE_FALLBACK"),
    )

    forward = choose_difficulty_shadow_target(values)
    reverse = choose_difficulty_shadow_target(tuple(reversed(values)))

    assert forward == reverse
    assert forward is not None
    assert (forward.key_mode, forward.difficulty) == (7, "EXPERT")


def test_duplicate_slot_is_rejected_instead_of_hidden():
    with pytest.raises(ValueError, match="duplicate difficulty shadow slot"):
        choose_difficulty_shadow_target(
            (slot(4, "HARD", 3.0), slot(4, "HARD", 4.0))
        )


def test_partial_plan_uses_difficult_hard_safe_single_leading_gap_candidate():
    target = DifficultyShadowTarget(4, "HARD", "EXPERT", 3.68, 1.94, 1.74)
    source = candidate()

    decision = plan_difficulty_shadow_partial_repair(
        target,
        (source,),
        (OsuBpmEvent(0, 67.0),),
        duration_ms=323_453,
    )

    assert decision.reason == "PARTIAL_NEAR_SOLUTION_SELECTED"
    assert decision.plan is not None
    assert decision.plan.source is source
    assert decision.plan.window.start_ms == 0
    assert decision.plan.window.end_ms == 17_623
    assert decision.plan.required_gameplay_interval is None


def test_partial_plan_integrates_tempo_changes_across_the_context_window():
    decision = plan_difficulty_shadow_partial_repair(
        DifficultyShadowTarget(4, "HARD", "EXPERT", 3.68, 1.94, 1.74),
        (candidate(first_rows=(12_697, 14_041)),),
        (
            OsuBpmEvent(0, 60.0),
            OsuBpmEvent(15_041, 120.0),
        ),
        duration_ms=323_453,
    )

    assert decision.plan is not None
    # One beat at 60 BPM reaches 15,041ms; three beats at 120 BPM reach 16,541ms.
    assert decision.plan.window.end_ms == 16_541


def test_observe_interval_is_bound_only_after_a_bounded_partial_plan_exists():
    decision = plan_difficulty_shadow_partial_repair(
        DifficultyShadowTarget(4, "HARD", "EXPERT", 3.68, 1.94, 1.74),
        (candidate(),),
        (OsuBpmEvent(0, 67.0),),
        duration_ms=323_453,
    )

    bound = bind_required_gameplay_interval(
        decision,
        evidence=required_gameplay_evidence(),
        bpm_events=(OsuBpmEvent(0, 67.0),),
        duration_ms=323_453,
        mode=RequiredGameplayIntervalMode.OBSERVE,
    )

    assert bound.reason == "BROADBAND_ATTACK_SUPPORTED"
    assert bound.plan is not None
    assert bound.plan.required_gameplay_interval is not None
    assert (
        bound.plan.required_gameplay_interval.mode
        is RequiredGameplayIntervalMode.OBSERVE
    )
    assert bound.plan.required_gameplay_interval.start_ms == 430
    assert bound.plan.required_gameplay_interval.end_ms == 570


def test_missing_required_evidence_does_not_invent_an_interval_or_run_a_plan():
    decision = plan_difficulty_shadow_partial_repair(
        DifficultyShadowTarget(4, "HARD", "EXPERT", 3.68, 1.94, 1.74),
        (candidate(),),
        (OsuBpmEvent(0, 67.0),),
        duration_ms=323_453,
    )

    bound = bind_required_gameplay_interval(
        decision,
        evidence=None,
        bpm_events=(OsuBpmEvent(0, 67.0),),
        duration_ms=323_453,
        mode=RequiredGameplayIntervalMode.OBSERVE,
    )

    assert bound.reason == "REQUIRED_GAMEPLAY_EVIDENCE_UNAVAILABLE"
    assert bound.plan is None


@pytest.mark.parametrize(
    ("source", "reason"),
    [
        (candidate(provenance="SAFE_FALLBACK"), "NO_NEAR_SOLUTION_CANDIDATE"),
        (candidate(hard_safe=False), "CANDIDATE_NOT_HARD_SAFE"),
        (candidate(rating=3.97), "CANDIDATE_DIFFICULTY_INSUFFICIENT"),
        (candidate(gaps=()), "GAP_NOT_SINGLE_LEADING"),
        (
            candidate(gaps=(active_gap(5_000, 10_000, "MIDDLE"),)),
            "GAP_NOT_SINGLE_LEADING",
        ),
        (candidate(intro_anchor_covered=None), "INTRO_EVIDENCE_UNAVAILABLE"),
        (candidate(first_rows=(12_697,)), "LOCAL_WINDOW_UNAVAILABLE"),
    ],
)
def test_partial_plan_fails_closed_with_typed_reason(source: Candidate, reason: str):
    decision = plan_difficulty_shadow_partial_repair(
        DifficultyShadowTarget(4, "HARD", "EXPERT", 3.68, 1.94, 1.74),
        (source,),
        (OsuBpmEvent(0, 67.0),),
        duration_ms=323_453,
    )

    assert decision.reason == reason
    assert decision.plan is None


def test_partial_plan_candidate_choice_is_independent_of_input_order():
    preferred = candidate(matched_f1=0.8, attempt=3, seed=17)
    other = candidate(matched_f1=0.7, attempt=1, seed=5)
    target = DifficultyShadowTarget(4, "HARD", "EXPERT", 3.68, 1.94, 1.74)

    forward = plan_difficulty_shadow_partial_repair(
        target,
        (preferred, other),
        (OsuBpmEvent(0, 67.0),),
        duration_ms=323_453,
    )
    reverse = plan_difficulty_shadow_partial_repair(
        target,
        (other, preferred),
        (OsuBpmEvent(0, 67.0),),
        duration_ms=323_453,
    )

    assert forward.plan is not None and reverse.plan is not None
    assert forward.plan.source is preferred
    assert reverse.plan.source is preferred


@pytest.mark.parametrize(
    ("challenger_factory", "reason"),
    [
        (
            lambda source: replace(
                source,
                generated=replace(
                    source.generated,
                    notes=[
                        NoteEvent(500, 0),
                        NoteEvent(1_000, 1),
                        NoteEvent(21_000, 2),
                    ],
                ),
                acceptance=FakeAcceptance(rating=4.50, gaps=()),  # type: ignore[arg-type]
                intro_anchor_covered=True,
            ),
            "SUFFIX_SEMANTIC_MUTATION",
        ),
        (
            lambda source: replace(
                source,
                generated=replace(
                    source.generated,
                    notes=[
                        NoteEvent(500, 0, "HOLD", 1_500),
                        NoteEvent(1_000, 1),
                        NoteEvent(20_000, 2),
                    ],
                ),
                acceptance=FakeAcceptance(rating=4.50, gaps=()),  # type: ignore[arg-type]
                intro_anchor_covered=True,
            ),
            "SUFFIX_SEMANTIC_MUTATION",
        ),
        (
            lambda source: replace(
                source,
                acceptance=FakeAcceptance(
                    rating=4.50,
                    gaps=(active_gap(0, 500),),
                ),  # type: ignore[arg-type]
                intro_anchor_covered=True,
            ),
            "INTRO_OR_GAP_UNRESOLVED",
        ),
        (
            lambda source: replace(
                source,
                acceptance=FakeAcceptance(rating=3.97, gaps=()),  # type: ignore[arg-type]
                intro_anchor_covered=True,
            ),
            "DIFFICULTY_LOST",
        ),
        (
            lambda source: replace(
                source,
                acceptance=FakeAcceptance(
                    rating=4.50,
                    gaps=(),
                    matched_f1=0.50,
                    matched_precision=0.44,
                ),  # type: ignore[arg-type]
                intro_anchor_covered=True,
            ),
            "QUALITY_NONREGRESSION_FAILED",
        ),
    ],
)
def test_shadow_partial_validation_fails_closed_with_typed_reason(
    challenger_factory,
    reason: str,
):
    source = candidate(first_rows=(500, 1_000, 20_000))
    plan = DifficultyShadowPartialPlan(
        target=DifficultyShadowTarget(4, "HARD", "EXPERT", 3.68, 1.94, 1.74),
        source=source,
        window=PartialRemapWindow(start_ms=0, end_ms=1_000),
    )

    assert difficulty_shadow_candidate_failure_reason(
        plan,
        challenger_factory(source),
    ) == reason


def test_shadow_partial_validation_accepts_safe_nonregressing_repair():
    source = candidate(first_rows=(500, 1_000, 20_000))
    challenger = replace(
        source,
        generated=replace(
            source.generated,
            notes=[NoteEvent(250, 0), NoteEvent(1_000, 1), NoteEvent(20_000, 2)],
        ),
        acceptance=FakeAcceptance(rating=4.50, gaps=()),  # type: ignore[arg-type]
        intro_anchor_covered=True,
    )
    plan = DifficultyShadowPartialPlan(
        target=DifficultyShadowTarget(4, "HARD", "EXPERT", 3.68, 1.94, 1.74),
        source=source,
        window=PartialRemapWindow(start_ms=0, end_ms=1_000),
    )

    assert difficulty_shadow_candidate_failure_reason(plan, challenger) is None


def test_required_gameplay_failure_keeps_exact_runtime_reason_in_attempt_evidence():
    error = WorkerError(
        ErrorCode.MANIA_REQUIRED_GAMEPLAY_FAILED,
        "required interval failed",
        context={"reason": "REQUIRED_GAMEPLAY_INTERVAL_NO_LEGAL_GROUP"},
    )

    assert difficulty_shadow_inference_failure_reason(
        error,
        inference_completed=False,
    ) == "REQUIRED_GAMEPLAY_INTERVAL_NO_LEGAL_GROUP"


def test_existing_candidate_must_meet_the_same_adjacent_gap_as_the_planner():
    current = candidate(
        provenance="SAFE_FALLBACK",
        rating=1.94,
        gaps=(active_gap(0, 500),),
    )
    below_gap = candidate(
        provenance="PRIMARY",
        rating=3.80,
        gaps=(),
        intro_anchor_covered=True,
        seed=29,
    )
    at_gap = candidate(
        provenance="PRIMARY",
        rating=3.98,
        gaps=(),
        intro_anchor_covered=True,
        seed=31,
    )

    assert not existing_candidate_resolves_fallback(
        current,
        easier_rating=3.68,
        candidates=(below_gap,),
    )
    assert existing_candidate_resolves_fallback(
        current,
        easier_rating=3.68,
        candidates=(at_gap,),
    )


@pytest.mark.parametrize("hold_expansion", [False, True])
def test_partial_plan_reports_phrase_or_hold_over_cap_as_window_too_large(
    hold_expansion: bool,
):
    if hold_expansion:
        source = candidate(first_rows=(500, 1_000))
        source = replace(
            source,
            generated=replace(
                source.generated,
                notes=[
                    NoteEvent(500, 0),
                    NoteEvent(1_000, 1, "HOLD", 322_453),
                ],
            ),
        )
    else:
        source = candidate(first_rows=(500, 260_000))

    decision = plan_difficulty_shadow_partial_repair(
        DifficultyShadowTarget(4, "HARD", "EXPERT", 3.68, 1.94, 1.74),
        (source,),
        (OsuBpmEvent(0, 67.0),),
        duration_ms=323_453,
    )

    assert decision.reason == "WINDOW_TOO_LARGE"
    assert decision.plan is None
