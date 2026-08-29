import hashlib
from collections import Counter
from dataclasses import replace
from itertools import pairwise
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from chart_worker.analysis.onset import OnsetAnalysis
from chart_worker.generation import difficulty_family_compiler
from chart_worker.generation.candidate_state import Candidate
from chart_worker.generation.difficulty_family_compiler import (
    DifficultyFamilyCompilerDecision,
    DifficultyFamilyCompilerProposal,
    DifficultyFamilyCompilerSlot,
    compile_difficulty_family_shadow,
    materialize_compiled_family,
    persist_difficulty_family_compiler_payloads,
)
from chart_worker.generation.mapperatorinator import GeneratedChart
from chart_worker.generation.osu_parser import OsuBpmEvent
from chart_worker.generation.params import GenerationRequest
from chart_worker.schema.note import NoteEvent
from chart_worker.schema.types import DIFFICULTIES
from chart_worker.stages.types import SongTimingAuthority
from chart_worker.validation.quality_gate import GateAction, GateAxis


def test_difficulty_family_compiler_contract_is_available() -> None:
    assert DifficultyFamilyCompilerSlot is not None
    assert DifficultyFamilyCompilerProposal is not None
    assert DifficultyFamilyCompilerDecision is not None
    assert compile_difficulty_family_shadow is not None


class FakeAcceptance:
    def __init__(
        self,
        rating: float,
        *,
        ordering_score: float | None = None,
        action: GateAction = GateAction.PASS,
        precision: float = 0.95,
        f1: float = 0.90,
        hard_safe: bool = True,
        gap_count: int = 0,
    ) -> None:
        self.action = action
        self.profile = SimpleNamespace(
            difficulty=SimpleNamespace(project_rating=rating),
            difficulty_vector_v2=SimpleNamespace(
                ordering_score=(rating if ordering_score is None else ordering_score)
            ),
        )
        gaps = tuple(
            SimpleNamespace(start_ms=index * 1_000, end_ms=index * 1_000 + 500)
            for index in range(gap_count)
        )
        self.timing = SimpleNamespace(
            coverage_gaps=gaps,
            overall=SimpleNamespace(
                matched_precision_50=precision,
                matched_f1_50=f1,
            ),
        )
        self.decisions = tuple(
            SimpleNamespace(axis=axis, action=self.decision(axis).action) for axis in GateAxis
        )
        self._hard_safe = hard_safe

    def decision(self, axis: GateAxis) -> SimpleNamespace:
        hard_axes = {
            GateAxis.STRUCTURE,
            GateAxis.TIMING_IDENTITY,
            GateAxis.SONG_BOUNDS,
        }
        return SimpleNamespace(
            action=(
                GateAction.PASS
                if getattr(self, "_hard_safe", True) or axis not in hard_axes
                else GateAction.RETRY_MAP
            )
        )


def _authority() -> SongTimingAuthority:
    return SongTimingAuthority(
        reference_path=Path("timing.osu"),
        sha256="a" * 64,
        audio_sha256="b" * 64,
        bpm_events=(OsuBpmEvent(0, 120.0), OsuBpmEvent(2_000, 240.0)),
        generator_name="fixture",
        seed=0,
        mode="STANDARD",
        attempt_count=1,
    )


def _onsets() -> OnsetAnalysis:
    strength = np.linspace(0.1, 1.0, 100, dtype=np.float64)
    return OnsetAnalysis(
        sample_rate_hz=1_000,
        hop_length=100,
        strength=strength,
        band_strength=np.vstack((strength, strength, strength)),
        onset_ms=tuple(range(0, 10_000, 100)),
    )


def _chart(note_count: int, *, key_mode: int = 4) -> GeneratedChart:
    notes = [
        NoteEvent(
            time_ms=index * 100,
            lane=index % key_mode,
            kind=("HOLD" if index in {8, 24} else "TAP"),
            duration_ms=(50 if index in {8, 24} else None),
            is_downbeat=index % 8 == 0,
        )
        for index in range(note_count)
    ]
    return GeneratedChart(
        notes=notes,
        key_mode=key_mode,
        osu_text="source",
        generator_name="fixture",
        seed=1,
        bpm_events=_authority().bpm_events,
    )


def _osu_text(chart: GeneratedChart, *, custom_samples: bool = False) -> str:
    lines = [
        "osu file format v14",
        "",
        "[General]",
        "Mode:3",
        "",
        "[Metadata]",
        "Version:Fixture",
        "",
        "[Difficulty]",
        f"CircleSize:{chart.key_mode}",
        "",
        "[TimingPoints]",
        "0,500.000000000000,4,2,0,60,1,0",
        "2000,250.000000000000,4,2,0,60,1,0",
        "",
        "[HitObjects]",
    ]
    for index, note in enumerate(chart.notes):
        x = round((note.lane + 0.5) * 512 / chart.key_mode)
        if custom_samples and index % 2 == 0:
            x -= 1
        sample = "2:2:-1:5:" if custom_samples else "0:0:0:0:"
        if note.kind == "HOLD":
            assert note.duration_ms is not None
            lines.append(f"{x},192,{note.time_ms},128,0,{note.time_ms + note.duration_ms}:{sample}")
        else:
            lines.append(f"{x},192,{note.time_ms},1,0,{sample}")
    return "\n".join(lines) + "\n"


def _slot(
    difficulty: str,
    rating: float,
    note_count: int,
    *,
    key_mode: int = 4,
) -> DifficultyFamilyCompilerSlot:
    chart = _chart(note_count, key_mode=key_mode)
    return DifficultyFamilyCompilerSlot(
        difficulty=difficulty,
        candidate_id=f"{key_mode}k:{difficulty}",
        candidate_payload_sha256=hashlib.sha256(
            f"{key_mode}:{difficulty}".encode()
        ).hexdigest(),
        generated=chart,
        acceptance=FakeAcceptance(rating),  # type: ignore[arg-type]
        osu_text=_osu_text(chart),
        provenance="PRIMARY",
    )


def _serialize(chart: GeneratedChart) -> str:
    return _osu_text(chart)


def _evaluate(chart: GeneratedChart, difficulty: str) -> FakeAcceptance:
    del difficulty
    rating = len(chart.notes) / 10.0
    return FakeAcceptance(rating, ordering_score=rating)


def _compile(
    slots: tuple[DifficultyFamilyCompilerSlot, ...],
) -> DifficultyFamilyCompilerDecision:
    return _compile_with(slots, evaluate=_evaluate)


def _runtime_candidate(slot: DifficultyFamilyCompilerSlot, tmp_path: Path) -> Candidate:
    return Candidate(
        request=GenerationRequest(
            audio_path=tmp_path / "audio.wav",
            key_mode=slot.generated.key_mode,
            difficulty=slot.difficulty,
            duration_ms=10_000,
            seed=1,
            requested_star=1.0,
        ),
        generated=slot.generated,
        acceptance=slot.acceptance,
        osu_text=slot.osu_text,
        workdir=tmp_path / "source",
        attempt=1,
        seed=1,
        provenance="PRIMARY",
    )


def _compile_with(
    slots: tuple[DifficultyFamilyCompilerSlot, ...],
    *,
    evaluate,
) -> DifficultyFamilyCompilerDecision:
    return compile_difficulty_family_shadow(
        slots,
        authority=_authority(),
        onset_analysis=_onsets(),
        duration_ms=10_000,
        boundary_policy_mode="HIGH_CONFIDENCE_ENFORCED",
        evaluate_candidate=evaluate,
        serialize_candidate=_serialize,
    )


class StepClock:
    def __init__(self, *, start: float = 100.0, step: float = 0.001) -> None:
        self.current = start - step
        self.step = step

    def __call__(self) -> float:
        self.current += self.step
        return self.current


def _identities(chart: GeneratedChart) -> set[tuple[int, int, str, int | None]]:
    return {(note.time_ms, note.lane, note.kind, note.duration_ms) for note in chart.notes}


def _section(text: str, name: str) -> tuple[str, ...]:
    lines = text.splitlines()
    start = lines.index(f"[{name}]") + 1
    result = []
    for line in lines[start:]:
        if line.startswith("[") and line.endswith("]"):
            break
        if line:
            result.append(line)
    return tuple(result)


def test_compiler_builds_deterministic_nested_subsets_for_an_inverted_family() -> None:
    slots = (
        _slot("EASY", 4.0, 40),
        _slot("NORMAL", 3.9, 39),
        _slot("HARD", 3.8, 38),
        _slot("EXPERT", 3.7, 37),
    )

    decision = _compile(slots)
    reordered = _compile(tuple(reversed(slots)))

    assert decision.status == "COMPILED"
    assert [item.target_difficulty for item in decision.proposals] == [
        "EASY",
        "NORMAL",
        "HARD",
        "EXPERT",
    ]
    assert [item.osu_text for item in decision.proposals] == [
        item.osu_text for item in reordered.proposals
    ]
    by_difficulty = {item.target_difficulty: item for item in decision.proposals}
    easy = _identities(by_difficulty["EASY"].generated)
    normal = _identities(by_difficulty["NORMAL"].generated)
    hard = _identities(by_difficulty["HARD"].generated)
    expert = _identities(by_difficulty["EXPERT"].generated)
    assert easy < normal < hard < expert
    assert expert == _identities(slots[0].generated)
    for proposal in decision.proposals:
        times = {note.time_ms for note in proposal.generated.notes}
        assert {0, 3_900}.issubset(times)
        assert _identities(proposal.generated).issubset(expert)
    report = decision.to_report()
    assert report["version"] == "difficulty-family-compiler-shadow-v2-runtime"
    assert report["status"] == "COMPILED"
    assert report["mutatesSelection"] is False
    assert report["mutatesPublishedCharts"] is False
    assert report["additionalModelCalls"] == 0
    assert len(report["proposals"]) == 4
    assert all(len(item["candidatePayloadSha256"]) == 64 for item in report["proposals"])


def test_compiled_family_materializes_four_unique_atomic_recovery_candidates(
    tmp_path: Path,
) -> None:
    slots = (
        _slot("EASY", 4.0, 40),
        _slot("NORMAL", 3.9, 39),
        _slot("HARD", 3.8, 38),
        _slot("EXPERT", 3.7, 37),
    )
    decision = _compile(slots)
    source_candidates = {
        slot.candidate_id: _runtime_candidate(slot, tmp_path) for slot in slots
    }
    # Exercise materialization after an earlier relative-difficulty relabel:
    # the request template in a target slot may originate from another label.
    current_assignment = {
        difficulty: source_candidates[slots[(index + 1) % len(slots)].candidate_id]
        for index, difficulty in enumerate(DIFFICULTIES)
    }

    materialized = materialize_compiled_family(
        decision,
        source_candidates=source_candidates,
        current_assignment=current_assignment,
    )

    assert tuple(materialized) == DIFFICULTIES
    assert len({candidate.osu_text for candidate in materialized.values()}) == 4
    assert all(
        candidate.request.difficulty == difficulty
        and candidate.provenance == "SAFE_FALLBACK"
        and candidate.recovery_reason == "DIFFICULTY_FAMILY_COMPILER_V1"
        for difficulty, candidate in materialized.items()
    )


def test_materialization_rejects_a_duplicate_compiler_payload(tmp_path: Path) -> None:
    slots = (
        _slot("EASY", 4.0, 40),
        _slot("NORMAL", 3.9, 39),
        _slot("HARD", 3.8, 38),
        _slot("EXPERT", 3.7, 37),
    )
    decision = _compile(slots)
    assert decision.status == "COMPILED"
    duplicated = replace(
        decision,
        proposals=(
            *decision.proposals[:-1],
            replace(
                decision.proposals[-1],
                osu_text=decision.proposals[-2].osu_text,
            ),
        ),
    )
    source_candidates = {
        slot.candidate_id: _runtime_candidate(slot, tmp_path) for slot in slots
    }

    with pytest.raises(ValueError, match="unique payloads"):
        materialize_compiled_family(
            duplicated,
            source_candidates=source_candidates,
            current_assignment={
                slot.difficulty: source_candidates[slot.candidate_id]
                for slot in slots
            },
        )


def test_compiler_records_deterministic_solver_and_candidate_evaluation_wall_time() -> None:
    slots = (
        _slot("EASY", 4.0, 40),
        _slot("NORMAL", 3.9, 39),
        _slot("HARD", 3.8, 38),
        _slot("EXPERT", 3.7, 37),
    )
    clock = StepClock()

    decision = compile_difficulty_family_shadow(
        slots,
        authority=_authority(),
        onset_analysis=_onsets(),
        duration_ms=10_000,
        boundary_policy_mode="HIGH_CONFIDENCE_ENFORCED",
        evaluate_candidate=_evaluate,
        serialize_candidate=_serialize,
        clock=clock,
    )

    assert decision.status == "COMPILED"
    assert decision.candidate_evaluation_wall_ms == pytest.approx(
        decision.proposals_evaluated * 1.0
    )
    assert decision.solver_wall_ms == pytest.approx((decision.proposals_evaluated * 2 + 1) * 1.0)
    assert decision.payload_persistence_wall_ms == 0.0
    report = decision.to_report()
    assert report["solverWallMs"] == decision.solver_wall_ms
    assert report["candidateEvaluationWallMs"] == decision.candidate_evaluation_wall_ms
    assert report["payloadPersistenceWallMs"] == 0.0


def test_compiler_records_zero_evaluation_time_for_early_decisions() -> None:
    separated = (
        _slot("EASY", 1.0, 10),
        _slot("NORMAL", 2.0, 20),
        _slot("HARD", 3.0, 30),
        _slot("EXPERT", 4.0, 40),
    )
    clock = StepClock(step=0.002)

    decision = compile_difficulty_family_shadow(
        separated,
        authority=_authority(),
        onset_analysis=_onsets(),
        duration_ms=10_000,
        boundary_policy_mode="HIGH_CONFIDENCE_ENFORCED",
        evaluate_candidate=_evaluate,
        serialize_candidate=_serialize,
        clock=clock,
    )

    assert decision.status == "NOT_NEEDED"
    assert decision.proposals_evaluated == 0
    assert decision.candidate_evaluation_wall_ms == 0.0
    assert decision.solver_wall_ms == pytest.approx(2.0)


def test_compiler_records_payload_persistence_wall_time_without_changing_payloads(
    tmp_path: Path,
) -> None:
    decision = _compile(
        (
            _slot("EASY", 4.0, 40),
            _slot("NORMAL", 3.9, 39),
            _slot("HARD", 3.8, 38),
            _slot("EXPERT", 3.7, 37),
        )
    )
    before = tuple(proposal.candidate_payload_sha256 for proposal in decision.proposals)

    persisted = persist_difficulty_family_compiler_payloads(
        decision,
        run_dir=tmp_path,
        clock=StepClock(step=0.004),
    )

    assert persisted.payload_persistence_wall_ms == pytest.approx(4.0)
    assert tuple(proposal.candidate_payload_sha256 for proposal in persisted.proposals) == before
    assert persisted.solver_wall_ms == decision.solver_wall_ms
    assert persisted.candidate_evaluation_wall_ms == decision.candidate_evaluation_wall_ms


def test_compiler_isolates_payload_persistence_failure_and_keeps_elapsed_time(
    monkeypatch,
    tmp_path: Path,
) -> None:
    decision = _compile(
        (
            _slot("EASY", 4.0, 40),
            _slot("NORMAL", 3.9, 39),
            _slot("HARD", 3.8, 38),
            _slot("EXPERT", 3.7, 37),
        )
    )

    def fail_persistence(**_kwargs):
        raise OSError("fixture persistence failure")

    monkeypatch.setattr(
        difficulty_family_compiler,
        "persist_candidate_payload",
        fail_persistence,
    )

    persisted = persist_difficulty_family_compiler_payloads(
        decision,
        run_dir=tmp_path,
        clock=StepClock(step=0.005),
    )

    assert persisted.status == "UNAVAILABLE"
    assert persisted.reason == "PAYLOAD_PERSISTENCE_FAILED"
    assert persisted.failure_type == "OSError"
    assert persisted.proposals == ()
    assert persisted.payload_persistence_wall_ms == pytest.approx(5.0)
    assert persisted.solver_wall_ms == decision.solver_wall_ms


def test_compiler_deletes_exact_anchor_rows_without_reserializing_survivors() -> None:
    slots = [
        _slot("EASY", 4.0, 40),
        _slot("NORMAL", 3.9, 39),
        _slot("HARD", 3.8, 38),
        _slot("EXPERT", 3.7, 37),
    ]
    anchor = slots[0]
    exact_anchor_text = _osu_text(anchor.generated, custom_samples=True)
    slots[0] = DifficultyFamilyCompilerSlot(
        difficulty=anchor.difficulty,
        candidate_id=anchor.candidate_id,
        candidate_payload_sha256=anchor.candidate_payload_sha256,
        generated=anchor.generated,
        acceptance=anchor.acceptance,
        osu_text=exact_anchor_text,
        provenance=anchor.provenance,
    )

    decision = _compile(tuple(slots))

    assert decision.status == "COMPILED"
    anchor_objects = Counter(_section(exact_anchor_text, "HitObjects"))
    anchor_timing = _section(exact_anchor_text, "TimingPoints")
    for proposal in decision.proposals:
        proposal_objects = Counter(_section(proposal.osu_text, "HitObjects"))
        assert not (proposal_objects - anchor_objects)
        assert _section(proposal.osu_text, "TimingPoints") == anchor_timing


@pytest.mark.parametrize("key_mode", (4, 6, 7))
def test_compiler_uses_the_same_subset_contract_for_every_supported_key_mode(
    key_mode: int,
) -> None:
    slots = tuple(
        _slot(
            difficulty,
            rating,
            note_count,
            key_mode=key_mode,
        )
        for difficulty, rating, note_count in (
            ("EASY", 4.0, 40),
            ("NORMAL", 3.9, 39),
            ("HARD", 3.8, 38),
            ("EXPERT", 3.7, 37),
        )
    )

    source_notes = tuple(_identities(slot.generated) for slot in slots)
    decision = _compile(slots)

    assert decision.status == "COMPILED"
    expert = _identities(decision.proposals[-1].generated)
    for easier, harder in pairwise(decision.proposals):
        assert _identities(easier.generated) < _identities(harder.generated)
    assert all(proposal.generated.key_mode == key_mode for proposal in decision.proposals)
    assert all(_identities(proposal.generated).issubset(expert) for proposal in decision.proposals)
    assert tuple(_identities(slot.generated) for slot in slots) == source_notes


def test_compiler_does_not_evaluate_an_already_separated_family() -> None:
    decision = _compile(
        (
            _slot("EASY", 1.0, 10),
            _slot("NORMAL", 2.0, 20),
            _slot("HARD", 3.0, 30),
            _slot("EXPERT", 4.0, 40),
        )
    )

    assert decision.status == "NOT_NEEDED"
    assert decision.proposals == ()
    assert decision.proposals_evaluated == 0


def test_compiler_reviews_a_small_positive_gap_without_rewriting_the_family() -> None:
    decision = _compile(
        (
            _slot("EASY", 1.00, 10),
            _slot("NORMAL", 1.50, 15),
            _slot("HARD", 2.70, 27),
            _slot("EXPERT", 2.75, 28),
        )
    )

    assert decision.status == "NOT_NEEDED"
    assert decision.reason == "FAMILY_ORDERED_NARROW_REVIEW"
    assert decision.proposals == ()
    assert decision.proposals_evaluated == 0


def test_compiler_abstains_when_no_anchor_is_hard_safe() -> None:
    slots = [
        _slot("EASY", 4.0, 40),
        _slot("NORMAL", 3.9, 39),
        _slot("HARD", 3.8, 38),
        _slot("EXPERT", 3.7, 37),
    ]
    slots = [
        DifficultyFamilyCompilerSlot(
            difficulty=slot.difficulty,
            candidate_id=slot.candidate_id,
            candidate_payload_sha256=slot.candidate_payload_sha256,
            generated=slot.generated,
            acceptance=FakeAcceptance(  # type: ignore[arg-type]
                slot.acceptance.profile.difficulty.project_rating,
                hard_safe=False,
            ),
            osu_text=slot.osu_text,
            provenance=slot.provenance,
        )
        for slot in slots
    ]

    decision = _compile(tuple(slots))

    assert decision.status == "UNAVAILABLE"
    assert decision.reason == "NO_SAFE_ANCHOR"
    assert decision.proposals == ()


def test_compiler_rejects_a_quality_rank_worse_than_either_comparator() -> None:
    slots = (
        DifficultyFamilyCompilerSlot(
            difficulty="EASY",
            candidate_id="4k:EASY",
            candidate_payload_sha256="e" * 64,
            generated=_chart(40),
            acceptance=FakeAcceptance(  # type: ignore[arg-type]
                4.0,
                action=GateAction.REVIEW,
            ),
            osu_text="source-EASY",
            provenance="PRIMARY",
        ),
        _slot("NORMAL", 3.9, 39),
        _slot("HARD", 3.8, 38),
        _slot("EXPERT", 3.7, 37),
    )

    def review_candidate(chart: GeneratedChart, difficulty: str) -> FakeAcceptance:
        del difficulty
        rating = len(chart.notes) / 10.0
        return FakeAcceptance(
            rating,
            ordering_score=rating,
            action=GateAction.REVIEW,
        )

    decision = _compile_with(slots, evaluate=review_candidate)

    assert decision.status == "UNAVAILABLE"
    assert decision.reason == "NO_SAFE_HARD_PROPOSAL"
    assert decision.proposals == ()


def test_compiler_does_not_promote_a_weaker_anchor_into_expert() -> None:
    weak_easy = DifficultyFamilyCompilerSlot(
        difficulty="EASY",
        candidate_id="4k:EASY:weak",
        candidate_payload_sha256="e" * 64,
        generated=_chart(40),
        acceptance=FakeAcceptance(  # type: ignore[arg-type]
            4.0,
            action=GateAction.REVIEW,
        ),
        osu_text="source-EASY-weak",
        provenance="PRIMARY",
    )
    slots = (
        weak_easy,
        _slot("NORMAL", 3.9, 39),
        _slot("HARD", 3.8, 38),
        _slot("EXPERT", 3.7, 37),
    )

    decision = _compile(slots)

    assert decision.status == "COMPILED"
    assert decision.anchor_candidate_id == "4k:NORMAL"
    assert decision.anchor_source_difficulty == "NORMAL"
    expert = decision.proposals[-1]
    assert expert.acceptance.action is GateAction.PASS


def test_compiler_rejects_timing_precision_regression() -> None:
    slots = (
        _slot("EASY", 4.0, 40),
        _slot("NORMAL", 3.9, 39),
        _slot("HARD", 3.8, 38),
        _slot("EXPERT", 3.7, 37),
    )

    def weak_precision(chart: GeneratedChart, difficulty: str) -> FakeAcceptance:
        del difficulty
        rating = len(chart.notes) / 10.0
        return FakeAcceptance(
            rating,
            ordering_score=rating,
            precision=0.80,
        )

    decision = _compile_with(slots, evaluate=weak_precision)

    assert decision.status == "UNAVAILABLE"
    assert decision.reason == "NO_SAFE_HARD_PROPOSAL"


def test_compiler_abstains_atomically_when_only_a_partial_family_is_feasible() -> None:
    slots = (
        _slot("EASY", 4.0, 40),
        _slot("NORMAL", 3.9, 39),
        _slot("HARD", 3.8, 38),
        _slot("EXPERT", 3.7, 37),
    )

    def easy_fails(chart: GeneratedChart, difficulty: str) -> FakeAcceptance:
        rating = len(chart.notes) / 10.0
        return FakeAcceptance(
            rating,
            ordering_score=rating,
            action=(GateAction.RETRY_MAP if difficulty == "EASY" else GateAction.PASS),
        )

    decision = _compile_with(slots, evaluate=easy_fails)

    assert decision.status == "UNAVAILABLE"
    assert decision.reason == "NO_SAFE_EASY_PROPOSAL"
    assert decision.proposals == ()
    assert decision.proposals_evaluated > 0
