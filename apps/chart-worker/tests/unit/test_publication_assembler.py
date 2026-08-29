from types import SimpleNamespace

import pytest

from chart_worker.errors import WorkerError
from chart_worker.generation.publication_assembler import (
    assemble_publication,
    classify_family_assignment_kinds,
    validate_monotonic_family_difficulty,
    validate_unique_family_payloads,
)
from chart_worker.schema.types import DIFFICULTIES


def _candidate(source_difficulty: str, payload: str):
    return SimpleNamespace(
        request=SimpleNamespace(difficulty=source_difficulty),
        osu_text=payload,
    )


def test_publication_assignment_kind_uses_target_and_payload_identity() -> None:
    easy = _candidate("EASY", "easy")
    normal = _candidate("NORMAL", "normal")
    hard = _candidate("HARD", "hard")

    kinds = classify_family_assignment_kinds(
        {
            "EASY": easy,
            "NORMAL": normal,
            "HARD": hard,
            "EXPERT": hard,
        }
    )

    assert kinds == {
        "EASY": "ORIGINAL",
        "NORMAL": "ORIGINAL",
        "HARD": "ORIGINAL",
        "EXPERT": "EMERGENCY_DUPLICATE",
    }


def test_unique_cross_difficulty_payload_is_reassigned_not_duplicated() -> None:
    easy = _candidate("EXPERT", "easy-target-payload")
    normal = _candidate("NORMAL", "normal")
    hard = _candidate("HARD", "hard")
    expert = _candidate("EASY", "expert-target-payload")

    kinds = classify_family_assignment_kinds(
        {
            "EASY": easy,
            "NORMAL": normal,
            "HARD": hard,
            "EXPERT": expert,
        }
    )

    assert kinds["EASY"] == "REASSIGNED"
    assert kinds["EXPERT"] == "REASSIGNED"
    assert "EMERGENCY_DUPLICATE" not in kinds.values()


def test_publication_rejects_duplicate_payloads_even_with_distinct_candidates() -> None:
    assignment = {
        "EASY": _candidate("EASY", "easy"),
        "NORMAL": _candidate("NORMAL", "normal"),
        "HARD": _candidate("HARD", "same"),
        "EXPERT": _candidate("EXPERT", "same"),
    }

    with pytest.raises(WorkerError, match="duplicate payload"):
        validate_unique_family_payloads(assignment, key_mode=4)


def test_publication_rejects_equal_or_inverted_final_difficulty_order() -> None:
    review = SimpleNamespace(
        status="PASS",
        ambiguous_pairs=(("HARD", "EXPERT"),),
        inverted_pairs=(),
        to_report=lambda: {"status": "PASS"},
    )

    with pytest.raises(WorkerError, match="strictly increasing"):
        validate_monotonic_family_difficulty(review, key_mode=4)


def test_narrow_but_strict_final_order_is_playtest_only() -> None:
    review = SimpleNamespace(
        status="PASS",
        ambiguous_pairs=(),
        inverted_pairs=(),
        narrow_pairs=(("HARD", "EXPERT"),),
        to_report=lambda: {"status": "PASS"},
    )

    with pytest.raises(WorkerError, match="materially separated") as caught:
        validate_monotonic_family_difficulty(review, key_mode=4)

    assert caught.value.context["failure_stage"] == (
        "DIFFICULTY_SEPARATION_PUBLICATION_GUARD"
    )


def test_unresolved_guard_preserves_unique_hard_safe_family_for_playtest(
    tmp_path, monkeypatch
) -> None:
    candidates = {
        difficulty: SimpleNamespace(
            request=SimpleNamespace(
                key_mode=4,
                difficulty=difficulty,
                requested_star=float(index + 1),
                cfg_scale=1.0,
            ),
            generated=SimpleNamespace(),
            acceptance=SimpleNamespace(),
            osu_text=f"unique-{difficulty}",
            attempt=1,
            seed=index,
            provenance="PRIMARY",
            recovery_reason=None,
            coverage_repair_gap_count=0,
        )
        for index, difficulty in enumerate(DIFFICULTIES)
    }
    states = {
        difficulty: SimpleNamespace(
            key_mode=4,
            difficulty=difficulty,
            publication_block_reason=None,
            attempt_evidence=[],
            attempt_errors=[],
            candidates=SimpleNamespace(
                admitted=(candidates[difficulty],),
                raw_rejected=(),
                safe_fallbacks=(),
            ),
            budget=SimpleNamespace(next_attempt=2),
            recovery=SimpleNamespace(was_attempted=lambda _kind: False),
            exhausted_error=None,
        )
        for difficulty in DIFFICULTIES
    }
    review = SimpleNamespace(
        status="PASS",
        ambiguous_pairs=(("HARD", "EXPERT"),),
        inverted_pairs=(),
        narrow_pairs=(),
        to_report=lambda: {"status": "PASS"},
    )
    monkeypatch.setattr(
        "chart_worker.generation.publication_assembler._record_unselected_candidates",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        "chart_worker.generation.publication_assembler._promote_key_mode",
        lambda assignments, **_kwargs: tuple(
            tmp_path / f"{difficulty.lower()}.osu"
            for difficulty, _candidate in assignments
        ),
    )

    assembly = assemble_publication(
        [(states, candidates, review)],
        prepared=SimpleNamespace(difficulty_family_resolution_enabled=True),
        authority=SimpleNamespace(sha256="a" * 64),
        run_dir=tmp_path,
    )

    assert len(assembly.variants) == 4
    assert assembly.missing == ()
    assert all(not variant.production_eligible for variant in assembly.variants)
    assert {
        variant.family_resolution_state for variant in assembly.variants
    } == {"UNRESOLVED"}
    assert len({candidate.osu_text for candidate in candidates.values()}) == 4


def test_hard_safe_raw_rejection_fills_only_its_playtest_slot(
    tmp_path, monkeypatch
) -> None:
    gate_report = {
        "action": "RETRY_MAP",
        "decisions": {
            "STRUCTURE": {"action": "PASS", "reasons": []},
            "TIMING_IDENTITY": {"action": "PASS", "reasons": []},
            "SONG_BOUNDS": {"action": "PASS", "reasons": []},
            "COVERAGE": {"action": "RETRY_MAP", "reasons": ["FIXTURE"]},
        },
        "timing": {"coverageGaps": [], "overall": {}},
    }
    candidates = {
        difficulty: SimpleNamespace(
            request=SimpleNamespace(
                key_mode=4,
                difficulty=difficulty,
                requested_star=float(index + 1),
                cfg_scale=1.0,
            ),
            generated=SimpleNamespace(),
            acceptance=SimpleNamespace(
                action=SimpleNamespace(value="RETRY_MAP")
                if difficulty == "EASY"
                else SimpleNamespace(value="PASS"),
                to_report=lambda report=gate_report: report,
            ),
            osu_text=f"unique-{difficulty}",
            workdir=tmp_path / f"source-{difficulty.lower()}",
            attempt=1,
            seed=index,
            provenance="PRIMARY",
            recovery_reason=None,
            coverage_repair_gap_count=0,
        )
        for index, difficulty in enumerate(DIFFICULTIES)
    }
    states = {
        difficulty: SimpleNamespace(
            key_mode=4,
            difficulty=difficulty,
            publication_block_reason=None,
            attempt_evidence=[],
            attempt_errors=[],
            candidates=SimpleNamespace(
                admitted=() if difficulty == "EASY" else (candidates[difficulty],),
                raw_rejected=(candidates[difficulty],) if difficulty == "EASY" else (),
                safe_fallbacks=(),
            ),
            budget=SimpleNamespace(next_attempt=2),
            recovery=SimpleNamespace(was_attempted=lambda _kind: False),
            exhausted_error=None,
        )
        for difficulty in DIFFICULTIES
    }
    assignment = {
        difficulty: None if difficulty == "EASY" else candidates[difficulty]
        for difficulty in DIFFICULTIES
    }
    monkeypatch.setattr(
        "chart_worker.generation.publication_assembler._record_unselected_candidates",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        "chart_worker.generation.publication_assembler._promote_key_mode",
        lambda assignments, **_kwargs: tuple(
            tmp_path / f"{difficulty.lower()}.osu"
            for difficulty, _candidate in assignments
        ),
    )

    assembly = assemble_publication(
        [(states, assignment, None)],
        prepared=SimpleNamespace(difficulty_family_resolution_enabled=False),
        authority=SimpleNamespace(sha256="a" * 64),
        run_dir=tmp_path,
    )

    assert len(assembly.variants) == 4
    assert assembly.missing == ()
    fallback = next(variant for variant in assembly.variants if variant.difficulty == "EASY")
    assert fallback.provenance == "RAW_UNVERIFIED"
    assert fallback.production_eligible is False
    assert fallback.family_resolution_state == "UNRESOLVED"
    assert fallback.family_resolution_reasons == (
        "QUALITY_REJECTED_HARD_SAFE_PLAYTEST_RETURN",
    )
