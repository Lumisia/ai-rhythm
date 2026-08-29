from dataclasses import replace

import pytest

from chart_worker.validation.difficulty_hazard_v1 import (
    CandidateDifficultyEvidenceV1,
    assess_assignment_difficulty_hazards,
    difficulty_evidence_corpus_sha256,
)
from chart_worker.validation.mania_star_evidence import ManiaStarEvidenceV1


def _metric(candidate_id: str, official: float, project: float):
    payload = candidate_id[0] * 64
    star = ManiaStarEvidenceV1(
        input_osu_sha256=payload,
        tool_binary_sha256="1" * 64,
        osu_tools_source_commit="2" * 40,
        osu_source_commit="3" * 40,
        calculator_version=20241007,
        star_rating=float(official),
        attributes_sha256="4" * 64,
        mods=(),
        verification_state="VERIFIED_PINNED_TOOL_EXECUTION",
    )
    return CandidateDifficultyEvidenceV1(
        candidate_id=candidate_id,
        candidate_payload_sha256=payload,
        official_star=star,
        project_rating=float(project),
        project_rating_evidence_sha256="5" * 64,
    )


def test_two_axis_hazard_flags_either_nonprogressing_metric():
    assignment = tuple(
        sorted(
            (
                ("4K:EASY", "a0"),
                ("4K:NORMAL", "b0"),
                ("4K:HARD", "c0"),
                ("4K:EXPERT", "d0"),
            )
        )
    )
    metrics = (
        _metric("a0", 1.0, 1.0),
        _metric("b0", 2.0, 2.0),
        _metric("c0", 3.0, 4.0),
        _metric("d0", 3.1, 3.5),
    )

    hazards = assess_assignment_difficulty_hazards(assignment, evidence=metrics)

    assert [item.status for item in hazards] == [
        "NO_OBSERVED_RISK",
        "NO_OBSERVED_RISK",
        "AT_RISK",
    ]
    assert hazards[-1].official_delta == pytest.approx(0.1)
    assert hazards[-1].project_delta == pytest.approx(-0.5)
    assert hazards[-1].reasons == ("PROJECT_NOT_STRICTLY_INCREASING",)


def test_missing_candidate_metric_is_unknown_not_normal():
    assignment = (("4K:EASY", "a0"), ("4K:NORMAL", "b0"))

    hazards = assess_assignment_difficulty_hazards(
        assignment,
        evidence=(_metric("a0", 1.0, 1.0),),
    )

    assert len(hazards) == 1
    assert hazards[0].status == "UNKNOWN"
    assert hazards[0].official_delta is hazards[0].project_delta is None
    assert hazards[0].reasons == ("MISSING_CANDIDATE_EVIDENCE:b0",)


def test_metric_requires_verified_official_result_bound_to_payload():
    valid = _metric("a0", 1.0, 1.0)

    with pytest.raises(ValueError, match="verified pinned"):
        replace(
            valid,
            official_star=replace(
                valid.official_star,
                verification_state="PINNED_TOOL_OUTPUT_UNVERIFIED",
            ),
        )
    with pytest.raises(ValueError, match="payload"):
        replace(
            valid,
            official_star=replace(valid.official_star, input_osu_sha256="9" * 64),
        )


def test_metric_corpus_digest_is_order_invariant_and_rejects_duplicates():
    first = _metric("a0", 1.0, 1.0)
    second = _metric("b0", 2.0, 2.0)

    assert difficulty_evidence_corpus_sha256((first, second)) == (
        difficulty_evidence_corpus_sha256((second, first))
    )
    with pytest.raises(ValueError, match="unique"):
        difficulty_evidence_corpus_sha256((first, first))
