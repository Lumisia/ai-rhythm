from __future__ import annotations

from copy import deepcopy

import pytest
from pydantic import ValidationError

from chart_worker.schema.boundary_label import (
    BoundaryLabelV1,
    BoundaryLabelV2,
    boundary_label_json_schema,
    boundary_label_v2_json_schema,
)


def valid_label() -> dict[str, object]:
    return {
        "version": 1,
        "labelId": "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa",
        "createdAt": "2026-08-10T00:00:00Z",
        "reviewerId": "reviewer-a",
        "run": {
            "runId": "40000000-0000-4000-8000-000000000001",
            "title": "Fixture",
            "songVersionId": "10000000-0000-4000-8000-000000000001",
            "gameAudioAssetId": "20000000-0000-4000-8000-000000000001",
        },
        "audio": {"sha256": "a" * 64, "durationMs": 10_000},
        "generationReport": {
            "path": "generation-report.json",
            "sha256": "b" * 64,
        },
        "group": {
            "groupId": "song-group-a",
            "relation": "EXACT_RECORDING",
            "confirmed": True,
        },
        "automaticEvidence": {
            "availability": "AVAILABLE",
            "unavailableReason": None,
            "evaluationVersion": "boundary-policy-evaluation-v1",
            "policyState": "PROVISIONAL",
            "policyConfidence": "UNKNOWN",
            "enforcementMode": "SHADOW",
            "observationSha256": "c" * 64,
            "lastDetectedOnsetMs": 8_800,
            "lastActiveRmsEndMs": 9_000,
            "lastEvidenceMs": 9_000,
            "provisionalMaxNoteStartMs": 9_070,
            "provisionalReleaseEndMs": 10_000,
            "effectiveMaxNoteStartMs": 10_000,
            "effectiveReleaseEndMs": 10_000,
        },
        "annotation": {
            "lastMeaningfulAttack": {"earliestMs": 8_800, "latestMs": 9_000},
            "lastAcceptableRelease": {"earliestMs": 9_000, "latestMs": 9_500},
            "provisionalBoundaryVerdict": "ACCEPTABLE",
            "tailCharacters": ["FADE_OR_REVERB"],
            "confidence": "MEDIUM",
            "comment": "잔향 경계가 약간 모호함",
        },
    }


def valid_v2_label() -> dict[str, object]:
    payload = deepcopy(valid_label())
    payload["version"] = 2
    payload["annotation"] = {
        "lastPlayableAttack": {"earliestMs": 8_700, "latestMs": 8_800},
        "primaryContentEnd": {"earliestMs": 9_000, "latestMs": 9_100},
        "acceptableReleaseEnd": {"earliestMs": 9_400, "latestMs": 9_500},
        "provisionalBoundaryVerdict": "ACCEPTABLE",
        "tailCharacters": ["FADE_OR_REVERB"],
        "confidence": "MEDIUM",
        "comment": "새 타점과 지속음 끝을 분리함",
    }
    return payload


def annotation(payload: dict[str, object]) -> dict[str, object]:
    value = payload["annotation"]
    assert isinstance(value, dict)
    return value


def automatic_evidence(payload: dict[str, object]) -> dict[str, object]:
    value = payload["automaticEvidence"]
    assert isinstance(value, dict)
    return value


def test_boundary_label_round_trips_with_aliases() -> None:
    model = BoundaryLabelV1.model_validate(valid_label())

    payload = model.model_dump_json(by_alias=True)

    assert '"lastMeaningfulAttack"' in payload
    assert BoundaryLabelV1.model_validate_json(payload) == model


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("reviewerId", " "),
        ("labelId", "not-a-uuid"),
    ],
)
def test_boundary_label_rejects_invalid_identity(field: str, value: object) -> None:
    payload = valid_label()
    payload[field] = value

    with pytest.raises(ValidationError):
        BoundaryLabelV1.model_validate(payload)


@pytest.mark.parametrize(
    ("interval_name", "earliest_ms", "latest_ms"),
    [
        ("lastMeaningfulAttack", 9_001, 9_000),
        ("lastMeaningfulAttack", -1, 9_000),
        ("lastAcceptableRelease", 9_000, 10_001),
        ("lastAcceptableRelease", 9_000.5, 9_500),
    ],
)
def test_boundary_label_rejects_invalid_time_intervals(
    interval_name: str,
    earliest_ms: object,
    latest_ms: object,
) -> None:
    payload = valid_label()
    annotation(payload)[interval_name] = {
        "earliestMs": earliest_ms,
        "latestMs": latest_ms,
    }

    with pytest.raises(ValidationError):
        BoundaryLabelV1.model_validate(payload)


def test_boundary_label_rejects_attack_after_release() -> None:
    payload = valid_label()
    annotation(payload)["lastMeaningfulAttack"] = {
        "earliestMs": 9_501,
        "latestMs": 9_600,
    }

    with pytest.raises(ValidationError, match="attack"):
        BoundaryLabelV1.model_validate(payload)


@pytest.mark.parametrize(
    "characters",
    [
        [],
        ["NOISE", "NOISE"],
        ["MIXED_OR_UNCERTAIN", "SILENCE"],
    ],
)
def test_boundary_label_rejects_invalid_tail_characters(characters: list[str]) -> None:
    payload = valid_label()
    annotation(payload)["tailCharacters"] = characters

    with pytest.raises(ValidationError, match="tail"):
        BoundaryLabelV1.model_validate(payload)


def test_unavailable_evidence_requires_not_available_verdict() -> None:
    payload = valid_label()
    evidence = automatic_evidence(payload)
    for key in tuple(evidence):
        evidence[key] = None
    evidence["availability"] = "UNAVAILABLE"
    evidence["unavailableReason"] = "musicBounds.boundaryPolicyEvaluation is missing"

    with pytest.raises(ValidationError, match="NOT_AVAILABLE"):
        BoundaryLabelV1.model_validate(payload)

    annotation(payload)["provisionalBoundaryVerdict"] = "NOT_AVAILABLE"
    model = BoundaryLabelV1.model_validate(payload)
    assert model.annotation.confidence == "MEDIUM"


def test_available_evidence_rejects_not_available_verdict() -> None:
    payload = valid_label()
    annotation(payload)["provisionalBoundaryVerdict"] = "NOT_AVAILABLE"

    with pytest.raises(ValidationError, match="comparable verdict"):
        BoundaryLabelV1.model_validate(payload)


def test_low_confidence_and_unknown_group_are_preserved() -> None:
    payload = deepcopy(valid_label())
    group = payload["group"]
    assert isinstance(group, dict)
    group.update({"relation": "UNKNOWN", "confirmed": False})
    annotation(payload)["confidence"] = "LOW"
    annotation(payload)["provisionalBoundaryVerdict"] = "UNCERTAIN"

    model = BoundaryLabelV1.model_validate(payload)

    assert model.group.relation == "UNKNOWN"
    assert model.group.confirmed is False
    assert model.annotation.confidence == "LOW"


def test_boundary_label_schema_uses_json_aliases() -> None:
    schema = boundary_label_json_schema()

    assert "labelId" in schema["properties"]
    assert "lastMeaningfulAttack" in schema["$defs"]["BoundaryHumanAnnotation"]["properties"]


def test_v2_preserves_three_distinct_human_intervals() -> None:
    label = BoundaryLabelV2.model_validate(valid_v2_label())

    assert label.annotation.last_playable_attack.latest_ms == 8_800
    assert label.annotation.primary_content_end.latest_ms == 9_100
    assert label.annotation.acceptable_release_end.latest_ms == 9_500
    assert BoundaryLabelV2.model_validate_json(
        label.model_dump_json(by_alias=True)
    ) == label


@pytest.mark.parametrize(
    ("field", "earliest_ms", "latest_ms", "message"),
    [
        ("lastPlayableAttack", 9_600, 9_700, "playable attack"),
        ("primaryContentEnd", 9_600, 9_700, "content end"),
    ],
)
def test_v2_rejects_impossible_interval_order(
    field: str,
    earliest_ms: int,
    latest_ms: int,
    message: str,
) -> None:
    payload = valid_v2_label()
    annotation(payload)[field] = {
        "earliestMs": earliest_ms,
        "latestMs": latest_ms,
    }

    with pytest.raises(ValidationError, match=message):
        BoundaryLabelV2.model_validate(payload)


def test_v2_allows_overlapping_uncertainty_intervals() -> None:
    payload = valid_v2_label()
    annotation(payload)["lastPlayableAttack"] = {
        "earliestMs": 8_900,
        "latestMs": 9_050,
    }
    annotation(payload)["primaryContentEnd"] = {
        "earliestMs": 9_000,
        "latestMs": 9_150,
    }

    label = BoundaryLabelV2.model_validate(payload)

    assert label.annotation.last_playable_attack.latest_ms == 9_050
    assert label.annotation.primary_content_end.earliest_ms == 9_000


def test_boundary_label_v2_schema_uses_semantic_field_names() -> None:
    schema = boundary_label_v2_json_schema()
    properties = schema["$defs"]["BoundaryHumanAnnotationV2"]["properties"]

    assert tuple(properties)[:3] == (
        "lastPlayableAttack",
        "primaryContentEnd",
        "acceptableReleaseEnd",
    )
