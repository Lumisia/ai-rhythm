import json
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

import pytest
from pydantic import ValidationError

from chart_worker.schema.export import export_schemas
from chart_worker.schema.playtest_run import (
    AudioFileRef,
    MissingChartRef,
    OutcomeStatusSnapshot,
    PlaytestRunManifest,
    PlaytestRunManifestV2,
    PublicationDecisionSnapshot,
    ReportFileRef,
    RunAudioRefs,
    RunChartRef,
)
from chart_worker.schema.types import DIFFICULTIES, KEY_MODES

SHA = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"


def _charts() -> list[RunChartRef]:
    return [
        RunChartRef(
            key_mode=key_mode,
            difficulty=difficulty,
            path=f"charts/{key_mode}k-{difficulty.lower()}.chart.json",
            sha256=SHA,
        )
        for key_mode in KEY_MODES
        for difficulty in DIFFICULTIES
    ]


def _manifest(**overrides) -> PlaytestRunManifest:
    values = {
        "run_id": UUID(int=1),
        "title": "테스트 곡",
        "generated_at": datetime(2026, 8, 2, tzinfo=UTC),
        "worker_version": "test-build",
        "audio": RunAudioRefs(game=AudioFileRef(path="audio/game.flac", sha256=SHA)),
        "charts": _charts(),
        "generation_report_path": "generation-report.json",
    }
    return PlaytestRunManifest(**(values | overrides))


def _manifest_v2(**overrides) -> PlaytestRunManifestV2:
    values = {
        "run_id": UUID(int=2),
        "title": "fixture",
        "generated_at": datetime(2026, 8, 10, tzinfo=UTC),
        "worker_version": "test-build",
        "audio": RunAudioRefs(game=AudioFileRef(path="audio/game.flac", sha256=SHA)),
        "charts": _charts(),
        "generation_report": ReportFileRef(
            path="generation-report.json",
            sha256=SHA,
        ),
        "outcome": OutcomeStatusSnapshot(
            execution="SUCCEEDED",
            completeness="COMPLETE",
            quality="PASS",
            failure_category="NONE",
            publishable_strict=True,
        ),
        "publication": PublicationDecisionSnapshot(
            policy_version="PUBLICATION_POLICY_V2",
            decision="ALLOW_PRODUCTION",
            reason_codes=[],
        ),
    }
    return PlaytestRunManifestV2(**(values | overrides))


@pytest.mark.parametrize(
    "path",
    ["../outside.flac", "/absolute.flac", "C:/outside.flac", "audio/../../outside.flac"],
)
def test_run_manifest_rejects_unsafe_paths(path):
    with pytest.raises(ValidationError, match="safe relative path"):
        AudioFileRef(path=path, sha256=SHA)


def test_run_manifest_normalizes_windows_separators():
    reference = AudioFileRef(path=r"audio\game.flac", sha256=SHA)
    assert reference.path == "audio/game.flac"


def test_run_manifest_requires_every_combination_to_be_published_or_declared_missing():
    with pytest.raises(ValidationError, match="cover all 12 combinations"):
        _manifest(charts=_charts()[:-1])
    duplicate = [*_charts()[:-1], _charts()[0]]
    with pytest.raises(ValidationError, match="duplicate chart combination"):
        _manifest(charts=duplicate)


def test_run_manifest_accepts_a_partial_run_that_declares_what_is_missing():
    """조합 하나가 빠져도 실행은 유효하다. 대신 무엇이 빠졌는지 밝힌다."""
    dropped = _charts()[-1]
    manifest = _manifest(
        charts=_charts()[:-1],
        missing_charts=[
            MissingChartRef(
                key_mode=dropped.key_mode,
                difficulty=dropped.difficulty,
                reason="NO_PUBLISHABLE_CANDIDATE",
            )
        ],
    )

    assert len(manifest.charts) == 11
    assert manifest.missing_charts[0].reason == "NO_PUBLISHABLE_CANDIDATE"


def test_run_manifest_rejects_a_chart_that_is_both_published_and_missing():
    published = _charts()[0]
    with pytest.raises(ValidationError, match="both published and missing"):
        _manifest(
            missing_charts=[
                MissingChartRef(
                    key_mode=published.key_mode,
                    difficulty=published.difficulty,
                    reason="NO_PUBLISHABLE_CANDIDATE",
                )
            ]
        )


def test_run_manifest_rejects_an_empty_chart_list():
    with pytest.raises(ValidationError):
        _manifest(charts=[])


def test_run_manifest_requires_all_keysound_references_together():
    audio = RunAudioRefs(
        game=AudioFileRef(path="audio/game.flac", sha256=SHA),
        no_drums=AudioFileRef(path="audio/no_drums.flac", sha256=SHA),
        keys=AudioFileRef(path="audio/drums.flac", sha256=SHA),
    )
    with pytest.raises(ValidationError, match="keysound references"):
        _manifest(audio=audio)


def test_run_manifest_json_uses_camel_case_and_round_trips():
    manifest = _manifest()
    payload = manifest.model_dump_json(by_alias=True)
    assert '"runId"' in payload
    assert '"keyMode"' in payload
    assert PlaytestRunManifest.model_validate_json(payload) == manifest


def test_v2_manifest_json_binds_report_and_round_trips():
    manifest = _manifest_v2()

    payload = manifest.model_dump_json(by_alias=True)

    assert '"version":2' in payload
    assert '"generationReport"' in payload
    assert '"publishableStrict":true' in payload
    assert PlaytestRunManifestV2.model_validate_json(payload) == manifest


def test_v2_manifest_accepts_review_only_with_matching_publication_decision():
    outcome = OutcomeStatusSnapshot(
        execution="SUCCEEDED",
        completeness="COMPLETE",
        quality="REVIEW",
        failure_category="NONE",
        publishable_strict=False,
    )
    publication = PublicationDecisionSnapshot(
        policy_version="PUBLICATION_POLICY_V2",
        decision="PLAYTEST_ONLY",
        reason_codes=["QUALITY_REVIEW_REQUIRED", "STRICT_OUTCOME_FALSE"],
    )

    manifest = _manifest_v2(outcome=outcome, publication=publication)

    assert manifest.publication.decision == "PLAYTEST_ONLY"


def test_v2_manifest_rejects_production_decision_for_review_outcome():
    outcome = OutcomeStatusSnapshot(
        execution="SUCCEEDED",
        completeness="COMPLETE",
        quality="REVIEW",
        failure_category="NONE",
        publishable_strict=False,
    )

    with pytest.raises(ValidationError, match="publication decision"):
        _manifest_v2(outcome=outcome)


def test_v2_manifest_rejects_unsorted_or_duplicate_reason_codes():
    with pytest.raises(ValidationError, match="reason codes"):
        PublicationDecisionSnapshot(
            policy_version="PUBLICATION_POLICY_V2",
            decision="PLAYTEST_ONLY",
            reason_codes=["STRICT_OUTCOME_FALSE", "STRICT_OUTCOME_FALSE"],
        )


def test_v2_manifest_preserves_boundary_blocker_and_recomputes_publication():
    publication = PublicationDecisionSnapshot(
        policy_version="PUBLICATION_POLICY_V2",
        decision="PLAYTEST_ONLY",
        reason_codes=["BOUNDARY_POLICY_UNCALIBRATED"],
    )

    manifest = _manifest_v2(
        strict_blockers=["BOUNDARY_POLICY_UNCALIBRATED"],
        publication=publication,
    )

    assert manifest.strict_blockers == ["BOUNDARY_POLICY_UNCALIBRATED"]


def test_v2_manifest_rejects_unsorted_or_duplicate_strict_blockers():
    with pytest.raises(ValidationError, match="strict blockers"):
        _manifest_v2(
            strict_blockers=[
                "BOUNDARY_POLICY_UNCALIBRATED",
                "BOUNDARY_POLICY_UNCALIBRATED",
            ]
        )


def test_v2_manifest_rejects_invalid_report_sha256():
    with pytest.raises(ValidationError):
        ReportFileRef(path="generation-report.json", sha256="not-a-sha")


def test_export_schemas_writes_six_parseable_contracts(tmp_path: Path):
    paths = export_schemas(tmp_path)
    assert {path.name for path in paths} == {
        "boundary-label-v1.schema.json",
        "boundary-label-v2.schema.json",
        "chart-v1.schema.json",
        "keysound-manifest-v1.schema.json",
        "playtest-run-v1.schema.json",
        "playtest-run-v2.schema.json",
    }
    schemas = {path.name: json.loads(path.read_text(encoding="utf-8")) for path in paths}
    assert "timeMs" in schemas["chart-v1.schema.json"]["$defs"]["ChartNote"]["properties"]
    assert "drumOnsets" in schemas["keysound-manifest-v1.schema.json"]["properties"]
    assert "runId" in schemas["playtest-run-v1.schema.json"]["properties"]
    assert "labelId" in schemas["boundary-label-v1.schema.json"]["properties"]
    v2_annotation = schemas["boundary-label-v2.schema.json"]["$defs"][
        "BoundaryHumanAnnotationV2"
    ]["properties"]
    assert {
        "lastPlayableAttack",
        "primaryContentEnd",
        "acceptableReleaseEnd",
    } <= v2_annotation.keys()
