import json
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

import pytest
from pydantic import ValidationError

from chart_worker.schema.export import export_schemas, schemas
from chart_worker.schema.playtest_run import (
    AudioFileRef,
    CoverageSummary,
    MissingChartRef,
    OutcomeStatusSnapshot,
    PlaytestRunManifest,
    PlaytestRunManifestV2,
    PublicationDecisionSnapshot,
    ReportFileRef,
    RunAudioRefs,
    RunChartRef,
    RunChartRefV2,
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
        "version": 2,
        "run_id": UUID(int=2),
        "title": "fixture",
        "generated_at": datetime(2026, 8, 10, tzinfo=UTC),
        "worker_version": "test-build",
        "audio": RunAudioRefs(game=AudioFileRef(path="audio/game.flac", sha256=SHA)),
        "charts": _charts(),
        "missing_charts": [],
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
        "strict_blockers": [],
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


@pytest.mark.parametrize("field", ("version", "missingCharts", "strictBlockers"))
def test_v2_manifest_rejects_missing_explicit_contract_fields(field):
    """Removing a V2 safety field must not silently activate a model default."""

    payload = json.loads(_manifest_v2().model_dump_json(by_alias=True))
    payload.pop(field)

    with pytest.raises(ValidationError, match=field):
        PlaytestRunManifestV2.model_validate(payload)


def test_v2_chart_ref_requires_playtest_tier_for_safe_fallback():
    fallback = RunChartRefV2(
        key_mode=4,
        difficulty="EASY",
        path="charts/4k-easy.chart.json",
        sha256=SHA,
        provenance="SAFE_FALLBACK",
        production_eligible=False,
        distribution_tier="PLAYTEST_ONLY",
    )

    assert fallback.production_eligible is False
    assert fallback.distribution_tier == "PLAYTEST_ONLY"
    with pytest.raises(ValidationError, match="fallback provenance"):
        RunChartRefV2(
            key_mode=4,
            difficulty="EASY",
            path="charts/4k-easy.chart.json",
            sha256=SHA,
            provenance="SAFE_FALLBACK",
            production_eligible=True,
            distribution_tier="PRODUCTION_CANDIDATE",
        )


@pytest.mark.parametrize("family_state", ("NARROW_REVIEW", "UNRESOLVED"))
def test_v2_chart_ref_keeps_non_resolved_family_out_of_production(family_state):
    """Removing the family-state guard must not publish unresolved evidence."""

    summary = CoverageSummary(
        first_note_time_ms=250,
        max_gap_ms=2_000,
        attack_required_gap_count=0,
        attack_required_gap_total_ms=0,
        repaired_gap_count=0,
    )
    chart = RunChartRefV2(
        key_mode=4,
        difficulty="HARD",
        path="charts/4k-hard.chart.json",
        sha256=SHA,
        family_resolution_state=family_state,
        family_resolution_reasons=["FAMILY_ORDER_NOT_PROVEN"],
        production_eligible=False,
        distribution_tier="PLAYTEST_ONLY",
        playability_tier="MODEL_PLAYABLE",
        coverage_summary=summary,
    )

    assert chart.family_resolution_state == family_state
    assert chart.production_eligible is False
    with pytest.raises(ValidationError, match="unresolved family must be playtest-only"):
        RunChartRefV2(
            key_mode=4,
            difficulty="HARD",
            path="charts/4k-hard.chart.json",
            sha256=SHA,
            family_resolution_state=family_state,
            family_resolution_reasons=["FAMILY_ORDER_NOT_PROVEN"],
            production_eligible=True,
            distribution_tier="PRODUCTION_CANDIDATE",
        )


def test_v2_chart_ref_requires_canonical_family_resolution_reasons():
    with pytest.raises(ValidationError, match="requires reason codes"):
        RunChartRefV2(
            key_mode=4,
            difficulty="HARD",
            path="charts/4k-hard.chart.json",
            sha256=SHA,
            family_resolution_state="UNRESOLVED",
            production_eligible=False,
            distribution_tier="PLAYTEST_ONLY",
        )
    with pytest.raises(ValidationError, match="sorted and unique"):
        RunChartRefV2(
            key_mode=4,
            difficulty="HARD",
            path="charts/4k-hard.chart.json",
            sha256=SHA,
            family_resolution_state="UNRESOLVED",
            family_resolution_reasons=["Z_REASON", "A_REASON", "Z_REASON"],
            production_eligible=False,
            distribution_tier="PLAYTEST_ONLY",
        )


def test_v2_chart_ref_carries_honest_coverage_repair_playability():
    summary = CoverageSummary(
        first_note_time_ms=250,
        max_gap_ms=2_000,
        attack_required_gap_count=0,
        attack_required_gap_total_ms=0,
        repaired_gap_count=2,
    )
    repaired = RunChartRefV2(
        key_mode=4,
        difficulty="EASY",
        path="charts/4k-easy.chart.json",
        sha256=SHA,
        provenance="COVERAGE_REPAIR",
        production_eligible=False,
        distribution_tier="PLAYTEST_ONLY",
        playability_tier="RECOVERY_PLAYABLE",
        coverage_summary=summary,
    )

    assert repaired.coverage_summary == summary
    assert repaired.playability_tier == "RECOVERY_PLAYABLE"
    with pytest.raises(ValidationError, match="RAW_UNVERIFIED"):
        RunChartRefV2(
            key_mode=4,
            difficulty="EASY",
            path="charts/4k-easy.chart.json",
            sha256=SHA,
            provenance="RAW_UNVERIFIED",
            production_eligible=False,
            distribution_tier="PLAYTEST_ONLY",
            playability_tier="MODEL_PLAYABLE",
            coverage_summary=summary,
        )


def test_v2_chart_ref_requires_summary_when_new_playability_tier_is_present():
    with pytest.raises(ValidationError, match="coverage summary"):
        RunChartRefV2(
            key_mode=4,
            difficulty="EASY",
            path="charts/4k-easy.chart.json",
            sha256=SHA,
            provenance="SAFE_FALLBACK",
            production_eligible=False,
            distribution_tier="PLAYTEST_ONLY",
            playability_tier="RECOVERY_PLAYABLE",
        )


def test_v2_chart_ref_marks_cross_difficulty_assignment_as_playtest_only():
    summary = CoverageSummary(
        first_note_time_ms=250,
        max_gap_ms=2_000,
        attack_required_gap_count=0,
        attack_required_gap_total_ms=0,
        repaired_gap_count=0,
    )
    reassigned = RunChartRefV2(
        key_mode=4,
        difficulty="EXPERT",
        path="charts/4k-expert.chart.json",
        sha256=SHA,
        provenance="PRIMARY",
        family_assignment_kind="REASSIGNED",
        source_difficulty="HARD",
        production_eligible=False,
        distribution_tier="PLAYTEST_ONLY",
        playability_tier="RECOVERY_PLAYABLE",
        coverage_summary=summary,
    )

    assert reassigned.family_assignment_kind == "REASSIGNED"
    assert reassigned.source_difficulty == "HARD"
    with pytest.raises(ValidationError, match="family assignment"):
        RunChartRefV2(
            key_mode=4,
            difficulty="EXPERT",
            path="charts/4k-expert.chart.json",
            sha256=SHA,
            provenance="PRIMARY",
            family_assignment_kind="EMERGENCY_DUPLICATE",
            source_difficulty="HARD",
            production_eligible=True,
            distribution_tier="PRODUCTION_CANDIDATE",
        )


def test_v2_chart_ref_requires_source_difficulty_for_adapted_assignment():
    with pytest.raises(ValidationError, match="source difficulty"):
        RunChartRefV2(
            key_mode=4,
            difficulty="EXPERT",
            path="charts/4k-expert.chart.json",
            sha256=SHA,
            provenance="PRIMARY",
            family_assignment_kind="REASSIGNED",
            production_eligible=False,
            distribution_tier="PLAYTEST_ONLY",
        )


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


def test_committed_playtest_v2_schema_matches_the_python_contract():
    project_root = Path(__file__).resolve().parents[4]
    committed = json.loads(
        (project_root / "packages/chart-schema/playtest-run-v2.schema.json")
        .read_text(encoding="utf-8")
    )

    assert committed == schemas()["playtest-run-v2.schema.json"]
