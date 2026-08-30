from __future__ import annotations

import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from chart_worker.boundary_review_migration import (
    build_migrated_report,
    migrate_boundary_review,
    preflight_boundary_review,
)
from chart_worker.schema.playtest_run import PlaytestRunManifestV2

FIXED_NOW = datetime(2026, 8, 10, 12, 34, 56, tzinfo=UTC)
RUN_ID = "12345678-1234-5678-9234-567812345678"
DIFFICULTIES = ("EASY", "NORMAL", "HARD", "EXPERT")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_json(path: Path, document: dict[str, Any]) -> bytes:
    body = (json.dumps(document, ensure_ascii=False, indent=2) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(body)
    return body


def make_v1_batch(tmp_path: Path) -> Path:
    batch = tmp_path / "source-batch"
    song = batch / "songs" / "01"
    audio = b"fixture-audio"
    chart = b'{"fixture":"chart"}\n'
    (song / "audio").mkdir(parents=True)
    (song / "charts").mkdir(parents=True)
    (song / "audio" / "game.flac").write_bytes(audio)
    (song / "charts" / "4k-easy.chart.json").write_bytes(chart)

    missing = [
        {"keyMode": key_mode, "difficulty": difficulty, "reason": "FIXTURE_MISSING"}
        for key_mode in (4, 6, 7)
        for difficulty in DIFFICULTIES
        if (key_mode, difficulty) != (4, "EASY")
    ]
    manifest = {
        "version": 1,
        "runId": RUN_ID,
        "title": "Fixture Song",
        "generatedAt": "2026-08-09T00:00:00Z",
        "workerVersion": "fixture-worker",
        "audio": {
            "game": {"path": "audio/game.flac", "sha256": _sha256(audio)},
            "noDrums": None,
            "keys": None,
        },
        "charts": [
            {
                "path": "charts/4k-easy.chart.json",
                "sha256": _sha256(chart),
                "keyMode": 4,
                "difficulty": "EASY",
            }
        ],
        "missingCharts": missing,
        "keysoundManifestPath": None,
        "generationReportPath": "generation-report.json",
    }
    report = {
        "version": 1,
        "runId": RUN_ID,
        "sourceName": "fixture.wav",
        "status": "PARTIAL",
        "publishable": True,
        "charts": [{"keyMode": 4, "difficulty": "EASY"}],
        "missingCharts": missing,
        "outcomeStatusV2": {
            "execution": "SUCCEEDED",
            "completeness": "PARTIAL",
            "quality": "REVIEW",
            "failureCategory": "NONE",
            "publishableStrict": False,
        },
    }
    _write_json(song / "playtest-run-v1.json", manifest)
    _write_json(song / "generation-report.json", report)
    return batch


def test_preflight_accepts_hash_bound_partial_v1_run(tmp_path: Path) -> None:
    songs = preflight_boundary_review(make_v1_batch(tmp_path))

    assert len(songs) == 1
    assert str(songs[0].manifest.run_id) == songs[0].report["runId"]
    assert songs[0].song_index == "01"


def test_preflight_rejects_changed_chart_bytes(tmp_path: Path) -> None:
    batch = make_v1_batch(tmp_path)
    (batch / "songs" / "01" / "charts" / "4k-easy.chart.json").write_bytes(
        b"tampered"
    )

    with pytest.raises(ValueError, match=r"songs[/\\]01.*SHA-256.*4k-easy"):
        preflight_boundary_review(batch)


def test_migrated_report_records_original_hashes_and_unavailable_evidence(
    tmp_path: Path,
) -> None:
    source = preflight_boundary_review(make_v1_batch(tmp_path))[0]

    document = json.loads(build_migrated_report(source, migrated_at=FIXED_NOW))

    assert document["strictBlockers"] == ["BOUNDARY_POLICY_UNCALIBRATED"]
    assert document["publicationDecision"] == {
        "decision": "PLAYTEST_ONLY",
        "policyVersion": "PUBLICATION_POLICY_V2",
        "reasonCodes": [
            "BOUNDARY_POLICY_UNCALIBRATED",
            "INCOMPLETE_CHART_SET",
            "QUALITY_REVIEW_REQUIRED",
            "STRICT_OUTCOME_FALSE",
        ],
    }
    assert document["publishable"] is False
    provenance = document["boundaryReviewMigration"]
    assert provenance["automaticEvidenceStatus"] == "UNAVAILABLE_SOURCE_REPORT"
    assert provenance["sourceManifestSha256"] == source.manifest_sha256
    assert provenance["sourceGenerationReportSha256"] == source.report_sha256
    assert provenance["migratedAt"] == "2026-08-10T12:34:56Z"


def test_migrate_builds_hardlinks_and_only_v2_manifest(tmp_path: Path) -> None:
    source = make_v1_batch(tmp_path)
    target = tmp_path / "boundary-label-v2-review"

    summary = migrate_boundary_review(source, target, migrated_at=FIXED_NOW)

    migrated = target / "songs" / "01"
    assert summary.song_count == 1
    assert (migrated / "playtest-run-v2.json").is_file()
    assert not (migrated / "playtest-run-v3.json").exists()
    assert not (migrated / "playtest-run-v1.json").exists()
    assert os.path.samefile(
        migrated / "audio" / "game.flac",
        source / "songs" / "01" / "audio" / "game.flac",
    )
    assert os.path.samefile(
        migrated / "charts" / "4k-easy.chart.json",
        source / "songs" / "01" / "charts" / "4k-easy.chart.json",
    )
    manifest = PlaytestRunManifestV2.model_validate_json(
        (migrated / "playtest-run-v2.json").read_text(encoding="utf-8")
    )
    assert manifest.generation_report.sha256 == _sha256(
        (migrated / "generation-report.json").read_bytes()
    )
    assert json.loads((target / "migration-summary.json").read_text())["songCount"] == 1


def test_migrate_removes_staging_when_hardlink_creation_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source = make_v1_batch(tmp_path)
    target = tmp_path / "boundary-label-v2-review"

    def fail_link(_source: Path, _target: Path) -> None:
        raise OSError("fixture hardlink failure")

    monkeypatch.setattr(os, "link", fail_link)

    with pytest.raises(ValueError, match="hardlink.*fixture hardlink failure"):
        migrate_boundary_review(source, target, migrated_at=FIXED_NOW)

    assert not target.exists()
    assert not tuple(tmp_path.glob(".boundary-label-v2-review.building-*"))


def test_migrate_refuses_to_overwrite_existing_target(tmp_path: Path) -> None:
    source = make_v1_batch(tmp_path)
    target = tmp_path / "boundary-label-v2-review"
    target.mkdir()

    with pytest.raises(ValueError, match="already exists"):
        migrate_boundary_review(source, target, migrated_at=FIXED_NOW)
