from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


def _load_audit_module():
    script_path = Path(__file__).parents[2] / "scripts" / "audit_intro_phrase_family.py"
    spec = importlib.util.spec_from_file_location("audit_intro_phrase_family", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_batch_fixture(tmp_path: Path, *, manifest_version: int) -> dict[str, Path | str]:
    song_dir = tmp_path / "songs" / "01"
    report_path = song_dir / "generation-report.json"
    hard_path = song_dir / "charts" / "4k-hard.chart.json"
    expert_path = song_dir / "charts" / "4k-expert.chart.json"
    _write_json(
        hard_path,
        {
            "notes": [{"timeMs": 100}, {"timeMs": 300}],
            "bpmEvents": [{"timeMs": 0, "bpm": 120.0}],
        },
    )
    _write_json(
        expert_path,
        {
            "notes": [{"timeMs": 0}, {"timeMs": 12_000}],
            "bpmEvents": [{"timeMs": 0, "bpm": 120.0}],
        },
    )
    _write_json(
        report_path,
        {
            "charts": [
                {
                    "keyMode": 4,
                    "difficulty": "HARD",
                    "chartPath": "charts/4k-hard.chart.json",
                    "selectedSeed": 2,
                    "attemptCount": 1,
                },
                {
                    "keyMode": 4,
                    "difficulty": "EXPERT",
                    "chartPath": "charts/4k-expert.chart.json",
                    "selectedSeed": 3,
                    "attemptCount": 1,
                },
            ],
            "introStartContract": {"candidates": []},
        },
    )
    audio_sha = "a" * 64
    if manifest_version == 1:
        manifest_path = song_dir / "playtest-run-v1.json"
        manifest = {
            "version": 1,
            "audio": {"game": {"path": "audio/game.flac", "sha256": audio_sha}},
        }
    else:
        manifest_path = song_dir / "playtest-run-v2.json"
        manifest = {
            "version": 2,
            "audio": {"asset": {"path": "audio/game.flac", "sha256": audio_sha}},
            "generationReport": {
                "path": "generation-report.json",
                "sha256": _sha256(report_path),
            },
        }
    _write_json(manifest_path, manifest)
    state_path = tmp_path / "batch-state.json"
    _write_json(
        state_path,
        {
            "startedAt": "2026-08-11T01:00:00Z",
            "finishedAt": "2026-08-11T02:00:00Z",
            "songs": [
                {
                    "index": 1,
                    "sourceName": "fixture.wav",
                    "outputPath": str(song_dir),
                }
            ]
        },
    )
    return {
        "songDir": song_dir,
        "reportPath": report_path,
        "hardPath": hard_path,
        "expertPath": expert_path,
        "audioSha": audio_sha,
        "statePath": state_path,
        "manifestPath": manifest_path,
    }


def test_audit_rows_bind_batch_report_audio_and_selected_candidates(tmp_path: Path):
    fixture = _write_batch_fixture(tmp_path, manifest_version=1)

    audit = _load_audit_module().audit_batch(tmp_path)
    row = next(item for item in audit["rows"] if item["keyMode"] == 4)

    assert row["batchStateSha256"] == _sha256(fixture["statePath"])
    assert audit["batchStartedAt"] == "2026-08-11T01:00:00Z"
    assert audit["batchFinishedAt"] == "2026-08-11T02:00:00Z"
    assert row["generationReportSha256"] == _sha256(fixture["reportPath"])
    assert row["audioSha256"] == fixture["audioSha"]
    assert row["review"]["hard"]["candidateId"] == (
        f"charts/4k-hard.chart.json@sha256:{_sha256(fixture['hardPath'])}"
    )
    assert row["review"]["expert"]["candidateId"] == (
        f"charts/4k-expert.chart.json@sha256:{_sha256(fixture['expertPath'])}"
    )


def test_audit_reads_v2_canonical_audio_identity(tmp_path: Path):
    fixture = _write_batch_fixture(tmp_path, manifest_version=2)

    audit = _load_audit_module().audit_batch(tmp_path)
    row = next(item for item in audit["rows"] if item["keyMode"] == 4)

    assert row["audioSha256"] == fixture["audioSha"]


def test_audit_rejects_v2_manifest_bound_to_a_different_report(tmp_path: Path):
    fixture = _write_batch_fixture(tmp_path, manifest_version=2)
    manifest_path = fixture["manifestPath"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["generationReport"]["sha256"] = "f" * 64
    _write_json(manifest_path, manifest)

    with pytest.raises(ValueError, match="generation report SHA-256 does not match"):
        _load_audit_module().audit_batch(tmp_path)


def test_audit_preserves_runtime_defect_when_expert_was_removed_from_publication(
    tmp_path: Path,
):
    fixture = _write_batch_fixture(tmp_path, manifest_version=1)
    report_path = fixture["reportPath"]
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["charts"] = [
        chart
        for chart in report["charts"]
        if not (chart["keyMode"] == 4 and chart["difficulty"] == "EXPERT")
    ]
    report["introPhraseFamilyReviews"] = [
        {
            "version": "intro-phrase-family-v1",
            "mode": "ACTIVE_FOR_HIGH_CONFIDENCE_DEFECT",
            "policyState": "PROVISIONAL",
            "status": "DEFECT",
            "reason": "ISOLATED_EXPERT_FIRST_ROW",
            "shouldRecover": True,
            "shouldBlockPublication": True,
            "hard": {"keyMode": 4, "difficulty": "HARD"},
            "expert": {"keyMode": 4, "difficulty": "EXPERT"},
        }
    ]
    _write_json(report_path, report)

    audit = _load_audit_module().audit_batch(tmp_path)
    row = next(item for item in audit["rows"] if item["keyMode"] == 4)

    assert row["runtimeReview"]["status"] == "DEFECT"
    assert row["publishedReview"]["status"] == "INSUFFICIENT"
    assert row["review"] == row["runtimeReview"]
    assert row["reviewSource"] == "GENERATION_RUNTIME"
    assert audit["statusCounts"]["DEFECT"] == 1
