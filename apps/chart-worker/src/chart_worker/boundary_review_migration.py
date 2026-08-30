"""Verified v1-to-v2 packaging for human song-boundary review."""

from __future__ import annotations

import copy
import json
import os
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from chart_worker.hashing import sha256_file
from chart_worker.schema.playtest_run import (
    OutcomeStatusSnapshot,
    PlaytestRunManifest,
    PlaytestRunManifestV2,
    PublicationDecisionSnapshot,
    ReportFileRef,
)
from chart_worker.validation.publication_policy import decide_publication

_STRICT_BLOCKERS = ("BOUNDARY_POLICY_UNCALIBRATED",)
_EXPECTED_COMBINATIONS = {
    (key_mode, difficulty)
    for key_mode in (4, 6, 7)
    for difficulty in ("EASY", "NORMAL", "HARD", "EXPERT")
}


@dataclass(frozen=True, slots=True)
class BoundaryReviewSongSource:
    song_index: str
    song_dir: Path
    manifest_path: Path
    manifest: PlaytestRunManifest
    manifest_sha256: str
    report_path: Path
    report: dict[str, Any]
    report_sha256: str


@dataclass(frozen=True, slots=True)
class BoundaryReviewMigrationSummary:
    target_root: Path
    song_count: int
    hardlink_count: int
    migrated_report_bytes: int
    migrated_at: datetime


def _load_json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        document = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not valid UTF-8 JSON: {path}: {error}") from error
    if not isinstance(document, dict):
        raise TypeError(f"{label} must contain a JSON object: {path}")
    return document


def _report_combinations(entries: object, *, field: str, song_label: str) -> set[tuple[int, str]]:
    if not isinstance(entries, list):
        raise TypeError(f"{song_label} report {field} must be a list")
    combinations: set[tuple[int, str]] = set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise TypeError(f"{song_label} report {field} contains a non-object entry")
        key_mode = entry.get("keyMode")
        difficulty = entry.get("difficulty")
        combination = (key_mode, difficulty)
        if key_mode not in (4, 6, 7) or difficulty not in {
            "EASY",
            "NORMAL",
            "HARD",
            "EXPERT",
        }:
            raise ValueError(f"{song_label} report {field} has an invalid combination")
        if combination in combinations:
            raise ValueError(f"{song_label} report {field} has a duplicate combination")
        combinations.add(combination)
    return combinations


def _verify_ref(song_dir: Path, *, relative_path: str, expected_sha256: str) -> None:
    path = song_dir / relative_path
    if not path.is_file():
        raise ValueError(f"{song_dir} referenced file is missing: {relative_path}")
    actual_sha256 = sha256_file(path)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"{song_dir} SHA-256 mismatch for {relative_path}: "
            f"expected {expected_sha256}, received {actual_sha256}"
        )


def _preflight_song(song_dir: Path) -> BoundaryReviewSongSource:
    manifest_path = song_dir / "playtest-run-v1.json"
    if (song_dir / "playtest-run-v2.json").exists():
        raise ValueError(f"{song_dir} contains both v1 and v2 manifests")
    if not manifest_path.is_file():
        raise ValueError(f"{song_dir} is missing playtest-run-v1.json")
    try:
        manifest = PlaytestRunManifest.model_validate_json(
            manifest_path.read_text(encoding="utf-8-sig")
        )
    except (OSError, UnicodeError, ValueError) as error:
        raise ValueError(f"{song_dir} has an invalid v1 manifest: {error}") from error

    refs = [manifest.audio.game]
    refs.extend(ref for ref in (manifest.audio.no_drums, manifest.audio.keys) if ref is not None)
    refs.extend(manifest.charts)
    for ref in refs:
        _verify_ref(song_dir, relative_path=ref.path, expected_sha256=ref.sha256)
    if manifest.keysound_manifest_path is not None:
        keysound_path = song_dir / manifest.keysound_manifest_path
        if not keysound_path.is_file():
            raise ValueError(f"{song_dir} keysound manifest is missing")

    report_path = song_dir / manifest.generation_report_path
    if not report_path.is_file():
        raise ValueError(f"{song_dir} generation report is missing")
    report = _load_json_object(report_path, label=f"{song_dir} generation report")
    if report.get("runId") != str(manifest.run_id):
        raise ValueError(f"{song_dir} report runId disagrees with the v1 manifest")

    manifest_charts = {(chart.key_mode, chart.difficulty) for chart in manifest.charts}
    manifest_missing = {
        (missing.key_mode, missing.difficulty) for missing in manifest.missing_charts
    }
    report_charts = _report_combinations(
        report.get("charts"), field="charts", song_label=str(song_dir)
    )
    report_missing = _report_combinations(
        report.get("missingCharts"), field="missingCharts", song_label=str(song_dir)
    )
    if report_charts != manifest_charts or report_missing != manifest_missing:
        raise ValueError(f"{song_dir} report chart coverage disagrees with the v1 manifest")
    if report_charts & report_missing or report_charts | report_missing != _EXPECTED_COMBINATIONS:
        raise ValueError(f"{song_dir} report chart coverage does not cover all 12 combinations")

    try:
        outcome = OutcomeStatusSnapshot.model_validate(report.get("outcomeStatusV2"))
        decide_publication(
            outcome=outcome.to_domain(),
            published_slots=len(manifest.charts),
            expected_slots=12,
            strict_blockers=_STRICT_BLOCKERS,
        )
    except ValueError as error:
        raise ValueError(f"{song_dir} outcomeStatusV2 is inconsistent: {error}") from error

    return BoundaryReviewSongSource(
        song_index=song_dir.name,
        song_dir=song_dir,
        manifest_path=manifest_path,
        manifest=manifest,
        manifest_sha256=sha256_file(manifest_path),
        report_path=report_path,
        report=report,
        report_sha256=sha256_file(report_path),
    )


def preflight_boundary_review(source_batch: Path) -> tuple[BoundaryReviewSongSource, ...]:
    songs_root = Path(source_batch) / "songs"
    if not songs_root.is_dir():
        raise ValueError(f"source batch songs directory does not exist: {songs_root}")
    song_dirs = tuple(sorted(path for path in songs_root.iterdir() if path.is_dir()))
    if not song_dirs:
        raise ValueError(f"source batch has no song directories: {songs_root}")
    return tuple(_preflight_song(song_dir) for song_dir in song_dirs)


def _utc_text(value: datetime) -> str:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("migrated_at must be timezone-aware")
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def build_migrated_report(
    source: BoundaryReviewSongSource,
    *,
    migrated_at: datetime,
) -> bytes:
    outcome = OutcomeStatusSnapshot.model_validate(source.report.get("outcomeStatusV2"))
    publication = decide_publication(
        outcome=outcome.to_domain(),
        published_slots=len(source.manifest.charts),
        expected_slots=12,
        strict_blockers=_STRICT_BLOCKERS,
    )
    report = copy.deepcopy(source.report)
    report["strictBlockers"] = list(_STRICT_BLOCKERS)
    report["publicationDecision"] = publication.to_report()
    report["publishable"] = False
    report["boundaryReviewMigration"] = {
        "version": "BOUNDARY_REVIEW_V1_TO_V2_MIGRATION_V1",
        "sourceRunId": str(source.manifest.run_id),
        "sourceManifestPath": f"songs/{source.song_index}/playtest-run-v1.json",
        "sourceManifestSha256": source.manifest_sha256,
        "sourceGenerationReportPath": (
            f"songs/{source.song_index}/{source.manifest.generation_report_path}"
        ),
        "sourceGenerationReportSha256": source.report_sha256,
        "migratedAt": _utc_text(migrated_at),
        "automaticEvidenceStatus": "UNAVAILABLE_SOURCE_REPORT",
        "changedFields": [
            "boundaryReviewMigration",
            "publicationDecision",
            "publishable",
            "strictBlockers",
        ],
        "sourcePublicationFields": {
            "publishable": source.report.get("publishable"),
            "strictBlockers": source.report.get("strictBlockers"),
            "publicationDecision": source.report.get("publicationDecision"),
        },
    }
    return (json.dumps(report, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def _asset_refs(source: BoundaryReviewSongSource) -> tuple[tuple[str, str], ...]:
    refs = [(source.manifest.audio.game.path, source.manifest.audio.game.sha256)]
    refs.extend(
        (ref.path, ref.sha256)
        for ref in (source.manifest.audio.no_drums, source.manifest.audio.keys)
        if ref is not None
    )
    refs.extend((ref.path, ref.sha256) for ref in source.manifest.charts)
    return tuple(refs)


def _hardlink(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, target)
    except OSError as error:
        raise ValueError(f"hardlink failed for {source} -> {target}: {error}") from error


def _write_v2_song(
    source: BoundaryReviewSongSource,
    target_song_dir: Path,
    *,
    migrated_at: datetime,
) -> tuple[int, int, dict[str, object]]:
    hardlink_count = 0
    for relative_path, expected_sha256 in _asset_refs(source):
        source_path = source.song_dir / relative_path
        target_path = target_song_dir / relative_path
        _hardlink(source_path, target_path)
        hardlink_count += 1
        if not os.path.samefile(source_path, target_path):
            raise ValueError(f"hardlink identity mismatch for {relative_path}")
        if sha256_file(target_path) != expected_sha256:
            raise ValueError(f"target SHA-256 mismatch for {relative_path}")

    if source.manifest.keysound_manifest_path is not None:
        relative_path = source.manifest.keysound_manifest_path
        source_path = source.song_dir / relative_path
        target_path = target_song_dir / relative_path
        _hardlink(source_path, target_path)
        hardlink_count += 1
        if not os.path.samefile(source_path, target_path):
            raise ValueError(f"hardlink identity mismatch for {relative_path}")

    report_bytes = build_migrated_report(source, migrated_at=migrated_at)
    report_path = target_song_dir / "generation-report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_bytes(report_bytes)

    outcome = OutcomeStatusSnapshot.model_validate(source.report.get("outcomeStatusV2"))
    publication = decide_publication(
        outcome=outcome.to_domain(),
        published_slots=len(source.manifest.charts),
        expected_slots=12,
        strict_blockers=_STRICT_BLOCKERS,
    )
    manifest_v2 = PlaytestRunManifestV2(
        version=2,
        run_id=source.manifest.run_id,
        title=source.manifest.title,
        generated_at=source.manifest.generated_at,
        worker_version=source.manifest.worker_version,
        audio=source.manifest.audio,
        charts=source.manifest.charts,
        missing_charts=source.manifest.missing_charts,
        keysound_manifest_path=source.manifest.keysound_manifest_path,
        generation_report=ReportFileRef(
            path="generation-report.json",
            sha256=sha256_file(report_path),
        ),
        outcome=outcome,
        strict_blockers=list(_STRICT_BLOCKERS),
        publication=PublicationDecisionSnapshot.model_validate(publication.to_report()),
    )
    manifest_path = target_song_dir / "playtest-run-v2.json"
    manifest_path.write_text(
        manifest_v2.model_dump_json(by_alias=True, indent=2) + "\n",
        encoding="utf-8",
    )
    verified_manifest = PlaytestRunManifestV2.model_validate_json(
        manifest_path.read_text(encoding="utf-8")
    )
    if sha256_file(report_path) != verified_manifest.generation_report.sha256:
        raise ValueError(f"migrated report SHA-256 mismatch for song {source.song_index}")
    if (target_song_dir / "playtest-run-v1.json").exists():
        raise ValueError(f"migrated song {source.song_index} unexpectedly contains a v1 manifest")

    return (
        hardlink_count,
        len(report_bytes),
        {
            "songIndex": source.song_index,
            "runId": str(source.manifest.run_id),
            "sourceManifestSha256": source.manifest_sha256,
            "sourceGenerationReportSha256": source.report_sha256,
            "migratedGenerationReportSha256": verified_manifest.generation_report.sha256,
            "publishedChartCount": len(source.manifest.charts),
            "missingChartCount": len(source.manifest.missing_charts),
            "hardlinkCount": hardlink_count,
        },
    )


def migrate_boundary_review(
    source_batch: Path,
    target_root: Path,
    *,
    migrated_at: datetime,
) -> BoundaryReviewMigrationSummary:
    source_batch = Path(source_batch).resolve()
    target_root = Path(target_root).resolve()
    if target_root.exists():
        raise ValueError(f"migration target already exists: {target_root}")
    sources = preflight_boundary_review(source_batch)
    migrated_at_text = _utc_text(migrated_at)
    staging = target_root.with_name(f".{target_root.name}.building-{uuid4()}")
    if staging.exists():
        raise ValueError(f"migration staging path already exists: {staging}")

    hardlink_count = 0
    migrated_report_bytes = 0
    song_records: list[dict[str, object]] = []
    try:
        for source in sources:
            song_hardlinks, report_bytes, record = _write_v2_song(
                source,
                staging / "songs" / source.song_index,
                migrated_at=migrated_at,
            )
            hardlink_count += song_hardlinks
            migrated_report_bytes += report_bytes
            song_records.append(record)
        summary_document = {
            "version": "BOUNDARY_LABEL_V2_REVIEW_TREE_V1",
            "sourceBatch": str(source_batch),
            "targetRoot": str(target_root),
            "migratedAt": migrated_at_text,
            "songCount": len(sources),
            "hardlinkCount": hardlink_count,
            "migratedReportBytes": migrated_report_bytes,
            "automaticEvidenceStatus": "UNAVAILABLE_SOURCE_REPORT",
            "strictBlockers": list(_STRICT_BLOCKERS),
            "songs": song_records,
        }
        (staging / "migration-summary.json").write_text(
            json.dumps(summary_document, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        staging.replace(target_root)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise

    return BoundaryReviewMigrationSummary(
        target_root=target_root,
        song_count=len(sources),
        hardlink_count=hardlink_count,
        migrated_report_bytes=migrated_report_bytes,
        migrated_at=migrated_at,
    )
