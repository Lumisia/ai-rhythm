"""Deterministic, path-redacted runtime evidence for generation reports."""

from __future__ import annotations

import hashlib
import json
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from chart_worker.analysis.activity import (
    OUTRO_POLICY_VERSION,
    SONG_BOUNDARY_CONTRACT_VERSION,
    build_song_boundary_contract,
)
from chart_worker.analysis.onset import OnsetAnalysis
from chart_worker.config import WorkerConfig
from chart_worker.generation.mapperatorinator_patch import (
    CONSTRAINT_PATCH_ID,
    EXPECTED_MAPPERATORINATOR_HEAD,
    REQUIRED_PATCHES,
)
from chart_worker.hashing import sha256_file
from chart_worker.stages.types import PreparedAudio, SongTimingAuthority
from chart_worker.validation.quality_gate import QUALITY_GATE_VERSION

_SOURCE_ROOT = Path(__file__).resolve().parents[1]
_TIMESTAMP_CONVENTION = "frame-index-times-hop-rounded-nearest-ms-v1"


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _source_manifest_sha256() -> str:
    entries = [
        (path.relative_to(_SOURCE_ROOT).as_posix(), sha256_file(path))
        for path in sorted(_SOURCE_ROOT.rglob("*.py"))
        if path.is_file()
    ]
    return _canonical_sha256(entries)


def _patch_manifest_sha256() -> str:
    entries = [
        (patch_id, sha256_file(path))
        for patch_id, path in REQUIRED_PATCHES
        if path.is_file()
    ]
    return _canonical_sha256(entries)


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def build_runtime_fingerprint(
    *,
    config: WorkerConfig,
    prepared: PreparedAudio,
    analysis: OnsetAnalysis,
    authority: SongTimingAuthority,
    generator: str,
    worker_version: str,
) -> dict[str, object]:
    """Return reproducible evidence without leaking local absolute paths."""
    normalized = prepared.normalized
    config_sha256 = _canonical_sha256(config.model_dump(mode="json"))
    normalization_sha256 = _canonical_sha256(
        {
            "profileVersion": normalized.profile_version,
            "sampleRateHz": normalized.sample_rate_hz,
            "channels": normalized.channels,
            "durationMs": normalized.duration_ms,
            "sourceDurationMs": normalized.source_duration_ms,
            "trimmedMs": normalized.trimmed_ms,
            "gainDb": normalized.gain_db,
            "achievedLufs": normalized.achieved_lufs,
            "achievedTruePeakDbtp": normalized.achieved_true_peak_dbtp,
            "limitedBy": normalized.limited_by,
        }
    )
    payload: dict[str, object] = {
        "canonicalAudioSha256": normalized.sha256,
        "normalizationConfigSha256": normalization_sha256,
        "timingAuthoritySha256": authority.sha256,
        "workerVersion": worker_version,
        "workerRuntimeManifestSha256": _source_manifest_sha256(),
        "policyConfigSha256": config_sha256,
        "qualityGateVersion": QUALITY_GATE_VERSION,
        "outroPolicyVersion": OUTRO_POLICY_VERSION,
        "songBoundaryContractVersion": SONG_BOUNDARY_CONTRACT_VERSION,
        "songBoundaryContractSha256": (
            build_song_boundary_contract(
                analysis.activity,
                normalized.duration_ms,
                enforcement_mode=prepared.boundary_policy_mode,
                terminal_silence=analysis.terminal_silence,
            ).stable_sha256()
            if analysis.activity is not None
            else None
        ),
        "generator": generator,
        "analyzer": {
            "id": "librosa-onset",
            "version": _package_version("librosa"),
            "sampleRateHz": analysis.sample_rate_hz,
            "hopLength": analysis.hop_length,
            "nFft": analysis.n_fft,
            "timestampConvention": _TIMESTAMP_CONVENTION,
        },
        "upstream": (
            {
                "expectedCommit": EXPECTED_MAPPERATORINATOR_HEAD,
                "constraintPatchId": CONSTRAINT_PATCH_ID,
                "patchManifestSha256": _patch_manifest_sha256(),
            }
            if generator == "mapperatorinator"
            else None
        ),
        "calibrationSetId": None,
        "evidenceGrade": "VERIFIED_CODE",
    }
    return {"id": f"sha256:{_canonical_sha256(payload)}", **payload}
