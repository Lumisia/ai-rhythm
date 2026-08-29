"""Canonical identity for one required-gameplay diagnostic invocation.

The worker and patched runtime calculate this value independently.  File paths,
song metadata, and public difficulty labels are intentionally absent: content
hashes and the exact generation semantics are the only identities.
"""

from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

from chart_worker.config import WorkerConfig
from chart_worker.generation.params import GAMEMODE_MANIA, PRECISION, GenerationRequest
from chart_worker.generation.required_gameplay_interval import (
    RequiredGameplayIntervalMode,
    RequiredGameplayIntervalV1,
)
from chart_worker.hashing import sha256_file

REQUIRED_GAMEPLAY_INVOCATION_VERSION = "required-gameplay-invocation-v1"


def _regular_file(path: Path, field_name: str) -> Path:
    if not isinstance(path, Path):
        raise TypeError(f"{field_name} must be a Path")
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(
            f"{field_name} must be an existing non-symlink regular file"
        )
    return path.resolve()


def _required_interval_payload(
    interval: RequiredGameplayIntervalV1,
) -> dict[str, object]:
    if type(interval) is not RequiredGameplayIntervalV1:
        raise TypeError(
            "required_gameplay_interval must be RequiredGameplayIntervalV1"
        )
    if interval.mode not in {
        RequiredGameplayIntervalMode.OBSERVE,
        RequiredGameplayIntervalMode.SHADOW_ENFORCE,
    }:
        raise ValueError("required gameplay invocation mode is unsupported")
    return {
        "allowedGroupTypes": [item.value for item in interval.allowed_group_types],
        "endTimeMs": interval.end_ms,
        "evidenceClass": interval.evidence_class.value,
        "evidenceDigest": interval.evidence_digest,
        "minimumCompleteGroups": interval.minimum_complete_groups,
        "mode": interval.mode.value,
        "startTimeMs": interval.start_ms,
    }


def required_gameplay_invocation_payload(
    config: WorkerConfig,
    request: GenerationRequest,
) -> dict[str, object]:
    """Return the path-free semantic payload independently reproducible upstream."""

    if type(config) is not WorkerConfig:
        raise TypeError("config must be WorkerConfig")
    if type(request) is not GenerationRequest:
        raise TypeError("request must be GenerationRequest")
    interval = request.required_gameplay_interval
    if interval is None:
        raise ValueError("required gameplay invocation requires an interval")
    if request.partial_start_ms is None or request.partial_end_ms is None:
        raise ValueError("required gameplay invocation requires a partial range")
    if type(request.seed) is not int or not 0 <= request.seed < 2**32:
        raise ValueError("required gameplay invocation requires an exact uint32 seed")

    audio_path = _regular_file(request.audio_path, "audio_path")
    reference_path = _regular_file(
        request.timing_reference_path,
        "timing_reference_path",
    )
    return {
        "version": REQUIRED_GAMEPLAY_INVOCATION_VERSION,
        "input": {
            "audioSha256": sha256_file(audio_path),
            "referenceSha256": sha256_file(reference_path),
        },
        "runtime": {
            "configName": "v32",
            "fastDecoderLoop": True,
            "maniaHoldStateMode": config.mapperatorinator_hold_state_mode,
            "precision": config.mapperatorinator_precision or PRECISION,
            "resnapEvents": True,
        },
        "generation": {
            "addToBeatmap": request.add_to_beatmap,
            "cfgScale": request.cfg_scale,
            "descriptors": list(request.descriptors),
            "difficulty": request.requested_star,
            "endTimeMs": request.partial_end_ms,
            "gamemode": GAMEMODE_MANIA,
            "inContext": ["TIMING"],
            "keycount": request.key_mode,
            "lastAttackTimeMs": request.max_note_start_ms,
            "outputType": ["MAP"],
            "parallel": False,
            "seed": request.seed,
            "startTimeMs": request.partial_start_ms,
            "superTiming": False,
            "year": request.year,
        },
        "requiredGameplayInterval": _required_interval_payload(interval),
    }


def required_gameplay_invocation_digest(
    config: WorkerConfig,
    request: GenerationRequest,
) -> str:
    payload = required_gameplay_invocation_payload(config, request)
    canonical = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("ascii")
    return sha256(canonical).hexdigest()
