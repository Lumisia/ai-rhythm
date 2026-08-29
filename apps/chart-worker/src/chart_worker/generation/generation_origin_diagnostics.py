"""Strict reader for required-gameplay generated-origin accounting."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Literal

from chart_worker.generation.required_gameplay_interval import (
    RequiredGameplayGroupType,
    RequiredGameplayIntervalMode,
    RequiredGameplayIntervalV1,
)
from chart_worker.generation.resnap_diagnostics import ResnapDiagnostics
from chart_worker.hashing import sha256_file

GENERATION_ORIGIN_DIAGNOSTICS_VERSION = "generation-origin-diagnostics-v1"
GENERATION_ORIGIN_DIAGNOSTICS_SUFFIX = ".origin.json"
_MAX_DIAGNOSTICS_BYTES = 1_048_576
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")
_STAGE_NAMES = (
    "decoder",
    "windowMerge",
    "canonical",
    "resnap",
    "finalSerialization",
)
GenerationOriginLossStage = Literal[
    "windowMerge",
    "canonical",
    "resnap",
    "finalSerialization",
]


class GenerationOriginDiagnosticsError(ValueError):
    """The accounting sidecar is missing, stale, malformed, or contradictory."""


@dataclass(frozen=True, slots=True)
class GameplayGroupStageCount:
    total_generated_complete_groups: int
    interval_generated_complete_groups: int
    tap_groups: int
    hold_start_groups: int


@dataclass(frozen=True, slots=True)
class GenerationOriginDiagnostics:
    evidence_digest: str
    invocation_digest: str
    decoder: GameplayGroupStageCount
    window_merge: GameplayGroupStageCount
    canonical: GameplayGroupStageCount
    resnap: GameplayGroupStageCount
    final_serialization: GameplayGroupStageCount
    first_loss_stage: GenerationOriginLossStage | None


def _fail(message: str) -> GenerationOriginDiagnosticsError:
    return GenerationOriginDiagnosticsError(message)


def _require_sha256(value: object, field_name: str) -> str:
    if type(value) is not str or _SHA256_PATTERN.fullmatch(value) is None:
        raise _fail(f"{field_name} must be a lowercase SHA-256 digest")
    return value


def _require_nonnegative_int(value: object, field_name: str) -> int:
    if type(value) is not int:
        raise _fail(f"{field_name} must be an integer")
    if value < 0:
        raise _fail(f"{field_name} must be non-negative")
    return value


def _require_exact_object(
    value: object,
    *,
    field_name: str,
    keys: set[str],
) -> dict[str, object]:
    if type(value) is not dict:
        raise _fail(f"{field_name} must be an object")
    if set(value) != keys:
        raise _fail(f"{field_name} has an invalid schema")
    return value


def _parse_stage_count(value: object, *, stage_name: str) -> GameplayGroupStageCount:
    payload = _require_exact_object(
        value,
        field_name=f"stages.{stage_name}",
        keys={
            "totalGeneratedCompleteGroups",
            "intervalGeneratedCompleteGroups",
            "tapGroups",
            "holdStartGroups",
        },
    )
    total = _require_nonnegative_int(
        payload["totalGeneratedCompleteGroups"],
        f"stages.{stage_name}.totalGeneratedCompleteGroups",
    )
    interval = _require_nonnegative_int(
        payload["intervalGeneratedCompleteGroups"],
        f"stages.{stage_name}.intervalGeneratedCompleteGroups",
    )
    taps = _require_nonnegative_int(
        payload["tapGroups"],
        f"stages.{stage_name}.tapGroups",
    )
    holds = _require_nonnegative_int(
        payload["holdStartGroups"],
        f"stages.{stage_name}.holdStartGroups",
    )
    if interval > total:
        raise _fail(f"stages.{stage_name} interval count exceeds total count")
    if taps + holds != interval:
        raise _fail(
            f"stages.{stage_name} group-type counts do not equal interval count"
        )
    return GameplayGroupStageCount(total, interval, taps, holds)


def _expected_first_loss_stage(
    counts: tuple[GameplayGroupStageCount, ...],
) -> GenerationOriginLossStage | None:
    for index in range(1, len(counts)):
        if (
            counts[index].interval_generated_complete_groups
            < counts[index - 1].interval_generated_complete_groups
        ):
            return _STAGE_NAMES[index]  # type: ignore[return-value]
    return None


def _validate_monotonic_counts(counts: tuple[GameplayGroupStageCount, ...]) -> None:
    for previous, current in pairwise(counts):
        if (
            current.total_generated_complete_groups
            > previous.total_generated_complete_groups
            or current.interval_generated_complete_groups
            > previous.interval_generated_complete_groups
        ):
            raise _fail("generated-origin stage counts must not increase")


def _validate_required_interval(
    value: object,
    *,
    interval: RequiredGameplayIntervalV1,
) -> None:
    payload = _require_exact_object(
        value,
        field_name="requiredInterval",
        keys={
            "startMs",
            "endMs",
            "minimumCompleteGroups",
            "allowedGroupTypes",
            "evidenceClass",
            "mode",
        },
    )
    expected = {
        "startMs": interval.start_ms,
        "endMs": interval.end_ms,
        "minimumCompleteGroups": interval.minimum_complete_groups,
        "allowedGroupTypes": sorted(item.value for item in interval.allowed_group_types),
        "evidenceClass": interval.evidence_class.value,
        "mode": interval.mode.value,
    }
    if payload != expected:
        raise _fail("requiredInterval does not match the worker contract")


def _generated_interval_group_count(
    diagnostics: ResnapDiagnostics,
    *,
    interval: RequiredGameplayIntervalV1,
) -> int:
    allowed = set(interval.allowed_group_types)
    count = 0
    for item in diagnostics.mania_objects:
        group_type = (
            RequiredGameplayGroupType.TAP
            if item.kind == "TAP"
            else RequiredGameplayGroupType.HOLD_START
        )
        if group_type not in allowed:
            continue
        if not interval.start_ms <= item.start_time_ms <= interval.end_ms:
            continue
        if not any(origin.kind == "GENERATED" for origin in item.start_origins):
            continue
        count += 1
    return count


def read_generation_origin_diagnostics(
    osu_path: Path,
    *,
    interval: RequiredGameplayIntervalV1,
    expected_invocation_digest: str,
    resnap_diagnostics: ResnapDiagnostics,
) -> GenerationOriginDiagnostics:
    """Read and independently verify the adjacent v35 accounting sidecar."""

    osu_path = Path(osu_path)
    if not osu_path.is_file() or osu_path.is_symlink():
        raise _fail("output .osu must be an existing regular file")
    if type(interval) is not RequiredGameplayIntervalV1:
        raise _fail("interval must be RequiredGameplayIntervalV1")
    _require_sha256(expected_invocation_digest, "expected invocation digest")
    if type(resnap_diagnostics) is not ResnapDiagnostics:
        raise _fail("resnap_diagnostics must be ResnapDiagnostics")

    sidecar_path = osu_path.with_suffix(GENERATION_ORIGIN_DIAGNOSTICS_SUFFIX)
    if not sidecar_path.is_file() or sidecar_path.is_symlink():
        raise _fail("generation-origin sidecar is missing or not a regular file")
    if sidecar_path.stat().st_size > _MAX_DIAGNOSTICS_BYTES:
        raise _fail("generation-origin sidecar exceeds 1 MiB")
    try:
        payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise _fail(f"generation-origin sidecar is unreadable: {error}") from error
    root = _require_exact_object(
        payload,
        field_name="sidecar",
        keys={
            "version",
            "output",
            "evidenceDigest",
            "invocationDigest",
            "requiredInterval",
            "stages",
            "firstLossStage",
        },
    )
    if root["version"] != GENERATION_ORIGIN_DIAGNOSTICS_VERSION:
        raise _fail("unsupported generation-origin diagnostics version")

    output = _require_exact_object(
        root["output"],
        field_name="output",
        keys={"fileName", "size", "sha256"},
    )
    if type(output["fileName"]) is not str or output["fileName"] != osu_path.name:
        raise _fail("output fileName must name the adjacent .osu file")
    if _require_nonnegative_int(output["size"], "output.size") != osu_path.stat().st_size:
        raise _fail("output size does not match .osu")
    output_sha = _require_sha256(output["sha256"], "output.sha256")
    if output_sha != sha256_file(osu_path):
        raise _fail("output SHA-256 does not match .osu")

    evidence_digest = _require_sha256(root["evidenceDigest"], "evidence digest")
    if evidence_digest != interval.evidence_digest:
        raise _fail("evidence digest does not match the worker contract")
    invocation_digest = _require_sha256(
        root["invocationDigest"],
        "invocation digest",
    )
    if invocation_digest != expected_invocation_digest:
        raise _fail("invocation digest does not match the worker request")
    _validate_required_interval(root["requiredInterval"], interval=interval)

    stages = _require_exact_object(
        root["stages"],
        field_name="stages",
        keys=set(_STAGE_NAMES),
    )
    counts = tuple(
        _parse_stage_count(stages[name], stage_name=name) for name in _STAGE_NAMES
    )
    _validate_monotonic_counts(counts)
    first_loss_stage = root["firstLossStage"]
    if first_loss_stage is not None and first_loss_stage not in _STAGE_NAMES[1:]:
        raise _fail("firstLossStage is unsupported")
    expected_first_loss = _expected_first_loss_stage(counts)
    if first_loss_stage != expected_first_loss:
        raise _fail("firstLossStage disagrees with stage counts")

    final_origin_count = _generated_interval_group_count(
        resnap_diagnostics,
        interval=interval,
    )
    if final_origin_count != counts[-1].interval_generated_complete_groups:
        raise _fail(
            "final generated interval count disagrees with resnap origin evidence"
        )
    if (
        interval.mode is RequiredGameplayIntervalMode.SHADOW_ENFORCE
        and final_origin_count < interval.minimum_complete_groups
    ):
        raise _fail(
            "SHADOW_ENFORCE final generated interval count is below minimum"
        )

    return GenerationOriginDiagnostics(
        evidence_digest=evidence_digest,
        invocation_digest=invocation_digest,
        decoder=counts[0],
        window_merge=counts[1],
        canonical=counts[2],
        resnap=counts[3],
        final_serialization=counts[4],
        first_loss_stage=first_loss_stage,
    )
