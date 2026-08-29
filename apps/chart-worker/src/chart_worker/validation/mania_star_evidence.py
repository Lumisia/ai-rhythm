"""Strict report-only evidence parsed from a pinned osu-tools invocation.

The parser follows PerformanceCalculator/Difficulty/DifficultyCommand.cs.  It does
not execute, download, or silently substitute an unofficial calculator.
"""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

_SUPPORTED_CALCULATOR_VERSION = 20241007
_MAX_OSU_TOOLS_OUTPUT_BYTES = 1024 * 1024
_OSU_TOOLS_TIMEOUT_SECONDS = 120


def _sha256(value: object, *, name: str) -> str:
    if type(value) is not str or len(value) != 64:
        raise TypeError(f"{name} must be a lowercase SHA-256 digest")
    if any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _commit(value: object, *, name: str) -> str:
    if type(value) is not str or len(value) != 40:
        raise TypeError(f"{name} must be a 40-character lowercase commit")
    if any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{name} must be a 40-character lowercase commit")
    return value


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _file_sha256(path: Path, *, name: str) -> str:
    if not isinstance(path, Path):
        raise TypeError(f"{name} must be a Path")
    resolved = path.resolve(strict=True)
    if not resolved.is_file():
        raise ValueError(f"{name} must be a regular file")
    with resolved.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


@dataclass(frozen=True, slots=True)
class PinnedOsuToolsManifestV1:
    tool_binary_sha256: str
    mania_ruleset_binary_sha256: str
    osu_tools_source_commit: str
    osu_source_commit: str
    calculator_version: int

    def __post_init__(self) -> None:
        _sha256(self.tool_binary_sha256, name="tool_binary_sha256")
        _sha256(
            self.mania_ruleset_binary_sha256,
            name="mania_ruleset_binary_sha256",
        )
        _commit(self.osu_tools_source_commit, name="osu_tools_source_commit")
        _commit(self.osu_source_commit, name="osu_source_commit")
        if type(self.calculator_version) is not int or (
            self.calculator_version != _SUPPORTED_CALCULATOR_VERSION
        ):
            raise ValueError("unsupported mania calculator version")


@dataclass(frozen=True, slots=True)
class ManiaStarEvidenceV1:
    input_osu_sha256: str
    tool_binary_sha256: str
    osu_tools_source_commit: str
    osu_source_commit: str
    calculator_version: int
    star_rating: float
    attributes_sha256: str
    mods: tuple[str, ...]
    verification_state: Literal[
        "PINNED_TOOL_OUTPUT_UNVERIFIED",
        "VERIFIED_PINNED_TOOL_EXECUTION",
    ] = "PINNED_TOOL_OUTPUT_UNVERIFIED"

    def __post_init__(self) -> None:
        _sha256(self.input_osu_sha256, name="input_osu_sha256")
        _sha256(self.tool_binary_sha256, name="tool_binary_sha256")
        _commit(self.osu_tools_source_commit, name="osu_tools_source_commit")
        _commit(self.osu_source_commit, name="osu_source_commit")
        if type(self.calculator_version) is not int or (
            self.calculator_version != _SUPPORTED_CALCULATOR_VERSION
        ):
            raise ValueError("unsupported mania calculator version")
        if type(self.star_rating) is not float or not math.isfinite(self.star_rating):
            raise ValueError("star_rating must be a finite exact float")
        if self.star_rating < 0:
            raise ValueError("star_rating must be non-negative")
        _sha256(self.attributes_sha256, name="attributes_sha256")
        if type(self.mods) is not tuple or any(type(mod) is not str for mod in self.mods):
            raise TypeError("mods must be a tuple of exact strings")
        if self.mods:
            raise ValueError("only the no-mod calibration feature is supported")
        if self.verification_state not in {
            "PINNED_TOOL_OUTPUT_UNVERIFIED",
            "VERIFIED_PINNED_TOOL_EXECUTION",
        }:
            raise ValueError("unsupported verification_state")

    @property
    def authorizes_calibration_feature(self) -> bool:
        return self.verification_state == "VERIFIED_PINNED_TOOL_EXECUTION"

    def to_report(self) -> dict[str, object]:
        return {
            "version": "mania-star-evidence-v1",
            "calculatorFamily": "OSU_TOOLS_MANIA",
            "calculatorVersion": self.calculator_version,
            "toolBinarySha256": self.tool_binary_sha256,
            "osuToolsSourceCommit": self.osu_tools_source_commit,
            "osuSourceCommit": self.osu_source_commit,
            "inputOsuSha256": self.input_osu_sha256,
            "mods": list(self.mods),
            "starRating": self.star_rating,
            "attributesSha256": self.attributes_sha256,
            "verificationState": self.verification_state,
            "authorizesCalibrationFeature": self.authorizes_calibration_feature,
        }


@dataclass(frozen=True, slots=True)
class ManiaStarBatchItemV1:
    osu_path: Path
    evidence: ManiaStarEvidenceV1

    def __post_init__(self) -> None:
        if not isinstance(self.osu_path, Path) or not self.osu_path.is_absolute():
            raise TypeError("osu_path must be an absolute Path")
        if not isinstance(self.evidence, ManiaStarEvidenceV1):
            raise TypeError("evidence must be ManiaStarEvidenceV1")


def parse_osu_tools_mania_difficulty(
    stdout: str,
    *,
    input_osu_sha256: str,
    tool_binary_sha256: str,
    osu_tools_source_commit: str,
    osu_source_commit: str,
    calculator_version: int,
) -> ManiaStarEvidenceV1:
    """Parse exactly one no-mod mania result from osu-tools `difficulty --json`."""
    if type(stdout) is not str:
        raise TypeError("stdout must be an exact string")
    try:
        payload = json.loads(stdout)
    except (json.JSONDecodeError, ValueError, RecursionError) as error:
        raise ValueError("osu-tools output is not strict JSON") from error
    if type(payload) is not dict or set(payload) != {"errors", "results"}:
        raise ValueError("osu-tools result set has unexpected keys")
    errors = payload["errors"]
    results = payload["results"]
    if type(errors) is not list or any(type(error) is not str for error in errors):
        raise TypeError("osu-tools errors must be a list of strings")
    if errors:
        raise ValueError("osu-tools reported a beatmap error")
    if type(results) is not list or len(results) != 1 or type(results[0]) is not dict:
        raise ValueError("osu-tools must return exactly one difficulty result")
    result = results[0]
    required = {"ruleset_id", "beatmap_id", "beatmap", "mods", "attributes"}
    if set(result) != required:
        raise ValueError("osu-tools difficulty result has unexpected keys")
    if type(result["ruleset_id"]) is not int or result["ruleset_id"] != 3:
        raise ValueError("osu-tools result is not osu!mania")
    if type(result["beatmap_id"]) is not int or type(result["beatmap"]) is not str:
        raise TypeError("osu-tools beatmap identity has invalid types")
    if type(result["mods"]) is not list or result["mods"]:
        raise ValueError("osu-tools calibration evidence must be no-mod")
    attributes = result["attributes"]
    if type(attributes) is not dict or any(type(key) is not str for key in attributes):
        raise TypeError("osu-tools attributes must be a string-keyed object")
    star = attributes.get("star_rating")
    if type(star) not in {int, float} or not math.isfinite(star) or star < 0:
        raise ValueError("osu-tools star_rating must be finite and non-negative")
    if type(calculator_version) is not int or calculator_version != _SUPPORTED_CALCULATOR_VERSION:
        raise ValueError("unsupported mania calculator version")
    return ManiaStarEvidenceV1(
        input_osu_sha256=_sha256(input_osu_sha256, name="input_osu_sha256"),
        tool_binary_sha256=_sha256(tool_binary_sha256, name="tool_binary_sha256"),
        osu_tools_source_commit=_commit(
            osu_tools_source_commit,
            name="osu_tools_source_commit",
        ),
        osu_source_commit=_commit(osu_source_commit, name="osu_source_commit"),
        calculator_version=calculator_version,
        star_rating=float(star),
        attributes_sha256=_canonical_sha256(attributes),
        mods=(),
    )


def run_pinned_osu_tools_mania_difficulty(
    *,
    osu_path: Path,
    tool_executable: Path,
    mania_ruleset_binary: Path,
    manifest: PinnedOsuToolsManifestV1,
) -> ManiaStarEvidenceV1:
    """Execute one exact pinned no-mod mania difficulty invocation."""
    if not isinstance(manifest, PinnedOsuToolsManifestV1):
        raise TypeError("manifest must be PinnedOsuToolsManifestV1")
    input_before = _file_sha256(osu_path, name="osu_path")
    tool_before = _file_sha256(tool_executable, name="tool_executable")
    ruleset_before = _file_sha256(mania_ruleset_binary, name="mania_ruleset_binary")
    if tool_before != manifest.tool_binary_sha256:
        raise ValueError("tool binary hash mismatch")
    if ruleset_before != manifest.mania_ruleset_binary_sha256:
        raise ValueError("mania ruleset binary hash mismatch")

    command = [
        str(tool_executable.resolve(strict=True)),
        "difficulty",
        str(osu_path.resolve(strict=True)),
        "-r:3",
        "-j",
    ]
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="strict",
            timeout=_OSU_TOOLS_TIMEOUT_SECONDS,
            check=False,
            shell=False,
        )
    except (OSError, subprocess.TimeoutExpired, UnicodeError) as error:
        raise ValueError("pinned osu-tools execution failed") from error
    if type(result.returncode) is not int or result.returncode != 0:
        raise ValueError("pinned osu-tools exited nonzero")
    if type(result.stderr) is not str or result.stderr:
        raise ValueError("pinned osu-tools wrote to stderr")
    if type(result.stdout) is not str:
        raise TypeError("pinned osu-tools stdout must be text")
    if len(result.stdout.encode("utf-8")) > _MAX_OSU_TOOLS_OUTPUT_BYTES:
        raise ValueError("pinned osu-tools stdout is too large")

    input_after = _file_sha256(osu_path, name="osu_path")
    tool_after = _file_sha256(tool_executable, name="tool_executable")
    ruleset_after = _file_sha256(mania_ruleset_binary, name="mania_ruleset_binary")
    if input_after != input_before:
        raise ValueError("input osu changed during pinned execution")
    if tool_after != tool_before or ruleset_after != ruleset_before:
        raise ValueError("pinned binary changed during execution")
    evidence = parse_osu_tools_mania_difficulty(
        result.stdout,
        input_osu_sha256=input_before,
        tool_binary_sha256=tool_before,
        osu_tools_source_commit=manifest.osu_tools_source_commit,
        osu_source_commit=manifest.osu_source_commit,
        calculator_version=manifest.calculator_version,
    )
    return replace(evidence, verification_state="VERIFIED_PINNED_TOOL_EXECUTION")


def run_pinned_osu_tools_mania_difficulty_batch(
    *,
    osu_paths: tuple[Path, ...],
    tool_executable: Path,
    mania_ruleset_binary: Path,
    manifest: PinnedOsuToolsManifestV1,
    max_workers: int = 4,
) -> tuple[ManiaStarBatchItemV1, ...]:
    """Run exact-file pinned calculations concurrently without losing input identity."""
    if type(osu_paths) is not tuple or not osu_paths:
        raise ValueError("osu_paths must be a non-empty tuple")
    if any(not isinstance(path, Path) for path in osu_paths):
        raise TypeError("osu_paths must contain only Path values")
    if type(max_workers) is not int or not 1 <= max_workers <= 8:
        raise ValueError("max_workers must be an exact integer between 1 and 8")
    if not isinstance(manifest, PinnedOsuToolsManifestV1):
        raise TypeError("manifest must be PinnedOsuToolsManifestV1")

    resolved = tuple(path.resolve(strict=True) for path in osu_paths)
    identities = tuple(str(path).casefold() for path in resolved)
    if len(set(identities)) != len(identities):
        raise ValueError("batch osu paths must be unique after resolution")
    ordered = tuple(sorted(resolved, key=lambda path: str(path).casefold()))
    inputs_before = {
        path: _file_sha256(path, name="batch osu path") for path in ordered
    }
    tool_before = _file_sha256(tool_executable, name="tool_executable")
    ruleset_before = _file_sha256(
        mania_ruleset_binary,
        name="mania_ruleset_binary",
    )
    if tool_before != manifest.tool_binary_sha256:
        raise ValueError("tool binary hash mismatch")
    if ruleset_before != manifest.mania_ruleset_binary_sha256:
        raise ValueError("mania ruleset binary hash mismatch")

    def calculate(path: Path) -> ManiaStarBatchItemV1:
        evidence = run_pinned_osu_tools_mania_difficulty(
            osu_path=path,
            tool_executable=tool_executable,
            mania_ruleset_binary=mania_ruleset_binary,
            manifest=manifest,
        )
        if evidence.input_osu_sha256 != inputs_before[path]:
            raise ValueError("batch evidence differs from its input digest")
        return ManiaStarBatchItemV1(osu_path=path, evidence=evidence)

    with ThreadPoolExecutor(max_workers=min(max_workers, len(ordered))) as executor:
        items = tuple(executor.map(calculate, ordered))

    for path, digest in inputs_before.items():
        if _file_sha256(path, name="batch osu path") != digest:
            raise ValueError("batch input osu changed during pinned execution")
    if (
        _file_sha256(tool_executable, name="tool_executable") != tool_before
        or _file_sha256(mania_ruleset_binary, name="mania_ruleset_binary")
        != ruleset_before
    ):
        raise ValueError("pinned binary changed during batch execution")
    return items
