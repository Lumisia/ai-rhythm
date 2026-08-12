#!/usr/bin/env python3
"""Preregister and replay historical mania HOLD structure failures.

This script is evidence tooling. It does not alter generation policy, retry limits,
or the source batch. Registration and execution are separate operations so that the
case list is frozen before GPU work starts.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from itertools import pairwise
from pathlib import Path
from typing import Any

_HOLD_FAILURES = (
    ("HOLD_END without HOLD_START", "ORPHAN_END", "end"),
    ("overlapping HOLD_START", "OVERLAP_START", "current"),
    ("HOLD_START without HOLD_END", "UNCLOSED_START", "start"),
    ("TAP while HOLD is active", "TAP_DURING_HOLD", "tap"),
)


def _stderr_from_error(raw_error: str) -> str:
    try:
        parsed = json.loads(raw_error)
    except json.JSONDecodeError:
        return raw_error
    if not isinstance(parsed, dict):
        return raw_error
    context = parsed.get("context")
    if not isinstance(context, dict):
        return raw_error
    stderr = context.get("stderr")
    return stderr if isinstance(stderr, str) else raw_error


def classify_hold_failure(raw_error: str) -> dict[str, int | str] | None:
    """Return the directly evidenced HOLD transition failure, if present."""
    stderr = _stderr_from_error(raw_error)
    for marker, kind, object_name in _HOLD_FAILURES:
        if marker not in stderr:
            continue
        lane_match = re.search(r"in lane (\d+)", stderr)
        object_match = re.search(
            rf"{object_name}=\{{.*?['\"]timeMs['\"]:\s*(\d+)",
            stderr,
            flags=re.DOTALL,
        )
        result: dict[str, int | str] = {"kind": kind}
        if lane_match is not None:
            result["lane"] = int(lane_match.group(1))
        if object_match is not None:
            result["timeMs"] = int(object_match.group(1))
        return result
    return None


def classify_generation_failure(raw_error: str) -> dict[str, int | str] | None:
    """Classify only structural failures directly named by Mapperatorinator."""
    hold_failure = classify_hold_failure(raw_error)
    if hold_failure is not None:
        return hold_failure
    stderr = _stderr_from_error(raw_error)
    missing_column = re.search(
        r"mania note group (\d+) has no gameplay column: "
        r"type=([A-Z_]+), time=([0-9]+(?:\.[0-9]+)?)",
        stderr,
    )
    if missing_column is not None:
        return {
            "kind": "MISSING_GAMEPLAY_COLUMN",
            "groupId": int(missing_column.group(1)),
            "eventType": missing_column.group(2),
            "timeMs": round(float(missing_column.group(3))),
        }
    duplicate = re.search(
        r"exact duplicate mania group is not cross-window idempotent: "
        r"groups (\d+) and (\d+), type=([A-Z_]+), lane=(\d+), "
        r"time=([0-9]+(?:\.[0-9]+)?)",
        stderr,
    )
    if duplicate is not None:
        return {
            "kind": "NON_IDEMPOTENT_DUPLICATE_GROUP",
            "firstGroupId": int(duplicate.group(1)),
            "secondGroupId": int(duplicate.group(2)),
            "eventType": duplicate.group(3),
            "lane": int(duplicate.group(4)),
            "timeMs": round(float(duplicate.group(5))),
        }
    incomplete = re.search(
        r"incomplete mania group at end of stream: lane=(\d+)",
        stderr,
    )
    if incomplete is not None:
        return {
            "kind": "INCOMPLETE_MANIA_GROUP",
            "lane": int(incomplete.group(1)),
        }
    return None


def _result_stderr(result: dict[str, Any]) -> str:
    inline = result.get("stderrText")
    if isinstance(inline, str):
        return inline
    path = result.get("stderrPath")
    if isinstance(path, str) and Path(path).is_file():
        return Path(path).read_text(encoding="utf-8", errors="replace")
    return ""


def build_results_document(
    registration: dict[str, Any],
    run_state: dict[str, Any],
    *,
    registration_sha256: str,
    run_state_sha256: str,
) -> dict[str, Any]:
    """Bind preregistered cases to replay results and compute an audit matrix."""
    expected_runtime = registration["runtime"]["fingerprintSha256"]
    actual_runtime = run_state["runtime"]["fingerprintSha256"]
    if actual_runtime != expected_runtime:
        raise ValueError("registration and run-state runtime fingerprints differ")
    cases_by_id = {case["caseId"]: case for case in registration["cases"]}
    results_by_id = {result["caseId"]: result for result in run_state["results"]}
    if len(cases_by_id) != len(registration["cases"]):
        raise ValueError("registration contains duplicate case IDs")
    if set(cases_by_id) != set(results_by_id):
        raise ValueError("registration and run-state case IDs differ")

    transition_matrix: dict[str, dict[str, int]] = {}
    cases: list[dict[str, Any]] = []
    counts = {
        "total": len(cases_by_id),
        "pass": 0,
        "generationFailure": 0,
        "otherFailure": 0,
    }
    for case in registration["cases"]:
        result = results_by_id[case["caseId"]]
        status = result["status"]
        current_failure = (
            classify_generation_failure(_result_stderr(result))
            if status == "GENERATION_FAILURE"
            else None
        )
        if status == "PASS":
            counts["pass"] += 1
            outcome = "PASS"
        elif status == "GENERATION_FAILURE":
            counts["generationFailure"] += 1
            outcome = (
                str(current_failure["kind"])
                if current_failure is not None
                else "UNCLASSIFIED_GENERATION_FAILURE"
            )
        else:
            counts["otherFailure"] += 1
            outcome = status
        prior = str(case["priorFailure"]["kind"])
        transition_matrix.setdefault(prior, {})[outcome] = (
            transition_matrix.setdefault(prior, {}).get(outcome, 0) + 1
        )
        cases.append(
            {
                "caseId": case["caseId"],
                "originalIndex": case.get("originalIndex"),
                "sourceName": case.get("sourceName"),
                "keyMode": case["keyMode"],
                "difficulty": case["difficulty"],
                "attemptNumber": case.get("attemptNumber"),
                "seed": case["seed"],
                "priorFailure": case["priorFailure"],
                "status": status,
                "currentFailure": current_failure,
                "elapsedSec": result.get("elapsedSec"),
                "stdoutPath": result.get("stdoutPath"),
                "stdoutSha256": result.get("stdoutSha256"),
                "stderrPath": result.get("stderrPath"),
                "stderrSha256": result.get("stderrSha256"),
                "analysis": result.get("analysis"),
            }
        )
    return {
        "version": "mania-hold-state-v1-targeted-results-v1",
        "sourceVersions": {
            "registration": registration.get("version"),
            "runState": run_state.get("version"),
        },
        "runStatus": run_state["status"],
        "evidence": {
            "registrationSha256": registration_sha256,
            "runStateSha256": run_state_sha256,
            "runtimeFingerprintSha256": actual_runtime,
        },
        "counts": counts,
        "transitionMatrix": transition_matrix,
        "cases": cases,
    }


def rewrite_generation_overrides(text: str, output_path: Path) -> list[str]:
    """Preserve frozen Hydra overrides while isolating only output_path."""
    rewritten: list[str] = []
    output_value = f"output_path='{output_path}'"
    found_output = False
    for line in text.splitlines():
        value = line.strip()
        if not value:
            continue
        if not value.startswith("- "):
            raise ValueError(f"invalid Hydra override line: {line!r}")
        value = value[2:]
        if value.startswith("output_path="):
            value = output_value
            found_output = True
        rewritten.append(value)
    if not found_output:
        raise ValueError("Hydra overrides do not contain output_path")
    return rewritten


def _override_mapping(text: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for override in rewrite_generation_overrides(text, Path("__isolated_output__")):
        key, separator, value = override.partition("=")
        if not separator:
            raise ValueError(f"Hydra override does not contain '=': {override!r}")
        values[key] = value
    return values


def _unquote_hydra_string(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] == "'":
        return value[1:-1].replace("\\'", "'")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _attempt_number(path: Path) -> int:
    match = re.fullmatch(r"attempt-(\d+)", path.name)
    if match is None:
        raise ValueError(f"invalid attempt directory: {path}")
    return int(match.group(1))


def _discover_attempts(attempt_root: Path) -> list[dict[str, Any]]:
    attempts: list[dict[str, Any]] = []
    paths = sorted(attempt_root.glob("attempt-*"), key=_attempt_number)
    for path in paths:
        overrides_path = path / ".hydra-run" / ".hydra" / "overrides.yaml"
        if not overrides_path.is_file():
            raise ValueError(f"missing Hydra overrides: {overrides_path}")
        overrides_text = overrides_path.read_text(encoding="utf-8")
        overrides = _override_mapping(overrides_text)
        attempts.append(
            {
                "attempt": path.name,
                "attemptNumber": _attempt_number(path),
                "seed": int(overrides["seed"]),
                "producedOsu": any(path.rglob("*.osu")),
                "attemptPath": str(path.resolve()),
                "overridesPath": str(overrides_path.resolve()),
                "overridesSha256": _sha256(overrides_path),
                "overrides": overrides,
            }
        )
    return attempts


def _song_output_path(batch_root: Path, song: dict[str, Any]) -> Path:
    declared = song.get("outputPath")
    if isinstance(declared, str) and Path(declared).is_dir():
        return Path(declared)
    original_index = int(song["originalIndex"])
    fallback = batch_root / "songs" / f"{original_index:02d}"
    if fallback.is_dir():
        return fallback
    raise ValueError(f"song output directory does not exist: {declared!r}")


def build_registration(
    batch_root: Path,
    *,
    target_root: Path,
    runtime: dict[str, Any],
    created_at: str,
) -> dict[str, Any]:
    """Build an immutable replay contract from source reports and attempt snapshots."""
    batch_root = batch_root.resolve()
    target_root = target_root.resolve()
    batch_state_path = batch_root / "batch-state.json"
    batch_state = json.loads(batch_state_path.read_text(encoding="utf-8-sig"))
    cases: list[dict[str, Any]] = []
    ambiguous: list[dict[str, Any]] = []

    for song in batch_state.get("songs", []):
        original_index = int(song["originalIndex"])
        song_dir = _song_output_path(batch_root, song)
        report_path = song_dir / "generation-report.json"
        if not report_path.is_file():
            continue
        report = json.loads(report_path.read_text(encoding="utf-8-sig"))
        entries = [*report.get("charts", []), *report.get("missingCharts", [])]
        for entry in entries:
            attempt_errors = list(entry.get("attemptErrors") or [])
            if not any(classify_hold_failure(error) for error in attempt_errors):
                continue
            key_mode = int(entry["keyMode"])
            difficulty = str(entry["difficulty"]).upper()
            combo = f"{key_mode}k-{difficulty.lower()}"
            attempt_root = song_dir / "raw" / "work" / "epoch-1" / combo
            attempts = _discover_attempts(attempt_root)
            try:
                bound = bind_attempt_errors(attempts, attempt_errors)
            except ValueError as exc:
                ambiguous.append(
                    {
                        "originalIndex": original_index,
                        "keyMode": key_mode,
                        "difficulty": difficulty,
                        "reason": str(exc),
                        "attemptRoot": str(attempt_root.resolve()),
                    }
                )
                continue
            for item in bound:
                failure = item["holdFailure"]
                if failure is None:
                    continue
                overrides = item.pop("overrides")
                audio_path = Path(_unquote_hydra_string(overrides["audio_path"]))
                timing_path = Path(_unquote_hydra_string(overrides["beatmap_path"]))
                if not audio_path.is_file() or not timing_path.is_file():
                    raise ValueError(
                        f"registered input is missing for original {original_index} "
                        f"{combo} {item['attempt']}"
                    )
                seed = int(item["seed"])
                case_id = (
                    f"o{original_index:02d}-{key_mode}k-{difficulty.lower()}-"
                    f"a{int(item['attemptNumber']):02d}-s{seed:03d}"
                )
                request = {
                    key: value
                    for key, value in overrides.items()
                    if key != "output_path"
                }
                cases.append(
                    {
                        "caseId": case_id,
                        "originalIndex": original_index,
                        "sourceName": report.get("sourceName")
                        or song.get("sourceName"),
                        "keyMode": key_mode,
                        "difficulty": difficulty,
                        "attempt": item["attempt"],
                        "attemptNumber": item["attemptNumber"],
                        "seed": seed,
                        "priorFailure": failure,
                        "priorErrorSha256": item["errorSha256"],
                        "generationReportPath": str(report_path.resolve()),
                        "generationReportSha256": _sha256(report_path),
                        "attemptPath": item["attemptPath"],
                        "overridesPath": item["overridesPath"],
                        "overridesSha256": item["overridesSha256"],
                        "audioPath": str(audio_path.resolve()),
                        "audioSha256": _sha256(audio_path),
                        "timingReferencePath": str(timing_path.resolve()),
                        "timingReferenceSha256": _sha256(timing_path),
                        "request": request,
                        "runOutputPath": str((target_root / "runs" / case_id).resolve()),
                    }
                )

    cases.sort(key=lambda case: case["caseId"])
    counts = {
        "caseCount": len(cases),
        "ambiguousCombinationCount": len(ambiguous),
        "ORPHAN_END": sum(
            case["priorFailure"]["kind"] == "ORPHAN_END" for case in cases
        ),
        "OVERLAP_START": sum(
            case["priorFailure"]["kind"] == "OVERLAP_START" for case in cases
        ),
        "UNCLOSED_START": sum(
            case["priorFailure"]["kind"] == "UNCLOSED_START" for case in cases
        ),
    }
    return {
        "version": "mania-hold-state-v1-targeted-registration-v1",
        "createdAt": created_at,
        "sourceBatchRoot": str(batch_root),
        "sourceBatchStatePath": str(batch_state_path.resolve()),
        "sourceBatchStateSha256": _sha256(batch_state_path),
        "sourceBatchStatus": batch_state.get("status"),
        "targetRoot": str(target_root),
        "policy": {
            "thresholdsFrozen": True,
            "retryPolicyFrozen": True,
            "oneExecutionPerRegisteredCase": True,
            "outputPathOnlyOverride": True,
        },
        "runtime": runtime,
        "counts": counts,
        "ambiguousCombinations": ambiguous,
        "cases": cases,
    }


def build_replay_command(
    case: dict[str, Any], *, mapper_python: Path
) -> list[str]:
    """Build the current-runtime replay command from the frozen old overrides."""
    output_path = Path(case["runOutputPath"])
    overrides_path = Path(case["overridesPath"])
    overrides = rewrite_generation_overrides(
        overrides_path.read_text(encoding="utf-8"), output_path
    )
    return [
        str(mapper_python),
        "inference.py",
        "-cn",
        "v32",
        f"hydra.run.dir='{output_path / '.hydra-run'}'",
        *overrides,
    ]


def analyze_generated_output(output_path: Path) -> dict[str, Any]:
    """Reparse one replay output independently of Mapperatorinator success status."""
    from chart_worker.analysis.hold_lane_state import analyze_hold_lane_state
    from chart_worker.generation.osu_parser import parse_osu_file
    from chart_worker.generation.resnap_diagnostics import read_resnap_diagnostics

    osu_paths = sorted(output_path.rglob("*.osu"))
    if len(osu_paths) != 1:
        raise ValueError(
            f"expected exactly one generated .osu, found {len(osu_paths)}"
        )
    osu_path = osu_paths[0]
    beatmap = parse_osu_file(osu_path)
    diagnostics = read_resnap_diagnostics(osu_path)
    lane_state = analyze_hold_lane_state(beatmap.notes, diagnostics)
    start_times = sorted({note.time_ms for note in beatmap.notes})
    release_ends = [
        note.time_ms + (note.duration_ms or 0) for note in beatmap.notes
    ]
    max_gap = max((right - left for left, right in pairwise(start_times)), default=0)
    resnap_path = osu_path.with_suffix(".resnap.json")
    return {
        "osuPath": str(osu_path.resolve()),
        "osuSha256": _sha256(osu_path),
        "keyMode": beatmap.key_mode,
        "noteCount": len(beatmap.notes),
        "tapCount": sum(note.kind == "TAP" for note in beatmap.notes),
        "holdCount": sum(note.kind == "HOLD" for note in beatmap.notes),
        "firstNoteTimeMs": min(start_times) if start_times else None,
        "lastNoteStartMs": max(start_times) if start_times else None,
        "lastReleaseEndMs": max(release_ends) if release_ends else None,
        "maxStartGapMs": max_gap,
        "laneState": lane_state.to_report(),
        "resnapDiagnostics": {
            "status": diagnostics.status,
            "collisionCount": len(diagnostics.collisions),
            "path": str(resnap_path.resolve()) if resnap_path.is_file() else None,
            "sha256": _sha256(resnap_path) if resnap_path.is_file() else None,
        },
    }


def _utc_now() -> str:
    return dt.datetime.now(dt.UTC).isoformat().replace("+00:00", "Z")


def execute_replay_case(
    case: dict[str, Any],
    *,
    mapper_home: Path,
    mapper_python: Path,
    timeout_seconds: int,
) -> dict[str, Any]:
    """Execute one frozen request once and retain complete stdout/stderr evidence."""
    output_path = Path(case["runOutputPath"])
    if output_path.exists() and any(output_path.iterdir()):
        raise ValueError(f"replay output is not empty: {output_path}")
    output_path.mkdir(parents=True, exist_ok=True)
    command = build_replay_command(case, mapper_python=mapper_python)
    command_sha = hashlib.sha256(
        json.dumps(command, ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    started_at = _utc_now()
    started = time.monotonic()
    environment = os.environ.copy()
    environment["HYDRA_FULL_ERROR"] = "1"
    stdout = ""
    stderr = ""
    exit_code: int | None = None
    timed_out = False
    try:
        completed = subprocess.run(
            command,
            cwd=mapper_home,
            env=environment,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_seconds,
            check=False,
        )
        stdout = completed.stdout
        stderr = completed.stderr
        exit_code = completed.returncode
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        stdout = exc.stdout if isinstance(exc.stdout, str) else ""
        stderr = exc.stderr if isinstance(exc.stderr, str) else ""
        stderr += f"\nTIMEOUT after {timeout_seconds} seconds\n"
    elapsed_seconds = round(time.monotonic() - started, 3)
    stdout_path = output_path / "runner.stdout.log"
    stderr_path = output_path / "runner.stderr.log"
    stdout_path.write_text(stdout, encoding="utf-8")
    stderr_path.write_text(stderr, encoding="utf-8")
    result: dict[str, Any] = {
        "caseId": case["caseId"],
        "startedAt": started_at,
        "finishedAt": _utc_now(),
        "elapsedSec": elapsed_seconds,
        "exitCode": exit_code,
        "timedOut": timed_out,
        "command": command,
        "commandSha256": command_sha,
        "stdoutPath": str(stdout_path.resolve()),
        "stdoutSha256": _sha256(stdout_path),
        "stderrPath": str(stderr_path.resolve()),
        "stderrSha256": _sha256(stderr_path),
    }
    if timed_out:
        result.update(
            {
                "status": "ENVIRONMENT_FAILURE",
                "failureClass": "TIMEOUT",
                "analysis": None,
            }
        )
        return result
    if exit_code != 0:
        generation_failure = classify_generation_failure(stderr)
        hold_failure = classify_hold_failure(stderr)
        result.update(
            {
                "status": "GENERATION_FAILURE",
                "failureClass": (
                    str(generation_failure["kind"])
                    if generation_failure is not None
                    else "UNCLASSIFIED"
                ),
                "generationFailure": generation_failure,
                "holdFailure": hold_failure,
                "analysis": None,
            }
        )
        return result
    try:
        analysis = analyze_generated_output(output_path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        result.update(
            {
                "status": "OUTPUT_INVALID",
                "failureClass": type(exc).__name__,
                "analysisError": str(exc),
                "analysis": None,
            }
        )
        return result
    result.update(
        {
            "status": (
                "PASS"
                if analysis["laneState"]["status"] == "PASS"
                else "OUTPUT_INVALID"
            ),
            "failureClass": None,
            "analysis": analysis,
        }
    )
    return result


def verify_case_inputs(case: dict[str, Any]) -> None:
    """Fail closed if any preregistered source artifact changed after registration."""
    artifacts = (
        ("generation report", "generationReportPath", "generationReportSha256"),
        ("overrides", "overridesPath", "overridesSha256"),
        ("audio", "audioPath", "audioSha256"),
        ("timing reference", "timingReferencePath", "timingReferenceSha256"),
    )
    for label, path_key, sha_key in artifacts:
        path = Path(case[path_key])
        if not path.is_file():
            raise ValueError(f"registered {label} is missing: {path}")
        if _sha256(path) != case[sha_key]:
            raise ValueError(f"{label} SHA-256 changed: {path}")


def _run_git(mapper_home: Path, *arguments: str, binary: bool = False):
    completed = subprocess.run(
        ["git", "-C", str(mapper_home), *arguments],
        capture_output=True,
        text=not binary,
        check=False,
    )
    if completed.returncode != 0:
        stderr = completed.stderr
        if isinstance(stderr, bytes):
            stderr = stderr.decode("utf-8", errors="replace")
        raise ValueError(f"git {' '.join(arguments)} failed: {stderr.strip()}")
    return completed.stdout


def inspect_runtime(mapper_home: Path, mapper_python: Path) -> dict[str, Any]:
    """Fingerprint the exact patched runtime used by every replay case."""
    from chart_worker.generation.mapperatorinator_patch import (
        EXPECTED_MAPPERATORINATOR_HEAD,
        REQUIRED_PATCHES,
        required_patch_statuses,
    )

    head = str(_run_git(mapper_home, "rev-parse", "HEAD")).strip()
    diff_bytes = _run_git(mapper_home, "diff", "--binary", "HEAD", binary=True)
    assert isinstance(diff_bytes, bytes)
    statuses = required_patch_statuses(mapper_home)
    patches = [
        {
            "patchId": patch_id,
            "path": str(patch_path.resolve()),
            "sha256": _sha256(patch_path),
            "status": statuses[patch_id],
        }
        for patch_id, patch_path in REQUIRED_PATCHES
    ]
    core = {
        "mapperatorinatorHome": str(mapper_home.resolve()),
        "mapperatorinatorPython": str(mapper_python.resolve()),
        "expectedHead": EXPECTED_MAPPERATORINATOR_HEAD,
        "head": head,
        "trackedDiffSha256": hashlib.sha256(diff_bytes).hexdigest(),
        "inferenceSha256": _sha256(mapper_home / "inference.py"),
        "patches": patches,
        "allRequiredPatchesApplied": all(
            item["status"] == "APPLIED" for item in patches
        ),
    }
    core["fingerprintSha256"] = hashlib.sha256(
        json.dumps(core, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return core


def _write_json_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _require_same_runtime(expected: dict[str, Any], actual: dict[str, Any]) -> None:
    if actual["head"] != actual["expectedHead"]:
        raise ValueError(
            f"Mapperatorinator HEAD is not pinned: {actual['head']} != "
            f"{actual['expectedHead']}"
        )
    if not actual["allRequiredPatchesApplied"]:
        raise ValueError("not every required Mapperatorinator patch is applied")
    if actual["fingerprintSha256"] != expected["fingerprintSha256"]:
        raise ValueError(
            "Mapperatorinator runtime changed after registration: "
            f"{actual['fingerprintSha256']} != {expected['fingerprintSha256']}"
        )


def run_registration(
    registration_path: Path,
    *,
    mapper_home: Path,
    mapper_python: Path,
    timeout_seconds: int,
) -> dict[str, Any]:
    registration = _load_json(registration_path)
    registration_sha = _sha256(registration_path)
    target_root = Path(registration["targetRoot"])
    state_path = target_root / "run-state.json"
    current_runtime = inspect_runtime(mapper_home, mapper_python)
    _require_same_runtime(registration["runtime"], current_runtime)
    if state_path.is_file():
        state = _load_json(state_path)
        if state["registrationSha256"] != registration_sha:
            raise ValueError("run-state is bound to a different registration SHA-256")
    else:
        state = {
            "version": "mania-hold-state-v1-targeted-run-v1",
            "registrationPath": str(registration_path.resolve()),
            "registrationSha256": registration_sha,
            "runtime": current_runtime,
            "startedAt": _utc_now(),
            "updatedAt": _utc_now(),
            "finishedAt": None,
            "status": "RUNNING",
            "totalCaseCount": len(registration["cases"]),
            "results": [],
        }
        _write_json_atomic(state_path, state)
    completed_ids = {result["caseId"] for result in state["results"]}
    for position, case in enumerate(registration["cases"], start=1):
        if case["caseId"] in completed_ids:
            continue
        verify_case_inputs(case)
        _require_same_runtime(
            registration["runtime"], inspect_runtime(mapper_home, mapper_python)
        )
        print(
            json.dumps(
                {
                    "event": "CASE_STARTED",
                    "position": position,
                    "total": len(registration["cases"]),
                    "caseId": case["caseId"],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        result = execute_replay_case(
            case,
            mapper_home=mapper_home,
            mapper_python=mapper_python,
            timeout_seconds=timeout_seconds,
        )
        state["results"].append(result)
        state["updatedAt"] = _utc_now()
        _write_json_atomic(state_path, state)
        print(
            json.dumps(
                {
                    "event": "CASE_FINISHED",
                    "position": position,
                    "total": len(registration["cases"]),
                    "caseId": case["caseId"],
                    "status": result["status"],
                    "elapsedSec": result["elapsedSec"],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    state["finishedAt"] = _utc_now()
    state["updatedAt"] = state["finishedAt"]
    state["status"] = (
        "COMPLETE"
        if all(result["status"] == "PASS" for result in state["results"])
        else "COMPLETE_WITH_ISSUES"
    )
    _write_json_atomic(state_path, state)
    return state


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    register = subparsers.add_parser("register")
    register.add_argument("--batch-root", type=Path, required=True)
    register.add_argument("--target-root", type=Path, required=True)
    register.add_argument("--mapper-home", type=Path, required=True)
    register.add_argument("--mapper-python", type=Path, required=True)
    run = subparsers.add_parser("run")
    run.add_argument("--registration", type=Path, required=True)
    run.add_argument("--mapper-home", type=Path, required=True)
    run.add_argument("--mapper-python", type=Path, required=True)
    run.add_argument("--timeout-seconds", type=int, default=1800)
    summarize = subparsers.add_parser("summarize")
    summarize.add_argument("--registration", type=Path, required=True)
    summarize.add_argument("--run-state", type=Path, required=True)
    summarize.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.command == "register":
        runtime = inspect_runtime(arguments.mapper_home, arguments.mapper_python)
        if not runtime["allRequiredPatchesApplied"]:
            raise ValueError("cannot register: required patch stack is not applied")
        registration = build_registration(
            arguments.batch_root,
            target_root=arguments.target_root,
            runtime=runtime,
            created_at=_utc_now(),
        )
        output_path = arguments.target_root / "registration.json"
        if output_path.exists():
            raise ValueError(f"registration already exists: {output_path}")
        _write_json_atomic(output_path, registration)
        print(
            json.dumps(
                {
                    "registrationPath": str(output_path.resolve()),
                    "registrationSha256": _sha256(output_path),
                    "counts": registration["counts"],
                },
                ensure_ascii=False,
            )
        )
        return 0
    if arguments.command == "summarize":
        registration = _load_json(arguments.registration)
        run_state = _load_json(arguments.run_state)
        results = build_results_document(
            registration,
            run_state,
            registration_sha256=_sha256(arguments.registration),
            run_state_sha256=_sha256(arguments.run_state),
        )
        _write_json_atomic(arguments.output, results)
        print(
            json.dumps(
                {
                    "resultsPath": str(arguments.output.resolve()),
                    "resultsSha256": _sha256(arguments.output),
                    "counts": results["counts"],
                },
                ensure_ascii=False,
            )
        )
        return 0
    state = run_registration(
        arguments.registration,
        mapper_home=arguments.mapper_home,
        mapper_python=arguments.mapper_python,
        timeout_seconds=arguments.timeout_seconds,
    )
    print(
        json.dumps(
            {
                "status": state["status"],
                "resultCount": len(state["results"]),
            },
            ensure_ascii=False,
        )
    )
    return 0


def bind_attempt_errors(
    attempts: list[dict[str, Any]], attempt_errors: list[str]
) -> list[dict[str, Any]]:
    """Bind HOLD errors to exact primary attempts without shifting over quality errors.

    The worker appends one `attemptErrors` item for every rejected primary attempt,
    including both inference crashes and successful `.osu` files rejected by the
    quality gate. A direct `seed=N` in stderr is strongest evidence. Otherwise the
    append position is the primary attempt number, as established by the generation
    loop. A structural failure must map to a unique attempt that produced no `.osu`.
    """
    attempts_by_number = {
        int(attempt.get("attemptNumber") or _attempt_number(Path(attempt["attempt"]))):
        attempt
        for attempt in attempts
    }
    bound: list[dict[str, Any]] = []
    used_attempts: set[str] = set()
    for error_number, raw_error in enumerate(attempt_errors, start=1):
        hold_failure = classify_hold_failure(raw_error)
        if hold_failure is None:
            continue
        stderr = _stderr_from_error(raw_error)
        seed_match = re.search(r"(?:^|[^A-Za-z0-9_])seed=(\d+)", stderr)
        attempt: dict[str, Any] | None = None
        if seed_match is not None:
            seed = int(seed_match.group(1))
            candidates = [
                item
                for item in attempts
                if int(item["seed"]) == seed and not item["producedOsu"]
            ]
            if len(candidates) == 1:
                attempt = candidates[0]
        if attempt is None:
            positioned = attempts_by_number.get(error_number)
            if positioned is not None and not positioned["producedOsu"]:
                attempt = positioned
        if attempt is None or attempt["attempt"] in used_attempts:
            raise ValueError(
                f"cannot bind HOLD error {error_number} to one attempt"
            )
        used_attempts.add(attempt["attempt"])
        bound.append(
            {
                **attempt,
                "errorSha256": hashlib.sha256(raw_error.encode("utf-8")).hexdigest(),
                "holdFailure": hold_failure,
            }
        )
    return bound


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(2) from error
