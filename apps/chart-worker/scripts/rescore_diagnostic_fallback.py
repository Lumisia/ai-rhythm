"""Re-score existing Mapperatorinator artifacts without invoking the model."""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

from chart_worker.analysis.onset import analyze_canonical_audio
from chart_worker.generation.diagnostic_fallback import (
    DIAGNOSTIC_FALLBACK_VERSION,
    DiagnosticFallbackIdentity,
    DiagnosticRawCandidate,
    export_diagnostic_fallback,
    select_diagnostic_candidate,
)
from chart_worker.generation.mapperatorinator import GeneratedChart
from chart_worker.generation.osu_parser import parse_osu_mania
from chart_worker.hashing import sha256_file
from chart_worker.stages.types import SongTimingAuthority
from chart_worker.validation.generated_chart import validate_generated_chart
from chart_worker.validation.quality_gate import (
    QUALITY_GATE_VERSION,
    evaluate_chart_candidate,
)
from chart_worker.validation.timing_authority import validate_timing_identity

RESCORE_VERSION = "coverage-v4-rescore-v3-preflight-safe-insufficient-evidence"
RESCORE_REPORT_NAME = "coverage-v4-rescore-v3.json"
DIAGNOSTIC_OUTPUT_ROOT = "diagnostic-raw-fallback-v3"


def _diagnostic_manifest(
    *,
    exports: list[dict[str, object]],
    failures: list[dict[str, object]],
) -> dict[str, object]:
    return {
        "version": DIAGNOSTIC_FALLBACK_VERSION,
        "decision": "PLAYTEST_ONLY",
        "modelInvocations": 0,
        "entries": exports,
        "failures": failures,
        "rescoreReport": RESCORE_REPORT_NAME,
    }


def _write_json_once_or_identical(path: Path, payload: dict[str, object]) -> None:
    encoded = (
        json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            indent=2,
        )
        + "\n"
    ).encode("utf-8")
    if path.exists():
        if path.read_bytes() != encoded:
            raise FileExistsError(f"conflicting evidence already exists: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    if temporary.exists():
        raise FileExistsError(f"conflicting temporary evidence: {temporary}")
    try:
        with temporary.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _compact_journal_evidence(
    rows: list[dict[str, object]],
    *,
    key_mode: int,
    difficulty: str,
    attempt: int,
    seed: int,
) -> list[dict[str, object]]:
    projected: list[dict[str, object]] = []
    for row in rows:
        if (
            row.get("keyMode") != key_mode
            or row.get("difficulty") != difficulty
            or row.get("attempt") != attempt
            or row.get("seed") != seed
        ):
            continue
        payload = row.get("payload")
        payload = payload if isinstance(payload, dict) else {}
        item: dict[str, object] = {
            "sequence": row.get("sequence"),
            "eventType": row.get("eventType"),
            "purpose": payload.get("purpose"),
        }
        if row.get("eventType") == "GATE_EVALUATED":
            old_gate = payload.get("gateReport")
            if isinstance(old_gate, dict):
                item["legacyGateAction"] = old_gate.get("action")
                decisions = old_gate.get("decisions")
                if isinstance(decisions, dict):
                    item["legacyDecisionReasons"] = {
                        axis: value.get("reasons")
                        for axis, value in decisions.items()
                        if isinstance(value, dict) and value.get("reasons")
                    }
        if row.get("eventType") == "INFERENCE_FAILED":
            error = payload.get("error")
            if isinstance(error, dict):
                item["failureCode"] = error.get("code")
                item["failureContext"] = error.get("context")
        projected.append(item)
    return projected


def rescore_existing_run(run_dir: Path) -> dict[str, object]:
    run_dir = run_dir.resolve(strict=True)
    report_path = run_dir / "generation-report.json"
    journal_path = run_dir / "attempt-journal.jsonl"
    audio_path = run_dir / "audio" / "game.flac"
    timing_path = run_dir / "audio" / "timing-reference.osu"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    rows = [
        json.loads(line)
        for line in journal_path.read_text(encoding="utf-8").splitlines()
        if line
    ]

    expected_audio_sha = report.get("canonicalAudioSha256")
    expected_timing_sha = report.get("timingAuthoritySha256")
    if sha256_file(audio_path) != expected_audio_sha:
        raise ValueError("canonical audio SHA-256 does not match generation report")
    if sha256_file(timing_path) != expected_timing_sha:
        raise ValueError("timing authority SHA-256 does not match generation report")
    if report.get("mapperatorinatorHoldStateMode") != "incremental":
        raise ValueError("diagnostic re-score requires incremental HOLD mode evidence")

    timing_text = timing_path.read_text(encoding="utf-8")
    timing = parse_osu_mania(timing_text)
    authority = SongTimingAuthority(
        reference_path=timing_path,
        sha256=expected_timing_sha,
        audio_sha256=expected_audio_sha,
        bpm_events=timing.bpm_events,
        generator_name="evidence-replay",
        seed=None,
        mode=report["timingGenerationMode"],
        attempt_count=report["timingAttemptCount"],
    )
    duration_ms = report["musicBounds"]["audioDurationMs"]
    if type(duration_ms) is not int or duration_ms <= 0:
        raise ValueError("audioDurationMs must be a positive exact integer")
    analysis = analyze_canonical_audio(audio_path)

    missing = {
        (entry["keyMode"], entry["difficulty"]): entry
        for entry in report.get("missingCharts", [])
    }
    starts_by_workdir = {
        row["payload"]["workdir"]: row
        for row in rows
        if row.get("eventType") == "INFERENCE_STARTED"
        and isinstance(row.get("payload"), dict)
        and isinstance(row["payload"].get("workdir"), str)
        and (row.get("keyMode"), row.get("difficulty")) in missing
    }
    candidates: dict[tuple[int, str], list[DiagnosticRawCandidate]] = defaultdict(list)
    rescored: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []

    for workdir_text, started in sorted(starts_by_workdir.items()):
        source_workdir = (run_dir / workdir_text).resolve()
        if run_dir not in source_workdir.parents:
            raise ValueError(f"journal workdir escapes run directory: {workdir_text}")
        osu_paths = tuple(source_workdir.glob("*.osu")) if source_workdir.is_dir() else ()
        if len(osu_paths) != 1:
            failures.append(
                {
                    "keyMode": started["keyMode"],
                    "difficulty": started["difficulty"],
                    "attempt": started["attempt"],
                    "seed": started["seed"],
                    "sourceWorkdir": workdir_text,
                    "reason": "EXPECTED_EXACTLY_ONE_EXISTING_OSU",
                    "observedCount": len(osu_paths),
                }
            )
            continue
        osu_path = osu_paths[0]
        text = osu_path.read_text(encoding="utf-8")
        key_mode = started["keyMode"]
        difficulty = started["difficulty"]
        attempt = started["attempt"]
        seed = started["seed"]
        try:
            beatmap = parse_osu_mania(text)
            generated = GeneratedChart(
                notes=beatmap.notes,
                key_mode=beatmap.key_mode,
                osu_text=text,
                generator_name="mapperatorinator-existing-artifact",
                seed=seed,
                bpm_events=beatmap.bpm_events,
            )
            acceptance = evaluate_chart_candidate(
                generated,
                authority,
                analysis,
                requested_key_mode=key_mode,
                requested_difficulty=difficulty,
                duration_ms=duration_ms,
            )
            evidence = _compact_journal_evidence(
                rows,
                key_mode=key_mode,
                difficulty=difficulty,
                attempt=attempt,
                seed=seed,
            )
            candidate = DiagnosticRawCandidate.create(
                key_mode=key_mode,
                difficulty=difficulty,
                seed=seed,
                attempt=attempt,
                osu_text=text,
                source_workdir=source_workdir,
                gate_report=acceptance.to_report(),
                attempt_errors=(missing[(key_mode, difficulty)]["reason"],),
                attempt_evidence=evidence,
            )
            candidates[(key_mode, difficulty)].append(candidate)
            rescored.append(
                {
                    "keyMode": key_mode,
                    "difficulty": difficulty,
                    "attempt": attempt,
                    "seed": seed,
                    "sourcePath": osu_path.relative_to(run_dir).as_posix(),
                    "sourceSha256": sha256_file(osu_path),
                    "newGateAction": acceptance.action.value,
                    "newGateReport": acceptance.to_report(),
                    "selectionScore": list(candidate.selection_score()),
                }
            )
        except (TypeError, ValueError) as error:
            failures.append(
                {
                    "keyMode": key_mode,
                    "difficulty": difficulty,
                    "attempt": attempt,
                    "seed": seed,
                    "sourcePath": osu_path.relative_to(run_dir).as_posix(),
                    "reason": "RESCORE_OR_HARD_GATE_FAILED",
                    "errorType": type(error).__name__,
                    "message": str(error),
                }
            )

    runtime = report["runtimeFingerprint"]
    upstream = runtime.get("upstream")
    if not isinstance(upstream, dict):
        raise TypeError("runtime fingerprint lacks Mapperatorinator upstream identity")
    identity = DiagnosticFallbackIdentity(
        audio_sha256=expected_audio_sha,
        timing_sha256=expected_timing_sha,
        model_identity=runtime["id"],
        patch_set_id=upstream["constraintPatchId"],
        hold_state_mode="incremental",
    )
    exports = []
    for key_mode, difficulty in sorted(missing):
        viable = candidates.get((key_mode, difficulty), [])
        if not viable:
            failures.append(
                {
                    "keyMode": key_mode,
                    "difficulty": difficulty,
                    "reason": "NO_HARD_SAFE_EXISTING_MODEL_ARTIFACT",
                }
            )
            continue
        selected = select_diagnostic_candidate(
            viable,
            key_mode=key_mode,
            difficulty=difficulty,
        )

        def validate_osu(
            text: str,
            *,
            candidate: DiagnosticRawCandidate = selected,
            expected_key_mode: int = key_mode,
        ) -> None:
            beatmap = parse_osu_mania(text)
            generated = GeneratedChart(
                notes=beatmap.notes,
                key_mode=beatmap.key_mode,
                osu_text=text,
                generator_name="diagnostic-raw-fallback",
                seed=candidate.seed,
                bpm_events=beatmap.bpm_events,
            )
            validate_generated_chart(
                generated,
                key_mode=expected_key_mode,
                duration_ms=duration_ms,
            )
            validate_timing_identity(beatmap.bpm_events, authority.bpm_events)

        exported = export_diagnostic_fallback(
            selected,
            run_dir=run_dir,
            identity=identity,
            validate_osu=validate_osu,
            output_root_name=DIAGNOSTIC_OUTPUT_ROOT,
        )
        exports.append(exported.to_report(relative_to=run_dir))

    result = {
        "version": RESCORE_VERSION,
        "modelInvocations": 0,
        "qualityGateVersion": QUALITY_GATE_VERSION,
        "diagnosticFallbackVersion": DIAGNOSTIC_FALLBACK_VERSION,
        "sourceGenerationReport": {
            "path": report_path.relative_to(run_dir).as_posix(),
            "sha256": sha256_file(report_path),
            "qualityGateVersion": report.get("qualityGateVersion"),
        },
        "sourceAttemptJournal": {
            "path": journal_path.relative_to(run_dir).as_posix(),
            "sha256": sha256_file(journal_path),
        },
        "identity": identity.to_report(),
        "rescoredCandidates": rescored,
        "diagnosticExports": exports,
        "failures": failures,
    }
    _write_json_once_or_identical(run_dir / RESCORE_REPORT_NAME, result)
    _write_json_once_or_identical(
        run_dir / DIAGNOSTIC_OUTPUT_ROOT / "manifest-v1.json",
        _diagnostic_manifest(exports=exports, failures=failures),
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    result = rescore_existing_run(args.run_dir)
    print(
        json.dumps(
            {
                "modelInvocations": result["modelInvocations"],
                "rescoredCandidates": len(result["rescoredCandidates"]),
                "diagnosticExports": result["diagnosticExports"],
                "failureCount": len(result["failures"]),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
