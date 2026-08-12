"""Evaluate evidence-bound human labels against frozen intro phrase audits."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Sequence
from pathlib import Path

from chart_worker.analysis.intro_phrase_calibration import (
    build_intro_phrase_review_queue,
    evaluate_intro_phrase_calibration,
)

APP_ROOT = Path(__file__).resolve().parents[1]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("audits", type=Path, nargs="+")
    parser.add_argument("--cohorts", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--queue-output", type=Path)
    return parser


def _load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _resolve_implementation_path(relative_path: object) -> Path:
    if not isinstance(relative_path, str):
        raise TypeError("implementation path must be relative to chart-worker")
    candidate = (APP_ROOT / relative_path).resolve()
    try:
        candidate.relative_to(APP_ROOT)
    except ValueError as exc:
        raise ValueError("implementation path escapes chart-worker root") from exc
    return candidate


def _validate_file_bindings(
    audit_paths: list[Path],
    audits: list[dict[str, object]],
    cohorts: dict[str, object],
) -> dict[str, object]:
    audit_files = {
        str(audit["batchStateSha256"]): {
            "batchStateSha256": str(audit["batchStateSha256"]),
            "auditSha256": _sha256(path),
        }
        for path, audit in zip(audit_paths, audits)
    }
    for cohort in cohorts.get("cohorts", []):
        batch_sha = str(cohort["batchStateSha256"])
        actual = audit_files[batch_sha]["auditSha256"]
        expected = cohort.get("auditSha256")
        if not isinstance(expected, str) or expected.lower() != actual:
            raise ValueError(f"cohort audit SHA-256 does not match input: {batch_sha}")

    policy_snapshot = cohorts.get("policySnapshot")
    implementation = (
        policy_snapshot.get("implementation")
        if isinstance(policy_snapshot, dict)
        else None
    )
    if not isinstance(implementation, list) or not implementation:
        raise ValueError("policySnapshot.implementation must bind at least one file")
    implementation_files: list[dict[str, str]] = []
    for item in implementation:
        if not isinstance(item, dict):
            raise TypeError("policySnapshot.implementation entries must be objects")
        path = _resolve_implementation_path(item.get("path"))
        if not path.is_file():
            raise ValueError(f"implementation file does not exist: {item.get('path')}")
        actual = _sha256(path)
        expected = item.get("sha256")
        if not isinstance(expected, str) or expected.lower() != actual:
            raise ValueError(
                f"implementation SHA-256 does not match: {item.get('path')}"
            )
        implementation_files.append(
            {"path": str(item["path"]), "sha256": actual}
        )
    return {
        "audits": sorted(audit_files.values(), key=lambda item: item["batchStateSha256"]),
        "implementation": implementation_files,
    }


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    audits = [_load_json(path) for path in args.audits]
    cohorts = _load_json(args.cohorts)
    labels = _load_json(args.labels)
    report = evaluate_intro_phrase_calibration(audits, cohorts, labels)
    bindings = _validate_file_bindings(args.audits, audits, cohorts)
    report["inputEvidence"] = {
        "cohortManifestSha256": _sha256(args.cohorts),
        "labelSetSha256": _sha256(args.labels),
        **bindings,
    }
    _write_json(args.output, report)
    if args.queue_output is not None:
        queue = build_intro_phrase_review_queue(audits, cohorts, labels)
        _write_json(args.queue_output, queue)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
