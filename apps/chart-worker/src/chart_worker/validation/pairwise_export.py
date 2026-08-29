"""Fail-closed export of private and blinded pairwise review bundles."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from chart_worker.validation.pairwise_labels import (
    build_pairwise_task_bundle,
    parse_candidate_label_binding_v1,
)

_SOURCE_VERSION = "difficulty-pairwise-export-source-v1"
_MAX_SOURCE_BYTES = 16 * 1024 * 1024


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _read_source(path: Path) -> tuple[dict[str, object], str]:
    if not isinstance(path, Path):
        raise TypeError("source_path must be a Path")
    source = path.resolve(strict=True)
    if not source.is_file():
        raise ValueError("pairwise export source must be a regular file")
    if source.stat().st_size > _MAX_SOURCE_BYTES:
        raise ValueError("pairwise export source exceeds the size limit")
    raw = source.read_bytes()
    try:
        value = json.loads(raw.decode("utf-8", errors="strict"))
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError, RecursionError) as error:
        raise ValueError("pairwise export source is not strict UTF-8 JSON") from error
    if type(value) is not dict or any(type(key) is not str for key in value):
        raise TypeError("pairwise export source must be an exact string-keyed object")
    required = {
        "version",
        "presentationSeed",
        "includeReversed",
        "bindings",
        "pairs",
    }
    if set(value) != required:
        raise ValueError(
            "pairwise export source keys differ: "
            f"missing={sorted(required - set(value))}, extra={sorted(set(value) - required)}"
        )
    return value, hashlib.sha256(raw).hexdigest()


def _write_exclusive_json(path: Path, value: object) -> str:
    payload = _canonical_bytes(value)
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    return hashlib.sha256(payload).hexdigest()


def export_pairwise_task_bundle_v1(source_path: Path, output_dir: Path) -> Path:
    """Create a new review export and write its terminal marker last."""
    if not isinstance(output_dir, Path):
        raise TypeError("output_dir must be a Path")
    if output_dir.exists():
        raise FileExistsError(f"pairwise export output already exists: {output_dir}")
    parent = output_dir.parent.resolve(strict=True)
    if not parent.is_dir():
        raise ValueError("pairwise export parent must be a directory")

    source, source_sha256 = _read_source(source_path)
    if source["version"] != _SOURCE_VERSION:
        raise ValueError("unsupported pairwise export source version")
    bindings_value = source["bindings"]
    pairs_value = source["pairs"]
    if type(bindings_value) is not list:
        raise TypeError("bindings must be an exact list")
    if type(pairs_value) is not list or any(type(pair) is not list for pair in pairs_value):
        raise TypeError("pairs must be an exact list of lists")
    bundle = build_pairwise_task_bundle(
        tuple(parse_candidate_label_binding_v1(binding) for binding in bindings_value),
        pairs=tuple(tuple(pair) for pair in pairs_value),
        presentation_seed=source["presentationSeed"],
        include_reversed=source["includeReversed"],
    )

    output_dir.mkdir()
    private_path = output_dir / "private-bundle.json"
    review_path = output_dir / "review-bundle.json"
    private_file_sha256 = _write_exclusive_json(private_path, bundle.to_private_report())
    review_file_sha256 = _write_exclusive_json(review_path, bundle.to_review_report())
    terminal_path = output_dir / "export-terminal-v1.json"
    _write_exclusive_json(
        terminal_path,
        {
            "version": "difficulty-pairwise-export-terminal-v1",
            "sourceFileSha256": source_sha256,
            "privateBundleSha256": bundle.stable_sha256(),
            "privateBundleFileSha256": private_file_sha256,
            "reviewBundleFileSha256": review_file_sha256,
            "taskCount": len(bundle.tasks),
            "ready": True,
        },
    )
    return terminal_path
