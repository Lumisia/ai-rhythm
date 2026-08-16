"""Deterministic, path-safe ownership for one Modal song request."""

from __future__ import annotations

import hashlib
from pathlib import Path


def request_output_dir(runs_root: Path, request_id: str) -> Path:
    """Return the unique run directory owned by ``request_id`` without creating it."""

    if not isinstance(runs_root, Path):
        raise TypeError("runs_root must be a Path")
    if type(request_id) is not str:
        raise TypeError("request_id must be a plain string")
    if not request_id or len(request_id) > 128:
        raise ValueError("request_id must contain 1 through 128 characters")
    resolved_root = runs_root.resolve(strict=True)
    digest = hashlib.sha256(request_id.encode("utf-8")).hexdigest()
    return resolved_root / digest
