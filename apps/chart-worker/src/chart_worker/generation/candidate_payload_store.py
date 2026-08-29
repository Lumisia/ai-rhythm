"""Content-addressed storage for immutable candidate osu payloads."""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path


class CandidatePayloadIntegrityError(ValueError):
    """Raised when an existing candidate payload does not match its address."""


@dataclass(frozen=True, slots=True)
class CandidatePayloadArtifact:
    path: Path
    relative_path: Path
    sha256: str


def candidate_payload_artifact(
    *,
    run_dir: Path,
    osu_text: str,
) -> CandidatePayloadArtifact:
    payload = osu_text.encode("utf-8")
    sha256 = hashlib.sha256(payload).hexdigest()
    relative_path = Path("raw", "candidates", "sha256", f"{sha256}.osu")
    return CandidatePayloadArtifact(
        path=run_dir / relative_path,
        relative_path=relative_path,
        sha256=sha256,
    )


def verify_candidate_payload(
    *,
    run_dir: Path,
    osu_text: str,
) -> CandidatePayloadArtifact:
    artifact = candidate_payload_artifact(run_dir=run_dir, osu_text=osu_text)
    if not artifact.path.is_file():
        raise CandidatePayloadIntegrityError(
            f"candidate payload is missing: {artifact.relative_path.as_posix()}"
        )
    with artifact.path.open("rb") as payload_file:
        actual_sha256 = hashlib.file_digest(payload_file, "sha256").hexdigest()
    if actual_sha256 != artifact.sha256:
        raise CandidatePayloadIntegrityError(
            "candidate payload hash mismatch: "
            f"{artifact.relative_path.as_posix()} expected={artifact.sha256} "
            f"actual={actual_sha256}"
        )
    return artifact


def persist_candidate_payload(
    *,
    run_dir: Path,
    osu_text: str,
) -> CandidatePayloadArtifact:
    artifact = candidate_payload_artifact(run_dir=run_dir, osu_text=osu_text)
    artifact.path.parent.mkdir(parents=True, exist_ok=True)
    payload = osu_text.encode("utf-8")
    try:
        with artifact.path.open("xb") as payload_file:
            payload_file.write(payload)
            payload_file.flush()
            os.fsync(payload_file.fileno())
    except FileExistsError:
        pass
    return verify_candidate_payload(run_dir=run_dir, osu_text=osu_text)
