"""Read-only replay of an archived V3 family-selection report."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath

from chart_worker.validation.family_evidence_v3 import parse_song_selection_evidence_v3
from chart_worker.validation.song_family_selector_v3 import evaluate_shadow_v3_proposal

_MAX_REPORT_BYTES = 64 * 1024 * 1024


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_report(path: Path) -> dict[str, object]:
    if not isinstance(path, Path):
        raise TypeError("report_path must be a Path")
    size = path.stat().st_size
    if size > _MAX_REPORT_BYTES:
        raise ValueError("generation report exceeds the replay size limit")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError, RecursionError) as error:
        raise ValueError("generation report is not strict UTF-8 JSON") from error
    if type(value) is not dict or any(type(key) is not str for key in value):
        raise TypeError("generation report must be a string-keyed object")
    return value


def _assignment_from_shadow(value: object) -> tuple[tuple[str, str | None], ...]:
    if type(value) is not dict:
        raise TypeError("songSelectionShadowV3 must be an object")
    proposal = value.get("proposedAssignment")
    if type(proposal) is not dict or any(
        type(slot) is not str
        or not slot
        or (candidate_id is not None and (type(candidate_id) is not str or not candidate_id))
        for slot, candidate_id in proposal.items()
    ):
        raise TypeError("proposedAssignment must map slots to candidate IDs or null")
    return tuple(sorted(proposal.items()))


def _verify_candidate_payloads(report_root: Path, evidence) -> int:
    root = report_root.resolve(strict=True)
    verified = 0
    for candidate in evidence.candidates:
        relative = PurePosixPath(candidate.candidate_payload_ref)
        payload_path = root.joinpath(*relative.parts).resolve(strict=True)
        try:
            payload_path.relative_to(root)
        except ValueError as error:
            raise ValueError("candidate payload escapes the report directory") from error
        if not payload_path.is_file():
            raise ValueError("candidate payload is not a regular file")
        if _sha256_file(payload_path) != candidate.candidate_payload_sha256:
            raise ValueError(f"candidate payload digest mismatch: {candidate.candidate_id}")
        verified += 1
    return verified


def replay_generation_report_v3(report_path: Path) -> dict[str, object]:
    """Verify evidence, source payloads, and the stored no-calibration SHADOW result."""
    report = _read_report(report_path)
    evidence = parse_song_selection_evidence_v3(report.get("songSelectionEvidenceV3"))
    digest = report.get("songSelectionEvidenceV3Sha256")
    if type(digest) is not str or digest != evidence.stable_sha256():
        raise ValueError("song selection evidence digest mismatch")
    stored_shadow = report.get("songSelectionShadowV3")
    proposal = _assignment_from_shadow(stored_shadow)
    evaluation = evaluate_shadow_v3_proposal(evidence, proposal=proposal)
    evaluation_report = evaluation.to_report()
    if _canonical_json(stored_shadow) != _canonical_json(evaluation_report):
        raise ValueError("stored SHADOW_V3 result differs from strict recalculation")
    verified_count = _verify_candidate_payloads(report_path.parent, evidence)
    return {
        "version": "family-v3-replay-v1",
        "contextId": evidence.context_id,
        "evidenceSha256": digest,
        "verifiedCandidatePayloadCount": verified_count,
        "storedShadowMatches": True,
        "evaluation": evaluation_report,
        "additionalModelCalls": 0,
        "mutatesArtifacts": False,
    }
