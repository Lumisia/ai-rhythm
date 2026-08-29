import hashlib
import json
import shutil
import uuid
from dataclasses import replace
from pathlib import Path

import pytest
from test_song_family_selector_v3 import _base_family, _evidence

from chart_worker.validation.family_evidence_v3 import SongSelectionEvidenceV3
from chart_worker.validation.family_replay_v3 import replay_generation_report_v3


@pytest.fixture
def workspace_tmp_path():
    root = Path.cwd() / ".pytest-family-replay-v3" / uuid.uuid4().hex
    root.mkdir(parents=True)
    try:
        yield root
    finally:
        shutil.rmtree(root)


def _write_report(tmp_path: Path) -> tuple[Path, dict[str, object]]:
    original = _evidence(*_base_family())
    candidates = []
    for candidate in original.candidates:
        payload_path = tmp_path / candidate.candidate_payload_ref
        payload_path.parent.mkdir(parents=True, exist_ok=True)
        payload = candidate.candidate_id.encode("ascii")
        payload_path.write_bytes(payload)
        candidates.append(
            replace(
                candidate,
                candidate_payload_sha256=hashlib.sha256(payload).hexdigest(),
            )
        )
    evidence = SongSelectionEvidenceV3(
        context_id=original.context_id,
        intro_selection=original.intro_selection,
        candidates=tuple(candidates),
        current_assignment=original.current_assignment,
    )
    evidence_report = evidence.to_report()
    proposal = dict(evidence.current_assignment)
    stored_shadow = {
        "mode": "SHADOW_V3",
        "selectedAssignment": proposal,
        "proposedAssignment": proposal,
        "shadowAssignment": proposal,
        "proposalEligible": False,
        "blockers": ["CALIBRATION_UNAVAILABLE"],
        "currentInversions": [],
        "proposedInversions": [],
        "resolvedInversions": [],
        "createdInversions": [],
        "calibrationSha256": None,
        "mutatesSelection": False,
    }
    report = {
        "songSelectionEvidenceV3": evidence_report,
        "songSelectionEvidenceV3Sha256": evidence.stable_sha256(),
        "songSelectionShadowV3": stored_shadow,
    }
    report_path = tmp_path / "generation-report.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    return report_path, report


def test_replay_verifies_evidence_payloads_and_stored_shadow(workspace_tmp_path: Path):
    report_path, _report = _write_report(workspace_tmp_path)

    replay = replay_generation_report_v3(report_path)

    assert replay["verifiedCandidatePayloadCount"] == 4
    assert replay["storedShadowMatches"] is True
    assert replay["evaluation"]["blockers"] == ["CALIBRATION_UNAVAILABLE"]
    assert replay["additionalModelCalls"] == 0
    assert replay["mutatesArtifacts"] is False


def test_replay_rejects_evidence_digest_mismatch(workspace_tmp_path: Path):
    report_path, report = _write_report(workspace_tmp_path)
    report["songSelectionEvidenceV3Sha256"] = "0" * 64
    report_path.write_text(json.dumps(report), encoding="utf-8")

    with pytest.raises(ValueError, match="evidence digest"):
        replay_generation_report_v3(report_path)


def test_replay_rejects_missing_or_modified_candidate_payload(workspace_tmp_path: Path):
    report_path, report = _write_report(workspace_tmp_path)
    candidate = report["songSelectionEvidenceV3"]["candidates"][0]
    payload_path = workspace_tmp_path / candidate["candidatePayloadRef"]
    payload_path.write_bytes(b"modified")

    with pytest.raises(ValueError, match="candidate payload digest"):
        replay_generation_report_v3(report_path)


def test_replay_rejects_stored_shadow_that_differs_from_recalculation(
    workspace_tmp_path: Path,
):
    report_path, report = _write_report(workspace_tmp_path)
    report["songSelectionShadowV3"]["blockers"] = []
    report_path.write_text(json.dumps(report), encoding="utf-8")

    with pytest.raises(ValueError, match="stored SHADOW_V3"):
        replay_generation_report_v3(report_path)
