import hashlib
import shutil
import uuid
from pathlib import Path

import pytest

from chart_worker.generation.candidate_payload_store import (
    CandidatePayloadIntegrityError,
    persist_candidate_payload,
    verify_candidate_payload,
)


@pytest.fixture
def workspace_tmp_path():
    root = Path.cwd() / ".pytest-candidate-payload-store" / uuid.uuid4().hex
    root.mkdir(parents=True)
    try:
        yield root
    finally:
        shutil.rmtree(root)


def test_persist_candidate_payload_writes_exact_content_addressed_bytes(
    workspace_tmp_path: Path,
):
    payload = b"[General]\nAudioFilename: song.wav\n"
    expected_sha256 = hashlib.sha256(payload).hexdigest()

    artifact = persist_candidate_payload(
        run_dir=workspace_tmp_path,
        osu_text=payload.decode("utf-8"),
    )

    assert artifact.relative_path == Path("raw", "candidates", "sha256", f"{expected_sha256}.osu")
    assert artifact.sha256 == expected_sha256
    assert artifact.path.read_bytes() == payload
    assert (
        verify_candidate_payload(
            run_dir=workspace_tmp_path,
            osu_text=payload.decode("utf-8"),
        )
        == artifact
    )


def test_persist_candidate_payload_rejects_corrupt_existing_artifact(
    workspace_tmp_path: Path,
):
    osu_text = "candidate\n"
    artifact = persist_candidate_payload(run_dir=workspace_tmp_path, osu_text=osu_text)
    artifact.path.write_bytes(b"corrupt")

    with pytest.raises(CandidatePayloadIntegrityError, match="hash mismatch"):
        persist_candidate_payload(run_dir=workspace_tmp_path, osu_text=osu_text)

    assert artifact.path.read_bytes() == b"corrupt"
