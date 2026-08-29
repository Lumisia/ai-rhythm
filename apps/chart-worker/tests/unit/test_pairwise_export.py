import hashlib
import json
import shutil
import uuid
from pathlib import Path

import pytest

from chart_worker.validation.pairwise_export import export_pairwise_task_bundle_v1


@pytest.fixture
def workspace_tmp_path():
    root = Path.cwd() / ".pytest-pairwise-export" / uuid.uuid4().hex
    root.mkdir(parents=True)
    try:
        yield root
    finally:
        shutil.rmtree(root)


def _binding(candidate_id: str, payload: str, feature: str) -> dict[str, object]:
    return {
        "candidateId": candidate_id,
        "audioSha256": "a" * 64,
        "keyMode": 4,
        "payloadSha256": payload * 64,
        "featureSha256": feature * 64,
    }


def _source() -> dict[str, object]:
    return {
        "version": "difficulty-pairwise-export-source-v1",
        "presentationSeed": "human-review-seed",
        "includeReversed": True,
        "bindings": [
            _binding("candidate-a", "b", "c"),
            _binding("candidate-b", "d", "e"),
        ],
        "pairs": [["candidate-a", "candidate-b"]],
    }


def test_export_writes_blinded_and_private_bundles_then_terminal_last(
    workspace_tmp_path: Path,
):
    source = workspace_tmp_path / "source.json"
    output = workspace_tmp_path / "export"
    source.write_text(json.dumps(_source()), encoding="utf-8")

    terminal_path = export_pairwise_task_bundle_v1(source, output)

    assert terminal_path == output / "export-terminal-v1.json"
    private_bytes = (output / "private-bundle.json").read_bytes()
    review_bytes = (output / "review-bundle.json").read_bytes()
    terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
    assert terminal["privateBundleFileSha256"] == hashlib.sha256(private_bytes).hexdigest()
    assert terminal["reviewBundleFileSha256"] == hashlib.sha256(review_bytes).hexdigest()
    review_text = review_bytes.decode("utf-8")
    assert '"payloadSha256"' in review_text
    for secret in ("candidateId", "audioSha256", "featureSha256", "keyMode"):
        assert f'"{secret}"' not in review_text


def test_export_rejects_existing_output_and_noncanonical_source(
    workspace_tmp_path: Path,
):
    source = workspace_tmp_path / "source.json"
    source.write_text(json.dumps(_source()), encoding="utf-8")
    output = workspace_tmp_path / "export"
    output.mkdir()

    with pytest.raises(FileExistsError, match="already exists"):
        export_pairwise_task_bundle_v1(source, output)

    bad_source = workspace_tmp_path / "bad-source.json"
    bad_source.write_text(json.dumps({**_source(), "difficulty": "HARD"}), encoding="utf-8")
    with pytest.raises(ValueError, match="keys differ"):
        export_pairwise_task_bundle_v1(bad_source, workspace_tmp_path / "bad-export")
