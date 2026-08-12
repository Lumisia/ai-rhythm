import json
from pathlib import Path

import pytest

from chart_worker.generation.attempt_journal import (
    AttemptJournal,
    AttemptJournalCorruptionError,
    build_attempt_journal_projection,
)
from chart_worker.hashing import sha256_file


def test_attempt_journal_appends_without_rewriting_previous_records(tmp_path: Path):
    path = tmp_path / "attempt-journal.jsonl"
    journal = AttemptJournal(path)

    first = journal.append(
        event_type="INFERENCE_STARTED",
        authority_epoch=1,
        key_mode=4,
        difficulty="EXPERT",
        attempt=1,
        seed=9,
        payload={"workdir": "raw/work/epoch-1/4k-expert/attempt-1"},
    )
    prefix = path.read_bytes()
    second = journal.append(
        event_type="INFERENCE_FAILED",
        authority_epoch=1,
        key_mode=4,
        difficulty="EXPERT",
        attempt=1,
        seed=9,
        payload={"code": "CHART_GENERATION_FAILED"},
    )

    assert first["sequence"] == 1
    assert second["sequence"] == 2
    assert path.read_bytes().startswith(prefix)
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert [record["eventType"] for record in records] == [
        "INFERENCE_STARTED",
        "INFERENCE_FAILED",
    ]
    assert all(record["version"] == "attempt-journal-v1" for record in records)
    assert "recordedAt" not in records[0]


def test_attempt_journal_reopens_and_continues_sequence(tmp_path: Path):
    path = tmp_path / "attempt-journal.jsonl"
    AttemptJournal(path).append(
        event_type="INFERENCE_STARTED",
        authority_epoch=2,
        key_mode=6,
        difficulty="HARD",
        attempt=3,
        seed=31,
    )

    reopened = AttemptJournal(path)
    event = reopened.append(
        event_type="INFERENCE_COMPLETED",
        authority_epoch=2,
        key_mode=6,
        difficulty="HARD",
        attempt=3,
        seed=31,
        payload={"noteCount": 512},
    )

    assert event["sequence"] == 2
    assert [record["sequence"] for record in reopened.records()] == [1, 2]


@pytest.mark.parametrize(
    "content",
    [
        b'{"version":"attempt-journal-v1","sequence":1}',
        (
            b'{"version":"attempt-journal-v1","sequence":1}\n'
            b'{"version":"attempt-journal-v1","sequence":3}\n'
        ),
        b"not-json\n",
    ],
)
def test_attempt_journal_rejects_truncated_noncontiguous_or_invalid_data(
    tmp_path: Path,
    content: bytes,
):
    path = tmp_path / "attempt-journal.jsonl"
    path.write_bytes(content)

    with pytest.raises(AttemptJournalCorruptionError):
        AttemptJournal(path)


def test_attempt_journal_refuses_to_append_after_external_mutation(tmp_path: Path):
    path = tmp_path / "attempt-journal.jsonl"
    journal = AttemptJournal(path)
    journal.append(
        event_type="INFERENCE_STARTED",
        authority_epoch=1,
        key_mode=7,
        difficulty="NORMAL",
        attempt=1,
        seed=3,
    )
    path.write_bytes(path.read_bytes() + b"{}\n")

    with pytest.raises(AttemptJournalCorruptionError, match="changed"):
        journal.append(
            event_type="INFERENCE_COMPLETED",
            authority_epoch=1,
            key_mode=7,
            difficulty="NORMAL",
            attempt=1,
            seed=3,
        )


def test_attempt_journal_projection_binds_complete_records_and_event_counts(
    tmp_path: Path,
):
    path = tmp_path / "attempt-journal.jsonl"
    journal = AttemptJournal(path)
    journal.append(
        event_type="INFERENCE_STARTED",
        authority_epoch=1,
        key_mode=4,
        difficulty="EXPERT",
        attempt=1,
        seed=9,
    )
    journal.append(
        event_type="INFERENCE_FAILED",
        authority_epoch=1,
        key_mode=4,
        difficulty="EXPERT",
        attempt=1,
        seed=9,
        payload={"code": "CHART_GENERATION_FAILED"},
    )

    projection = build_attempt_journal_projection(path, relative_to=tmp_path)

    assert projection == {
        "version": "attempt-journal-projection-v1",
        "status": "AVAILABLE",
        "path": "attempt-journal.jsonl",
        "sha256": sha256_file(path),
        "journalVersion": "attempt-journal-v1",
        "recordCount": 2,
        "eventCounts": {"INFERENCE_FAILED": 1, "INFERENCE_STARTED": 1},
        "records": list(journal.records()),
    }


def test_attempt_journal_projection_reports_missing_or_corrupt_evidence(
    tmp_path: Path,
):
    path = tmp_path / "attempt-journal.jsonl"

    assert build_attempt_journal_projection(path, relative_to=tmp_path) == {
        "version": "attempt-journal-projection-v1",
        "status": "UNAVAILABLE",
        "path": "attempt-journal.jsonl",
        "reason": "NOT_CREATED",
    }

    path.write_bytes(b'{"version":"attempt-journal-v1","sequence":1}')
    projection = build_attempt_journal_projection(path, relative_to=tmp_path)

    assert projection["version"] == "attempt-journal-projection-v1"
    assert projection["status"] == "CORRUPT"
    assert projection["path"] == "attempt-journal.jsonl"
    assert projection["sha256"] == sha256_file(path)
    assert projection["reason"] == "TRUNCATED_OR_INVALID"
    assert "truncated line" in projection["error"]
    assert "records" not in projection
