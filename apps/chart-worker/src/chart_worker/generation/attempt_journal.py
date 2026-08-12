"""Durable, append-only evidence for MAP generation attempts.

The journal is intentionally narrower than event sourcing: it is a single-writer
diagnostic trace. Pipeline state and published reports remain authoritative.
"""

from __future__ import annotations

import json
import os
from collections import Counter
from collections.abc import Mapping
from pathlib import Path

from chart_worker.hashing import sha256_file

JOURNAL_VERSION = "attempt-journal-v1"
ATTEMPT_JOURNAL_PROJECTION_VERSION = "attempt-journal-projection-v1"


class AttemptJournalCorruptionError(ValueError):
    """Raised when an existing journal is truncated, reordered, or mutated."""


class AttemptJournal:
    """Append deterministic JSON records and fsync every complete line.

    One pipeline process owns a journal. Size checks detect accidental second
    writers or external edits; this class does not pretend to provide a
    multi-process locking protocol.
    """

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        records, byte_length = self._read_validated()
        self._sequence = len(records)
        self._byte_length = byte_length

    def append(
        self,
        *,
        event_type: str,
        authority_epoch: int,
        key_mode: int,
        difficulty: str,
        attempt: int | None,
        seed: int | None,
        payload: Mapping[str, object] | None = None,
    ) -> dict[str, object]:
        current_length = self.path.stat().st_size if self.path.exists() else 0
        if current_length != self._byte_length:
            raise AttemptJournalCorruptionError(
                "attempt journal changed after it was opened"
            )
        if not event_type:
            raise ValueError("event_type must not be empty")
        if authority_epoch < 1:
            raise ValueError("authority_epoch must be positive")
        if key_mode < 1:
            raise ValueError("key_mode must be positive")
        if not difficulty:
            raise ValueError("difficulty must not be empty")
        if attempt is not None and attempt < 1:
            raise ValueError("attempt must be positive when provided")

        record: dict[str, object] = {
            "version": JOURNAL_VERSION,
            "sequence": self._sequence + 1,
            "eventType": event_type,
            "authorityEpoch": authority_epoch,
            "keyMode": key_mode,
            "difficulty": difficulty,
            "attempt": attempt,
            "seed": seed,
            "payload": dict(payload or {}),
        }
        encoded = (
            json.dumps(
                record,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            )
            + "\n"
        ).encode("utf-8")
        flags = os.O_APPEND | os.O_CREAT | os.O_WRONLY
        if hasattr(os, "O_BINARY"):
            flags |= os.O_BINARY
        descriptor = os.open(self.path, flags, 0o644)
        try:
            remaining = memoryview(encoded)
            while remaining:
                written = os.write(descriptor, remaining)
                if written <= 0:
                    raise OSError("could not append attempt journal record")
                remaining = remaining[written:]
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        self._sequence += 1
        self._byte_length += len(encoded)
        return record

    def records(self) -> tuple[dict[str, object], ...]:
        records, byte_length = self._read_validated()
        if byte_length != self._byte_length:
            raise AttemptJournalCorruptionError(
                "attempt journal changed after it was opened"
            )
        return records

    def _read_validated(self) -> tuple[tuple[dict[str, object], ...], int]:
        if not self.path.exists():
            return (), 0
        raw = self.path.read_bytes()
        if raw and not raw.endswith(b"\n"):
            raise AttemptJournalCorruptionError("attempt journal has a truncated line")
        records: list[dict[str, object]] = []
        for expected_sequence, raw_line in enumerate(raw.splitlines(), start=1):
            try:
                decoded = json.loads(raw_line)
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise AttemptJournalCorruptionError(
                    "attempt journal contains invalid JSON"
                ) from error
            if not isinstance(decoded, dict):
                raise AttemptJournalCorruptionError(
                    "attempt journal record must be an object"
                )
            if decoded.get("version") != JOURNAL_VERSION:
                raise AttemptJournalCorruptionError(
                    "attempt journal version is missing or unsupported"
                )
            if decoded.get("sequence") != expected_sequence:
                raise AttemptJournalCorruptionError(
                    "attempt journal sequence is not contiguous"
                )
            records.append(decoded)
        return tuple(records), len(raw)


def build_attempt_journal_projection(
    path: Path,
    *,
    relative_to: Path,
) -> dict[str, object]:
    """Return a report-safe, hash-bound snapshot of durable attempt evidence.

    Generation failure reporting must not destroy the original failure merely
    because its diagnostic journal is absent or corrupt. Those states are
    represented explicitly instead of being silently dropped or guessed.
    """

    path = Path(path)
    relative_path = path.relative_to(relative_to).as_posix()
    base: dict[str, object] = {
        "version": ATTEMPT_JOURNAL_PROJECTION_VERSION,
        "path": relative_path,
    }
    if not path.exists():
        return {
            **base,
            "status": "UNAVAILABLE",
            "reason": "NOT_CREATED",
        }

    digest = sha256_file(path)
    try:
        records = AttemptJournal(path).records()
    except AttemptJournalCorruptionError as error:
        return {
            **base,
            "status": "CORRUPT",
            "sha256": digest,
            "reason": "TRUNCATED_OR_INVALID",
            "error": str(error),
        }

    event_counts = Counter(str(record["eventType"]) for record in records)
    return {
        **base,
        "status": "AVAILABLE",
        "sha256": digest,
        "journalVersion": JOURNAL_VERSION,
        "recordCount": len(records),
        "eventCounts": dict(sorted(event_counts.items())),
        "records": list(records),
    }
