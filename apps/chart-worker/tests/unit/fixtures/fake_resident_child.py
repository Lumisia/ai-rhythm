from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

SCENARIO = sys.argv[1]
JOB_ROOT = Path(sys.argv[2]).resolve()


def emit(record: dict[str, object]) -> None:
    sys.stdout.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
    sys.stdout.flush()


emit({"version": 1, "type": "READY"})

for line in sys.stdin:
    record = json.loads(line)
    kind = record["type"]
    if kind == "BEGIN_SONG":
        emit(
            {
                "version": 1,
                "type": "SONG_STARTED",
                "songHash": record["songHash"],
            }
        )
        continue
    if kind == "END_SONG":
        emit({"version": 1, "type": "SONG_ENDED", "songHash": record["songHash"]})
        continue
    if kind == "SHUTDOWN":
        if SCENARIO == "ignore_shutdown":
            time.sleep(10)
        emit({"version": 1, "type": "SHUTDOWN_COMPLETE"})
        raise SystemExit(0)
    if kind != "INVOKE":
        raise RuntimeError(f"unexpected record: {kind}")

    if SCENARIO == "eof_before":
        raise SystemExit(7)
    if SCENARIO == "polluted_stdout":
        print("this is not protocol JSON", flush=True)
        continue
    if SCENARIO == "malformed":
        print("{broken", flush=True)
        continue
    if SCENARIO == "oversized":
        print("x" * (1_048_576 + 1), flush=True)
        continue
    if SCENARIO in ("invocation_conflict", "corrupt_marker"):
        emit(
            {
                "version": 1,
                "type": "REJECTED",
                "code": (
                    "INVOCATION_CONFLICT"
                    if SCENARIO == "invocation_conflict"
                    else "UNKNOWN_COMPLETION"
                ),
                "invocationId": record["invocationId"],
                "requestHash": record["requestHash"],
                "reason": SCENARIO,
            }
        )
        continue
    if SCENARIO == "replayed_terminal":
        output = JOB_ROOT / record["workdir"] / "generated.osu"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"osu file")
        emit(
            {
                "version": 1,
                "type": "SUCCESS",
                "invocationId": record["invocationId"],
                "requestHash": record["requestHash"],
                "returncode": 0,
                "stdout": "replayed-output",
                "stderr": "",
                "replayed": True,
                "artifacts": [
                    {
                        "relativePath": "generated.osu",
                        "size": output.stat().st_size,
                        "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
                    }
                ],
            }
        )
        continue

    accepted = {
        "version": 1,
        "type": "ACCEPTED",
        "invocationId": record["invocationId"],
        "requestHash": record["requestHash"],
    }
    emit(accepted)
    if SCENARIO == "eof_after":
        raise SystemExit(8)
    if SCENARIO == "timeout_after":
        time.sleep(10)
        continue
    if SCENARIO == "large_stderr":
        remaining = 2 * 1024 * 1024
        payload = b"z" * 8192
        while remaining:
            chunk = payload[: min(len(payload), remaining)]
            sys.stderr.buffer.write(chunk)
            sys.stderr.buffer.flush()
            remaining -= len(chunk)

    output = JOB_ROOT / record["workdir"] / "generated.osu"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(b"osu file")

    emit(
        {
            "version": 1,
            "type": "SUCCESS",
            "invocationId": record["invocationId"],
            "requestHash": record["requestHash"],
            "returncode": 0,
            "stdout": "child-output",
            "stderr": "",
            "replayed": False,
            "artifacts": [
                {
                    "relativePath": "generated.osu",
                    "size": output.stat().st_size,
                    "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
                }
            ],
        }
    )
