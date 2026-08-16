"""One-song inference session lifecycle and terminal taxonomy.

This module intentionally does not implement the resident child transport yet.  It freezes
the ownership/state/error contract shared by the current one-shot runner and the later
bounded NDJSON transport.
"""

from __future__ import annotations

import hashlib
import json
import os
import queue
import subprocess
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path, PurePosixPath
from typing import Literal, Protocol

from chart_worker.audio.runner import CommandResult
from chart_worker.errors import ErrorCode, WorkerError

_DIGEST_LENGTH = 64


def _require_digest(value: object, *, name: str) -> str:
    if type(value) is not str or len(value) != _DIGEST_LENGTH:
        raise ValueError(f"{name} must be an exact 64-character SHA-256")
    if any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{name} must be lowercase hexadecimal")
    return value


@dataclass(frozen=True, slots=True)
class SongIdentity:
    song_id: str
    audio_sha256: str
    config_digest: str

    def __post_init__(self) -> None:
        if type(self.song_id) is not str or not self.song_id:
            raise ValueError("song_id must be a non-empty plain string")
        object.__setattr__(
            self,
            "audio_sha256",
            _require_digest(self.audio_sha256, name="audio_sha256"),
        )
        object.__setattr__(
            self,
            "config_digest",
            _require_digest(self.config_digest, name="config_digest"),
        )


@dataclass(frozen=True, slots=True)
class InvocationArtifact:
    relative_path: str
    size: int
    sha256: str

    def __post_init__(self) -> None:
        if type(self.relative_path) is not str or not self.relative_path or "\\" in self.relative_path:
            raise ValueError("artifact relative_path must be a non-empty POSIX path")
        relative = PurePosixPath(self.relative_path)
        if relative.is_absolute() or any(part in ("", ".", "..") for part in relative.parts):
            raise ValueError("artifact relative_path is unsafe")
        if type(self.size) is not int or self.size < 0:
            raise ValueError("artifact size must be a non-negative plain int")
        _require_digest(self.sha256, name="artifact sha256")


@dataclass(frozen=True, slots=True)
class InvocationResult:
    status: Literal["SUCCESS", "FAILURE"]
    command: CommandResult
    accepted: bool
    invocation_id: str | None = None
    request_hash: str | None = None
    artifacts: tuple[InvocationArtifact, ...] = ()

    def __post_init__(self) -> None:
        if self.status not in ("SUCCESS", "FAILURE"):
            raise ValueError("status must be SUCCESS or FAILURE")
        if type(self.accepted) is not bool:
            raise TypeError("accepted must be a plain bool")
        if self.accepted:
            _require_digest(self.invocation_id, name="invocation_id")
            _require_digest(self.request_hash, name="request_hash")
        elif self.invocation_id is not None or self.request_hash is not None:
            raise ValueError("an unaccepted result cannot carry resident invocation identity")
        if type(self.artifacts) is not tuple or any(
            not isinstance(artifact, InvocationArtifact) for artifact in self.artifacts
        ):
            raise TypeError("artifacts must be a tuple of InvocationArtifact")


class SessionTransportError(RuntimeError):
    """Transport stopped before or after the resident child accepted an invocation."""

    def __init__(
        self,
        message: str,
        *,
        accepted: bool,
        invocation_id: str | None = None,
        request_hash: str | None = None,
    ) -> None:
        if type(accepted) is not bool:
            raise TypeError("accepted must be a plain bool")
        if accepted:
            _require_digest(invocation_id, name="invocation_id")
            _require_digest(request_hash, name="request_hash")
        elif invocation_id is not None or request_hash is not None:
            raise ValueError("a pre-accept failure cannot carry accepted invocation identity")
        super().__init__(message)
        self.accepted = accepted
        self.invocation_id = invocation_id
        self.request_hash = request_hash


class SessionState(StrEnum):
    IDLE = "IDLE"
    SONG_ACTIVE = "SONG_ACTIVE"
    POISONED = "POISONED"
    CLOSED = "CLOSED"


class InferenceSession(Protocol):
    @property
    def state(self) -> SessionState: ...

    def begin_song(self, identity: SongIdentity) -> None: ...

    def invoke(self, argv: list[str], workdir: Path) -> InvocationResult: ...

    def end_song(self) -> None: ...

    def close(self) -> None: ...


class ResidentTransport(Protocol):
    def begin_song(self, identity: SongIdentity) -> None: ...

    def invoke(self, argv: list[str], workdir: Path) -> InvocationResult: ...

    def end_song(self) -> None: ...

    def close(self, timeout_sec: float) -> None: ...


class _SessionLifecycle:
    def __init__(self) -> None:
        self._state = SessionState.IDLE

    @property
    def state(self) -> SessionState:
        return self._state

    def _require_active(self) -> None:
        if self._state is not SessionState.SONG_ACTIVE:
            raise RuntimeError(f"inference session is not song-active: {self._state}")

    def _begin(self) -> None:
        if self._state is SessionState.POISONED:
            raise RuntimeError("inference session is poisoned")
        if self._state is SessionState.CLOSED:
            raise RuntimeError("inference session is closed")
        if self._state is not SessionState.IDLE:
            raise RuntimeError("an inference song is already active")
        self._state = SessionState.SONG_ACTIVE

    def _end(self) -> None:
        self._require_active()
        self._state = SessionState.IDLE

    def _poison(self) -> None:
        self._state = SessionState.POISONED

    def _mark_closed(self) -> bool:
        if self._state is SessionState.CLOSED:
            return False
        self._state = SessionState.CLOSED
        return True


RunOneshot = Callable[[list[str], Path], CommandResult]


class OneshotSession(_SessionLifecycle):
    def __init__(self, *, run: RunOneshot) -> None:
        super().__init__()
        self._run = run

    def begin_song(self, identity: SongIdentity) -> None:
        del identity
        self._begin()

    def invoke(self, argv: list[str], workdir: Path) -> InvocationResult:
        self._require_active()
        command = self._run(list(argv), workdir)
        return InvocationResult(
            status="SUCCESS" if command.returncode == 0 else "FAILURE",
            command=command,
            accepted=False,
        )

    def end_song(self) -> None:
        self._end()

    def close(self) -> None:
        self._mark_closed()


class ResidentProcessSession(_SessionLifecycle):
    def __init__(self, *, transport: ResidentTransport, close_timeout_sec: float = 5.0) -> None:
        super().__init__()
        if type(close_timeout_sec) is not float or not 0 < close_timeout_sec <= 30:
            raise ValueError("close_timeout_sec must be a float in (0, 30]")
        self._transport = transport
        self._close_timeout_sec = close_timeout_sec

    def begin_song(self, identity: SongIdentity) -> None:
        self._begin()
        try:
            self._transport.begin_song(identity)
        except BaseException:
            self._poison()
            raise

    def invoke(self, argv: list[str], workdir: Path) -> InvocationResult:
        self._require_active()
        try:
            result = self._transport.invoke(list(argv), workdir)
        except SessionTransportError as error:
            context = {
                "accepted": error.accepted,
                "invocationId": error.invocation_id,
                "requestHash": error.request_hash,
            }
            self._poison()
            if error.accepted:
                raise WorkerError(
                    ErrorCode.INFERENCE_COMPLETION_UNKNOWN,
                    str(error),
                    context=context,
                ) from error
            raise WorkerError(
                ErrorCode.INFERENCE_START_FAILED,
                str(error),
                context=context,
            ) from error
        except WorkerError:
            self._poison()
            raise
        if not result.accepted:
            self._poison()
            raise WorkerError(
                ErrorCode.INFERENCE_PROTOCOL_FAILED,
                "resident invocation returned a terminal result without ACCEPTED",
                context={"accepted": False},
            )
        return result

    def end_song(self) -> None:
        self._require_active()
        try:
            self._transport.end_song()
        except BaseException:
            self._poison()
            raise
        self._end()

    def close(self) -> None:
        if not self._mark_closed():
            return
        self._transport.close(self._close_timeout_sec)


class _StreamClosed(RuntimeError):
    pass


class _LineReadFailure(RuntimeError):
    pass


_EOF = object()


def _protocol_failure(reason: str, **context: object) -> WorkerError:
    return WorkerError(
        ErrorCode.INFERENCE_PROTOCOL_FAILED,
        f"resident inference protocol failed: {reason}",
        context={"reason": reason, **context},
    )


def _json_without_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


class _RotatingStderrSink:
    def __init__(self, path: Path, max_bytes: int) -> None:
        self.path = path
        self.rotated_path = path.with_suffix(path.suffix + ".1")
        self.max_bytes = max_bytes
        path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, payload: bytes) -> None:
        if not payload:
            return
        current_size = self.path.stat().st_size if self.path.exists() else 0
        if current_size and current_size + len(payload) > self.max_bytes:
            os.replace(self.path, self.rotated_path)
            current_size = 0
        remaining = self.max_bytes - current_size
        if remaining <= 0:
            return
        with self.path.open("ab") as stream:
            stream.write(payload[:remaining])


class SubprocessResidentTransport:
    """Single-owner binary stdio transport for the later resident worker entrypoint."""

    def __init__(
        self,
        *,
        command: Sequence[str],
        cwd: Path,
        job_root: Path,
        stderr_path: Path,
        env: Mapping[str, str] | None = None,
        startup_timeout_sec: float = 30.0,
        invocation_timeout_sec: float = 7200.0,
        shutdown_timeout_sec: float = 5.0,
        max_line_bytes: int = 1_048_576,
        max_stderr_bytes: int = 8 * 1024 * 1024,
    ) -> None:
        if not command or any(type(part) is not str or not part for part in command):
            raise ValueError("command must contain non-empty plain strings")
        for name, value in (
            ("startup_timeout_sec", startup_timeout_sec),
            ("invocation_timeout_sec", invocation_timeout_sec),
            ("shutdown_timeout_sec", shutdown_timeout_sec),
        ):
            if type(value) is not float or not 0 < value <= 10800:
                raise ValueError(f"{name} must be a float in (0, 10800]")
        if type(max_line_bytes) is not int or max_line_bytes != 1_048_576:
            raise ValueError("max_line_bytes must be exactly 1 MiB")
        if type(max_stderr_bytes) is not int or not 4096 <= max_stderr_bytes <= 64 * 1024 * 1024:
            raise ValueError("max_stderr_bytes must be between 4 KiB and 64 MiB")
        for name, path in (("cwd", cwd), ("job_root", job_root), ("stderr_path", stderr_path)):
            if not isinstance(path, Path) or not path.is_absolute():
                raise ValueError(f"{name} must be absolute")
        if not cwd.is_dir() or not job_root.is_dir():
            raise ValueError("cwd and job_root must already exist")
        self._job_root = job_root.resolve(strict=True)
        self._startup_timeout_sec = startup_timeout_sec
        self._invocation_timeout_sec = invocation_timeout_sec
        self._shutdown_timeout_sec = shutdown_timeout_sec
        self._max_line_bytes = max_line_bytes
        self._records: queue.Queue[bytes | BaseException | object] = queue.Queue()
        self._stderr_sink = _RotatingStderrSink(stderr_path, max_stderr_bytes)
        self._song_hash: str | None = None
        self._closed = False
        self._process = subprocess.Popen(
            list(command),
            cwd=str(cwd),
            env=None if env is None else dict(env),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
        )
        if self._process.stdin is None or self._process.stdout is None or self._process.stderr is None:
            self._force_stop(1.0)
            raise RuntimeError("resident child pipes were not created")
        self._stdout_thread = threading.Thread(
            target=self._drain_stdout,
            name="resident-stdout",
            daemon=True,
        )
        self._stderr_thread = threading.Thread(
            target=self._drain_stderr,
            name="resident-stderr",
            daemon=True,
        )
        self._stdout_thread.start()
        self._stderr_thread.start()
        try:
            ready = self._read_record(time.monotonic() + startup_timeout_sec)
            self._require_record(ready, {"version", "type"}, expected_type="READY")
        except BaseException:
            self._force_stop(1.0)
            raise

    def _drain_stdout(self) -> None:
        assert self._process.stdout is not None
        try:
            while True:
                line = self._process.stdout.readline(self._max_line_bytes + 1)
                if line == b"":
                    self._records.put(_EOF)
                    return
                if len(line) > self._max_line_bytes:
                    self._records.put(_LineReadFailure("protocol line exceeds 1 MiB"))
                    return
                if not line.endswith(b"\n"):
                    self._records.put(_LineReadFailure("protocol line is not newline terminated"))
                    return
                self._records.put(line)
        except (OSError, ValueError) as error:
            self._records.put(error)

    def _drain_stderr(self) -> None:
        assert self._process.stderr is not None
        try:
            while True:
                payload = self._process.stderr.read(8192)
                if not payload:
                    return
                self._stderr_sink.write(payload)
        except OSError:
            return

    def _read_record(self, deadline: float) -> dict[str, object]:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("resident protocol deadline expired")
        try:
            item = self._records.get(timeout=remaining)
        except queue.Empty:
            raise TimeoutError("resident protocol deadline expired") from None
        if item is _EOF:
            raise _StreamClosed("resident child closed stdout")
        if isinstance(item, BaseException):
            if isinstance(item, _LineReadFailure):
                raise _protocol_failure("INVALID_LINE", detail=str(item)) from item
            raise _protocol_failure("STDOUT_READER_FAILED", detail=str(item)) from item
        try:
            record = json.loads(
                item.decode("utf-8", errors="strict"),
                object_pairs_hook=_json_without_duplicate_keys,
            )
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError, RecursionError) as error:
            raise _protocol_failure("MALFORMED_JSON", detail=str(error)) from error
        if type(record) is not dict:
            raise _protocol_failure("RECORD_NOT_OBJECT")
        return record

    @staticmethod
    def _require_record(
        record: dict[str, object],
        keys: set[str],
        *,
        expected_type: str,
    ) -> None:
        if set(record) != keys:
            raise _protocol_failure("INVALID_SCHEMA", expectedType=expected_type)
        if type(record["version"]) is not int or record["version"] != 1:
            raise _protocol_failure("UNSUPPORTED_VERSION", expectedType=expected_type)
        if record["type"] != expected_type:
            raise _protocol_failure(
                "UNEXPECTED_RECORD",
                expectedType=expected_type,
                actualType=record["type"],
            )

    def _send(self, record: dict[str, object]) -> None:
        if self._closed or self._process.stdin is None:
            raise _StreamClosed("resident child stdin is closed")
        payload = json.dumps(
            record,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8") + b"\n"
        if len(payload) > self._max_line_bytes:
            raise _protocol_failure("REQUEST_TOO_LARGE", bytes=len(payload))
        try:
            self._process.stdin.write(payload)
            self._process.stdin.flush()
        except (BrokenPipeError, OSError) as error:
            raise _StreamClosed("resident child stdin write failed") from error

    @staticmethod
    def _song_digest(identity: SongIdentity) -> str:
        payload = json.dumps(
            {
                "audioSha256": identity.audio_sha256,
                "configDigest": identity.config_digest,
                "songId": identity.song_id,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def begin_song(self, identity: SongIdentity) -> None:
        if self._song_hash is not None:
            raise RuntimeError("transport already has an active song")
        song_hash = self._song_digest(identity)
        self._send(
            {
                "version": 1,
                "type": "BEGIN_SONG",
                "songHash": song_hash,
                "songId": identity.song_id,
                "audioSha256": identity.audio_sha256,
                "configDigest": identity.config_digest,
            }
        )
        try:
            record = self._read_record(time.monotonic() + self._startup_timeout_sec)
        except (_StreamClosed, TimeoutError) as error:
            raise SessionTransportError(str(error), accepted=False) from error
        self._require_record(
            record,
            {"version", "type", "songHash"},
            expected_type="SONG_STARTED",
        )
        if record["songHash"] != song_hash:
            raise _protocol_failure("SONG_HASH_MISMATCH")
        self._song_hash = song_hash

    def _relative_workdir(self, workdir: Path) -> str:
        if not isinstance(workdir, Path) or not workdir.is_absolute():
            raise ValueError("resident workdir must be absolute")
        resolved = workdir.resolve(strict=False)
        if not resolved.is_relative_to(self._job_root) or resolved == self._job_root:
            raise ValueError("resident workdir must be a child of job_root")
        return resolved.relative_to(self._job_root).as_posix()

    @staticmethod
    def _sha256_path(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _validate_artifacts(
        self,
        raw_artifacts: object,
        *,
        workdir: Path,
        terminal_type: object,
    ) -> tuple[InvocationArtifact, ...]:
        if type(raw_artifacts) is not list or len(raw_artifacts) > 4096:
            raise _protocol_failure("INVALID_ARTIFACT_LIST")
        artifacts: list[InvocationArtifact] = []
        expected_paths: set[str] = set()
        resolved_workdir = workdir.resolve(strict=True)
        for raw in raw_artifacts:
            if type(raw) is not dict or set(raw) != {"relativePath", "size", "sha256"}:
                raise _protocol_failure("INVALID_ARTIFACT_SCHEMA")
            try:
                artifact = InvocationArtifact(
                    relative_path=raw["relativePath"],
                    size=raw["size"],
                    sha256=raw["sha256"],
                )
            except (TypeError, ValueError) as error:
                raise _protocol_failure("INVALID_ARTIFACT_VALUE", detail=str(error)) from error
            if artifact.relative_path in expected_paths:
                raise _protocol_failure("DUPLICATE_ARTIFACT_PATH")
            expected_paths.add(artifact.relative_path)
            relative = PurePosixPath(artifact.relative_path)
            path = resolved_workdir.joinpath(*relative.parts)
            if path.is_symlink() or not path.is_file():
                raise _protocol_failure("ARTIFACT_MISSING_OR_SYMLINK", path=artifact.relative_path)
            resolved = path.resolve(strict=True)
            if not resolved.is_relative_to(resolved_workdir):
                raise _protocol_failure("ARTIFACT_ESCAPES_WORKDIR", path=artifact.relative_path)
            if path.stat().st_size != artifact.size or self._sha256_path(path) != artifact.sha256:
                raise _protocol_failure("ARTIFACT_HASH_MISMATCH", path=artifact.relative_path)
            artifacts.append(artifact)

        actual_paths: set[str] = set()
        for path in resolved_workdir.rglob("*"):
            if path.is_symlink():
                raise _protocol_failure("ARTIFACT_TREE_CONTAINS_SYMLINK")
            if not path.is_file():
                continue
            relative_text = path.relative_to(resolved_workdir).as_posix()
            if relative_text == "resident-terminal-v1.json" or (
                relative_text.startswith(".resident-terminal-v1.json.")
                and relative_text.endswith(".tmp")
            ):
                continue
            actual_paths.add(relative_text)
        if actual_paths != expected_paths:
            raise _protocol_failure(
                "ARTIFACT_SET_MISMATCH",
                missing=sorted(expected_paths - actual_paths),
                extra=sorted(actual_paths - expected_paths),
            )
        if terminal_type == "SUCCESS" and not any(
            artifact.relative_path.lower().endswith(".osu") for artifact in artifacts
        ):
            raise _protocol_failure("SUCCESS_WITHOUT_OSU_ARTIFACT")
        return tuple(artifacts)

    def invoke(self, argv: list[str], workdir: Path) -> InvocationResult:
        if self._song_hash is None:
            raise RuntimeError("transport has no active song")
        if not argv or any(type(part) is not str or not part for part in argv):
            raise ValueError("argv must contain non-empty plain strings")
        request_body = {
            "argv": list(argv),
            "songHash": self._song_hash,
            "workdir": self._relative_workdir(workdir),
        }
        canonical = json.dumps(
            request_body,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        request_hash = hashlib.sha256(canonical).hexdigest()
        invocation_id = hashlib.sha256(
            f"chart-worker-resident-invocation-v1:{request_hash}".encode("ascii")
        ).hexdigest()
        self._send(
            {
                "version": 1,
                "type": "INVOKE",
                "invocationId": invocation_id,
                "requestHash": request_hash,
                **request_body,
            }
        )
        accepted = False
        deadline = time.monotonic() + self._invocation_timeout_sec
        try:
            record = self._read_record(deadline)
            first_type = record.get("type")
            if first_type == "REJECTED":
                self._require_record(
                    record,
                    {
                        "version",
                        "type",
                        "code",
                        "invocationId",
                        "requestHash",
                        "reason",
                    },
                    expected_type="REJECTED",
                )
                if (
                    record["invocationId"] != invocation_id
                    or record["requestHash"] != request_hash
                    or type(record["reason"]) is not str
                ):
                    raise _protocol_failure("REJECTION_IDENTITY_OR_REASON_INVALID")
                context = {
                    "invocationId": invocation_id,
                    "requestHash": request_hash,
                    "reason": record["reason"],
                }
                if record["code"] == "INVOCATION_CONFLICT":
                    context["accepted"] = False
                    raise WorkerError(
                        ErrorCode.INFERENCE_INVOCATION_CONFLICT,
                        "resident invocation conflicts with existing output identity",
                        context=context,
                    )
                if record["code"] == "UNKNOWN_COMPLETION":
                    context["accepted"] = True
                    raise WorkerError(
                        ErrorCode.INFERENCE_COMPLETION_UNKNOWN,
                        "existing resident output has no valid matching terminal marker",
                        context=context,
                    )
                raise _protocol_failure("UNSUPPORTED_REJECTION_CODE", code=record["code"])
            if first_type == "ACCEPTED":
                self._require_record(
                    record,
                    {"version", "type", "invocationId", "requestHash"},
                    expected_type="ACCEPTED",
                )
                if (
                    record["invocationId"] != invocation_id
                    or record["requestHash"] != request_hash
                ):
                    raise _protocol_failure("ACCEPTED_IDENTITY_MISMATCH")
                accepted = True
                terminal = self._read_record(deadline)
            elif first_type in ("SUCCESS", "FAILURE"):
                terminal = record
                accepted = True
            else:
                raise _protocol_failure("UNEXPECTED_FIRST_INVOCATION_RECORD", actualType=first_type)
        except (_StreamClosed, TimeoutError) as error:
            raise SessionTransportError(
                str(error),
                accepted=accepted,
                invocation_id=invocation_id if accepted else None,
                request_hash=request_hash if accepted else None,
            ) from error

        expected_common = {
            "version",
            "type",
            "invocationId",
            "requestHash",
            "returncode",
            "stdout",
            "stderr",
            "replayed",
            "artifacts",
        }
        terminal_type = terminal.get("type")
        if terminal_type not in ("SUCCESS", "FAILURE"):
            raise _protocol_failure("INVALID_TERMINAL_TYPE", actualType=terminal_type)
        self._require_record(terminal, expected_common, expected_type=terminal_type)
        if terminal["invocationId"] != invocation_id or terminal["requestHash"] != request_hash:
            raise _protocol_failure("TERMINAL_IDENTITY_MISMATCH")
        returncode = terminal["returncode"]
        stdout = terminal["stdout"]
        stderr = terminal["stderr"]
        replayed = terminal["replayed"]
        if (
            type(returncode) is not int
            or type(stdout) is not str
            or type(stderr) is not str
            or type(replayed) is not bool
        ):
            raise _protocol_failure("INVALID_TERMINAL_PAYLOAD_TYPES")
        if (first_type == "ACCEPTED" and replayed) or (
            first_type in ("SUCCESS", "FAILURE") and not replayed
        ):
            raise _protocol_failure("TERMINAL_REPLAY_FLAG_MISMATCH")
        if (terminal_type == "SUCCESS") != (returncode == 0):
            raise _protocol_failure("TERMINAL_RETURN_CODE_MISMATCH")
        artifacts = self._validate_artifacts(
            terminal["artifacts"],
            workdir=workdir,
            terminal_type=terminal_type,
        )
        return InvocationResult(
            status=terminal_type,
            command=CommandResult(list(argv), returncode, stdout, stderr),
            accepted=True,
            invocation_id=invocation_id,
            request_hash=request_hash,
            artifacts=artifacts,
        )

    def end_song(self) -> None:
        if self._song_hash is None:
            raise RuntimeError("transport has no active song")
        song_hash = self._song_hash
        self._send({"version": 1, "type": "END_SONG", "songHash": song_hash})
        try:
            record = self._read_record(time.monotonic() + self._startup_timeout_sec)
        except (_StreamClosed, TimeoutError) as error:
            raise SessionTransportError(str(error), accepted=False) from error
        self._require_record(
            record,
            {"version", "type", "songHash"},
            expected_type="SONG_ENDED",
        )
        if record["songHash"] != song_hash:
            raise _protocol_failure("SONG_HASH_MISMATCH")
        self._song_hash = None

    def _force_stop(self, timeout_sec: float) -> None:
        if self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=max(0.05, timeout_sec / 2))
            except subprocess.TimeoutExpired:
                self._process.kill()
                try:
                    self._process.wait(timeout=max(0.05, timeout_sec / 2))
                except subprocess.TimeoutExpired:
                    pass

    def close(self, timeout_sec: float) -> None:
        if self._closed:
            return
        self._closed = True
        deadline = time.monotonic() + min(timeout_sec, self._shutdown_timeout_sec)
        pending_interrupt: KeyboardInterrupt | SystemExit | None = None
        shutdown_failed = False
        if self._process.poll() is None and self._process.stdin is not None:
            try:
                payload = b'{"type":"SHUTDOWN","version":1}\n'
                self._process.stdin.write(payload)
                self._process.stdin.flush()
                record = self._read_record(deadline)
                self._require_record(
                    record,
                    {"version", "type"},
                    expected_type="SHUTDOWN_COMPLETE",
                )
            except (KeyboardInterrupt, SystemExit) as error:
                pending_interrupt = error
            except (OSError, TimeoutError, _StreamClosed, WorkerError):
                shutdown_failed = True
        if shutdown_failed:
            self._force_stop(max(0.1, min(timeout_sec, self._shutdown_timeout_sec)))
        remaining = max(0.0, deadline - time.monotonic())
        try:
            self._process.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            self._force_stop(max(0.1, timeout_sec))
        if self._process.stdin is not None:
            try:
                self._process.stdin.close()
            except OSError:
                pass
        self._stdout_thread.join(timeout=0.2)
        self._stderr_thread.join(timeout=0.2)
        if pending_interrupt is not None:
            raise pending_interrupt


@contextmanager
def inference_song_scope(
    session: InferenceSession,
    identity: SongIdentity,
    *,
    close_on_exit: bool,
):
    """Bracket exactly one song and close only sessions owned by the caller."""

    began = False
    try:
        session.begin_song(identity)
        began = True
        yield session
    finally:
        try:
            if began and session.state is SessionState.SONG_ACTIVE:
                session.end_song()
        finally:
            if close_on_exit:
                session.close()
