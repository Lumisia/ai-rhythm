from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from chart_worker.audio.runner import CommandResult
from chart_worker.errors import Disposition, ErrorCode, WorkerError
from chart_worker.generation.inference_session import (
    InvocationResult,
    OneshotSession,
    ResidentProcessSession,
    SessionState,
    SessionTransportError,
    SongIdentity,
    SubprocessResidentTransport,
    inference_song_scope,
)


def _song(label: str) -> SongIdentity:
    digest = (label.encode("utf-8").hex() + "0" * 64)[:64]
    return SongIdentity(song_id=label, audio_sha256=digest, config_digest="1" * 64)


@dataclass
class _FakeTransport:
    events: list[tuple[str, object]] = field(default_factory=list)
    active_song: SongIdentity | None = None
    request_state: list[str] = field(default_factory=list)
    model_registry_loads: int = 1
    next_error: SessionTransportError | None = None
    close_calls: int = 0

    def begin_song(self, identity: SongIdentity) -> None:
        assert self.active_song is None
        assert self.request_state == []
        self.active_song = identity
        self.events.append(("begin", identity.song_id))

    def invoke(self, argv: list[str], workdir: Path) -> InvocationResult:
        assert self.active_song is not None
        self.request_state.append(argv[-1])
        self.events.append(("invoke", argv[-1]))
        if self.next_error is not None:
            error = self.next_error
            self.next_error = None
            raise error
        return InvocationResult(
            status="SUCCESS",
            command=CommandResult(list(argv), 0, "ok", ""),
            accepted=True,
            invocation_id="a" * 64,
            request_hash="b" * 64,
        )

    def end_song(self) -> None:
        assert self.active_song is not None
        self.events.append(("end", self.active_song.song_id))
        self.active_song = None
        self.request_state.clear()

    def close(self, timeout_sec: float) -> None:
        self.close_calls += 1
        self.events.append(("close", timeout_sec))


def test_resident_session_preserves_registry_but_clears_song_local_state_between_songs(
    tmp_path: Path,
):
    transport = _FakeTransport()
    session = ResidentProcessSession(transport=transport)

    for label in ("A", "B"):
        with inference_song_scope(session, _song(label), close_on_exit=False):
            result = session.invoke(["python", label], tmp_path)
            assert result.status == "SUCCESS"
            assert transport.request_state == [label]
        assert transport.request_state == []
        assert session.state is SessionState.IDLE

    assert transport.model_registry_loads == 1
    assert transport.events[:6] == [
        ("begin", "A"),
        ("invoke", "A"),
        ("end", "A"),
        ("begin", "B"),
        ("invoke", "B"),
        ("end", "B"),
    ]


def test_owned_scope_closes_once_but_attached_scope_does_not(tmp_path: Path):
    owned_transport = _FakeTransport()
    owned = ResidentProcessSession(transport=owned_transport)
    with inference_song_scope(owned, _song("owned"), close_on_exit=True):
        owned.invoke(["python", "owned"], tmp_path)
    owned.close()

    attached_transport = _FakeTransport()
    attached = ResidentProcessSession(transport=attached_transport)
    with inference_song_scope(attached, _song("attached"), close_on_exit=False):
        attached.invoke(["python", "attached"], tmp_path)

    assert owned_transport.close_calls == 1
    assert owned.state is SessionState.CLOSED
    assert attached_transport.close_calls == 0
    assert attached.state is SessionState.IDLE


def test_failure_before_acceptance_is_retryable_but_dead_session_is_poisoned(tmp_path: Path):
    transport = _FakeTransport(
        next_error=SessionTransportError("child failed before ACCEPTED", accepted=False)
    )
    session = ResidentProcessSession(transport=transport)
    session.begin_song(_song("before"))

    with pytest.raises(WorkerError) as caught:
        session.invoke(["python", "before"], tmp_path)

    assert caught.value.code is ErrorCode.INFERENCE_START_FAILED
    assert caught.value.disposition is Disposition.RETRYABLE
    assert session.state is SessionState.POISONED


def test_failure_after_acceptance_is_unknown_nonretryable_and_poisons_session(
    tmp_path: Path,
):
    transport = _FakeTransport(
        next_error=SessionTransportError(
            "EOF after ACCEPTED",
            accepted=True,
            invocation_id="a" * 64,
            request_hash="b" * 64,
        )
    )
    session = ResidentProcessSession(transport=transport)
    session.begin_song(_song("after"))

    with pytest.raises(WorkerError) as caught:
        session.invoke(["python", "after"], tmp_path)

    assert caught.value.code is ErrorCode.INFERENCE_COMPLETION_UNKNOWN
    assert caught.value.disposition is Disposition.FINAL_ALERT
    assert caught.value.context == {
        "accepted": True,
        "invocationId": "a" * 64,
        "requestHash": "b" * 64,
    }
    assert session.state is SessionState.POISONED
    with pytest.raises(RuntimeError, match="poisoned"):
        session.begin_song(_song("next"))


def test_oneshot_session_uses_same_lifecycle_without_claiming_acceptance(tmp_path: Path):
    calls: list[tuple[list[str], Path]] = []

    def run(argv: list[str], workdir: Path) -> CommandResult:
        calls.append((list(argv), workdir))
        return CommandResult(list(argv), 0, "one-shot", "")

    session = OneshotSession(run=run)
    with inference_song_scope(session, _song("one"), close_on_exit=True):
        result = session.invoke(["python", "one"], tmp_path)

    assert result.accepted is False
    assert result.command.stdout == "one-shot"
    assert calls == [(["python", "one"], tmp_path)]
    assert session.state is SessionState.CLOSED


_FAKE_CHILD = Path(__file__).parent / "fixtures" / "fake_resident_child.py"


def _subprocess_session(tmp_path: Path, scenario: str) -> tuple[ResidentProcessSession, Path, Path]:
    job_root = (tmp_path / "jobs").resolve()
    job_root.mkdir(parents=True)
    stderr_path = (tmp_path / "logs" / "resident.stderr.log").resolve()
    transport = SubprocessResidentTransport(
        command=[sys.executable, "-u", str(_FAKE_CHILD), scenario, str(job_root)],
        cwd=_FAKE_CHILD.parent,
        job_root=job_root,
        stderr_path=stderr_path,
        startup_timeout_sec=1.0,
        invocation_timeout_sec=1.0,
        shutdown_timeout_sec=0.5,
        max_line_bytes=1_048_576,
        max_stderr_bytes=64 * 1024,
    )
    return ResidentProcessSession(transport=transport, close_timeout_sec=1.0), job_root, stderr_path


def test_subprocess_transport_success_and_two_song_boundaries(tmp_path: Path):
    session, job_root, _ = _subprocess_session(tmp_path, "success")

    for label in ("first", "second"):
        with inference_song_scope(session, _song(label), close_on_exit=False):
            workdir = job_root / label
            result = session.invoke(["inference.py", f"seed={label}"], workdir)
            assert result.status == "SUCCESS"
            assert result.accepted is True
            assert result.command.stdout == "child-output"
            assert [artifact.relative_path for artifact in result.artifacts] == ["generated.osu"]
            assert result.artifacts[0].size == len(b"osu file")

    session.close()
    assert session.state is SessionState.CLOSED


@pytest.mark.parametrize(
    ("scenario", "expected_code", "expected_accepted"),
    [
        ("eof_before", ErrorCode.INFERENCE_START_FAILED, False),
        ("eof_after", ErrorCode.INFERENCE_COMPLETION_UNKNOWN, True),
        ("timeout_after", ErrorCode.INFERENCE_COMPLETION_UNKNOWN, True),
    ],
)
def test_subprocess_transport_preserves_acceptance_boundary(
    tmp_path: Path,
    scenario: str,
    expected_code: ErrorCode,
    expected_accepted: bool,
):
    session, job_root, _ = _subprocess_session(tmp_path, scenario)
    session.begin_song(_song(scenario))

    with pytest.raises(WorkerError) as caught:
        session.invoke(["inference.py", "seed=1"], job_root / scenario)

    assert caught.value.code is expected_code
    assert caught.value.context["accepted"] is expected_accepted
    assert session.state is SessionState.POISONED
    session.close()


@pytest.mark.parametrize("scenario", ["polluted_stdout", "malformed", "oversized"])
def test_subprocess_transport_fails_closed_on_protocol_corruption(
    tmp_path: Path,
    scenario: str,
):
    session, job_root, _ = _subprocess_session(tmp_path, scenario)
    session.begin_song(_song(scenario))

    with pytest.raises(WorkerError) as caught:
        session.invoke(["inference.py", "seed=2"], job_root / scenario)

    assert caught.value.code is ErrorCode.INFERENCE_PROTOCOL_FAILED
    assert session.state is SessionState.POISONED
    session.close()


def test_large_stderr_is_drained_without_blocking_protocol(tmp_path: Path):
    session, job_root, stderr_path = _subprocess_session(tmp_path, "large_stderr")
    with inference_song_scope(session, _song("stderr"), close_on_exit=True):
        result = session.invoke(["inference.py", "seed=3"], job_root / "stderr")

    assert result.status == "SUCCESS"
    assert stderr_path.is_file()
    rotated = stderr_path.with_suffix(stderr_path.suffix + ".1")
    assert stderr_path.stat().st_size <= 64 * 1024
    assert not rotated.exists() or rotated.stat().st_size <= 64 * 1024


def test_matching_terminal_replay_returns_without_second_accepted(tmp_path: Path):
    session, job_root, _ = _subprocess_session(tmp_path, "replayed_terminal")
    with inference_song_scope(session, _song("replay"), close_on_exit=True):
        result = session.invoke(["inference.py", "seed=4"], job_root / "replay")

    assert result.status == "SUCCESS"
    assert result.accepted is True
    assert result.command.stdout == "replayed-output"
    assert result.artifacts[0].relative_path == "generated.osu"


@pytest.mark.parametrize(
    ("scenario", "code"),
    [
        ("invocation_conflict", ErrorCode.INFERENCE_INVOCATION_CONFLICT),
        ("corrupt_marker", ErrorCode.INFERENCE_COMPLETION_UNKNOWN),
    ],
)
def test_rejected_existing_output_never_executes_again(
    tmp_path: Path,
    scenario: str,
    code: ErrorCode,
):
    session, job_root, _ = _subprocess_session(tmp_path, scenario)
    session.begin_song(_song(scenario))

    with pytest.raises(WorkerError) as caught:
        session.invoke(["inference.py", "seed=5"], job_root / scenario)

    assert caught.value.code is code
    assert caught.value.retryable is False
    assert session.state is SessionState.POISONED
    session.close()


def test_close_is_bounded_when_child_ignores_shutdown(tmp_path: Path):
    session, _, _ = _subprocess_session(tmp_path, "ignore_shutdown")
    started = time.monotonic()

    session.close()

    assert time.monotonic() - started < 2.0
    assert session.state is SessionState.CLOSED
