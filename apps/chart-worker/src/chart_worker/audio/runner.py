"""외부 프로세스 실행.

명령행 조립은 commands 가, 단계 조립은 normalize 가 한다. 여기서는
프로세스를 띄우고 결과를 돌려주는 일만 한다.
"""

import os
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_TIMEOUT_SEC = 600.0


@dataclass(frozen=True, slots=True)
class CommandResult:
    argv: list[str]
    returncode: int
    stdout: str
    stderr: str


class CommandError(Exception):
    """프로세스를 못 띄웠거나, 시간이 초과했거나, 0 이 아닌 값으로 끝났다."""

    def __init__(
        self,
        argv: list[str],
        message: str,
        *,
        returncode: int | None = None,
        stderr: str = "",
    ) -> None:
        super().__init__(f"{argv[0]}: {message}")
        self.argv = list(argv)
        self.returncode = returncode
        self.stderr = stderr


def build_env(
    shared_bin_dir: Path | None,
    base: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """FFmpeg shared 빌드 bin 디렉터리를 PATH 앞에 붙인다.

    static 빌드는 DLL 이 없어 torchcodec 이 죽는다(Phase5 §4).

    Windows 의 환경변수 이름은 대소문자를 무시하지만 dict 는 아니다.
    그냥 env["PATH"] 로 덮어쓰면 기존 "Path" 항목이 남아 원래 값이 이긴다.
    """
    env = dict(os.environ if base is None else base)
    if shared_bin_dir is None:
        return env

    key = next((name for name in env if name.upper() == "PATH"), "PATH")
    existing = env.get(key, "")
    env[key] = f"{shared_bin_dir}{os.pathsep}{existing}" if existing else str(shared_bin_dir)
    return env


@dataclass(frozen=True, slots=True)
class CommandRunner:
    """실행 설정을 묶어둔 호출 가능 객체. normalize 가 이걸 주입받는다."""

    shared_bin_dir: Path | None = None
    timeout_sec: float = DEFAULT_TIMEOUT_SEC
    check: bool = True
    env: dict[str, str] = field(default_factory=dict, compare=False)

    def __call__(self, argv: list[str]) -> CommandResult:
        return run_command(
            argv,
            shared_bin_dir=self.shared_bin_dir,
            timeout_sec=self.timeout_sec,
            check=self.check,
        )


def run_command(
    argv: list[str],
    *,
    shared_bin_dir: Path | None = None,
    timeout_sec: float = DEFAULT_TIMEOUT_SEC,
    check: bool = True,
) -> CommandResult:
    """프로세스를 실행하고 stdout·stderr 를 문자열로 돌려준다.

    ffmpeg 는 파일 이름을 시스템 인코딩으로 찍기도 한다. errors="replace"
    로 읽어 디코딩 실패가 실행 실패로 둔갑하지 않게 한다.
    """
    try:
        completed = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=build_env(shared_bin_dir),
            timeout=timeout_sec,
            check=False,
        )
    except FileNotFoundError:
        raise CommandError(argv, "executable not found") from None
    except subprocess.TimeoutExpired:
        raise CommandError(argv, f"timed out after {timeout_sec:g}s") from None

    result = CommandResult(
        argv=list(argv),
        returncode=completed.returncode,
        stdout=completed.stdout or "",
        stderr=completed.stderr or "",
    )
    if check and result.returncode != 0:
        raise CommandError(
            argv,
            f"exited with {result.returncode}",
            returncode=result.returncode,
            stderr=result.stderr,
        )
    return result
