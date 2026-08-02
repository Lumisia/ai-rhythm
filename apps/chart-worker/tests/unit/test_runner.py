import os
import sys
from pathlib import Path

import pytest

from chart_worker.audio.runner import CommandError, build_env, run_command

SHARED = Path("C:/ffmpeg/bin")


def test_env_is_untouched_without_a_shared_build():
    base = {"PATH": "/usr/bin"}
    assert build_env(None, base) == base


def test_shared_bin_dir_goes_in_front_of_path():
    env = build_env(SHARED, {"PATH": "/usr/bin"})
    assert env["PATH"] == f"{SHARED}{os.pathsep}/usr/bin"


def test_windows_lowercase_path_key_is_reused_not_shadowed():
    """dict 는 대소문자를 구분한다. PATH 를 새로 넣으면 기존 Path 가 이긴다."""
    env = build_env(SHARED, {"Path": "C:/windows"})
    assert "PATH" not in env
    assert env["Path"] == f"{SHARED}{os.pathsep}C:/windows"


def test_missing_path_variable_is_created():
    assert build_env(SHARED, {}) == {"PATH": str(SHARED)}


def test_base_mapping_is_not_mutated():
    base = {"PATH": "/usr/bin"}
    build_env(SHARED, base)
    assert base == {"PATH": "/usr/bin"}


def test_runs_a_process_and_captures_both_streams():
    result = run_command(
        [sys.executable, "-c", "import sys; print('out'); print('err', file=sys.stderr)"]
    )
    assert result.returncode == 0
    assert "out" in result.stdout
    assert "err" in result.stderr


def test_raises_on_a_non_zero_exit():
    with pytest.raises(CommandError, match="exited with 3") as caught:
        run_command([sys.executable, "-c", "import sys; sys.stderr.write('bad'); sys.exit(3)"])
    assert caught.value.returncode == 3
    assert "bad" in caught.value.stderr


def test_check_false_returns_the_failing_result():
    result = run_command([sys.executable, "-c", "raise SystemExit(3)"], check=False)
    assert result.returncode == 3


def test_raises_when_the_executable_is_missing():
    with pytest.raises(CommandError, match="executable or working directory not found"):
        run_command(["definitely-not-a-real-binary-xyz"])


def test_raises_on_timeout():
    with pytest.raises(CommandError, match="timed out"):
        run_command([sys.executable, "-c", "import time; time.sleep(5)"], timeout_sec=0.2)


def test_undecodable_output_does_not_become_a_failure():
    """ffmpeg 는 파일 이름을 시스템 인코딩으로 찍기도 한다."""
    result = run_command(
        [sys.executable, "-c", "import sys; sys.stdout.buffer.write(b'\\xff\\xfe')"]
    )
    assert result.returncode == 0
    assert result.stdout  # 디코딩 실패가 실행 실패로 둔갑하지 않는다
