import subprocess
from pathlib import Path

import pytest

from chart_worker.generation.mapperatorinator_patch import (
    MapperatorinatorPatchError,
    apply_mapperatorinator_patch,
    patch_status,
)


def _git(home: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=home,
        check=True,
        capture_output=True,
        text=True,
    )


def make_upstream_fixture(tmp_path: Path) -> tuple[Path, str, Path]:
    home = tmp_path / "upstream"
    home.mkdir()
    _git(home, "init")
    _git(home, "config", "user.name", "Chart Worker Tests")
    _git(home, "config", "user.email", "chart-worker-tests@example.invalid")
    (home / "module.py").write_text("before\n", encoding="utf-8")
    _git(home, "add", "module.py")
    _git(home, "commit", "-m", "fixture")
    head = _git(home, "rev-parse", "HEAD").stdout.strip()
    patch = tmp_path / "constraint.patch"
    patch.write_text(
        "diff --git a/module.py b/module.py\n"
        "--- a/module.py\n"
        "+++ b/module.py\n"
        "@@ -1 +1 @@\n"
        "-before\n"
        "+after\n",
        encoding="utf-8",
    )
    return home, head, patch


def test_patch_lifecycle_is_applicable_then_applied(tmp_path: Path):
    home, head, patch = make_upstream_fixture(tmp_path)

    assert patch_status(home, patch, head) == "APPLICABLE"
    apply_mapperatorinator_patch(home, patch_path=patch, expected_head=head)

    assert patch_status(home, patch, head) == "APPLIED"
    assert (home / "module.py").read_text(encoding="utf-8") == "after\n"


def test_patch_application_is_idempotent(tmp_path: Path):
    home, head, patch = make_upstream_fixture(tmp_path)

    apply_mapperatorinator_patch(home, patch_path=patch, expected_head=head)
    apply_mapperatorinator_patch(home, patch_path=patch, expected_head=head)

    assert (home / "module.py").read_text(encoding="utf-8") == "after\n"


def test_patch_rejects_an_unexpected_upstream_commit(tmp_path: Path):
    home, _, patch = make_upstream_fixture(tmp_path)

    with pytest.raises(MapperatorinatorPatchError, match="commit"):
        patch_status(home, patch, "0" * 40)


def test_patch_rejects_a_partial_or_conflicting_tree(tmp_path: Path):
    home, head, patch = make_upstream_fixture(tmp_path)
    (home / "module.py").write_text("partially changed\n", encoding="utf-8")

    with pytest.raises(MapperatorinatorPatchError, match="partial|conflict"):
        patch_status(home, patch, head)
