import subprocess
from pathlib import Path

import pytest

from chart_worker.generation.mapperatorinator_patch import (
    MapperatorinatorPatchError,
    apply_mapperatorinator_patch,
    apply_required_mapperatorinator_patches,
    patch_status,
    require_mapperatorinator_patch_set,
    required_patch_statuses,
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


def make_patch_set_fixture(
    tmp_path: Path,
) -> tuple[Path, str, tuple[tuple[str, Path], ...]]:
    home = tmp_path / "upstream-set"
    home.mkdir()
    _git(home, "init")
    _git(home, "config", "user.name", "Chart Worker Tests")
    _git(home, "config", "user.email", "chart-worker-tests@example.invalid")
    (home / "first.py").write_text("first-before\n", encoding="utf-8")
    (home / "second.py").write_text("second-before\n", encoding="utf-8")
    (home / "third.py").write_text("third-before\n", encoding="utf-8")
    _git(home, "add", "first.py", "second.py", "third.py")
    _git(home, "commit", "-m", "fixture")
    head = _git(home, "rev-parse", "HEAD").stdout.strip()

    patches = []
    for patch_id, filename, before, after in (
        ("first-v1", "first.py", "first-before", "first-after"),
        ("second-v1", "second.py", "second-before", "second-after"),
        ("third-v1", "third.py", "third-before", "third-after"),
    ):
        patch = tmp_path / f"{patch_id}.patch"
        patch.write_text(
            f"diff --git a/{filename} b/{filename}\n"
            f"--- a/{filename}\n"
            f"+++ b/{filename}\n"
            "@@ -1 +1 @@\n"
            f"-{before}\n"
            f"+{after}\n",
            encoding="utf-8",
        )
        patches.append((patch_id, patch))
    return home, head, tuple(patches)


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


def test_required_patch_set_applies_every_patch(tmp_path: Path):
    home, head, patches = make_patch_set_fixture(tmp_path)

    assert required_patch_statuses(home, patches=patches, expected_head=head) == {
        "first-v1": "APPLICABLE",
        "second-v1": "APPLICABLE",
        "third-v1": "APPLICABLE",
    }
    apply_required_mapperatorinator_patches(
        home,
        patches=patches,
        expected_head=head,
    )

    assert required_patch_statuses(home, patches=patches, expected_head=head) == {
        "first-v1": "APPLIED",
        "second-v1": "APPLIED",
        "third-v1": "APPLIED",
    }
    assert (home / "first.py").read_text(encoding="utf-8") == "first-after\n"
    assert (home / "second.py").read_text(encoding="utf-8") == "second-after\n"
    assert (home / "third.py").read_text(encoding="utf-8") == "third-after\n"


@pytest.mark.parametrize("missing_index", [1, 2])
def test_required_patch_set_rejects_a_partially_applied_checkout(
    tmp_path: Path,
    missing_index: int,
):
    home, head, patches = make_patch_set_fixture(tmp_path)
    for index, (_, patch) in enumerate(patches):
        if index != missing_index:
            apply_mapperatorinator_patch(
                home,
                patch_path=patch,
                expected_head=head,
            )

    with pytest.raises(MapperatorinatorPatchError, match=patches[missing_index][0]):
        require_mapperatorinator_patch_set(
            home,
            patches=patches,
            expected_head=head,
        )


def test_patch_rejects_an_unexpected_upstream_commit(tmp_path: Path):
    home, _, patch = make_upstream_fixture(tmp_path)

    with pytest.raises(MapperatorinatorPatchError, match="commit"):
        patch_status(home, patch, "0" * 40)


def test_patch_rejects_a_partial_or_conflicting_tree(tmp_path: Path):
    home, head, patch = make_upstream_fixture(tmp_path)
    (home / "module.py").write_text("partially changed\n", encoding="utf-8")

    with pytest.raises(MapperatorinatorPatchError, match="partial|conflict"):
        patch_status(home, patch, head)
