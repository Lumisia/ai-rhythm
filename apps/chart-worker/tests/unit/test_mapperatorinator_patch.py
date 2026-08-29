import hashlib
import os
import shutil
import subprocess
from contextlib import contextmanager
from pathlib import Path

import pytest

from chart_worker.generation import mapperatorinator_patch
from chart_worker.generation.mapperatorinator_patch import (
    REQUIRED_PATCHES,
    MapperatorinatorPatchError,
    apply_mapperatorinator_patch,
    apply_required_mapperatorinator_patches,
    patch_status,
    require_mapperatorinator_patch_set,
    required_patch_statuses,
)


def test_required_patch_manifest_includes_hold_state_grammar():
    assert [patch_id for patch_id, _ in REQUIRED_PATCHES] == [
        "mania-keycount-v1",
        "mania-output-safety-v1",
        (
            "mania-event-times-v2-group-indexed+"
            "mania-resnap-v3-hold-pairs"
        ),
        "mania-origin-v3-hold-ir-column-boundary",
        "mania-sidecar-v4-osu-bound",
        "mania-hold-state-v1-window-context",
        "mania-grammar-v2-atomic-groups",
        "mania-grammar-v3-budget-trim-resnap-order",
        "mania-grammar-v4-prompt-end-state",
        "mania-grammar-v5-serialization-boundaries",
        "mania-grammar-v6-effective-generation-limit",
        "mania-grammar-v7-last-slot-atomic-completion",
        "mania-grammar-v8-transition-evidence",
        "mania-grammar-v9-prompt-timeline-state",
        "mania-grammar-v10-full-prompt-groups",
        "mania-grammar-v11-window-failure-evidence",
        "mania-grammar-v12-addressable-negative-time",
        "mania-grammar-v13-prompt-append-time",
        "mania-grammar-v14-prompt-boundary-evidence",
        "mania-grammar-v15-require-addressable-time",
        "mania-grammar-v16-canonical-append-authority",
        "mania-grammar-v17-addressable-prompt-boundary",
        "mania-grammar-v18-prompt-source-evidence",
        "mania-grammar-v19-negative-bucket-boundary",
        "mania-grammar-v20-token-domain-initial-state",
        "mania-grammar-v21-lane-timeline-evidence",
        "mania-grammar-v22-canonical-column-authority",
        "mania-grammar-v23-start-boundary-feasibility",
        "mania-grammar-v24-resnap-boundary-contract",
        "mania-grammar-v25-incremental-hold-state",
        "mania-tail-repair-v26-checkpointed-suffix",
        "resident-runtime-v27-offline-session-cache",
        "mania-temporal-horizon-v28-time-and-token-reachability",
        "mania-decoder-termination-v29-hard-cap-eos",
        "generation-telemetry-v30-hash-bound-evidence",
        "resnap-lane-order-v31-integer-milliseconds",
        "canonical-sidecar-v32-final-serialization",
        "mania-terminal-budget-v33-reserve-eos",
        "mania-group-fragment-v34-explicit-failure",
        "required-gameplay-interval-v35-origin-accounting",
        "required-gameplay-enforcement-v36-shadow-fsm",
        "intro-evidence-v37-region-class",
    ]
    assert all(path.is_file() for _, path in REQUIRED_PATCHES)
    assert (
        mapperatorinator_patch.INTRO_EVIDENCE_V37_PATCH_PATH
        == REQUIRED_PATCHES[-1][1]
    )
    assert (
        mapperatorinator_patch.REQUIRED_GAMEPLAY_ENFORCEMENT_V36_PATCH_PATH
        == REQUIRED_PATCHES[-2][1]
    )
    assert (
        mapperatorinator_patch.REQUIRED_GAMEPLAY_INTERVAL_V35_PATCH_PATH
        == REQUIRED_PATCHES[-3][1]
    )
    assert REQUIRED_PATCHES[-4][0] == (
        "mania-group-fragment-v34-explicit-failure"
    )
    assert (
        mapperatorinator_patch.MANIA_GROUP_FRAGMENT_V34_PATCH_PATH
        == REQUIRED_PATCHES[-4][1]
    )
    assert REQUIRED_PATCHES[-5][0] == (
        "mania-terminal-budget-v33-reserve-eos"
    )
    assert (
        mapperatorinator_patch.MANIA_TERMINAL_BUDGET_V33_PATCH_PATH
        == REQUIRED_PATCHES[-5][1]
    )
    assert (
        mapperatorinator_patch.LEGACY_V36_REQUIRED_PATCHES
        == REQUIRED_PATCHES[:-1]
    )
    assert (
        mapperatorinator_patch.LEGACY_V35_REQUIRED_PATCHES
        == mapperatorinator_patch.LEGACY_V36_REQUIRED_PATCHES[:-1]
    )
    assert (
        mapperatorinator_patch.LEGACY_V34_REQUIRED_PATCHES
        == mapperatorinator_patch.LEGACY_V35_REQUIRED_PATCHES[:-1]
    )
    assert (
        mapperatorinator_patch.LEGACY_V33_REQUIRED_PATCHES
        == mapperatorinator_patch.LEGACY_V34_REQUIRED_PATCHES[:-1]
    )
    assert (
        mapperatorinator_patch.LEGACY_V32_REQUIRED_PATCHES
        == mapperatorinator_patch.LEGACY_V33_REQUIRED_PATCHES[:-1]
    )
    assert REQUIRED_PATCHES[-6][0] == (
        "canonical-sidecar-v32-final-serialization"
    )
    assert (
        mapperatorinator_patch.CANONICAL_SIDECAR_V32_PATCH_PATH
        == REQUIRED_PATCHES[-6][1]
    )
    assert REQUIRED_PATCHES[-7][0] == (
        "resnap-lane-order-v31-integer-milliseconds"
    )
    assert (
        mapperatorinator_patch.LEGACY_V31_REQUIRED_PATCHES
        == mapperatorinator_patch.LEGACY_V32_REQUIRED_PATCHES[:-1]
    )
    assert REQUIRED_PATCHES[-8][0] == (
        "generation-telemetry-v30-hash-bound-evidence"
    )
    assert (
        mapperatorinator_patch.LEGACY_V30_REQUIRED_PATCHES
        == mapperatorinator_patch.LEGACY_V31_REQUIRED_PATCHES[:-1]
    )
    assert REQUIRED_PATCHES[-9][0] == (
        "mania-decoder-termination-v29-hard-cap-eos"
    )
    assert (
        mapperatorinator_patch.LEGACY_V29_REQUIRED_PATCHES
        == mapperatorinator_patch.LEGACY_V30_REQUIRED_PATCHES[:-1]
    )
    assert (
        mapperatorinator_patch.LEGACY_V28_REQUIRED_PATCHES
        == mapperatorinator_patch.LEGACY_V29_REQUIRED_PATCHES[:-1]
    )
    assert (
        mapperatorinator_patch.LEGACY_V27_REQUIRED_PATCHES
        == mapperatorinator_patch.LEGACY_V28_REQUIRED_PATCHES[:-1]
    )
    assert (
        mapperatorinator_patch.LEGACY_V26_REQUIRED_PATCHES
        == mapperatorinator_patch.LEGACY_V27_REQUIRED_PATCHES[:-1]
    )
    assert (
        mapperatorinator_patch.LEGACY_V25_REQUIRED_PATCHES
        == mapperatorinator_patch.LEGACY_V26_REQUIRED_PATCHES[:-1]
    )
    assert (
        mapperatorinator_patch.LEGACY_V24_REQUIRED_PATCHES
        == mapperatorinator_patch.LEGACY_V25_REQUIRED_PATCHES[:-1]
    )
    assert (
        mapperatorinator_patch.LEGACY_V23_REQUIRED_PATCHES
        == mapperatorinator_patch.LEGACY_V24_REQUIRED_PATCHES[:-1]
    )
    assert (
        mapperatorinator_patch.LEGACY_V22_REQUIRED_PATCHES
        == mapperatorinator_patch.LEGACY_V23_REQUIRED_PATCHES[:-1]
    )
    assert (
        mapperatorinator_patch.LEGACY_V21_REQUIRED_PATCHES
        == mapperatorinator_patch.LEGACY_V22_REQUIRED_PATCHES[:-1]
    )
    assert (
        mapperatorinator_patch.LEGACY_V20_REQUIRED_PATCHES
        == mapperatorinator_patch.LEGACY_V21_REQUIRED_PATCHES[:-1]
    )
    assert (
        mapperatorinator_patch.LEGACY_V19_REQUIRED_PATCHES
        == mapperatorinator_patch.LEGACY_V20_REQUIRED_PATCHES[:-1]
    )
    assert (
        mapperatorinator_patch.LEGACY_V18_REQUIRED_PATCHES
        == mapperatorinator_patch.LEGACY_V19_REQUIRED_PATCHES[:-1]
    )
    assert (
        mapperatorinator_patch.LEGACY_V17_REQUIRED_PATCHES
        == mapperatorinator_patch.LEGACY_V18_REQUIRED_PATCHES[:-1]
    )
    assert (
        mapperatorinator_patch.LEGACY_V16_REQUIRED_PATCHES
        == mapperatorinator_patch.LEGACY_V17_REQUIRED_PATCHES[:-1]
    )
    assert (
        mapperatorinator_patch.LEGACY_V15_REQUIRED_PATCHES
        == mapperatorinator_patch.LEGACY_V16_REQUIRED_PATCHES[:-1]
    )
    assert (
        mapperatorinator_patch.LEGACY_V14_REQUIRED_PATCHES
        == mapperatorinator_patch.LEGACY_V15_REQUIRED_PATCHES[:-1]
    )
    assert (
        mapperatorinator_patch.LEGACY_V13_REQUIRED_PATCHES
        == mapperatorinator_patch.LEGACY_V14_REQUIRED_PATCHES[:-1]
    )
    assert (
        mapperatorinator_patch.LEGACY_V12_REQUIRED_PATCHES
        == mapperatorinator_patch.LEGACY_V13_REQUIRED_PATCHES[:-1]
    )
    assert (
        mapperatorinator_patch.LEGACY_V11_REQUIRED_PATCHES
        == mapperatorinator_patch.LEGACY_V12_REQUIRED_PATCHES[:-1]
    )
    assert (
        mapperatorinator_patch.LEGACY_V10_REQUIRED_PATCHES
        == mapperatorinator_patch.LEGACY_V11_REQUIRED_PATCHES[:-1]
    )
    assert (
        mapperatorinator_patch.LEGACY_V9_REQUIRED_PATCHES
        == mapperatorinator_patch.LEGACY_V10_REQUIRED_PATCHES[:-1]
    )
    assert (
        mapperatorinator_patch.LEGACY_V8_REQUIRED_PATCHES
        == mapperatorinator_patch.LEGACY_V9_REQUIRED_PATCHES[:-1]
    )
    assert (
        mapperatorinator_patch.LEGACY_V7_REQUIRED_PATCHES
        == mapperatorinator_patch.LEGACY_V8_REQUIRED_PATCHES[:-1]
    )
    assert (
        mapperatorinator_patch.LEGACY_V6_REQUIRED_PATCHES
        == mapperatorinator_patch.LEGACY_V7_REQUIRED_PATCHES[:-1]
    )
    assert (
        mapperatorinator_patch.LEGACY_V5_REQUIRED_PATCHES
        == mapperatorinator_patch.LEGACY_V6_REQUIRED_PATCHES[:-1]
    )
    assert (
        mapperatorinator_patch.LEGACY_V4_REQUIRED_PATCHES
        == mapperatorinator_patch.LEGACY_V5_REQUIRED_PATCHES[:-1]
    )
    assert (
        mapperatorinator_patch.LEGACY_REQUIRED_PATCH_SETS[0]
        == mapperatorinator_patch.LEGACY_V36_REQUIRED_PATCHES
    )
    assert (
        mapperatorinator_patch.LEGACY_REQUIRED_PATCH_SETS[1]
        == mapperatorinator_patch.LEGACY_V35_REQUIRED_PATCHES
    )
    assert (
        mapperatorinator_patch.LEGACY_REQUIRED_PATCH_SETS[2]
        == mapperatorinator_patch.LEGACY_V34_REQUIRED_PATCHES
    )
    assert (
        mapperatorinator_patch.LEGACY_REQUIRED_PATCH_SETS[3]
        == mapperatorinator_patch.LEGACY_V33_REQUIRED_PATCHES
    )


def test_required_patch_files_are_syntactically_valid():
    for patch_id, patch_path in REQUIRED_PATCHES:
        result = subprocess.run(
            ["git", "apply", "--numstat", str(patch_path)],
            check=False,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, (
            f"{patch_id} is not a valid git patch: {result.stderr.strip()}"
        )


def _clone_pinned_upstream(source: Path, destination: Path) -> Path:
    destination.mkdir()
    shutil.copytree(source / ".git", destination / ".git", copy_function=shutil.copy2)
    _git(destination, "remote", "set-url", "origin", str(source))
    _git(destination, "config", "core.autocrlf", "false")
    _git(destination, "config", "core.eol", "lf")
    _git(
        destination,
        "checkout",
        "--detach",
        "--force",
        mapperatorinator_patch.EXPECTED_MAPPERATORINATOR_HEAD,
    )
    assert (
        _git(destination, "rev-parse", "HEAD").stdout.strip()
        == mapperatorinator_patch.EXPECTED_MAPPERATORINATOR_HEAD
    )
    return destination


def _patch_paths(patches: tuple[tuple[str, Path], ...]) -> set[str]:
    paths: set[str] = set()
    for patch_id, patch_path in patches:
        result = subprocess.run(
            ["git", "apply", "--numstat", "-z", str(patch_path)],
            check=False,
            capture_output=True,
        )
        assert result.returncode == 0, (
            f"could not inspect paths for {patch_id}: "
            f"{result.stderr.decode('utf-8', errors='replace')}"
        )
        for entry in result.stdout.split(b"\0"):
            if entry:
                paths.add(entry.split(b"\t", 2)[2].decode("utf-8"))
    return paths


def _path_snapshot(home: Path, paths: set[str]) -> dict[str, tuple[str, str, str]]:
    snapshot: dict[str, tuple[str, str, str]] = {}
    for relative in sorted(paths):
        path = home.joinpath(*relative.split("/"))
        if not path.exists():
            snapshot[relative] = ("ABSENT", "ABSENT", "ABSENT")
            continue
        raw = path.read_bytes()
        snapshot[relative] = (
            hashlib.sha256(raw).hexdigest(),
            hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest(),
            _git(home, "hash-object", f"--path={relative}", relative).stdout.strip(),
        )
    return snapshot


def test_v37_patch_chain_round_trips_from_pinned_local_upstream(
    monkeypatch: pytest.MonkeyPatch,
):
    source_value = os.environ.get("MAPPERATORINATOR_INTEGRATION_SOURCE")
    root_value = os.environ.get("MAPPERATORINATOR_INTEGRATION_ROOT")
    if not source_value:
        pytest.fail(
            "MAPPERATORINATOR_INTEGRATION_SOURCE is required and must point "
            "to the local Mapperatorinator checkout at the pinned commit"
        )
    if not root_value:
        pytest.fail(
            "MAPPERATORINATOR_INTEGRATION_ROOT is required and must name a "
            "new, absent evidence directory"
        )
    source = Path(source_value).resolve()
    integration_root = Path(root_value).resolve()
    if not source.is_dir():
        pytest.fail(
            "MAPPERATORINATOR_INTEGRATION_SOURCE does not exist or is not a "
            f"directory: {source}"
        )
    source_head = subprocess.run(
        ["git", "-C", str(source), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    if source_head.returncode != 0:
        pytest.fail(
            "MAPPERATORINATOR_INTEGRATION_SOURCE must be a local Git checkout: "
            f"{source_head.stderr.strip()}"
        )
    actual_source_head = source_head.stdout.strip()
    if actual_source_head != mapperatorinator_patch.EXPECTED_MAPPERATORINATOR_HEAD:
        pytest.fail(
            "MAPPERATORINATOR_INTEGRATION_SOURCE must be pinned to "
            f"{mapperatorinator_patch.EXPECTED_MAPPERATORINATOR_HEAD}, got "
            f"{actual_source_head}"
        )
    if integration_root.exists():
        pytest.fail(
            "MAPPERATORINATOR_INTEGRATION_ROOT must be absent before the proof: "
            f"{integration_root}"
        )
    integration_root.mkdir(parents=True)
    assert source.resolve() != integration_root.resolve()

    temp_index = 0

    @contextmanager
    def accessible_temporary_directory(*, prefix: str):
        nonlocal temp_index
        path = integration_root / f"{prefix}{temp_index}"
        temp_index += 1
        path.mkdir()
        yield str(path)

    monkeypatch.setattr(
        mapperatorinator_patch.tempfile,
        "TemporaryDirectory",
        accessible_temporary_directory,
    )
    legacy_v36 = mapperatorinator_patch.LEGACY_V36_REQUIRED_PATCHES
    v37_patch = REQUIRED_PATCHES[-1][1]
    all_paths = _patch_paths(REQUIRED_PATCHES)
    v37_paths = _patch_paths((REQUIRED_PATCHES[-1],))
    earlier_only_paths = all_paths - v37_paths

    full_home = _clone_pinned_upstream(source, integration_root / "full-v37")
    assert (full_home / ".git" / "objects").resolve() != (
        source / ".git" / "objects"
    ).resolve()
    apply_required_mapperatorinator_patches(full_home)
    expected_statuses = dict.fromkeys(
        (patch_id for patch_id, _ in REQUIRED_PATCHES),
        "APPLIED",
    )
    assert required_patch_statuses(full_home) == expected_statuses
    full_snapshot = _path_snapshot(full_home, all_paths)
    first_status = _git(full_home, "status", "--porcelain=v1").stdout

    apply_required_mapperatorinator_patches(full_home)
    assert required_patch_statuses(full_home) == expected_statuses
    assert _path_snapshot(full_home, all_paths) == full_snapshot
    assert _git(full_home, "status", "--porcelain=v1").stdout == first_status

    v36_home = _clone_pinned_upstream(source, integration_root / "exact-v36")
    apply_required_mapperatorinator_patches(
        v36_home,
        patches=legacy_v36,
    )
    v36_snapshot = _path_snapshot(v36_home, all_paths)

    reverse = _git(full_home, "apply", "--reverse", str(v37_patch))
    assert reverse.returncode == 0
    reversed_snapshot = _path_snapshot(full_home, all_paths)
    assert reversed_snapshot == v36_snapshot
    assert {
        path: reversed_snapshot[path] for path in earlier_only_paths
    } == {
        path: full_snapshot[path] for path in earlier_only_paths
    }

    reapply = _git(full_home, "apply", str(v37_patch))
    assert reapply.returncode == 0
    assert _path_snapshot(full_home, all_paths) == full_snapshot

    apply_required_mapperatorinator_patches(v36_home)
    assert required_patch_statuses(v36_home) == expected_statuses
    assert _path_snapshot(v36_home, all_paths) == full_snapshot


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


def make_layered_patch_set_fixture(
    tmp_path: Path,
) -> tuple[Path, str, tuple[tuple[str, Path], ...]]:
    home = tmp_path / "upstream-layered"
    home.mkdir()
    _git(home, "init")
    _git(home, "config", "user.name", "Chart Worker Tests")
    _git(home, "config", "user.email", "chart-worker-tests@example.invalid")
    (home / "module.py").write_text("before\n", encoding="utf-8")
    _git(home, "add", "module.py")
    _git(home, "commit", "-m", "fixture")
    head = _git(home, "rev-parse", "HEAD").stdout.strip()
    first = tmp_path / "first-layer.patch"
    first.write_text(
        "diff --git a/module.py b/module.py\n"
        "--- a/module.py\n"
        "+++ b/module.py\n"
        "@@ -1 +1 @@\n"
        "-before\n"
        "+middle\n",
        encoding="utf-8",
    )
    second = tmp_path / "second-layer.patch"
    second.write_text(
        "diff --git a/module.py b/module.py\n"
        "--- a/module.py\n"
        "+++ b/module.py\n"
        "@@ -1 +1 @@\n"
        "-middle\n"
        "+after\n",
        encoding="utf-8",
    )
    return home, head, (("first-v1", first), ("second-v1", second))


def make_legacy_migration_fixture(
    tmp_path: Path,
) -> tuple[
    Path,
    str,
    tuple[tuple[str, Path], ...],
    tuple[tuple[str, Path], ...],
]:
    home = tmp_path / "upstream-migration"
    home.mkdir()
    _git(home, "init")
    _git(home, "config", "user.name", "Chart Worker Tests")
    _git(home, "config", "user.email", "chart-worker-tests@example.invalid")
    (home / "module.py").write_text("before\n", encoding="utf-8")
    _git(home, "add", "module.py")
    _git(home, "commit", "-m", "fixture")
    head = _git(home, "rev-parse", "HEAD").stdout.strip()

    common = tmp_path / "common.patch"
    common.write_text(
        "diff --git a/module.py b/module.py\n"
        "--- a/module.py\n"
        "+++ b/module.py\n"
        "@@ -1 +1 @@\n"
        "-before\n"
        "+common\n",
        encoding="utf-8",
    )
    legacy = tmp_path / "legacy.patch"
    legacy.write_text(
        "diff --git a/module.py b/module.py\n"
        "--- a/module.py\n"
        "+++ b/module.py\n"
        "@@ -1 +1 @@\n"
        "-common\n"
        "+legacy\n",
        encoding="utf-8",
    )
    replacement = tmp_path / "replacement.patch"
    replacement.write_text(
        "diff --git a/module.py b/module.py\n"
        "--- a/module.py\n"
        "+++ b/module.py\n"
        "@@ -1 +1 @@\n"
        "-common\n"
        "+replacement\n",
        encoding="utf-8",
    )
    legacy_patches = (("common-v1", common), ("legacy-v1", legacy))
    replacement_patches = (("common-v1", common), ("replacement-v2", replacement))
    return home, head, legacy_patches, replacement_patches


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


def test_required_patch_set_recognizes_overlapping_layered_patches(tmp_path: Path):
    home, head, patches = make_layered_patch_set_fixture(tmp_path)

    apply_required_mapperatorinator_patches(
        home,
        patches=patches,
        expected_head=head,
    )


def test_layered_patch_verification_ignores_parent_git_repository_of_tempdir(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    home, head, patches = make_layered_patch_set_fixture(tmp_path)
    apply_required_mapperatorinator_patches(
        home,
        patches=patches,
        expected_head=head,
    )

    temp_parent = tmp_path / "unrelated-parent"
    temp_parent.mkdir()
    _git(temp_parent, "init")
    _git(temp_parent, "config", "user.name", "Chart Worker Tests")
    _git(temp_parent, "config", "user.email", "chart-worker-tests@example.invalid")
    sentinel = temp_parent / "sentinel.txt"
    sentinel.write_text("must-not-change\n", encoding="utf-8")
    _git(temp_parent, "add", "sentinel.txt")
    _git(temp_parent, "commit", "-m", "unrelated fixture")
    temp_root = temp_parent / "runtime-temp"
    temp_root.mkdir()
    monkeypatch.setattr(mapperatorinator_patch.tempfile, "tempdir", str(temp_root))

    assert required_patch_statuses(
        home,
        patches=patches,
        expected_head=head,
    ) == {
        "first-v1": "APPLIED",
        "second-v1": "APPLIED",
    }
    assert sentinel.read_text(encoding="utf-8") == "must-not-change\n"


def test_required_patch_set_appends_to_an_exact_layered_legacy_stack(tmp_path: Path):
    home, head, legacy = make_layered_patch_set_fixture(tmp_path)
    third = tmp_path / "third-layer.patch"
    third.write_text(
        "diff --git a/module.py b/module.py\n"
        "--- a/module.py\n"
        "+++ b/module.py\n"
        "@@ -1 +1 @@\n"
        "-after\n"
        "+final\n",
        encoding="utf-8",
    )
    replacement = (*legacy, ("third-v1", third))
    apply_required_mapperatorinator_patches(
        home,
        patches=legacy,
        expected_head=head,
    )

    apply_required_mapperatorinator_patches(
        home,
        patches=replacement,
        legacy_patches=legacy,
        expected_head=head,
    )

    assert (home / "module.py").read_text(encoding="utf-8") == "final\n"


def test_required_patch_set_ignores_only_crlf_representation(tmp_path: Path):
    home, head, patches = make_layered_patch_set_fixture(tmp_path)
    apply_required_mapperatorinator_patches(
        home,
        patches=patches,
        expected_head=head,
    )
    (home / "module.py").write_bytes(b"after\r\n")

    assert required_patch_statuses(
        home,
        patches=patches,
        expected_head=head,
    ) == {"first-v1": "APPLIED", "second-v1": "APPLIED"}

    assert required_patch_statuses(
        home,
        patches=patches,
        expected_head=head,
    ) == {"first-v1": "APPLIED", "second-v1": "APPLIED"}
    apply_required_mapperatorinator_patches(
        home,
        patches=patches,
        expected_head=head,
    )


def test_required_patch_set_migrates_an_exact_legacy_stack(tmp_path: Path):
    home, head, legacy_patches, replacement_patches = make_legacy_migration_fixture(
        tmp_path
    )
    apply_required_mapperatorinator_patches(
        home,
        patches=legacy_patches,
        expected_head=head,
    )

    apply_required_mapperatorinator_patches(
        home,
        patches=replacement_patches,
        legacy_patches=legacy_patches,
        expected_head=head,
    )

    assert (home / "module.py").read_text(encoding="utf-8") == "replacement\n"
    assert required_patch_statuses(
        home,
        patches=replacement_patches,
        expected_head=head,
    ) == {"common-v1": "APPLIED", "replacement-v2": "APPLIED"}


def test_required_patch_set_tries_each_known_legacy_stack(tmp_path: Path):
    home, head, legacy_patches, replacement_patches = make_legacy_migration_fixture(
        tmp_path
    )
    apply_required_mapperatorinator_patches(
        home,
        patches=legacy_patches,
        expected_head=head,
    )

    apply_required_mapperatorinator_patches(
        home,
        patches=replacement_patches,
        legacy_patch_sets=(replacement_patches, legacy_patches),
        expected_head=head,
    )

    assert (home / "module.py").read_text(encoding="utf-8") == "replacement\n"


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
