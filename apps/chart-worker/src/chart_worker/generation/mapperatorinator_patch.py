"""Verify and apply the project-owned Mapperatorinator compatibility patches."""

from __future__ import annotations

import subprocess
from functools import cache
from pathlib import Path
from typing import Literal

EXPECTED_MAPPERATORINATOR_HEAD = "2a70eb89004da20e39b0fcbaad2686b264d5a040"
KEYCOUNT_PATCH_ID = "mania-keycount-v1"
OUTPUT_SAFETY_PATCH_ID = "mania-output-safety-v1"
EVENT_TIMES_PATCH_ID = "mania-event-times-v1"
CONSTRAINT_PATCH_ID = (
    f"{KEYCOUNT_PATCH_ID}+{OUTPUT_SAFETY_PATCH_ID}+{EVENT_TIMES_PATCH_ID}"
)
DEFAULT_PATCH_PATH = (
    Path(__file__).resolve().parents[3]
    / "patches"
    / "mapperatorinator-v32-mania-keycount.patch"
)
OUTPUT_SAFETY_PATCH_PATH = (
    Path(__file__).resolve().parents[3]
    / "patches"
    / "mapperatorinator-v32-output-safety.patch"
)
EVENT_TIMES_PATCH_PATH = (
    Path(__file__).resolve().parents[3]
    / "patches"
    / "mapperatorinator-v32-event-times.patch"
)
REQUIRED_PATCHES = (
    (KEYCOUNT_PATCH_ID, DEFAULT_PATCH_PATH),
    (OUTPUT_SAFETY_PATCH_ID, OUTPUT_SAFETY_PATCH_PATH),
    (EVENT_TIMES_PATCH_ID, EVENT_TIMES_PATCH_PATH),
)

PatchStatus = Literal["APPLIED", "APPLICABLE"]
PatchSpec = tuple[str, Path]


class MapperatorinatorPatchError(RuntimeError):
    """The configured Mapperatorinator checkout cannot use the required patch."""


def _git(home: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=home,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )


def patch_status(home: Path, patch_path: Path, expected_head: str) -> PatchStatus:
    """Return whether an exact upstream checkout can accept or already has the patch."""
    home = Path(home).resolve()
    patch_path = Path(patch_path).resolve()
    if not home.is_dir():
        raise MapperatorinatorPatchError(f"Mapperatorinator home does not exist: {home}")
    if not patch_path.is_file():
        raise MapperatorinatorPatchError(f"Mapperatorinator patch does not exist: {patch_path}")

    head_result = _git(home, "rev-parse", "HEAD")
    if head_result.returncode != 0:
        raise MapperatorinatorPatchError(
            f"could not read Mapperatorinator commit: {head_result.stderr.strip()}"
        )
    actual_head = head_result.stdout.strip()
    if actual_head != expected_head:
        raise MapperatorinatorPatchError(
            f"unexpected Mapperatorinator commit: expected {expected_head}, got {actual_head}"
        )

    patch_arg = str(patch_path)
    reverse = _git(home, "apply", "--reverse", "--check", patch_arg)
    if reverse.returncode == 0:
        return "APPLIED"
    applicable = _git(home, "apply", "--check", patch_arg)
    if applicable.returncode == 0:
        return "APPLICABLE"
    detail = applicable.stderr.strip() or reverse.stderr.strip()
    raise MapperatorinatorPatchError(
        f"Mapperatorinator patch is partial or conflicts with the checkout: {detail}"
    )


def apply_mapperatorinator_patch(
    home: Path,
    *,
    patch_path: Path = DEFAULT_PATCH_PATH,
    expected_head: str = EXPECTED_MAPPERATORINATOR_HEAD,
) -> None:
    """Apply the compatibility patch once, leaving an already patched checkout unchanged."""
    status = patch_status(home, patch_path, expected_head)
    if status == "APPLIED":
        return
    result = _git(Path(home).resolve(), "apply", str(Path(patch_path).resolve()))
    if result.returncode != 0:
        raise MapperatorinatorPatchError(
            f"failed to apply Mapperatorinator patch: {result.stderr.strip()}"
        )
    if patch_status(home, patch_path, expected_head) != "APPLIED":
        raise MapperatorinatorPatchError("Mapperatorinator patch did not reach APPLIED state")


def required_patch_statuses(
    home: Path,
    *,
    patches: tuple[PatchSpec, ...] = REQUIRED_PATCHES,
    expected_head: str = EXPECTED_MAPPERATORINATOR_HEAD,
) -> dict[str, PatchStatus]:
    """Return each required patch status for the pinned checkout."""
    return {
        patch_id: patch_status(home, patch_path, expected_head)
        for patch_id, patch_path in patches
    }


def apply_required_mapperatorinator_patches(
    home: Path,
    *,
    patches: tuple[PatchSpec, ...] = REQUIRED_PATCHES,
    expected_head: str = EXPECTED_MAPPERATORINATOR_HEAD,
) -> None:
    """Apply every project-owned patch without reapplying completed patches."""
    for _, patch_path in patches:
        apply_mapperatorinator_patch(
            home,
            patch_path=patch_path,
            expected_head=expected_head,
        )


def require_mapperatorinator_patch_set(
    home: Path,
    *,
    patches: tuple[PatchSpec, ...] = REQUIRED_PATCHES,
    expected_head: str = EXPECTED_MAPPERATORINATOR_HEAD,
) -> None:
    """Fail unless every project-owned patch is already applied."""
    statuses = required_patch_statuses(
        home,
        patches=patches,
        expected_head=expected_head,
    )
    missing = [patch_id for patch_id, status in statuses.items() if status != "APPLIED"]
    if missing:
        raise MapperatorinatorPatchError(
            f"required Mapperatorinator patches are not applied: {', '.join(missing)}; "
            "run scripts/apply_mapperatorinator_patch.py first"
        )


@cache
def _require_cached(home: str) -> None:
    require_mapperatorinator_patch_set(Path(home))


def require_mapperatorinator_patch(home: Path) -> None:
    """Fail fast unless the configured checkout has the exact required patch."""
    _require_cached(str(Path(home).resolve()))
