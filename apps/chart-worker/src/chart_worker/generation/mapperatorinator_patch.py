"""Verify and apply the project-owned Mapperatorinator compatibility patches."""

from __future__ import annotations

import os
import subprocess
import tempfile
from functools import cache
from pathlib import Path, PurePosixPath
from typing import Literal

EXPECTED_MAPPERATORINATOR_HEAD = "2a70eb89004da20e39b0fcbaad2686b264d5a040"
KEYCOUNT_PATCH_ID = "mania-keycount-v1"
OUTPUT_SAFETY_PATCH_ID = "mania-output-safety-v1"
EVENT_TIMES_PATCH_ID = "mania-event-times-v2-group-indexed"
RESNAP_COLLISIONS_PATCH_ID = "mania-resnap-v3-hold-pairs"
MANIA_RESNAP_PATCH_ID = f"{EVENT_TIMES_PATCH_ID}+{RESNAP_COLLISIONS_PATCH_ID}"
MANIA_ORIGIN_PATCH_ID = "mania-origin-v3-hold-ir-column-boundary"
MANIA_SIDECAR_PATCH_ID = "mania-sidecar-v4-osu-bound"
MANIA_HOLD_STATE_PATCH_ID = "mania-hold-state-v1-window-context"
MANIA_GRAMMAR_PATCH_ID = "mania-grammar-v2-atomic-groups"
MANIA_GRAMMAR_V3_PATCH_ID = "mania-grammar-v3-budget-trim-resnap-order"
MANIA_GRAMMAR_V4_PATCH_ID = "mania-grammar-v4-prompt-end-state"
MANIA_GRAMMAR_V5_PATCH_ID = "mania-grammar-v5-serialization-boundaries"
MANIA_GRAMMAR_V6_PATCH_ID = "mania-grammar-v6-effective-generation-limit"
MANIA_GRAMMAR_V7_PATCH_ID = "mania-grammar-v7-last-slot-atomic-completion"
MANIA_GRAMMAR_V8_PATCH_ID = "mania-grammar-v8-transition-evidence"
MANIA_GRAMMAR_V9_PATCH_ID = "mania-grammar-v9-prompt-timeline-state"
MANIA_GRAMMAR_V10_PATCH_ID = "mania-grammar-v10-full-prompt-groups"
MANIA_GRAMMAR_V11_PATCH_ID = "mania-grammar-v11-window-failure-evidence"
MANIA_GRAMMAR_V12_PATCH_ID = "mania-grammar-v12-addressable-negative-time"
MANIA_GRAMMAR_V13_PATCH_ID = "mania-grammar-v13-prompt-append-time"
MANIA_GRAMMAR_V14_PATCH_ID = "mania-grammar-v14-prompt-boundary-evidence"
MANIA_GRAMMAR_V15_PATCH_ID = "mania-grammar-v15-require-addressable-time"
MANIA_GRAMMAR_V16_PATCH_ID = "mania-grammar-v16-canonical-append-authority"
MANIA_GRAMMAR_V17_PATCH_ID = "mania-grammar-v17-addressable-prompt-boundary"
MANIA_GRAMMAR_V18_PATCH_ID = "mania-grammar-v18-prompt-source-evidence"
MANIA_GRAMMAR_V19_PATCH_ID = "mania-grammar-v19-negative-bucket-boundary"
MANIA_GRAMMAR_V20_PATCH_ID = "mania-grammar-v20-token-domain-initial-state"
MANIA_GRAMMAR_V21_PATCH_ID = "mania-grammar-v21-lane-timeline-evidence"
MANIA_GRAMMAR_V22_PATCH_ID = "mania-grammar-v22-canonical-column-authority"
MANIA_GRAMMAR_V23_PATCH_ID = "mania-grammar-v23-start-boundary-feasibility"
MANIA_GRAMMAR_V24_PATCH_ID = "mania-grammar-v24-resnap-boundary-contract"
MANIA_GRAMMAR_V25_PATCH_ID = "mania-grammar-v25-incremental-hold-state"
MANIA_TAIL_REPAIR_V26_PATCH_ID = "mania-tail-repair-v26-checkpointed-suffix"
RESIDENT_RUNTIME_V27_PATCH_ID = "resident-runtime-v27-offline-session-cache"
MANIA_TEMPORAL_HORIZON_V28_PATCH_ID = (
    "mania-temporal-horizon-v28-time-and-token-reachability"
)
MANIA_DECODER_TERMINATION_V29_PATCH_ID = (
    "mania-decoder-termination-v29-hard-cap-eos"
)
GENERATION_TELEMETRY_V30_PATCH_ID = (
    "generation-telemetry-v30-hash-bound-evidence"
)
RESNAP_LANE_ORDER_V31_PATCH_ID = (
    "resnap-lane-order-v31-integer-milliseconds"
)
CANONICAL_SIDECAR_V32_PATCH_ID = (
    "canonical-sidecar-v32-final-serialization"
)
MANIA_TERMINAL_BUDGET_V33_PATCH_ID = (
    "mania-terminal-budget-v33-reserve-eos"
)
MANIA_GROUP_FRAGMENT_V34_PATCH_ID = (
    "mania-group-fragment-v34-explicit-failure"
)
CONSTRAINT_PATCH_ID = (
    f"{KEYCOUNT_PATCH_ID}+{OUTPUT_SAFETY_PATCH_ID}+{MANIA_RESNAP_PATCH_ID}+"
    f"{MANIA_ORIGIN_PATCH_ID}+{MANIA_SIDECAR_PATCH_ID}+"
    f"{MANIA_HOLD_STATE_PATCH_ID}+{MANIA_GRAMMAR_PATCH_ID}+"
    f"{MANIA_GRAMMAR_V3_PATCH_ID}+{MANIA_GRAMMAR_V4_PATCH_ID}+"
    f"{MANIA_GRAMMAR_V5_PATCH_ID}+{MANIA_GRAMMAR_V6_PATCH_ID}+"
    f"{MANIA_GRAMMAR_V7_PATCH_ID}+{MANIA_GRAMMAR_V8_PATCH_ID}+"
    f"{MANIA_GRAMMAR_V9_PATCH_ID}+{MANIA_GRAMMAR_V10_PATCH_ID}+"
    f"{MANIA_GRAMMAR_V11_PATCH_ID}+{MANIA_GRAMMAR_V12_PATCH_ID}+"
    f"{MANIA_GRAMMAR_V13_PATCH_ID}+{MANIA_GRAMMAR_V14_PATCH_ID}+"
    f"{MANIA_GRAMMAR_V15_PATCH_ID}+{MANIA_GRAMMAR_V16_PATCH_ID}+"
    f"{MANIA_GRAMMAR_V17_PATCH_ID}+{MANIA_GRAMMAR_V18_PATCH_ID}+"
    f"{MANIA_GRAMMAR_V19_PATCH_ID}+{MANIA_GRAMMAR_V20_PATCH_ID}+"
    f"{MANIA_GRAMMAR_V21_PATCH_ID}+{MANIA_GRAMMAR_V22_PATCH_ID}+"
    f"{MANIA_GRAMMAR_V23_PATCH_ID}+{MANIA_GRAMMAR_V24_PATCH_ID}+"
    f"{MANIA_GRAMMAR_V25_PATCH_ID}+{MANIA_TAIL_REPAIR_V26_PATCH_ID}+"
    f"{RESIDENT_RUNTIME_V27_PATCH_ID}+{MANIA_TEMPORAL_HORIZON_V28_PATCH_ID}+"
    f"{MANIA_DECODER_TERMINATION_V29_PATCH_ID}+"
    f"{GENERATION_TELEMETRY_V30_PATCH_ID}+"
    f"{RESNAP_LANE_ORDER_V31_PATCH_ID}+"
    f"{CANONICAL_SIDECAR_V32_PATCH_ID}+"
    f"{MANIA_TERMINAL_BUDGET_V33_PATCH_ID}+"
    f"{MANIA_GROUP_FRAGMENT_V34_PATCH_ID}"
)
_PACKAGE_ROOT = Path(__file__).resolve().parents[1]
_PACKAGED_PATCH_DIR = _PACKAGE_ROOT / "patches"
_SOURCE_PATCH_DIR = Path(__file__).resolve().parents[3] / "patches"
PATCH_DIR = (
    _PACKAGED_PATCH_DIR
    if _PACKAGED_PATCH_DIR.is_dir()
    else _SOURCE_PATCH_DIR
)
DEFAULT_PATCH_PATH = PATCH_DIR / "mapperatorinator-v32-mania-keycount.patch"
OUTPUT_SAFETY_PATCH_PATH = PATCH_DIR / "mapperatorinator-v32-output-safety.patch"
LEGACY_EVENT_TIMES_PATCH_PATH = PATCH_DIR / "mapperatorinator-v32-event-times.patch"
LEGACY_RESNAP_COLLISIONS_PATCH_PATH = (
    PATCH_DIR / "mapperatorinator-v32-resnap-collisions.patch"
)
MANIA_RESNAP_PATCH_PATH = PATCH_DIR / "mapperatorinator-v32-mania-resnap-v3.patch"
MANIA_ORIGIN_PATCH_PATH = PATCH_DIR / "mapperatorinator-v32-mania-origin-v3.patch"
MANIA_SIDECAR_PATCH_PATH = PATCH_DIR / "mapperatorinator-v32-mania-sidecar-v4.patch"
MANIA_HOLD_STATE_PATCH_PATH = (
    PATCH_DIR / "mapperatorinator-v32-mania-hold-state-v1.patch"
)
MANIA_GRAMMAR_PATCH_PATH = (
    PATCH_DIR / "mapperatorinator-v32-mania-grammar-v2.patch"
)
MANIA_GRAMMAR_V3_PATCH_PATH = (
    PATCH_DIR / "mapperatorinator-v32-mania-grammar-v3.patch"
)
MANIA_GRAMMAR_V4_PATCH_PATH = (
    PATCH_DIR / "mapperatorinator-v32-mania-grammar-v4.patch"
)
MANIA_GRAMMAR_V5_PATCH_PATH = (
    PATCH_DIR / "mapperatorinator-v32-mania-grammar-v5.patch"
)
MANIA_GRAMMAR_V6_PATCH_PATH = (
    PATCH_DIR / "mapperatorinator-v32-mania-grammar-v6.patch"
)
MANIA_GRAMMAR_V7_PATCH_PATH = (
    PATCH_DIR / "mapperatorinator-v32-mania-grammar-v7.patch"
)
MANIA_GRAMMAR_V8_PATCH_PATH = (
    PATCH_DIR / "mapperatorinator-v32-mania-grammar-v8.patch"
)
MANIA_GRAMMAR_V9_PATCH_PATH = (
    PATCH_DIR / "mapperatorinator-v32-mania-grammar-v9.patch"
)
MANIA_GRAMMAR_V10_PATCH_PATH = (
    PATCH_DIR / "mapperatorinator-v32-mania-grammar-v10.patch"
)
MANIA_GRAMMAR_V11_PATCH_PATH = (
    PATCH_DIR / "mapperatorinator-v32-mania-grammar-v11.patch"
)
MANIA_GRAMMAR_V12_PATCH_PATH = (
    PATCH_DIR / "mapperatorinator-v32-mania-grammar-v12.patch"
)
MANIA_GRAMMAR_V13_PATCH_PATH = (
    PATCH_DIR / "mapperatorinator-v32-mania-grammar-v13.patch"
)
MANIA_GRAMMAR_V14_PATCH_PATH = (
    PATCH_DIR / "mapperatorinator-v32-mania-grammar-v14.patch"
)
MANIA_GRAMMAR_V15_PATCH_PATH = (
    PATCH_DIR / "mapperatorinator-v32-mania-grammar-v15.patch"
)
MANIA_GRAMMAR_V16_PATCH_PATH = (
    PATCH_DIR / "mapperatorinator-v32-mania-grammar-v16.patch"
)
MANIA_GRAMMAR_V17_PATCH_PATH = (
    PATCH_DIR / "mapperatorinator-v32-mania-grammar-v17.patch"
)
MANIA_GRAMMAR_V18_PATCH_PATH = (
    PATCH_DIR / "mapperatorinator-v32-mania-grammar-v18.patch"
)
MANIA_GRAMMAR_V19_PATCH_PATH = (
    PATCH_DIR / "mapperatorinator-v32-mania-grammar-v19.patch"
)
MANIA_GRAMMAR_V20_PATCH_PATH = (
    PATCH_DIR / "mapperatorinator-v32-mania-grammar-v20.patch"
)
MANIA_GRAMMAR_V21_PATCH_PATH = (
    PATCH_DIR / "mapperatorinator-v32-mania-grammar-v21.patch"
)
MANIA_GRAMMAR_V22_PATCH_PATH = (
    PATCH_DIR / "mapperatorinator-v32-mania-grammar-v22.patch"
)
MANIA_GRAMMAR_V23_PATCH_PATH = (
    PATCH_DIR / "mapperatorinator-v32-mania-grammar-v23.patch"
)
MANIA_GRAMMAR_V24_PATCH_PATH = (
    PATCH_DIR / "mapperatorinator-v32-mania-grammar-v24.patch"
)
MANIA_GRAMMAR_V25_PATCH_PATH = (
    PATCH_DIR / "mapperatorinator-v32-mania-grammar-v25.patch"
)
MANIA_TAIL_REPAIR_V26_PATCH_PATH = (
    PATCH_DIR / "mapperatorinator-v32-mania-tail-repair-v26.patch"
)
RESIDENT_RUNTIME_V27_PATCH_PATH = (
    PATCH_DIR / "mapperatorinator-v32-resident-runtime-v27.patch"
)
MANIA_TEMPORAL_HORIZON_V28_PATCH_PATH = (
    PATCH_DIR / "mapperatorinator-v32-mania-temporal-horizon-v28.patch"
)
MANIA_DECODER_TERMINATION_V29_PATCH_PATH = (
    PATCH_DIR / "mapperatorinator-v32-mania-decoder-termination-v29.patch"
)
GENERATION_TELEMETRY_V30_PATCH_PATH = (
    PATCH_DIR / "mapperatorinator-v32-generation-telemetry-v30.patch"
)
RESNAP_LANE_ORDER_V31_PATCH_PATH = (
    PATCH_DIR / "mapperatorinator-v32-resnap-lane-order-v31.patch"
)
CANONICAL_SIDECAR_V32_PATCH_PATH = (
    PATCH_DIR / "mapperatorinator-v32-canonical-sidecar-v32.patch"
)
MANIA_TERMINAL_BUDGET_V33_PATCH_PATH = (
    PATCH_DIR / "mapperatorinator-v32-terminal-budget-v33.patch"
)
MANIA_GROUP_FRAGMENT_V34_PATCH_PATH = (
    PATCH_DIR / "mapperatorinator-v32-group-fragment-v34.patch"
)
LEGACY_V2_MANIA_ORIGIN_PATCH_PATH = (
    PATCH_DIR / "mapperatorinator-v32-mania-origin-v2.patch"
)
LEGACY_V2_MANIA_RESNAP_PATCH_PATH = (
    PATCH_DIR / "mapperatorinator-v32-mania-resnap-v2.patch"
)
REQUIRED_PATCHES = (
    (KEYCOUNT_PATCH_ID, DEFAULT_PATCH_PATH),
    (OUTPUT_SAFETY_PATCH_ID, OUTPUT_SAFETY_PATCH_PATH),
    (MANIA_RESNAP_PATCH_ID, MANIA_RESNAP_PATCH_PATH),
    (MANIA_ORIGIN_PATCH_ID, MANIA_ORIGIN_PATCH_PATH),
    (MANIA_SIDECAR_PATCH_ID, MANIA_SIDECAR_PATCH_PATH),
    (MANIA_HOLD_STATE_PATCH_ID, MANIA_HOLD_STATE_PATCH_PATH),
    (MANIA_GRAMMAR_PATCH_ID, MANIA_GRAMMAR_PATCH_PATH),
    (MANIA_GRAMMAR_V3_PATCH_ID, MANIA_GRAMMAR_V3_PATCH_PATH),
    (MANIA_GRAMMAR_V4_PATCH_ID, MANIA_GRAMMAR_V4_PATCH_PATH),
    (MANIA_GRAMMAR_V5_PATCH_ID, MANIA_GRAMMAR_V5_PATCH_PATH),
    (MANIA_GRAMMAR_V6_PATCH_ID, MANIA_GRAMMAR_V6_PATCH_PATH),
    (MANIA_GRAMMAR_V7_PATCH_ID, MANIA_GRAMMAR_V7_PATCH_PATH),
    (MANIA_GRAMMAR_V8_PATCH_ID, MANIA_GRAMMAR_V8_PATCH_PATH),
    (MANIA_GRAMMAR_V9_PATCH_ID, MANIA_GRAMMAR_V9_PATCH_PATH),
    (MANIA_GRAMMAR_V10_PATCH_ID, MANIA_GRAMMAR_V10_PATCH_PATH),
    (MANIA_GRAMMAR_V11_PATCH_ID, MANIA_GRAMMAR_V11_PATCH_PATH),
    (MANIA_GRAMMAR_V12_PATCH_ID, MANIA_GRAMMAR_V12_PATCH_PATH),
    (MANIA_GRAMMAR_V13_PATCH_ID, MANIA_GRAMMAR_V13_PATCH_PATH),
    (MANIA_GRAMMAR_V14_PATCH_ID, MANIA_GRAMMAR_V14_PATCH_PATH),
    (MANIA_GRAMMAR_V15_PATCH_ID, MANIA_GRAMMAR_V15_PATCH_PATH),
    (MANIA_GRAMMAR_V16_PATCH_ID, MANIA_GRAMMAR_V16_PATCH_PATH),
    (MANIA_GRAMMAR_V17_PATCH_ID, MANIA_GRAMMAR_V17_PATCH_PATH),
    (MANIA_GRAMMAR_V18_PATCH_ID, MANIA_GRAMMAR_V18_PATCH_PATH),
    (MANIA_GRAMMAR_V19_PATCH_ID, MANIA_GRAMMAR_V19_PATCH_PATH),
    (MANIA_GRAMMAR_V20_PATCH_ID, MANIA_GRAMMAR_V20_PATCH_PATH),
    (MANIA_GRAMMAR_V21_PATCH_ID, MANIA_GRAMMAR_V21_PATCH_PATH),
    (MANIA_GRAMMAR_V22_PATCH_ID, MANIA_GRAMMAR_V22_PATCH_PATH),
    (MANIA_GRAMMAR_V23_PATCH_ID, MANIA_GRAMMAR_V23_PATCH_PATH),
    (MANIA_GRAMMAR_V24_PATCH_ID, MANIA_GRAMMAR_V24_PATCH_PATH),
    (MANIA_GRAMMAR_V25_PATCH_ID, MANIA_GRAMMAR_V25_PATCH_PATH),
    (MANIA_TAIL_REPAIR_V26_PATCH_ID, MANIA_TAIL_REPAIR_V26_PATCH_PATH),
    (RESIDENT_RUNTIME_V27_PATCH_ID, RESIDENT_RUNTIME_V27_PATCH_PATH),
    (
        MANIA_TEMPORAL_HORIZON_V28_PATCH_ID,
        MANIA_TEMPORAL_HORIZON_V28_PATCH_PATH,
    ),
    (
        MANIA_DECODER_TERMINATION_V29_PATCH_ID,
        MANIA_DECODER_TERMINATION_V29_PATCH_PATH,
    ),
    (
        GENERATION_TELEMETRY_V30_PATCH_ID,
        GENERATION_TELEMETRY_V30_PATCH_PATH,
    ),
    (
        RESNAP_LANE_ORDER_V31_PATCH_ID,
        RESNAP_LANE_ORDER_V31_PATCH_PATH,
    ),
    (
        CANONICAL_SIDECAR_V32_PATCH_ID,
        CANONICAL_SIDECAR_V32_PATCH_PATH,
    ),
    (
        MANIA_TERMINAL_BUDGET_V33_PATCH_ID,
        MANIA_TERMINAL_BUDGET_V33_PATCH_PATH,
    ),
    (
        MANIA_GROUP_FRAGMENT_V34_PATCH_ID,
        MANIA_GROUP_FRAGMENT_V34_PATCH_PATH,
    ),
)
LEGACY_V33_REQUIRED_PATCHES = REQUIRED_PATCHES[:-1]
LEGACY_V32_REQUIRED_PATCHES = LEGACY_V33_REQUIRED_PATCHES[:-1]
LEGACY_V31_REQUIRED_PATCHES = LEGACY_V32_REQUIRED_PATCHES[:-1]
LEGACY_V30_REQUIRED_PATCHES = LEGACY_V31_REQUIRED_PATCHES[:-1]
LEGACY_V29_REQUIRED_PATCHES = LEGACY_V30_REQUIRED_PATCHES[:-1]
LEGACY_V28_REQUIRED_PATCHES = LEGACY_V29_REQUIRED_PATCHES[:-1]
LEGACY_V27_REQUIRED_PATCHES = LEGACY_V28_REQUIRED_PATCHES[:-1]
LEGACY_V26_REQUIRED_PATCHES = LEGACY_V27_REQUIRED_PATCHES[:-1]
LEGACY_V25_REQUIRED_PATCHES = LEGACY_V26_REQUIRED_PATCHES[:-1]
LEGACY_V24_REQUIRED_PATCHES = LEGACY_V25_REQUIRED_PATCHES[:-1]
LEGACY_V23_REQUIRED_PATCHES = LEGACY_V24_REQUIRED_PATCHES[:-1]
LEGACY_V22_REQUIRED_PATCHES = LEGACY_V23_REQUIRED_PATCHES[:-1]
LEGACY_V21_REQUIRED_PATCHES = LEGACY_V22_REQUIRED_PATCHES[:-1]
LEGACY_V20_REQUIRED_PATCHES = LEGACY_V21_REQUIRED_PATCHES[:-1]
LEGACY_V19_REQUIRED_PATCHES = LEGACY_V20_REQUIRED_PATCHES[:-1]
LEGACY_V18_REQUIRED_PATCHES = LEGACY_V19_REQUIRED_PATCHES[:-1]
LEGACY_V17_REQUIRED_PATCHES = LEGACY_V18_REQUIRED_PATCHES[:-1]
LEGACY_V16_REQUIRED_PATCHES = LEGACY_V17_REQUIRED_PATCHES[:-1]
LEGACY_V15_REQUIRED_PATCHES = LEGACY_V16_REQUIRED_PATCHES[:-1]
LEGACY_V14_REQUIRED_PATCHES = LEGACY_V15_REQUIRED_PATCHES[:-1]
LEGACY_V13_REQUIRED_PATCHES = LEGACY_V14_REQUIRED_PATCHES[:-1]
LEGACY_V12_REQUIRED_PATCHES = LEGACY_V13_REQUIRED_PATCHES[:-1]
LEGACY_V11_REQUIRED_PATCHES = LEGACY_V12_REQUIRED_PATCHES[:-1]
LEGACY_V10_REQUIRED_PATCHES = LEGACY_V11_REQUIRED_PATCHES[:-1]
LEGACY_V9_REQUIRED_PATCHES = LEGACY_V10_REQUIRED_PATCHES[:-1]
LEGACY_V8_REQUIRED_PATCHES = LEGACY_V9_REQUIRED_PATCHES[:-1]
LEGACY_V7_REQUIRED_PATCHES = LEGACY_V8_REQUIRED_PATCHES[:-1]
LEGACY_V6_REQUIRED_PATCHES = LEGACY_V7_REQUIRED_PATCHES[:-1]
LEGACY_V5_REQUIRED_PATCHES = LEGACY_V6_REQUIRED_PATCHES[:-1]
LEGACY_V4_REQUIRED_PATCHES = LEGACY_V5_REQUIRED_PATCHES[:-1]
LEGACY_V3_REQUIRED_PATCHES = LEGACY_V4_REQUIRED_PATCHES[:-1]
LEGACY_ORIGIN_V2_REQUIRED_PATCHES = (
    *REQUIRED_PATCHES[:3],
    (
        "mania-origin-v2-hold-ir-column-grammar",
        LEGACY_V2_MANIA_ORIGIN_PATCH_PATH,
    ),
)
LEGACY_REQUIRED_PATCHES = (
    (KEYCOUNT_PATCH_ID, DEFAULT_PATCH_PATH),
    (OUTPUT_SAFETY_PATCH_ID, OUTPUT_SAFETY_PATCH_PATH),
    ("mania-event-times-v1", LEGACY_EVENT_TIMES_PATCH_PATH),
    ("mania-resnap-collisions-v1", LEGACY_RESNAP_COLLISIONS_PATCH_PATH),
)
LEGACY_V2_REQUIRED_PATCHES = (
    (KEYCOUNT_PATCH_ID, DEFAULT_PATCH_PATH),
    (OUTPUT_SAFETY_PATCH_ID, OUTPUT_SAFETY_PATCH_PATH),
    (
        f"{EVENT_TIMES_PATCH_ID}+mania-resnap-collisions-v2-preserve-raw",
        LEGACY_V2_MANIA_RESNAP_PATCH_PATH,
    ),
)
LEGACY_REQUIRED_PATCH_SETS = (
    LEGACY_V33_REQUIRED_PATCHES,
    LEGACY_V32_REQUIRED_PATCHES,
    LEGACY_V31_REQUIRED_PATCHES,
    LEGACY_V30_REQUIRED_PATCHES,
    LEGACY_V29_REQUIRED_PATCHES,
    LEGACY_V28_REQUIRED_PATCHES,
    LEGACY_V27_REQUIRED_PATCHES,
    LEGACY_V26_REQUIRED_PATCHES,
    LEGACY_V25_REQUIRED_PATCHES,
    LEGACY_V24_REQUIRED_PATCHES,
    LEGACY_V23_REQUIRED_PATCHES,
    LEGACY_V22_REQUIRED_PATCHES,
    LEGACY_V21_REQUIRED_PATCHES,
    LEGACY_V20_REQUIRED_PATCHES,
    LEGACY_V19_REQUIRED_PATCHES,
    LEGACY_V18_REQUIRED_PATCHES,
    LEGACY_V17_REQUIRED_PATCHES,
    LEGACY_V16_REQUIRED_PATCHES,
    LEGACY_V15_REQUIRED_PATCHES,
    LEGACY_V14_REQUIRED_PATCHES,
    LEGACY_V13_REQUIRED_PATCHES,
    LEGACY_V12_REQUIRED_PATCHES,
    LEGACY_V11_REQUIRED_PATCHES,
    LEGACY_V10_REQUIRED_PATCHES,
    LEGACY_V9_REQUIRED_PATCHES,
    LEGACY_V8_REQUIRED_PATCHES,
    LEGACY_V7_REQUIRED_PATCHES,
    LEGACY_V6_REQUIRED_PATCHES,
    LEGACY_V5_REQUIRED_PATCHES,
    LEGACY_V4_REQUIRED_PATCHES,
    LEGACY_ORIGIN_V2_REQUIRED_PATCHES,
    LEGACY_V3_REQUIRED_PATCHES,
    LEGACY_V2_REQUIRED_PATCHES,
    LEGACY_REQUIRED_PATCHES,
)

PatchStatus = Literal["APPLIED", "APPLICABLE"]
PatchSpec = tuple[str, Path]


class MapperatorinatorPatchError(RuntimeError):
    """The configured Mapperatorinator checkout cannot use the required patch."""


def _git(
    home: Path,
    *args: str,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=home,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )


def _patch_stack_matches_worktree(
    home: Path,
    patches: tuple[PatchSpec, ...],
) -> bool:
    """Compare the worktree with the complete layered patch result."""
    with tempfile.TemporaryDirectory(prefix="mapperatorinator-patch-index-") as temp_dir:
        expected_root = Path(temp_dir)
        touched_paths: set[PurePosixPath] = set()
        for patch_id, patch_path in patches:
            result = _git(
                home,
                "apply",
                "--numstat",
                "-z",
                str(Path(patch_path).resolve()),
            )
            if result.returncode != 0:
                raise MapperatorinatorPatchError(
                    f"could not inspect patch {patch_id}: {result.stderr.strip()}"
                )
            for entry in result.stdout.split("\0"):
                if not entry:
                    continue
                fields = entry.split("\t", 2)
                if len(fields) != 3:
                    raise MapperatorinatorPatchError(
                        f"could not parse paths from patch {patch_id}"
                    )
                relative = PurePosixPath(fields[2])
                if relative.is_absolute() or ".." in relative.parts:
                    raise MapperatorinatorPatchError(
                        f"patch {patch_id} contains an unsafe path: {relative}"
                    )
                touched_paths.add(relative)

        for relative in touched_paths:
            source = subprocess.run(
                [
                    "git",
                    "cat-file",
                    "--filters",
                    f"--path={relative.as_posix()}",
                    f"HEAD:{relative.as_posix()}",
                ],
                cwd=home,
                check=False,
                capture_output=True,
            )
            if source.returncode == 0:
                destination = expected_root.joinpath(*relative.parts)
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(source.stdout)

        for patch_id, patch_path in patches:
            isolated_env = os.environ.copy()
            isolated_env["GIT_CEILING_DIRECTORIES"] = str(expected_root.parent)
            result = subprocess.run(
                ["git", "apply", "--no-index", str(Path(patch_path).resolve())],
                cwd=expected_root,
                env=isolated_env,
                check=False,
                capture_output=True,
                text=True,
                encoding="utf-8",
            )
            if result.returncode != 0:
                raise MapperatorinatorPatchError(
                    f"patch stack is invalid at {patch_id}: {result.stderr.strip()}"
                )

        def normalized_bytes(path: Path) -> bytes:
            # Git patches are line-oriented. Windows checkouts can contain a
            # mixture of CRLF base lines and LF inserted lines even when their
            # parsed source is identical. Ignore only that representation;
            # every other byte must still match.
            return path.read_bytes().replace(b"\r\n", b"\n")

        return all(
            expected_root.joinpath(*relative.parts).is_file()
            and home.joinpath(*relative.parts).is_file()
            and normalized_bytes(expected_root.joinpath(*relative.parts))
            == normalized_bytes(home.joinpath(*relative.parts))
            for relative in touched_paths
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
    home = Path(home).resolve()
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
    if _patch_stack_matches_worktree(home, patches):
        return dict.fromkeys((patch_id for patch_id, _ in patches), "APPLIED")
    return {
        patch_id: patch_status(home, patch_path, expected_head)
        for patch_id, patch_path in patches
    }


def apply_required_mapperatorinator_patches(
    home: Path,
    *,
    patches: tuple[PatchSpec, ...] = REQUIRED_PATCHES,
    legacy_patches: tuple[PatchSpec, ...] | None = None,
    legacy_patch_sets: tuple[tuple[PatchSpec, ...], ...] | None = None,
    expected_head: str = EXPECTED_MAPPERATORINATOR_HEAD,
) -> None:
    """Apply every patch, migrating an exact legacy stack when configured."""
    home = Path(home).resolve()
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
    if _patch_stack_matches_worktree(home, patches):
        return
    if legacy_patches is not None and legacy_patch_sets is not None:
        raise ValueError("provide legacy_patches or legacy_patch_sets, not both")
    if legacy_patch_sets is None:
        if legacy_patches is not None:
            legacy_patch_sets = (legacy_patches,)
        elif patches == REQUIRED_PATCHES:
            legacy_patch_sets = LEGACY_REQUIRED_PATCH_SETS
        else:
            legacy_patch_sets = ()
    matching_legacy_patches = next(
        (
            candidate
            for candidate in legacy_patch_sets
            if candidate and _patch_stack_matches_worktree(home, candidate)
        ),
        None,
    )
    apply_start_index = 0
    if matching_legacy_patches is not None:
        shared_prefix_length = 0
        for legacy_patch, replacement_patch in zip(
            matching_legacy_patches,
            patches,
        ):
            if legacy_patch != replacement_patch:
                break
            shared_prefix_length += 1
        for patch_id, patch_path in reversed(
            matching_legacy_patches[shared_prefix_length:]
        ):
            result = _git(home, "apply", "--reverse", str(Path(patch_path).resolve()))
            if result.returncode != 0:
                raise MapperatorinatorPatchError(
                    f"failed to remove legacy Mapperatorinator patch {patch_id}: "
                    f"{result.stderr.strip()}"
                )
        apply_start_index = shared_prefix_length
    for _, patch_path in patches[apply_start_index:]:
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
