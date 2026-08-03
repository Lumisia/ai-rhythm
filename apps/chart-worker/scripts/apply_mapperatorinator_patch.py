"""Apply the pinned Mapperatorinator compatibility patch to a local checkout."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from chart_worker.generation.mapperatorinator_patch import (
    CONSTRAINT_PATCH_ID,
    DEFAULT_PATCH_PATH,
    EXPECTED_MAPPERATORINATOR_HEAD,
    apply_mapperatorinator_patch,
    patch_status,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("home", type=Path)
    args = parser.parse_args()
    apply_mapperatorinator_patch(args.home)
    status = patch_status(
        args.home,
        DEFAULT_PATCH_PATH,
        EXPECTED_MAPPERATORINATOR_HEAD,
    )
    print(f"{CONSTRAINT_PATCH_ID} {status}")


if __name__ == "__main__":
    main()
