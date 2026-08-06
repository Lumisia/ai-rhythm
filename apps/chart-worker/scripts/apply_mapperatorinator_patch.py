"""Apply the pinned Mapperatorinator compatibility patches to a local checkout."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from chart_worker.generation.mapperatorinator_patch import (
    apply_required_mapperatorinator_patches,
    required_patch_statuses,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("home", type=Path)
    args = parser.parse_args()
    apply_required_mapperatorinator_patches(args.home)
    for patch_id, status in required_patch_statuses(args.home).items():
        print(f"{patch_id} {status}")


if __name__ == "__main__":
    main()
