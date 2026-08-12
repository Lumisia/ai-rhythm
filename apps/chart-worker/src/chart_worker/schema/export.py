"""Pydantic 계약을 프론트가 읽을 JSON Schema로 내보낸다."""

import json
from pathlib import Path
from typing import Any

from chart_worker.schema.boundary_label import BoundaryLabelV1, BoundaryLabelV2
from chart_worker.schema.chart import ChartDocument
from chart_worker.schema.keysound import KeysoundManifest
from chart_worker.schema.playtest_run import PlaytestRunManifest, PlaytestRunManifestV2


def schemas() -> dict[str, dict[str, Any]]:
    return {
        "boundary-label-v1.schema.json": BoundaryLabelV1.model_json_schema(by_alias=True),
        "boundary-label-v2.schema.json": BoundaryLabelV2.model_json_schema(by_alias=True),
        "chart-v1.schema.json": ChartDocument.model_json_schema(by_alias=True),
        "keysound-manifest-v1.schema.json": KeysoundManifest.model_json_schema(by_alias=True),
        "playtest-run-v1.schema.json": PlaytestRunManifest.model_json_schema(by_alias=True),
        "playtest-run-v2.schema.json": PlaytestRunManifestV2.model_json_schema(by_alias=True),
    }


def export_schemas(target: Path) -> tuple[Path, ...]:
    target.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, schema in schemas().items():
        path = target / name
        path.write_text(
            json.dumps(schema, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        paths.append(path)
    return tuple(paths)
