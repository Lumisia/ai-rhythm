import json
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

import pytest
from pydantic import ValidationError

from chart_worker.schema.export import export_schemas
from chart_worker.schema.playtest_run import (
    AudioFileRef,
    PlaytestRunManifest,
    RunAudioRefs,
    RunChartRef,
)
from chart_worker.schema.types import DIFFICULTIES, KEY_MODES

SHA = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"


def _charts() -> list[RunChartRef]:
    return [
        RunChartRef(
            key_mode=key_mode,
            difficulty=difficulty,
            path=f"charts/{key_mode}k-{difficulty.lower()}.chart.json",
            sha256=SHA,
        )
        for key_mode in KEY_MODES
        for difficulty in DIFFICULTIES
    ]


def _manifest(**overrides) -> PlaytestRunManifest:
    values = {
        "run_id": UUID(int=1),
        "title": "테스트 곡",
        "generated_at": datetime(2026, 8, 2, tzinfo=UTC),
        "worker_version": "test-build",
        "audio": RunAudioRefs(game=AudioFileRef(path="audio/game.flac", sha256=SHA)),
        "charts": _charts(),
        "generation_report_path": "generation-report.json",
    }
    return PlaytestRunManifest(**(values | overrides))


@pytest.mark.parametrize(
    "path",
    ["../outside.flac", "/absolute.flac", "C:/outside.flac", "audio/../../outside.flac"],
)
def test_run_manifest_rejects_unsafe_paths(path):
    with pytest.raises(ValidationError, match="safe relative path"):
        AudioFileRef(path=path, sha256=SHA)


def test_run_manifest_normalizes_windows_separators():
    reference = AudioFileRef(path=r"audio\game.flac", sha256=SHA)
    assert reference.path == "audio/game.flac"


def test_run_manifest_requires_exactly_twelve_unique_chart_combinations():
    with pytest.raises(ValidationError, match="12 chart combinations"):
        _manifest(charts=_charts()[:-1])
    duplicate = [*_charts()[:-1], _charts()[0]]
    with pytest.raises(ValidationError, match="duplicate chart combination"):
        _manifest(charts=duplicate)


def test_run_manifest_requires_all_keysound_references_together():
    audio = RunAudioRefs(
        game=AudioFileRef(path="audio/game.flac", sha256=SHA),
        no_drums=AudioFileRef(path="audio/no_drums.flac", sha256=SHA),
        keys=AudioFileRef(path="audio/drums.flac", sha256=SHA),
    )
    with pytest.raises(ValidationError, match="keysound references"):
        _manifest(audio=audio)


def test_run_manifest_json_uses_camel_case_and_round_trips():
    manifest = _manifest()
    payload = manifest.model_dump_json(by_alias=True)
    assert '"runId"' in payload
    assert '"keyMode"' in payload
    assert PlaytestRunManifest.model_validate_json(payload) == manifest


def test_export_schemas_writes_three_parseable_contracts(tmp_path: Path):
    paths = export_schemas(tmp_path)
    assert {path.name for path in paths} == {
        "chart-v1.schema.json",
        "keysound-manifest-v1.schema.json",
        "playtest-run-v1.schema.json",
    }
    schemas = {path.name: json.loads(path.read_text(encoding="utf-8")) for path in paths}
    assert "timeMs" in schemas["chart-v1.schema.json"]["$defs"]["ChartNote"]["properties"]
    assert "drumOnsets" in schemas["keysound-manifest-v1.schema.json"]["properties"]
    assert "runId" in schemas["playtest-run-v1.schema.json"]["properties"]
