from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path


def _load_script(name: str):
    script_path = Path(__file__).parents[2] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_outro_audit_consumes_v3_manifest(tmp_path: Path) -> None:
    song_root = tmp_path / "songs" / "01"
    audio_path = song_root / "audio" / "game.flac"
    audio_path.parent.mkdir(parents=True)
    audio_path.write_bytes(b"fixture audio")
    chart_path = song_root / "charts" / "4k-easy.chart.json"
    _write_json(
        chart_path,
        {
            "notes": [{"timeMs": 1_000, "durationMs": None}],
        },
    )
    _write_json(
        song_root / "playtest-run-v3.json",
        {
            "version": 3,
            "title": "fixture",
            "audio": {
                "game": {
                    "path": "audio/game.flac",
                    "sha256": _sha256(audio_path),
                }
            },
            "charts": [
                {
                    "path": "charts/4k-easy.chart.json",
                    "sha256": _sha256(chart_path),
                    "keyMode": 4,
                    "difficulty": "EASY",
                }
            ],
        },
    )
    batch_state_path = tmp_path / "batch-state.json"
    _write_json(
        batch_state_path,
        {
            "status": "COMPLETE",
            "songs": [
                {
                    "index": 1,
                    "sourceName": "fixture.wav",
                    "outputPath": str(song_root),
                }
            ],
        },
    )

    report = _load_script("audit_outro_family").audit_batch(batch_state_path)

    assert report["hashMismatchCount"] == 0
    assert report["songs"][0]["runManifest"] == "playtest-run-v3.json"


def test_rescore_diagnostic_manifest_has_no_chart_level_authority() -> None:
    module = _load_script("rescore_diagnostic_fallback")

    manifest = module._diagnostic_manifest(
        exports=[{"path": "diagnostic/map.osu"}],
        failures=[],
    )

    assert manifest == {
        "version": module.DIAGNOSTIC_FALLBACK_VERSION,
        "decision": "PLAYTEST_ONLY",
        "modelInvocations": 0,
        "entries": [{"path": "diagnostic/map.osu"}],
        "failures": [],
        "rescoreReport": module.RESCORE_REPORT_NAME,
    }
    assert "productionEligible" not in manifest
