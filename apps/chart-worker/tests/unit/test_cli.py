import json
from pathlib import Path
from types import SimpleNamespace

from typer.testing import CliRunner

from chart_worker.cli import app
from chart_worker.errors import ErrorCode, WorkerError

runner = CliRunner()


def test_generate_cli_requires_existing_source(tmp_path: Path):
    result = runner.invoke(
        app,
        ["generate", str(tmp_path / "missing.wav"), "--out", str(tmp_path / "run")],
    )
    assert result.exit_code == 2
    assert "does not exist" in result.output


def test_generate_cli_writes_worker_error_as_json(monkeypatch, tmp_path: Path):
    source = tmp_path / "fixture.wav"
    source.write_bytes(b"source")

    def fail(*args, **kwargs):
        del args, kwargs
        raise WorkerError(
            ErrorCode.CHART_GENERATION_FAILED,
            "generator stopped",
            context={"key_mode": 4},
        )

    monkeypatch.setattr("chart_worker.cli.run_pipeline", fail)
    result = runner.invoke(
        app,
        ["generate", str(source), "--out", str(tmp_path / "run")],
    )

    assert result.exit_code == 1
    payload = json.loads(result.stderr)
    assert payload == {
        "code": "CHART_GENERATION_FAILED",
        "message": "generator stopped",
        "context": {"key_mode": 4},
    }


def test_postprocess_cli_rejects_the_input_directory_as_output(tmp_path: Path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    result = runner.invoke(
        app,
        ["postprocess", str(run_dir), "--out", str(run_dir)],
    )
    assert result.exit_code == 2
    assert "must be different" in result.output


def test_bench_cli_is_exposed():
    result = runner.invoke(app, ["bench", "--help"])
    assert result.exit_code == 0
    assert "--generator" in result.output


def test_generate_cli_passes_human_reference_onsets_to_pipeline(monkeypatch, tmp_path: Path):
    source = tmp_path / "fixture.wav"
    source.write_bytes(b"source")
    reference = tmp_path / "reference.json"
    reference.write_text("{}", encoding="utf-8")
    captured = {}

    def succeed(options):
        captured["options"] = options
        return SimpleNamespace(manifest_path=tmp_path / "manifest.json")

    monkeypatch.setattr("chart_worker.cli.run_pipeline", succeed)
    result = runner.invoke(
        app,
        [
            "generate",
            str(source),
            "--out",
            str(tmp_path / "run"),
            "--reference-onsets",
            str(reference),
        ],
    )

    assert result.exit_code == 0
    assert captured["options"].reference_onsets_path == reference


def test_bench_cli_passes_human_reference_onsets_to_pipeline(monkeypatch, tmp_path: Path):
    source = tmp_path / "fixture.wav"
    source.write_bytes(b"source")
    reference = tmp_path / "reference.json"
    reference.write_text("{}", encoding="utf-8")
    captured = {}

    def succeed(options):
        captured["options"] = options
        return SimpleNamespace(report_path=tmp_path / "benchmark-report.json")

    monkeypatch.setattr("chart_worker.cli.run_benchmark", succeed)
    result = runner.invoke(
        app,
        [
            "bench",
            str(source),
            "--out",
            str(tmp_path / "run"),
            "--reference-onsets",
            str(reference),
        ],
    )

    assert result.exit_code == 0
    assert captured["options"].reference_onsets_path == reference
