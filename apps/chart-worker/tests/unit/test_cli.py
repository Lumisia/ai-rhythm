import json
from pathlib import Path

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
