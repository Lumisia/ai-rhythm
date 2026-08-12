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
    assert json.loads(result.stderr) == {
        "code": "CHART_GENERATION_FAILED",
        "message": "generator stopped",
        "context": {"key_mode": 4},
    }


def test_bench_cli_is_exposed():
    result = runner.invoke(app, ["bench", "--help"])
    assert result.exit_code == 0
    assert "--generator" in result.output


def test_generate_cli_defaults_to_mapperatorinator(monkeypatch, tmp_path: Path):
    source = tmp_path / "fixture.wav"
    source.write_bytes(b"source")
    captured = {}

    def succeed(options):
        captured["options"] = options
        return SimpleNamespace(manifest_path=tmp_path / "manifest.json")

    monkeypatch.setattr("chart_worker.cli.run_pipeline", succeed)
    result = runner.invoke(
        app, ["generate", str(source), "--out", str(tmp_path / "run")]
    )

    assert result.exit_code == 0
    assert captured["options"].generator == "mapperatorinator"


def test_generate_cli_does_not_offer_destructive_overwrite():
    result = runner.invoke(app, ["generate", "--help"])
    assert result.exit_code == 0
    assert "--overwrite" not in result.output


def test_bench_cli_can_select_the_gpu_free_fake_generator(monkeypatch, tmp_path: Path):
    source = tmp_path / "fixture.wav"
    source.write_bytes(b"source")
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
            "--generator",
            "fake",
        ],
    )

    assert result.exit_code == 0
    assert captured["options"].generator == "fake"


def test_removed_postprocess_command_is_not_exposed():
    result = runner.invoke(app, ["postprocess", "run", "--out", "output"])
    assert result.exit_code == 2


def test_recalculate_difficulty_cli_writes_shadow_report(monkeypatch, tmp_path: Path):
    batch = tmp_path / "batch"
    batch.mkdir()
    output = tmp_path / "shadow.json"
    captured = {}

    def succeed(batch_dir, output_path):
        captured["batch_dir"] = batch_dir
        captured["output_path"] = output_path
        return SimpleNamespace(chart_count=12)

    monkeypatch.setattr("chart_worker.cli.recalculate_batch", succeed)
    result = runner.invoke(
        app,
        ["recalculate-difficulty", str(batch), "--out", str(output)],
    )

    assert result.exit_code == 0
    assert captured == {"batch_dir": batch, "output_path": output}
    assert str(output) in result.output


def test_migrate_boundary_review_cli_prints_summary_path(monkeypatch, tmp_path: Path):
    source = tmp_path / "source-batch"
    source.mkdir()
    target = tmp_path / "boundary-label-v2-review"
    captured = {}

    def succeed(source_batch, target_root, *, migrated_at):
        captured["source_batch"] = source_batch
        captured["target_root"] = target_root
        captured["migrated_at"] = migrated_at
        return SimpleNamespace(target_root=target_root, song_count=33)

    monkeypatch.setattr("chart_worker.cli.migrate_boundary_review", succeed)
    result = runner.invoke(
        app,
        ["migrate-boundary-review", str(source), "--out", str(target)],
    )

    assert result.exit_code == 0
    assert captured["source_batch"] == source
    assert captured["target_root"] == target
    assert captured["migrated_at"].tzinfo is not None
    assert str(target / "migration-summary.json") in result.output
