"""로컬 chart-worker 명령줄 인터페이스."""

import json
import os
import sys
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Annotated

import click
import typer

from chart_worker.analysis.difficulty_shadow import recalculate_batch
from chart_worker.bench import run_benchmark
from chart_worker.boundary_review_migration import migrate_boundary_review
from chart_worker.errors import WorkerError
from chart_worker.pipeline import PipelineOptions, run_pipeline
from chart_worker.validation.family_replay_v3 import replay_generation_report_v3
from chart_worker.validation.pairwise_export import export_pairwise_task_bundle_v1

# Rich error panels contain box-drawing characters that Windows redirected
# CP949 streams cannot represent reliably.  Plain Click/Typer formatting keeps
# parser errors portable while command output uses the explicit UTF-8 helper
# below.
app = typer.Typer(no_args_is_help=True, rich_markup_mode=None)


def _echo_cli_text(message: str, *, err: bool = False) -> None:
    """Write CLI output through an explicit UTF-8 redirect-safe stream."""

    stream = sys.stderr if err else sys.stdout
    binary_stream = getattr(stream, "buffer", None)
    if binary_stream is not None and not stream.isatty():
        payload = f"{message}{os.linesep}".encode("utf-8", errors="backslashreplace")
        click.echo(payload, file=binary_stream, nl=False)
        return
    click.echo(message, file=stream)


class GeneratorOption(str, Enum):
    FAKE = "fake"
    MAPPERATORINATOR = "mapperatorinator"


@app.callback()
def main() -> None:
    """리듬 게임 채보를 직접 생성한다."""


def _worker_error_payload(error: WorkerError) -> dict[str, object]:
    prefix = f"{error.code.value}: "
    message = str(error).removeprefix(prefix)
    return {
        "code": error.code.value,
        "message": message,
        "context": error.context,
    }


def _run_or_exit(options: PipelineOptions) -> Path:
    try:
        return run_pipeline(options).manifest_path
    except WorkerError as error:
        _echo_cli_text(
            json.dumps(_worker_error_payload(error), ensure_ascii=False),
            err=True,
        )
        raise typer.Exit(1) from error
    except ValueError as error:
        raise typer.BadParameter(str(error)) from error


@app.command()
def generate(
    source: Annotated[
        Path,
        typer.Argument(exists=True, file_okay=True, dir_okay=False, readable=True),
    ],
    out: Annotated[Path, typer.Option("--out", "-o")],
    title: Annotated[str | None, typer.Option("--title")] = None,
    generator: Annotated[
        GeneratorOption, typer.Option("--generator")
    ] = GeneratorOption.MAPPERATORINATOR,
    seed: Annotated[int, typer.Option("--seed")] = 0,
) -> None:
    """SOURCE에서 4K·6K·7K의 난이도별 채보 12개를 생성한다."""
    manifest_path = _run_or_exit(
        PipelineOptions(
            source=source,
            output_dir=out,
            title=title or source.stem,
            generator=generator.value,
            seed=seed,
        )
    )
    _echo_cli_text(str(manifest_path))


@app.command()
def bench(
    source: Annotated[
        Path,
        typer.Argument(exists=True, file_okay=True, dir_okay=False, readable=True),
    ],
    out: Annotated[Path, typer.Option("--out", "-o")],
    title: Annotated[str | None, typer.Option("--title")] = None,
    generator: Annotated[
        GeneratorOption, typer.Option("--generator")
    ] = GeneratorOption.MAPPERATORINATOR,
    seed: Annotated[int, typer.Option("--seed")] = 0,
) -> None:
    """12개 채보와 생성 시간·구조 진단 보고서를 기록한다."""
    try:
        result = run_benchmark(
            PipelineOptions(
                source=source,
                output_dir=out,
                title=title or source.stem,
                generator=generator.value,
                seed=seed,
            )
        )
    except WorkerError as error:
        _echo_cli_text(
            json.dumps(_worker_error_payload(error), ensure_ascii=False),
            err=True,
        )
        raise typer.Exit(1) from error
    except ValueError as error:
        raise typer.BadParameter(str(error)) from error
    _echo_cli_text(str(result.report_path))


@app.command("recalculate-difficulty")
def recalculate_difficulty(
    batch_dir: Annotated[
        Path,
        typer.Argument(exists=True, file_okay=False, dir_okay=True, readable=True),
    ],
    out: Annotated[Path, typer.Option("--out", "-o")],
) -> None:
    """Recalculate DifficultyVector v2 from archived chart JSON files."""
    try:
        recalculate_batch(batch_dir, out)
    except ValueError as error:
        raise typer.BadParameter(str(error)) from error
    _echo_cli_text(str(out))


@app.command("migrate-boundary-review")
def migrate_boundary_review_command(
    source_batch: Annotated[
        Path,
        typer.Argument(exists=True, file_okay=False, dir_okay=True, readable=True),
    ],
    out: Annotated[Path, typer.Option("--out", "-o")],
) -> None:
    """Package a verified legacy batch for local human boundary review."""
    try:
        summary = migrate_boundary_review(
            source_batch,
            out,
            migrated_at=datetime.now(UTC),
        )
    except (TypeError, ValueError) as error:
        raise typer.BadParameter(str(error)) from error
    _echo_cli_text(str(summary.target_root / "migration-summary.json"))


@app.command("replay-family-v3")
def replay_family_v3(
    report: Annotated[
        Path,
        typer.Argument(exists=True, file_okay=True, dir_okay=False, readable=True),
    ],
) -> None:
    """Verify and replay one archived report without model calls or file mutation."""
    try:
        replay = replay_generation_report_v3(report)
    except (OSError, TypeError, ValueError) as error:
        raise typer.BadParameter(str(error)) from error
    _echo_cli_text(json.dumps(replay, ensure_ascii=False, sort_keys=True))


@app.command("export-pairwise-v3")
def export_pairwise_v3(
    source: Annotated[
        Path,
        typer.Argument(exists=True, file_okay=True, dir_okay=False, readable=True),
    ],
    out: Annotated[Path, typer.Option("--out", "-o")],
) -> None:
    """Export hash-bound private and blinded human-review task bundles."""
    try:
        terminal_path = export_pairwise_task_bundle_v1(source, out)
    except (OSError, TypeError, ValueError) as error:
        raise typer.BadParameter(str(error)) from error
    _echo_cli_text(str(terminal_path))


if __name__ == "__main__":  # pragma: no cover
    app()
