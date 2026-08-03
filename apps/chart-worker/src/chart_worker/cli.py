"""로컬 chart-worker 명령줄 인터페이스."""

import json
from enum import Enum
from pathlib import Path
from typing import Annotated

import typer

from chart_worker.bench import run_benchmark
from chart_worker.errors import WorkerError
from chart_worker.pipeline import PipelineOptions, run_pipeline

app = typer.Typer(no_args_is_help=True)


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
        typer.echo(json.dumps(_worker_error_payload(error), ensure_ascii=False), err=True)
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
    typer.echo(str(manifest_path))


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
        typer.echo(json.dumps(_worker_error_payload(error), ensure_ascii=False), err=True)
        raise typer.Exit(1) from error
    except ValueError as error:
        raise typer.BadParameter(str(error)) from error
    typer.echo(str(result.report_path))


if __name__ == "__main__":  # pragma: no cover
    app()
