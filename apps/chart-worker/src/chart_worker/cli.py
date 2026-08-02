"""로컬 chart-worker 명령행."""

import json
from enum import Enum
from pathlib import Path
from typing import Annotated

import typer

from chart_worker.bench import run_benchmark
from chart_worker.errors import WorkerError
from chart_worker.pipeline import PipelineOptions, run_pipeline
from chart_worker.reprocess import PostprocessOptions, run_postprocess_only

app = typer.Typer(no_args_is_help=True)


class GeneratorOption(str, Enum):
    FAKE = "fake"
    MAPPERATORINATOR = "mapperatorinator"


@app.callback()
def main() -> None:
    """리듬 게임 채보 생성 워커."""


def _worker_error_payload(error: WorkerError) -> dict[str, object]:
    prefix = f"{error.code.value}: "
    message = str(error).removeprefix(prefix)
    return {
        "code": error.code.value,
        "message": message,
        "context": error.context,
    }


@app.command()
def generate(
    source: Annotated[
        Path,
        typer.Argument(exists=True, file_okay=True, dir_okay=False, readable=True),
    ],
    out: Annotated[Path, typer.Option("--out", "-o")],
    title: Annotated[str | None, typer.Option("--title")] = None,
    generator: Annotated[GeneratorOption, typer.Option("--generator")] = GeneratorOption.FAKE,
    seed: Annotated[int, typer.Option("--seed")] = 0,
    keysounds: Annotated[
        bool,
        typer.Option("--keysounds/--no-keysounds"),
    ] = False,
    overwrite: Annotated[bool, typer.Option("--overwrite")] = False,
    reference_onsets: Annotated[
        Path | None,
        typer.Option(
            "--reference-onsets",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
        ),
    ] = None,
) -> None:
    """SOURCE에서 로컬 플레이테스트용 12개 채보를 생성한다."""
    try:
        result = run_pipeline(
            PipelineOptions(
                source=source,
                output_dir=out,
                title=title or source.stem,
                generator=generator.value,
                keysounds=keysounds,
                seed=seed,
                overwrite=overwrite,
                reference_onsets_path=reference_onsets,
            )
        )
    except WorkerError as error:
        typer.echo(
            json.dumps(_worker_error_payload(error), ensure_ascii=False),
            err=True,
        )
        raise typer.Exit(1) from error
    except ValueError as error:
        raise typer.BadParameter(str(error)) from error

    typer.echo(str(result.manifest_path))


@app.command()
def postprocess(
    input_dir: Annotated[
        Path,
        typer.Argument(exists=True, file_okay=False, dir_okay=True, readable=True),
    ],
    out: Annotated[Path, typer.Option("--out", "-o")],
    keysounds: Annotated[
        bool,
        typer.Option("--keysounds/--no-keysounds"),
    ] = False,
) -> None:
    """기존 실행의 분석값과 raw osu로 S4를 다시 실행한다.

    --keysounds를 주고 입력에 스템이 없으면 S3도 같이 돌린다.
    """
    try:
        result = run_postprocess_only(
            PostprocessOptions(input_dir=input_dir, output_dir=out, keysounds=keysounds)
        )
    except WorkerError as error:
        typer.echo(
            json.dumps(_worker_error_payload(error), ensure_ascii=False),
            err=True,
        )
        raise typer.Exit(1) from error
    except ValueError as error:
        raise typer.BadParameter(str(error)) from error
    typer.echo(str(result.manifest_path))


@app.command()
def bench(
    source: Annotated[
        Path,
        typer.Argument(exists=True, file_okay=True, dir_okay=False, readable=True),
    ],
    out: Annotated[Path, typer.Option("--out", "-o")],
    title: Annotated[str | None, typer.Option("--title")] = None,
    generator: Annotated[GeneratorOption, typer.Option("--generator")] = GeneratorOption.FAKE,
    seed: Annotated[int, typer.Option("--seed")] = 0,
    keysounds: Annotated[
        bool,
        typer.Option("--keysounds/--no-keysounds"),
    ] = False,
    reference_onsets: Annotated[
        Path | None,
        typer.Option(
            "--reference-onsets",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
        ),
    ] = None,
) -> None:
    """12개 채보를 생성하고 benchmark-report.json을 기록한다."""
    try:
        result = run_benchmark(
            PipelineOptions(
                source=source,
                output_dir=out,
                title=title or source.stem,
                generator=generator.value,
                keysounds=keysounds,
                seed=seed,
                reference_onsets_path=reference_onsets,
            )
        )
    except WorkerError as error:
        typer.echo(
            json.dumps(_worker_error_payload(error), ensure_ascii=False),
            err=True,
        )
        raise typer.Exit(1) from error
    except ValueError as error:
        raise typer.BadParameter(str(error)) from error
    typer.echo(str(result.report_path))


if __name__ == "__main__":  # pragma: no cover
    app()
