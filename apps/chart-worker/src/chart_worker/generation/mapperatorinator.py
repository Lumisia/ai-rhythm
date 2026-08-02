"""Mapperatorinator subprocess 어댑터.

명령행 조립은 순수 함수로 떼어 GPU 없이 검증한다. ffmpeg 어댑터와
같은 구조다.
"""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from chart_worker.audio.runner import CommandError, CommandResult, CommandRunner
from chart_worker.config import WorkerConfig
from chart_worker.errors import ErrorCode, WorkerError
from chart_worker.generation.osu_parser import parse_osu_file
from chart_worker.generation.params import (
    GAMEMODE_MANIA,
    MANIA_COLUMN_TEMPERATURE,
    PRECISION,
    GenerationRequest,
)
from chart_worker.schema.note import Chart

INFERENCE_SCRIPT = "inference.py"
CONFIG_NAME = "v32"
RunCommand = Callable[[list[str]], CommandResult]


@dataclass(frozen=True, slots=True)
class GeneratedChart:
    notes: Chart
    key_mode: int
    osu_text: str
    generator_name: str
    seed: int | None


class ChartGenerator(Protocol):
    def __call__(self, request: GenerationRequest, workdir: Path) -> GeneratedChart: ...


def _hydra_list(values: tuple[str, ...]) -> str:
    """Hydra 목록 인자. 값에 콤마가 없으므로 따옴표 없이 적는다."""
    for value in values:
        if "," in value or "[" in value or "]" in value:
            raise ValueError(f"descriptor cannot contain list syntax: {value!r}")
    return "[" + ",".join(values) + "]"


def _absolute(path: Path) -> Path:
    """실행 위치에 흔들리지 않는 경로로 만든다.

    inference.py 를 체크아웃 안에서 돌리므로 cwd 가 우리 작업 디렉터리가
    아니다. 상대 경로를 그대로 넘기면 Mapperatorinator 홈 아래로 해석되어
    `storage/game.flac` 이 `<home>/storage/game.flac` 이 된다.

    구분자가 없는 이름은 건드리지 않는다. PATH 에서 찾는 실행 파일이다.
    """
    if path.parent == Path():
        return path
    return path.resolve()


def build_command(
    config: WorkerConfig,
    request: GenerationRequest,
    output_dir: Path,
) -> list[str]:
    """Hydra 인자를 조립한다.

    precision 을 항상 명시한다. v32.yaml 기본값이 bf16 인데 Turing(sm_75)은
    bf16 을 지원하지 않는다. 설정 파일 기본값에 기대면 GPU 를 바꿀 때
    조용히 죽는다.
    """
    if config.mapperatorinator_python is None or config.mapperatorinator_home is None:
        raise ValueError("mapperatorinator_python and mapperatorinator_home must be configured")

    argv = [
        str(_absolute(config.mapperatorinator_python)),
        INFERENCE_SCRIPT,
        "-cn",
        CONFIG_NAME,
        f"audio_path={_absolute(request.audio_path)}",
        f"output_path={_absolute(output_dir)}",
        f"gamemode={GAMEMODE_MANIA}",
        f"keycount={request.key_mode}",
        f"difficulty={request.requested_star}",
        f"year={request.year}",
        f"hold_note_ratio={request.hold_note_ratio}",
        f"cfg_scale={request.cfg_scale}",
        f"mania_column_temperature={MANIA_COLUMN_TEMPERATURE}",
        f"descriptors={_hydra_list(request.descriptors)}",
        f"negative_descriptors={_hydra_list(request.negative_descriptors)}",
        f"precision={config.mapperatorinator_precision or PRECISION}",
        "export_osz=false",
        "resnap_events=true",
    ]
    if request.seed is not None:
        argv.append(f"seed={request.seed}")
    if request.timing_osu_path is not None:
        # 참조 .osu 없이 in_context=[TIMING] 을 켜면 줄 타이밍이 없다.
        argv.append(f"beatmap_path={_absolute(request.timing_osu_path)}")
        argv.append("in_context=[TIMING]")
    return argv


def _require_clean_output_dir(output_dir: Path) -> None:
    """이전 실행의 .osu 를 이번 실행 결과로 오인하지 않게 한다."""
    existing = sorted(output_dir.rglob("*.osu"))
    if existing:
        raise WorkerError(
            ErrorCode.CHART_GENERATION_FAILED,
            "output directory already contains .osu files",
            context={"files": [str(path) for path in existing]},
        )


def find_generated_osu(output_dir: Path) -> Path:
    """생성된 .osu 를 찾는다. 정확히 하나여야 한다."""
    produced = sorted(output_dir.rglob("*.osu"))
    if not produced:
        raise WorkerError(
            ErrorCode.CHART_GENERATION_FAILED,
            "no .osu file was produced",
            context={"output_dir": str(output_dir)},
        )
    if len(produced) > 1:
        # 어느 것이 이번 실행의 산출물인지 알 수 없다. 조용히 하나를 고르면
        # 이전 실행 결과를 채보로 내보내게 된다.
        raise WorkerError(
            ErrorCode.CHART_GENERATION_FAILED,
            f"expected exactly one .osu file, found {len(produced)}",
            context={"files": [str(path) for path in produced]},
        )
    return produced[0]


@dataclass(frozen=True, slots=True)
class MapperatorinatorGenerator:
    config: WorkerConfig
    run: RunCommand | None = None

    def __call__(self, request: GenerationRequest, workdir: Path) -> GeneratedChart:
        workdir.mkdir(parents=True, exist_ok=True)
        _require_clean_output_dir(workdir)
        run = self.run or CommandRunner(
            shared_bin_dir=self.config.ffmpeg_shared_bin_dir,
            timeout_sec=1800.0,
            # inference.py 는 상대 경로이고 Hydra 는 실행 위치에서 configs/ 를
            # 찾는다. 체크아웃 밖에서 돌리면 스크립트도 설정도 못 찾는다.
            cwd=self.config.mapperatorinator_home,
        )
        argv = build_command(self.config, request, workdir)
        try:
            run(argv)
        except CommandError as error:
            raise WorkerError(
                ErrorCode.CHART_GENERATION_FAILED,
                f"inference failed: {error}",
                context={"stderr": error.stderr[-2000:], "key_mode": request.key_mode},
            ) from error

        osu_path = find_generated_osu(workdir)
        try:
            beatmap = parse_osu_file(osu_path)
        except ValueError as error:
            raise WorkerError(
                ErrorCode.CHART_OSU_PARSE_FAILED,
                f"could not parse the generated beatmap: {error}",
                context={"path": str(osu_path)},
            ) from error
        if beatmap.key_mode != request.key_mode:
            raise WorkerError(
                ErrorCode.CHART_GENERATION_FAILED,
                f"asked for {request.key_mode}K but got {beatmap.key_mode}K",
            )
        return GeneratedChart(
            notes=beatmap.notes,
            key_mode=beatmap.key_mode,
            osu_text=osu_path.read_text(encoding="utf-8-sig"),
            generator_name="mapperatorinator-v32",
            seed=request.seed,
        )
