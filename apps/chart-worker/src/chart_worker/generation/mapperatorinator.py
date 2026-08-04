"""Mapperatorinator subprocess 어댑터.

명령행 조립은 순수 함수로 떼어 GPU 없이 검증한다. ffmpeg 어댑터와
같은 구조다.
"""

import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

from chart_worker.audio.runner import CommandError, CommandResult, CommandRunner
from chart_worker.config import WorkerConfig
from chart_worker.errors import ErrorCode, WorkerError
from chart_worker.generation.mapperatorinator_patch import require_mapperatorinator_patch
from chart_worker.generation.osu_parser import OsuBeatmap, OsuBpmEvent, parse_osu_file
from chart_worker.generation.params import (
    GAMEMODE_MANIA,
    PRECISION,
    GenerationRequest,
    TimingGenerationRequest,
)
from chart_worker.schema.note import Chart

INFERENCE_SCRIPT = "inference.py"
CONFIG_NAME = "v32"
RunCommand = Callable[[list[str]], CommandResult]
PatchVerifier = Callable[[Path], None]


@dataclass(frozen=True, slots=True)
class GeneratedChart:
    notes: Chart
    key_mode: int
    osu_text: str
    generator_name: str
    seed: int | None
    bpm_events: tuple[OsuBpmEvent, ...] = ()


@dataclass(frozen=True, slots=True)
class GeneratedTiming:
    osu_text: str
    bpm_events: tuple[OsuBpmEvent, ...]
    generator_name: str
    seed: int | None
    mode: Literal["STANDARD", "SUPER_TIMING"]


class ChartGenerator(Protocol):
    def generate_timing(
        self, request: TimingGenerationRequest, workdir: Path
    ) -> GeneratedTiming: ...

    def generate_map(
        self, request: GenerationRequest, workdir: Path
    ) -> GeneratedChart: ...


def inference_env(base: Mapping[str, str] | None = None) -> dict[str, str]:
    """Unicode 제목도 Windows 자식 프로세스가 UTF-8로 출력하게 한다."""
    env = dict(os.environ if base is None else base)
    forced = {"PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}
    for name in list(env):
        if name.upper() in forced:
            del env[name]
    env.update(forced)
    return env


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


def _hydra_path(path: Path) -> str:
    """Quote a path as one Hydra string value.

    Passing one subprocess argv item is not enough: Hydra parses spaces, parentheses and
    brackets again using its override grammar. Mapperatorinator's own CLI examples wrap
    Windows paths in single quotes inside the override value for this reason.
    """
    value = str(path.resolve())
    return "'" + value.replace("'", "\\'") + "'"


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
        f"hydra.run.dir={_hydra_path(output_dir / '.hydra-run')}",
        f"audio_path={_hydra_path(request.audio_path)}",
        f"output_path={_hydra_path(output_dir)}",
        f"gamemode={GAMEMODE_MANIA}",
        f"keycount={request.key_mode}",
        f"difficulty={request.requested_star}",
        f"year={request.year}",
        f"end_time={request.duration_ms}",
        f"cfg_scale={request.cfg_scale}",
        f"descriptors={_hydra_list(request.descriptors)}",
        f"precision={config.mapperatorinator_precision or PRECISION}",
        "export_osz=false",
        "output_type=[TIMING,MAP]",
        "hitsounded=false",
        "fast_decoder_loop=true",
        "resnap_events=true",
    ]
    if request.seed is not None:
        argv.append(f"seed={request.seed}")
    return argv


def _command_prefix(config: WorkerConfig, output_dir: Path) -> list[str]:
    if config.mapperatorinator_python is None or config.mapperatorinator_home is None:
        raise ValueError("mapperatorinator_python and mapperatorinator_home must be configured")
    return [
        str(_absolute(config.mapperatorinator_python)),
        INFERENCE_SCRIPT,
        "-cn",
        CONFIG_NAME,
        f"hydra.run.dir={_hydra_path(output_dir / '.hydra-run')}",
    ]


def _common_generation_arguments(
    config: WorkerConfig,
    audio_path: Path,
    duration_ms: int,
    year: int,
    output_dir: Path,
) -> list[str]:
    return [
        *_command_prefix(config, output_dir),
        f"audio_path={_hydra_path(audio_path)}",
        f"output_path={_hydra_path(output_dir)}",
        f"gamemode={GAMEMODE_MANIA}",
        f"year={year}",
        f"end_time={duration_ms}",
        f"precision={config.mapperatorinator_precision or PRECISION}",
        "export_osz=false",
        "hitsounded=false",
        "fast_decoder_loop=true",
        "resnap_events=true",
    ]


def build_timing_command(
    config: WorkerConfig,
    request: TimingGenerationRequest,
    output_dir: Path,
) -> list[str]:
    argv = _common_generation_arguments(
        config,
        request.audio_path,
        request.duration_ms,
        request.year,
        output_dir,
    )
    argv.extend(
        [
            "output_type=[TIMING]",
            f"super_timing={'true' if request.super_timing else 'false'}",
        ]
    )
    if request.seed is not None:
        argv.append(f"seed={request.seed}")
    return argv


def build_map_command(
    config: WorkerConfig,
    request: GenerationRequest,
    output_dir: Path,
) -> list[str]:
    """Build MAP inference that consumes a stable timing reference."""
    argv = _common_generation_arguments(
        config,
        request.audio_path,
        request.duration_ms,
        request.year,
        output_dir,
    )
    argv.extend(
        [
            f"keycount={request.key_mode}",
            f"difficulty={request.requested_star}",
            f"cfg_scale={request.cfg_scale}",
            f"descriptors={_hydra_list(request.descriptors)}",
            f"beatmap_path={_hydra_path(request.timing_reference_path)}",
            "in_context=[TIMING]",
            "output_type=[MAP]",
            "super_timing=false",
        ]
    )
    if request.seed is not None:
        argv.append(f"seed={request.seed}")
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
    verify_patch: PatchVerifier = require_mapperatorinator_patch

    def _run_and_parse(self, argv: list[str], workdir: Path) -> tuple[str, OsuBeatmap]:
        if self.config.mapperatorinator_home is None:
            raise ValueError("mapperatorinator_home must be configured")
        self.verify_patch(self.config.mapperatorinator_home)
        workdir.mkdir(parents=True, exist_ok=True)
        _require_clean_output_dir(workdir)
        run = self.run or CommandRunner(
            shared_bin_dir=self.config.ffmpeg_shared_bin_dir,
            timeout_sec=1800.0,
            # inference.py 는 상대 경로이고 Hydra 는 실행 위치에서 configs/ 를
            # 찾는다. 체크아웃 밖에서 돌리면 스크립트도 설정도 못 찾는다.
            cwd=self.config.mapperatorinator_home,
            env=inference_env(),
        )
        try:
            run(argv)
        except CommandError as error:
            raise WorkerError(
                ErrorCode.CHART_GENERATION_FAILED,
                f"inference failed: {error}",
                context={"stderr": error.stderr[-2000:]},
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
        return osu_path.read_text(encoding="utf-8-sig"), beatmap

    def generate_timing(
        self, request: TimingGenerationRequest, workdir: Path
    ) -> GeneratedTiming:
        osu_text, beatmap = self._run_and_parse(
            build_timing_command(self.config, request, workdir), workdir
        )
        if not beatmap.bpm_events:
            raise WorkerError(
                ErrorCode.CHART_GENERATION_FAILED,
                "timing inference produced no BPM events",
            )
        return GeneratedTiming(
            osu_text=osu_text,
            bpm_events=beatmap.bpm_events,
            generator_name="mapperatorinator-v32",
            seed=request.seed,
            mode="SUPER_TIMING" if request.super_timing else "STANDARD",
        )

    def generate_map(
        self, request: GenerationRequest, workdir: Path
    ) -> GeneratedChart:
        osu_text, beatmap = self._run_and_parse(
            build_map_command(self.config, request, workdir), workdir
        )
        if beatmap.key_mode != request.key_mode:
            raise WorkerError(
                ErrorCode.CHART_GENERATION_FAILED,
                f"asked for {request.key_mode}K but got {beatmap.key_mode}K",
            )
        return GeneratedChart(
            notes=beatmap.notes,
            key_mode=beatmap.key_mode,
            osu_text=osu_text,
            generator_name="mapperatorinator-v32",
            seed=request.seed,
            bpm_events=beatmap.bpm_events,
        )

    def __call__(self, request: GenerationRequest, workdir: Path) -> GeneratedChart:
        """Compatibility bridge until the generation stage uses generate_map directly."""
        osu_text, beatmap = self._run_and_parse(
            build_command(self.config, request, workdir), workdir
        )
        if beatmap.key_mode != request.key_mode:
            raise WorkerError(
                ErrorCode.CHART_GENERATION_FAILED,
                f"asked for {request.key_mode}K but got {beatmap.key_mode}K",
            )
        return GeneratedChart(
            notes=beatmap.notes,
            key_mode=beatmap.key_mode,
            osu_text=osu_text,
            generator_name="mapperatorinator-v32",
            seed=request.seed,
            bpm_events=beatmap.bpm_events,
        )
