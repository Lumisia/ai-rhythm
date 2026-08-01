"""audio-profile-v1 표준화 — 단계 조립.

프로브 -> 측정 -> 게인 계산 -> 인코딩 -> 결과 검증 순서로 돈다.
각 단계의 실패를 Phase5 §1 오류 코드로 옮긴다.
"""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from chart_worker.audio import commands, profile
from chart_worker.audio.loudness import compute_gain_db, is_silent, parse_loudnorm_json
from chart_worker.audio.probe import AudioProbe, parse_probe_json
from chart_worker.audio.runner import CommandError, CommandResult, CommandRunner
from chart_worker.config import WorkerConfig
from chart_worker.errors import ErrorCode, WorkerError
from chart_worker.hashing import sha256_file

RunCommand = Callable[[list[str]], CommandResult]


@dataclass(frozen=True, slots=True)
class NormalizedAudio:
    path: Path
    profile_version: str
    sha256: str
    duration_ms: int
    sample_rate_hz: int
    channels: int
    source_duration_ms: int
    trimmed_ms: int
    gain_db: float
    achieved_lufs: float
    achieved_true_peak_dbtp: float
    shortfall_lu: float
    limited_by: str


def _probe(run: RunCommand, ffprobe_bin: Path, path: Path) -> AudioProbe:
    argv = commands.probe_command(ffprobe_bin, path)
    try:
        result = run(argv)
    except CommandError as error:
        raise WorkerError(
            ErrorCode.AUDIO_INVALID,
            f"ffprobe failed: {error}",
            context={"path": str(path), "stderr": error.stderr[-2000:]},
        ) from error
    try:
        return parse_probe_json(result.stdout)
    except ValueError as error:
        raise WorkerError(
            ErrorCode.AUDIO_INVALID,
            f"ffprobe output unusable: {error}",
            context={"path": str(path)},
        ) from error


def _measure(run: RunCommand, ffmpeg_bin: Path, source: Path) -> str:
    argv = commands.measure_command(ffmpeg_bin, source)
    try:
        return run(argv).stderr
    except CommandError as error:
        raise WorkerError(
            ErrorCode.AUDIO_NORMALIZATION_FAILED,
            f"loudness measurement failed: {error}",
            context={"path": str(source), "stderr": error.stderr[-2000:]},
        ) from error


def _too_long(path: Path, duration_ms: int, limit_ms: int, *, remove: bool) -> WorkerError:
    if remove:
        # 남겨두면 다음 실행이 유효한 산출물로 오인한다.
        path.unlink(missing_ok=True)
    return WorkerError(
        ErrorCode.AUDIO_TOO_LONG,
        f"{duration_ms} ms exceeds the {limit_ms} ms limit",
        context={"path": str(path), "duration_ms": duration_ms, "limit_ms": limit_ms},
    )


def normalize_audio(
    source: Path,
    target: Path,
    *,
    config: WorkerConfig,
    run: RunCommand | None = None,
) -> NormalizedAudio:
    """원본 오디오를 audio-profile-v1 FLAC 으로 굳힌다."""
    if run is None:
        run = CommandRunner(shared_bin_dir=config.ffmpeg_shared_bin_dir)

    ffmpeg_bin = config.ffmpeg_bin
    ffprobe_bin = config.ffprobe_bin

    source_probe = _probe(run, ffprobe_bin, source)
    if source_probe.duration_ms > profile.MAX_INPUT_DURATION_MS:
        raise _too_long(
            source, source_probe.duration_ms, profile.MAX_INPUT_DURATION_MS, remove=False
        )

    stderr = _measure(run, ffmpeg_bin, source)
    try:
        measurement = parse_loudnorm_json(stderr)
    except ValueError as error:
        # loudnorm 은 프레임을 하나라도 받으면 JSON 을 찍는다. 명령이 0 으로
        # 끝났는데 블록이 없다는 건 트림이 전부를 잘라냈다는 뜻이다.
        raise WorkerError(
            ErrorCode.AUDIO_SILENT,
            "the trimmed signal produced no loudness measurement",
            context={"path": str(source)},
        ) from error
    if is_silent(measurement):
        raise WorkerError(
            ErrorCode.AUDIO_SILENT,
            f"integrated loudness {measurement.input_i} LUFS is at or below the threshold",
            context={"path": str(source), "input_i": measurement.input_i},
        )

    plan = compute_gain_db(measurement)

    target.parent.mkdir(parents=True, exist_ok=True)
    argv = commands.normalize_command(ffmpeg_bin, source, target, gain_db=plan.gain_db)
    try:
        run(argv)
    except CommandError as error:
        target.unlink(missing_ok=True)
        raise WorkerError(
            ErrorCode.AUDIO_NORMALIZATION_FAILED,
            f"encoding failed: {error}",
            context={"path": str(target), "stderr": error.stderr[-2000:]},
        ) from error

    # 명령행에 프로파일을 박아도 확인은 별개다.
    result_probe = _probe(run, ffprobe_bin, target)
    _require_profile(target, result_probe)
    if result_probe.duration_ms > profile.MAX_DURATION_MS:
        raise _too_long(target, result_probe.duration_ms, profile.MAX_DURATION_MS, remove=True)

    return NormalizedAudio(
        path=target,
        profile_version=profile.PROFILE_VERSION,
        sha256=sha256_file(target),
        duration_ms=result_probe.duration_ms,
        sample_rate_hz=result_probe.sample_rate_hz,
        channels=result_probe.channels,
        source_duration_ms=source_probe.duration_ms,
        trimmed_ms=max(0, source_probe.duration_ms - result_probe.duration_ms),
        gain_db=plan.gain_db,
        achieved_lufs=plan.achieved_lufs,
        achieved_true_peak_dbtp=plan.achieved_true_peak_dbtp,
        shortfall_lu=plan.shortfall_lu,
        limited_by=plan.limited_by,
    )


def _require_profile(target: Path, result: AudioProbe) -> None:
    expected = {
        "codec": (result.codec_name, profile.AUDIO_CODEC),
        "sample_rate_hz": (result.sample_rate_hz, profile.SAMPLE_RATE_HZ),
        "channels": (result.channels, profile.CHANNELS),
    }
    mismatched = {
        name: {"actual": actual, "expected": wanted}
        for name, (actual, wanted) in expected.items()
        if actual != wanted
    }
    if mismatched:
        target.unlink(missing_ok=True)
        raise WorkerError(
            ErrorCode.AUDIO_NORMALIZATION_FAILED,
            f"output does not match {profile.PROFILE_VERSION}: {mismatched}",
            context={"path": str(target), "mismatched": mismatched},
        )
