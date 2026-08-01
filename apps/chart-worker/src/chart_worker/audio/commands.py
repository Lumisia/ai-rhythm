"""ffmpeg · ffprobe 명령행 조립.

전부 순수 함수다. 실행은 runner 가 한다. 실수하기 쉬운 부분이 정확히
여기 몰려 있어서 ffmpeg 없이 전부 검증할 수 있게 떼어놨다.
"""

from pathlib import Path

from chart_worker.audio.profile import (
    AUDIO_CODEC,
    CHANNELS,
    FLAC_COMPRESSION_LEVEL,
    SAMPLE_FORMAT,
    SAMPLE_RATE_HZ,
    SILENCE_THRESHOLD_DB,
    TARGET_LUFS,
    TARGET_TRUE_PEAK_DBTP,
)


def _number(value: float) -> str:
    """ffmpeg 인자에 넣을 수를 군더더기 없이 적는다 (-60.0 -> -60)."""
    return format(value, "g")


def silence_trim_filter(threshold_db: float = SILENCE_THRESHOLD_DB) -> str:
    """앞 무음 절단 필터.

    측정 패스와 인코딩 패스가 **반드시 같은 문자열**을 써야 한다.
    다르면 측정값이 실제로 인코딩할 신호를 설명하지 못한다.

    detection 은 기본값 rms 를 그대로 쓴다. peak 로 바꾸면 디더 잡음
    한 샘플에 트림이 멈춘다.
    """
    return (
        "silenceremove="
        "start_periods=1"
        ":start_duration=0"
        f":start_threshold={_number(threshold_db)}dB"
        ":detection=rms"
    )


def ffprobe_for(ffmpeg_bin: Path) -> Path:
    """ffprobe 는 ffmpeg 옆에 같은 확장자로 설치된다."""
    return ffmpeg_bin.with_name(f"ffprobe{ffmpeg_bin.suffix}")


def probe_command(ffprobe_bin: Path, source: Path) -> list[str]:
    return [
        str(ffprobe_bin),
        "-hide_banner",
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        str(source),
    ]


def measure_command(
    ffmpeg_bin: Path,
    source: Path,
    *,
    threshold_db: float = SILENCE_THRESHOLD_DB,
) -> list[str]:
    """1패스 — 트림한 신호의 라우드니스를 잰다. 파일을 쓰지 않는다.

    loudnorm 을 정규화가 아니라 **측정기로만** 쓴다. 실제 게인은
    compute_gain_db 가 계산해 volume 필터로 건다.
    """
    filters = f"{silence_trim_filter(threshold_db)},loudnorm=I={_number(TARGET_LUFS)}:TP={_number(TARGET_TRUE_PEAK_DBTP)}:print_format=json"
    return [
        str(ffmpeg_bin),
        "-hide_banner",
        "-nostdin",
        "-v",
        "info",
        "-i",
        str(source),
        "-map",
        "0:a:0",
        "-af",
        filters,
        "-f",
        "null",
        "-",
    ]


def normalize_command(
    ffmpeg_bin: Path,
    source: Path,
    target: Path,
    *,
    gain_db: float,
    threshold_db: float = SILENCE_THRESHOLD_DB,
) -> list[str]:
    """2패스 — 트림하고 상수 게인을 걸어 FLAC 으로 굳힌다.

    -ar 을 명시하는 이유: loudnorm 이 dynamic 으로 넘어가면 트루피크
    검출을 위해 192 kHz 로 업샘플하고 출력이 그대로 나온다. 우리는
    dynamic 을 쓰지 않지만 프로파일을 명령행에 못 박는다.

    -bitexact 를 붙이는 이유: 없으면 FLAC vendor string 에 ffmpeg
    버전이 박혀 같은 입력이 다른 sha256 을 낸다. 자산 식별이 sha256 이라
    그대로 두면 중복 제거가 깨진다.
    """
    filters = f"{silence_trim_filter(threshold_db)},volume={gain_db:.6f}dB"
    return [
        str(ffmpeg_bin),
        "-hide_banner",
        "-nostdin",
        "-v",
        "error",
        "-y",
        "-i",
        str(source),
        "-map",
        "0:a:0",
        "-af",
        filters,
        "-ar",
        str(SAMPLE_RATE_HZ),
        "-ac",
        str(CHANNELS),
        "-sample_fmt",
        SAMPLE_FORMAT,
        "-c:a",
        AUDIO_CODEC,
        "-compression_level",
        str(FLAC_COMPRESSION_LEVEL),
        "-map_metadata",
        "-1",
        "-bitexact",
        str(target),
    ]
