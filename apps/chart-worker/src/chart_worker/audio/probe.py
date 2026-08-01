"""ffprobe 출력 파싱.

format.duration 을 믿지 않는다. 컨테이너가 적어둔 근삿값이라 채보의
durationMs 로 쓰면 경계 노트가 어긋난다. 스트림의 duration_ts 와
time_base 로 정확한 샘플 수를 복원한다.
"""

import json
from dataclasses import dataclass
from fractions import Fraction
from typing import Any


@dataclass(frozen=True, slots=True)
class AudioProbe:
    codec_name: str
    sample_rate_hz: int
    channels: int
    duration_ms: int
    duration_is_exact: bool
    """duration_ts 로 복원했으면 True. 컨테이너 근삿값으로 떨어졌으면 False."""


def _first_audio_stream(payload: dict[str, Any]) -> dict[str, Any]:
    for stream in payload.get("streams", []):
        if isinstance(stream, dict) and stream.get("codec_type") == "audio":
            return stream
    raise ValueError("no audio stream found")


def _exact_duration_ms(stream: dict[str, Any]) -> int | None:
    """duration_ts x time_base. FLAC 은 time_base 가 1/샘플레이트라 정확하다."""
    raw_ts = stream.get("duration_ts")
    raw_base = stream.get("time_base")
    if raw_ts is None or not isinstance(raw_base, str):
        return None
    try:
        ticks = int(raw_ts)
        base = Fraction(raw_base)
    except (TypeError, ValueError, ZeroDivisionError):
        return None
    if ticks < 0 or base <= 0:
        return None
    # Fraction 으로 곱해 부동소수 반올림 오차를 없앤다.
    return round(ticks * base * 1000)


def _approximate_duration_ms(payload: dict[str, Any], stream: dict[str, Any]) -> int:
    for source in (stream, payload.get("format", {})):
        raw = source.get("duration") if isinstance(source, dict) else None
        if raw is None:
            continue
        try:
            seconds = float(raw)
        except (TypeError, ValueError):
            continue
        if seconds >= 0:
            return round(seconds * 1000)
    raise ValueError("ffprobe reported no usable duration")


def parse_probe_json(payload: str | dict[str, Any]) -> AudioProbe:
    """ffprobe -print_format json 출력을 읽는다."""
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except json.JSONDecodeError as error:
            raise ValueError(f"ffprobe output is not JSON: {error}") from None
    if not isinstance(payload, dict):
        # 이 패키지는 입력 검증 실패를 ValueError 로 통일한다. TRY004 를 의도적으로 무시한다.
        raise ValueError("ffprobe output must be a JSON object")  # noqa: TRY004

    stream = _first_audio_stream(payload)
    try:
        sample_rate = int(stream["sample_rate"])
        channels = int(stream["channels"])
        codec_name = str(stream["codec_name"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"ffprobe audio stream is incomplete: {error}") from None
    if sample_rate <= 0 or channels <= 0:
        raise ValueError("ffprobe reported a non-positive sample rate or channel count")

    exact = _exact_duration_ms(stream)
    duration_ms = exact if exact is not None else _approximate_duration_ms(payload, stream)
    return AudioProbe(
        codec_name=codec_name,
        sample_rate_hz=sample_rate,
        channels=channels,
        duration_ms=duration_ms,
        duration_is_exact=exact is not None,
    )
