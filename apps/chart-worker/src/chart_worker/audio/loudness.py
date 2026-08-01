"""loudnorm 측정값 파싱과 게인 계산.

loudnorm 의 2패스 정규화를 쓰지 않는다. linear=true 는 조건을 만족하지
못하면 문서 그대로 dynamic 으로 되돌아가고, dynamic 은 시변 AGC 라
onset strength 포락선을 변형한다. 그건 librosa 가 재는 값 그 자체다.

대신 loudnorm 을 측정기로만 쓰고 게인은 여기서 계산해 volume 필터로
건다. 상수 스칼라 곱은 상대적 onset 세기를 수학적으로 바꿀 수 없다.
"""

import json
import math
import re
from dataclasses import dataclass
from typing import Literal

from chart_worker.audio.profile import (
    SILENCE_THRESHOLD_DB,
    TARGET_LUFS,
    TARGET_TRUE_PEAK_DBTP,
)

# loudnorm 이 찍는 JSON 은 중첩이 없다. stderr 에 진행 로그와 섞여 나오므로
# 중괄호 블록을 전부 찾아 마지막 것을 쓴다.
_JSON_BLOCK = re.compile(r"\{[^{}]*\}")

_REQUIRED_FIELDS = ("input_i", "input_tp", "input_lra", "input_thresh")


@dataclass(frozen=True, slots=True)
class LoudnessMeasurement:
    """1패스가 잰 값. 단위는 input_i · input_thresh 가 LUFS, input_tp 가 dBTP."""

    input_i: float
    input_tp: float
    input_lra: float
    input_thresh: float


@dataclass(frozen=True, slots=True)
class GainPlan:
    gain_db: float
    achieved_lufs: float
    achieved_true_peak_dbtp: float
    shortfall_lu: float
    """목표 라우드니스에 못 미친 양. 0 이면 명중."""

    limited_by: Literal["LOUDNESS", "TRUE_PEAK"]


def _to_float(raw: object, field_name: str) -> float:
    """loudnorm 은 값을 전부 문자열로 찍는다. 무음이면 "-inf" 가 온다."""
    if not isinstance(raw, bool) and isinstance(raw, (str, int, float)):
        try:
            return float(raw)
        except ValueError:
            pass
    # 이 패키지는 입력 검증 실패를 ValueError 로 통일한다. 타입 오류에도 TypeError 를 쓰지 않는다.
    raise ValueError(f"loudnorm {field_name} is not a number: {raw!r}")


def parse_loudnorm_json(stderr: str) -> LoudnessMeasurement:
    """ffmpeg stderr 에서 loudnorm 측정 블록을 뽑는다."""
    blocks = _JSON_BLOCK.findall(stderr)
    for block in reversed(blocks):
        try:
            payload = json.loads(block)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and all(key in payload for key in _REQUIRED_FIELDS):
            return LoudnessMeasurement(
                input_i=_to_float(payload["input_i"], "input_i"),
                input_tp=_to_float(payload["input_tp"], "input_tp"),
                input_lra=_to_float(payload["input_lra"], "input_lra"),
                input_thresh=_to_float(payload["input_thresh"], "input_thresh"),
            )
    raise ValueError("loudnorm measurement JSON not found in ffmpeg output")


def is_silent(
    measurement: LoudnessMeasurement,
    *,
    threshold_db: float = SILENCE_THRESHOLD_DB,
) -> bool:
    """트림 후에도 임계값 아래면 소리가 없는 파일이다."""
    return math.isinf(measurement.input_i) or measurement.input_i <= threshold_db


def compute_gain_db(
    measurement: LoudnessMeasurement,
    *,
    target_lufs: float = TARGET_LUFS,
    target_true_peak_dbtp: float = TARGET_TRUE_PEAK_DBTP,
) -> GainPlan:
    """트루피크를 넘지 않는 선에서 목표 라우드니스에 가장 가까운 상수 게인.

    피크가 심한 곡은 목표에 미달한다. 그걸 받아들인다. 목표 -14 LUFS 는
    청취 테스트로 확정할 초안이고, onset 왜곡은 채보 정확도를 직접 깎는다.
    미달분은 shortfall_lu 로 남겨 나중에 판단한다.
    """
    if not math.isfinite(measurement.input_i) or not math.isfinite(measurement.input_tp):
        raise ValueError("cannot plan gain from a non-finite measurement")

    loudness_gain = target_lufs - measurement.input_i
    peak_gain = target_true_peak_dbtp - measurement.input_tp
    if peak_gain < loudness_gain:
        gain_db, limited_by = peak_gain, "TRUE_PEAK"
    else:
        gain_db, limited_by = loudness_gain, "LOUDNESS"

    achieved_lufs = measurement.input_i + gain_db
    return GainPlan(
        gain_db=round(gain_db, 6),
        achieved_lufs=round(achieved_lufs, 6),
        achieved_true_peak_dbtp=round(measurement.input_tp + gain_db, 6),
        shortfall_lu=round(max(0.0, target_lufs - achieved_lufs), 6),
        limited_by=limited_by,
    )
