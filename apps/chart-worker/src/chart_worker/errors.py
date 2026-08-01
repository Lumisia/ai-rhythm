"""워커 공용 오류 코드와 재시도 분류.

재시도 판단 기준은 하나다 — 다시 해서 달라질 수 있나.
그 답을 코드마다 표로 박아둔다. 호출부가 except 블록마다 매번 판단하면
같은 오류를 곳에 따라 다르게 다루게 된다.
"""

from enum import StrEnum
from typing import Any

MAX_ATTEMPTS = 3
BACKOFF_SECONDS: tuple[int, ...] = (30, 120, 480)


class ErrorCode(StrEnum):
    # 오디오
    AUDIO_INVALID = "AUDIO_INVALID"
    AUDIO_TOO_LONG = "AUDIO_TOO_LONG"
    AUDIO_SILENT = "AUDIO_SILENT"
    AUDIO_NORMALIZATION_FAILED = "AUDIO_NORMALIZATION_FAILED"
    STEMS_SEPARATION_FAILED = "STEMS_SEPARATION_FAILED"

    # 채보
    CHART_ANALYSIS_FAILED = "CHART_ANALYSIS_FAILED"
    CHART_GENERATION_FAILED = "CHART_GENERATION_FAILED"
    CHART_OSU_PARSE_FAILED = "CHART_OSU_PARSE_FAILED"
    CHART_VALIDATION_FAILED = "CHART_VALIDATION_FAILED"
    CHART_TIMING_INVARIANT_VIOLATED = "CHART_TIMING_INVARIANT_VIOLATED"
    CHART_TARGET_RATING_UNREACHABLE = "CHART_TARGET_RATING_UNREACHABLE"

    # 인프라
    LEASE_INVALID = "LEASE_INVALID"
    LEASE_EXPIRED = "LEASE_EXPIRED"
    STORAGE_UPLOAD_FAILED = "STORAGE_UPLOAD_FAILED"
    STORAGE_OBJECT_MISSING = "STORAGE_OBJECT_MISSING"
    ASSET_HASH_MISMATCH = "ASSET_HASH_MISMATCH"


class Disposition(StrEnum):
    """오류를 만났을 때 무엇을 할지."""

    RETRYABLE = "RETRYABLE"
    """다시 하면 달라질 수 있다. 지수 백오프 후 재시도한다."""

    FINAL = "FINAL"
    """입력이 문제다. 재시도가 무의미하다."""

    FINAL_ALERT = "FINAL_ALERT"
    """코드나 저장소 문제다. 재시도하지 않고 사람이 봐야 한다."""

    WARN = "WARN"
    """실패가 아니다. 결과는 살리고 사실만 기록한다."""


_DISPOSITION: dict[ErrorCode, Disposition] = {
    # 입력 자체가 못 쓰는 파일이다. 몇 번을 돌려도 같다.
    ErrorCode.AUDIO_INVALID: Disposition.FINAL,
    ErrorCode.AUDIO_TOO_LONG: Disposition.FINAL,
    ErrorCode.AUDIO_SILENT: Disposition.FINAL,
    # 멀쩡한 입력에서 ffmpeg 나 Demucs 가 죽는 건 대개 환경·자원 문제다.
    ErrorCode.AUDIO_NORMALIZATION_FAILED: Disposition.RETRYABLE,
    ErrorCode.STEMS_SEPARATION_FAILED: Disposition.RETRYABLE,
    ErrorCode.CHART_ANALYSIS_FAILED: Disposition.RETRYABLE,
    ErrorCode.CHART_GENERATION_FAILED: Disposition.RETRYABLE,
    # 생성기가 뱉은 .osu 가 깨졌다. 같은 파일을 다시 읽어도 같다.
    ErrorCode.CHART_OSU_PARSE_FAILED: Disposition.FINAL,
    # 워커가 이미 안에서 3회 복구를 시도한 뒤의 결과다. 바깥에서 또 돌려도 같다.
    ErrorCode.CHART_VALIDATION_FAILED: Disposition.FINAL,
    # 후처리가 time_ms 를 건드렸다. 데이터가 아니라 코드 버그다.
    ErrorCode.CHART_TIMING_INVARIANT_VIOLATED: Disposition.FINAL_ALERT,
    # 목표 ★에 못 미쳤을 뿐 채보는 쓸 수 있다. 실제 ★를 표시한다.
    ErrorCode.CHART_TARGET_RATING_UNREACHABLE: Disposition.WARN,
    ErrorCode.LEASE_INVALID: Disposition.FINAL,
    ErrorCode.LEASE_EXPIRED: Disposition.RETRYABLE,
    ErrorCode.STORAGE_UPLOAD_FAILED: Disposition.RETRYABLE,
    ErrorCode.STORAGE_OBJECT_MISSING: Disposition.FINAL,
    # 내용 해시 불일치는 재시도로 고쳐지지 않고 저장소 손상이나 배선 버그를
    # 뜻한다. Phase5 §2 표는 이 코드를 명시하지 않지만 조용히 FINAL 로 묻으면
    # 원인이 안 드러난다.
    ErrorCode.ASSET_HASH_MISMATCH: Disposition.FINAL_ALERT,
}


def disposition_of(code: ErrorCode) -> Disposition:
    """오류 코드의 처리 방침을 돌려준다.

    표에 없는 코드에 기본값을 주지 않는다. 코드만 추가하고 분류를
    빠뜨리면 여기서 KeyError 로 드러나야 한다.
    """
    return _DISPOSITION[code]


class WorkerError(Exception):
    """워커 단계가 실패했을 때 던지는 예외.

    입력 검증 실패는 ValueError 로 남긴다(Global Constraints 예외 종류).
    이 예외는 작업 단위가 끝난 뒤 큐가 무엇을 할지 정하기 위한 것이다.
    """

    def __init__(
        self,
        code: ErrorCode,
        message: str,
        *,
        context: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(f"{code.value}: {message}")
        self.code = code
        self.context: dict[str, Any] = dict(context or {})

    @property
    def disposition(self) -> Disposition:
        return disposition_of(self.code)

    @property
    def retryable(self) -> bool:
        return self.disposition is Disposition.RETRYABLE
