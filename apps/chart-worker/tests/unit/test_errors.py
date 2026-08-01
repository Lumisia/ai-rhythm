import pytest

from chart_worker.errors import (
    BACKOFF_SECONDS,
    MAX_ATTEMPTS,
    Disposition,
    ErrorCode,
    WorkerError,
    disposition_of,
)


def test_every_error_code_has_a_disposition():
    """분류를 빠뜨린 코드는 호출부가 조용히 기본값으로 처리한다."""
    for code in ErrorCode:
        assert isinstance(disposition_of(code), Disposition)


@pytest.mark.parametrize(
    ("code", "expected"),
    [
        (ErrorCode.AUDIO_INVALID, Disposition.FINAL),
        (ErrorCode.AUDIO_TOO_LONG, Disposition.FINAL),
        (ErrorCode.AUDIO_SILENT, Disposition.FINAL),
        (ErrorCode.AUDIO_NORMALIZATION_FAILED, Disposition.RETRYABLE),
        (ErrorCode.STEMS_SEPARATION_FAILED, Disposition.RETRYABLE),
        (ErrorCode.CHART_ANALYSIS_FAILED, Disposition.RETRYABLE),
        (ErrorCode.CHART_GENERATION_FAILED, Disposition.RETRYABLE),
        (ErrorCode.CHART_OSU_PARSE_FAILED, Disposition.FINAL),
        (ErrorCode.CHART_VALIDATION_FAILED, Disposition.FINAL),
        (ErrorCode.CHART_TIMING_INVARIANT_VIOLATED, Disposition.FINAL_ALERT),
        (ErrorCode.ASSET_HASH_MISMATCH, Disposition.FINAL_ALERT),
        (ErrorCode.CHART_TARGET_RATING_UNREACHABLE, Disposition.WARN),
        (ErrorCode.LEASE_EXPIRED, Disposition.RETRYABLE),
        (ErrorCode.LEASE_INVALID, Disposition.FINAL),
        (ErrorCode.STORAGE_UPLOAD_FAILED, Disposition.RETRYABLE),
        (ErrorCode.STORAGE_OBJECT_MISSING, Disposition.FINAL),
    ],
)
def test_documented_dispositions(code, expected):
    assert disposition_of(code) is expected


def test_only_retryable_is_retryable():
    for code in ErrorCode:
        expected = disposition_of(code) is Disposition.RETRYABLE
        assert WorkerError(code, "boom").retryable is expected


def test_timing_invariant_violation_is_never_retried():
    """후처리가 time_ms 를 건드린 건 데이터가 아니라 코드 버그다."""
    error = WorkerError(ErrorCode.CHART_TIMING_INVARIANT_VIOLATED, "3 notes moved")
    assert error.retryable is False
    assert error.disposition is Disposition.FINAL_ALERT


def test_backoff_matches_attempt_limit():
    assert MAX_ATTEMPTS == 3
    assert BACKOFF_SECONDS == (30, 120, 480)
    assert len(BACKOFF_SECONDS) >= MAX_ATTEMPTS


def test_error_carries_code_and_context():
    error = WorkerError(
        ErrorCode.AUDIO_TOO_LONG, "190s", context={"duration_ms": 190_000}
    )
    assert error.code is ErrorCode.AUDIO_TOO_LONG
    assert error.context == {"duration_ms": 190_000}
    assert error.disposition is Disposition.FINAL
    assert "AUDIO_TOO_LONG" in str(error)
    assert "190s" in str(error)


def test_context_defaults_to_empty_dict_and_is_copied():
    source = {"path": "a.flac"}
    error = WorkerError(ErrorCode.AUDIO_INVALID, "bad", context=source)
    source["path"] = "b.flac"
    assert error.context == {"path": "a.flac"}
    assert WorkerError(ErrorCode.AUDIO_INVALID, "bad").context == {}


def test_error_code_str_value_matches_name():
    for code in ErrorCode:
        assert code.value == code.name


def test_worker_error_is_not_a_value_error():
    """입력 검증 실패(ValueError)와 단계 실패를 섞지 않는다."""
    assert not isinstance(WorkerError(ErrorCode.AUDIO_INVALID, "bad"), ValueError)
