"""audio-profile-v2 — 채보 시간축의 정의 그 자체.

여기가 흔들리면 전부 흔들린다. 프로파일을 바꿔도 같은 원본의 파일
sha256은 유지될 수 있지만 프로파일 identity는 달라진다. 기존
채보 계약과 섞지 않도록 값을 고칠 때는 PROFILE_VERSION을 함께 올린다.
"""

PROFILE_VERSION = "audio-profile-v2"

SAMPLE_RATE_HZ = 48_000
CHANNELS = 2
SAMPLE_FORMAT = "s16"
AUDIO_CODEC = "flac"
FLAC_COMPRESSION_LEVEL = 8

MAX_DURATION_MS = 600_000
"""프로파일 상한. 트림이 끝난 최종 파일에 적용한다."""

MAX_INPUT_DURATION_MS = MAX_DURATION_MS
"""비싼 2-pass 디코딩 전에 적용하는 동일한 10분 입력 방어선."""

SILENCE_THRESHOLD_DB = -60.0
TARGET_LUFS = -14.0
TARGET_TRUE_PEAK_DBTP = -1.0
