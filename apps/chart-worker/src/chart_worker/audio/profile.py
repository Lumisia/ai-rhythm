"""audio-profile-v1 — 채보 시간축의 정의 그 자체.

여기가 흔들리면 전부 흔들린다. 프로파일을 바꾸면 같은 원본이 다른
sha256 을 내므로 기존 채보가 전부 무효가 된다. 값을 고칠 때는
PROFILE_VERSION 을 함께 올리고 재생성한다.
"""

PROFILE_VERSION = "audio-profile-v1"

SAMPLE_RATE_HZ = 48_000
CHANNELS = 2
SAMPLE_FORMAT = "s16"
AUDIO_CODEC = "flac"
FLAC_COMPRESSION_LEVEL = 8

MAX_DURATION_MS = 184_000
"""프로파일 상한. 트림이 끝난 최종 파일에 적용한다."""

MAX_INPUT_DURATION_MS = 600_000
"""입력 방어선.

MAX_DURATION_MS 와 다른 이유로 존재한다. 앞 무음을 넉넉히 감안해도
10분을 넘는 입력은 2패스 디코딩을 태울 값어치가 없다.
"""

SILENCE_THRESHOLD_DB = -60.0
TARGET_LUFS = -14.0
TARGET_TRUE_PEAK_DBTP = -1.0
