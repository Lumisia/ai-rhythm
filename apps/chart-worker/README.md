# chart-worker 직접 생성과 로컬 검증

`chart-worker`는 한 곡에서 4K·6K·7K와 EASY·NORMAL·HARD·EXPERT 조합
12개를 생성한다. Mapperatorinator가 만든 timing, 레인, TAP/HOLD 종류와 길이를
후처리로 바꾸지 않는다. `generate`와 `bench` 명령을 실행하는 동안에만 모델이
동작하며 상시 GPU 서비스가 아니다.

## 생성 계약

정상 출력은 조합마다 한 번만 추론한다. 파싱 또는 구조 검증에 실패한 조합만 다른
seed로 최대 두 번 재시도하며, 이미 성공한 조합은 다시 만들지 않는다.

| 난이도 | 요청 별 | descriptor |
| --- | ---: | --- |
| EASY | 1.0 | `expression/simple` |
| NORMAL | 1.5 | `style/mixed rice` |
| HARD | 3.5 | `style/generic hybrid` |
| EXPERT | 4.5 | `tech/technical hybrid` |

공통 요청은 `cfg_scale=1.0`, `output_type=[TIMING,MAP]`,
`hitsounded=false`, `fast_decoder_loop=true`다. `hold_note_ratio`를 보내지 않으므로
롱노트는 모델이 곡과 패턴에 맞게 정한다. 룰 기반 채보, Beat This timing,
super timing, 정상 채보의 다중 seed 후보 경쟁, solver와 레인 재배치는 실행하지 않는다.

`end_time`에는 정규화 오디오 길이를 전달한다. Mapperatorinator가 패딩 구간에
오디오 밖 노트를 만드는 것을 모델 자체 크롭 단계에서 막고, 내보내기 단계는
그 결과를 다시 수정하지 않는다. Hydra 실행 로그도 각 조합 출력의 `.hydra-run`
아래에 두므로 Mapperatorinator 체크아웃이 읽기 전용이어도 실행할 수 있다.

## 실제 Mapperatorinator 실행 준비

PowerShell의 현재 프로세스에만 Mapperatorinator와 공유 FFmpeg 경로를 설정한다.
환경에 맞게 실제 경로를 바꾼다.

```powershell
$mapperRoot = "C:\Users\PC\mapperatorinator"
$mapperPython = "C:\Users\PC\mapperatorinator\.venv\Scripts\python.exe"
$sharedFfmpegBin = "C:\path\to\shared-ffmpeg\bin"

if (-not (Test-Path -LiteralPath $mapperRoot -PathType Container)) { throw "Mapperatorinator home is missing" }
if (-not (Test-Path -LiteralPath $mapperPython -PathType Leaf)) { throw "Mapperatorinator Python is missing" }
if (-not (Test-Path -LiteralPath "$sharedFfmpegBin\ffmpeg.exe" -PathType Leaf)) { throw "shared FFmpeg is missing" }
if (-not (Test-Path -LiteralPath "$sharedFfmpegBin\ffprobe.exe" -PathType Leaf)) { throw "shared FFprobe is missing" }

$env:MAPPERATORINATOR_HOME = $mapperRoot
$env:MAPPERATORINATOR_PYTHON = $mapperPython
$env:MAPPERATORINATOR_PRECISION = "fp16"
$env:FFMPEG_BIN = "$sharedFfmpegBin\ffmpeg.exe"
$env:FFMPEG_SHARED_BIN_DIR = $sharedFfmpegBin
```

로컬 RTX 2070은 `fp16`을 사용한다. `bf16`은 지수 범위가 넓어 수치적으로
안정적이지만 Turing GPU에서는 지원되지 않는다. Modal의 L4·A10·A100처럼
지원되는 GPU에서만 `MAPPERATORINATOR_PRECISION=bf16`을 선택한다.

## 생성과 벤치마크

기존 산출물을 다른 실행 결과로 오인하지 않도록 비어 있는 새 출력 폴더를 쓴다.

```powershell
uv run --project apps/chart-worker chart-worker generate `
  "C:\Users\PC\Desktop\Koe no Yukue (声の行く先) - Take 2.wav" `
  --out ".data\playtests\koe-direct" `
  --title "Koe no Yukue"
```

단계별 시간과 12개 채보 참조, 난이도 역전 경고까지 기록하려면 `bench`를 쓴다.

```powershell
uv run --project apps/chart-worker chart-worker bench `
  "C:\Users\PC\Desktop\Koe no Yukue (声の行く先) - Take 2.wav" `
  --out ".data\playtests\koe-direct-bench" `
  --title "Koe no Yukue"
```

주요 산출물은 다음과 같다.

- `raw/<key>k-<difficulty>.osu`: 검증을 통과한 Mapperatorinator 원본
- `raw/work/<variant>/attempt-1..3/`: 시도별 원문과 Hydra 로그
- `charts/*.json`: 원본 노트와 timing을 보존한 chart-v1
- `audio/game.flac`: 브라우저 재생용 정규화 음원
- `generation-report.json`: descriptor, 정밀도, 생성 시간, 노트·HOLD 수, 첫 노트와 최대 공백
- `benchmark-report.json`: 실행 요약과 읽기 전용 구조 경고
- `playtest-run-v1.json`: 프론트엔드가 읽는 실행 폴더 manifest

`generation-report.json`에는 `attemptsPerChartMax=3`, 실제 시도 횟수·seed·실패
원문, `canonicalAudioSha256`와 채보별 `timingDiagnostics`가 기록된다. 분석은
`audio/game.flac`에서 곡당 한 번 실행하고 12개 채보가 같은 onset 배열을 공유한다.
30초 단위 구간과 전체 고유 노트 행을 평가하며, 전체 ±50ms precision 70% 미만,
충분한 행이 있는 구간의 60% 미만 또는 전체 median과 25ms를 넘게 벗어난 구간은
`REVIEW`다. 노트 사이 또는 곡 앞뒤에 8초 이상 공백이 있고 그 안에 onset이 8개
이상 있으면 `coverageGaps`로 기록해 누락된 활성 구간도 `REVIEW`한다. 이는 사람
라벨 정확도가 아니라 수동 검수 신호이며 노트를 수정하지 않는다.

원본 osu!mania 키 수, X 좌표와 변환 레인 범위, `0 <= 시작 < 음원 길이`와
`HOLD 끝 <= 음원 길이`, 빈 채보, 퇴화 HOLD, 같은 레인·시각 중복, 겹친 HOLD,
정렬되고 중복 없는 양의 유한 BPM timing point를 검사한다. 실패 조합은
최대 두 번 재시도하고 세 출력이 모두 잘못되면 실행을 실패시킨다. 정상 출력의
추론 횟수는 여전히 한 번이다.

canonical `game.flac`의 SHA-256은 onset 분석 전과 외부 Mapperatorinator 생성 직후
다시 계산한다. 중간에 오디오가 바뀌면 export·manifest 작성 전에 실패한다. 채보 전체
timing 진단이 `INSUFFICIENT`여도 공개 가능한 `PASS`가 아니라 `REVIEW`로 기록한다.

## GPU 없는 구조 검사

```powershell
uv run --project apps/chart-worker chart-worker generate `
  ".\song.wav" --out ".data\fake-run" --generator fake
```

fake는 12개 파일과 manifest 계약을 검사하기 위한 테스트 대역이다. 실제 채보
품질이나 Mapperatorinator 추론 성공을 증명하지 않는다.

## 2026-08-03 실제 4K NORMAL smoke

`Koe no Yukue`와 RTX 2070 SUPER, `fp16`, seed 0으로 직접 생성한 결과다.

| 항목 | 결과 |
| --- | ---: |
| 캐시·오프라인 웜 실행 | 41.7초 |
| 최초 실행 | 12분 7초 |
| 노트 / 행 | 628 / 549 |
| HOLD | 58개 (9.24%) |
| 첫 / 마지막 노트 | 1.445초 / 181.079초 |
| 오디오 길이 | 181.227초 |
| 최대 행 공백 | 3.110초 |
| raw→chart 노트 signature | 일치 |
| raw→chart timing signature | 일치 |
| librosa onset 행 precision ±20ms | 55.92% |
| librosa onset 행 precision ±50ms | 83.24% |

최초 실행에는 차단된 Hugging Face 네트워크 확인의 재시도가 반복된 시간이 포함됐다.
모델이 이미 캐시돼 있고 오프라인 모드인 두 번째 실행은 41.7초였다. librosa 수치는
677개 자동 검출 onset과 549개 고유 노트 행을 일대일 대응한 진단값이다. ±50ms에서
median signed error는 -12ms, p95 절대 오차는 34ms였다. 사람 라벨 정확도는 아니다.

### 4K·6K·7K NORMAL 반복 실측

같은 곡과 설정으로 세 키 모드를 별도 프로세스에서 순차 실행했다. 모델 캐시를
오프라인으로 재사용했으며 각 실행에 모델 로딩 시간이 포함된다.

| 키 | 시간 | 노트/행 | HOLD | 측정 난이도 | onset ±50ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| 4K | 48.520초 | 628 / 549 | 58 (9.24%) | 2.96 NORMAL | 83.24% |
| 6K | 59.942초 | 761 / 627 | 69 (9.07%) | 3.97 HARD | 79.90% |
| 7K | 46.528초 | 671 / 545 | 89 (13.26%) | 3.52 HARD | 83.49% |
| **합계** | **154.990초** | **2,060 / 1,721** | **216** | — | — |

전체 wall time은 155.001초, 약 2분 35초였다. 세 채보 모두 마지막 노트가
181.227초 오디오 안에 있고 raw→chart 노트·timing signature가 일치했다. 같은
NORMAL 요청이어도 6K·7K 측정 결과는 HARD이므로 키 수별 requested star 보정은
아직 필요하다.

### 12개 전체 회귀

새 구조 검증·선별 재시도·공유 onset 진단을 적용한 같은 곡의 12개 전체 `fp16`
회귀는 약 17분 22초가 걸렸다. 4K EASY·HARD와 7K EXPERT는 두 번째 시도,
6K EXPERT는 세 번째 시도에 통과했고 나머지는 첫 시도에 통과했다. 전체 ±50ms
precision은 62.35~94.55%였지만, 활성 onset이 있는 시작 공백까지 검사하면 12개 중
10개가 수동 검수 대상이다. 상세 표와 오류 원문 요약은
`../../docs/현재 구현 상태와 운영 가이드.md`에 기록한다.

## 개발 검증

```powershell
uv run --project apps/chart-worker ruff check apps/chart-worker/src apps/chart-worker/tests
uv run --project apps/chart-worker pytest apps/chart-worker/tests/unit apps/chart-worker/tests/integration/test_fake_pipeline.py -q
uv build --project apps/chart-worker
```

librosa와 soundfile은 기본 `uv sync --project apps/chart-worker`에 포함된다.
Beat This, Demucs와 키음 stem은 현재 패키지의 생성 의존성이 아니다.

## Modal 배포 경로 계약

`game.flac`은 실행 패키지 안의 고정 역할명이다. 동시 요청 충돌은 파일명을 UUID로
바꾸는 대신 실행 디렉터리를 UUID `runId`로 격리해 막는다.

```text
/tmp/jobs/<runId>/audio/game.flac
/songs/<songId>/runs/<runId>/audio/game.flac
```

공유 Volume의 `/data/game.flac` 같은 한 경로에 여러 요청이 쓰면 안 된다. 이 계약은
문서와 로컬 산출물 구조에 반영됐지만 Modal 실행·object storage 업로드 어댑터는
아직 구현하지 않았다.
