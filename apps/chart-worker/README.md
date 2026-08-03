# chart-worker 직접 생성과 로컬 검증

`chart-worker`는 한 곡에서 4K·6K·7K와 EASY·NORMAL·HARD·EXPERT 조합
12개를 생성한다. Mapperatorinator가 만든 timing, 레인, TAP/HOLD 종류와 길이를
후처리로 바꾸지 않는다. `generate`와 `bench` 명령을 실행하는 동안에만 모델이
동작하며 상시 GPU 서비스가 아니다.

## 생성 계약

각 조합은 한 번만 추론한다.

| 난이도 | 요청 별 | descriptor |
| --- | ---: | --- |
| EASY | 1.0 | `expression/simple` |
| NORMAL | 1.5 | `style/mixed rice` |
| HARD | 3.5 | `style/generic hybrid` |
| EXPERT | 4.5 | `tech/technical hybrid` |

공통 요청은 `cfg_scale=1.0`, `output_type=[TIMING,MAP]`,
`hitsounded=false`, `fast_decoder_loop=true`다. `hold_note_ratio`를 보내지 않으므로
롱노트는 모델이 곡과 패턴에 맞게 정한다. 룰 기반 채보, Beat This timing,
super timing, 다중 seed 후보, solver와 레인 재배치는 실행하지 않는다.

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

- `raw/<key>-<difficulty>/...osu`: Mapperatorinator 원본
- `charts/*.json`: 원본 노트와 timing을 보존한 chart-v1
- `audio/game.flac`: 브라우저 재생용 정규화 음원
- `generation-report.json`: descriptor, 정밀도, 생성 시간, 노트·HOLD 수, 첫 노트와 최대 공백
- `benchmark-report.json`: 실행 요약과 읽기 전용 구조 경고
- `playtest-run-v1.json`: 프론트엔드가 읽는 실행 폴더 manifest

보고서의 노트 수·공백·난이도는 원본을 설명할 뿐 결과를 수정하거나 실패로
바꾸지 않는다. 사람 라벨이 없는 onset 일치율을 “정확도”로 단정하지 않는다.
librosa 비교가 필요하면 `diagnostics` extra를 설치해 생성 뒤 별도 진단으로
실행하며, 생성 경로와 성공 여부에는 연결하지 않는다.

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

## 개발 검증

```powershell
uv run --project apps/chart-worker ruff check apps/chart-worker/src apps/chart-worker/tests
uv run --project apps/chart-worker pytest apps/chart-worker/tests/unit apps/chart-worker/tests/integration/test_fake_pipeline.py -q
uv build --project apps/chart-worker
```

선택 진단 의존성은 `uv sync --project apps/chart-worker --extra diagnostics`로
설치한다. Beat This, Demucs와 키음 stem은 현재 패키지의 생성 의존성이 아니다.
