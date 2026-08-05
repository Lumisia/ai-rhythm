# chart-worker 직접 생성과 로컬 검증

`chart-worker`는 한 곡에서 4K·6K·7K와 EASY·NORMAL·HARD·EXPERT 조합
12개를 생성한다. Mapperatorinator가 만든 timing, 레인, TAP/HOLD 종류와 길이를
후처리로 바꾸지 않는다. `generate`와 `bench` 명령을 실행하는 동안에만 모델이
동작하며 상시 GPU 서비스가 아니다.

## 생성 계약

곡마다 표준 timing을 한 번 생성하고, timing 구조 검증이 실패할 때만 Super Timing을
한 번 더 시도한다. 검증된 기준은 `audio/timing-reference.osu`에 저장한다. 그 뒤
4K·6K·7K와 네 난이도 조합의 MAP 12개가 모두 같은 reference를 재사용한다. 각 후보는
안정된 raw 경로로 승격되기 전에 `STRUCTURE`, `TIMING_IDENTITY`, `TIMING_ALIGNMENT`,
`COVERAGE` 네 축에서 독립적으로 판정된다. 전체 우선순위는
`RETRY_MAP > REVIEW > PASS`다. 구조 오류나 충분한 근거가 함께 있는 MAP 결함처럼
명확한 `RETRY_MAP`만 다음 seed를 소비하며, 한 조합당 총 세 번까지 시도한다. 약하거나
모호한 근거인 `REVIEW`는 새 seed를 소비하지 않고 실행을 보류한다. 이미 성공한 조합은
다시 만들지 않는다.

| 난이도 | 요청 별 | descriptor |
| --- | ---: | --- |
| EASY | 1.0 | `expression/simple` |
| NORMAL | 1.5 | `style/mixed rice` |
| HARD | 2.0 | `style/mixed rice` |
| EXPERT | 2.75 | `style/mixed rice` |

timing 요청은 `output_type=[TIMING]`이고, MAP 요청은 `output_type=[MAP]`,
`beatmap_path=audio/timing-reference.osu`, `in_context=[TIMING]`을 사용한다. MAP의 공통
설정은 `cfg_scale=1.0`, `hitsounded=false`, `fast_decoder_loop=true`다.
`hold_note_ratio`를 보내지 않으므로 롱노트는 모델이 곡과 패턴에 맞게 정한다. 룰 기반
채보, Beat This timing, 정상 채보의 다중 seed 후보 경쟁, solver와 레인 재배치는
실행하지 않는다.

각 MAP 시도 직전에 timing reference SHA-256과 canonical audio SHA-256 identity를
검사한다. 생성된 MAP의 BPM event가 reference와 하나라도 다르면 그 MAP만 재시도하고
안정 경로로 승격하지 않는다. BPM event나 노트 시각·레인·종류·HOLD 길이를 reference에
맞추기 위해 재작성하지 않는다. timing authority 검토는 full-beat 기준의 base·half·double
pulse 증거와 onset-envelope autocorrelation이 같은 대안을 지지하는지 확인한다. 노트 행의
grid subdivision 정렬은 이 tempo 검토와 별개의 검증 신호다.

`end_time`에는 정규화 오디오 길이를 전달한다. Mapperatorinator가 패딩 구간에
오디오 밖 노트를 만드는 것을 모델 자체 크롭 단계에서 막고, 내보내기 단계는
그 결과를 다시 수정하지 않는다. Hydra 실행 로그도 각 조합 출력의 `.hydra-run`
아래에 두므로 Mapperatorinator 체크아웃이 읽기 전용이어도 실행할 수 있다.

## 실제 Mapperatorinator 실행 준비

먼저 프로젝트가 고정한 Mapperatorinator commit에 keycount 출력 제약 패치를 한 번
적용한다. 이미 적용된 상태에서는 그대로 성공한다.

```powershell
Set-Location apps\chart-worker
.\.venv\Scripts\python.exe scripts\apply_mapperatorinator_patch.py C:\Users\PC\mapperatorinator
```

worker는 실행 전에 commit과 패치 상태를 확인한다. 다른 commit, 일부만 적용된 패치,
미적용 상태에서는 추론을 시작하지 않는다. 이 제약은 4K·6K·7K에서 요청 범위 밖
`MANIA_COLUMN` 토큰만 차단하며 timing, TAP/HOLD, descriptor는 변경하지 않는다.

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

- `raw/<key>k-<difficulty>.osu`: 네 축이 모두 `PASS`한 Mapperatorinator 원본
- `raw/work/<variant>/attempt-1..3/`: 시도별 진단 작업 위치. 생성기가 만든 경우 원문과 Hydra 로그를 보존
- `charts/*.json`: 전체 실행이 공개 가능할 때 원본 노트와 timing을 보존한 chart-v1
- `audio/game.flac`: 브라우저 재생용 정규화 음원
- `audio/timing-reference.osu`: 12개 MAP이 공유하는 SHA-256 고정 timing 기준
- `generation-report.json`: 성공 또는 보류 결과와 품질 게이트 근거
- `benchmark-report.json`: 실행 요약과 읽기 전용 구조 경고
- `playtest-run-v1.json`: 프론트엔드가 읽는 실행 폴더 manifest

성공한 `generation-report.json`은 `qualityGateVersion=quality-gate-v1`,
`publishable=true`, `status=PASS`와 채보별 acceptance status·reason·축별 결정을 기록한다.
`REVIEW` 또는 재시도 소진으로 보류된 실행도 같은 위치에 보고서를 남긴다. 이때
`publishable=false`, `status=REVIEW` 또는 `EXHAUSTED`, error code/context,
`canonicalAudioSha256`와 timing authority identity를 기록한다. 생성기가 후보 원문이나
로그를 만든 경우에는 진단 작업 위치인 `raw/work/<variant>/attempt-*`에 보존한다.
작업 산출물의 존재 여부와 관계없이 거절된 후보는 안정된 `raw/<variant>.osu`에 도달하지
않는다. 보류된 실행은 export하지 않으며 `charts/`와 `playtest-run-v1.json`도 만들지 않는다.

onset과 활성 구간은 canonical `audio/game.flac`의 `OnsetAnalysis`에서 곡당 한 번 계산하고
12개 채보가 공유한다. nearest-onset ±20ms·±50ms 비율, 15초 구간, coverage metric도
후보 acceptance에 한 번 기록하며 보고서는 이를 다시 계산하지 않고 직렬화한다. 이 비율은
자동 proxy 진단이다. 예를 들어 70%는 사람이 라벨링한 Mapperatorinator 정확도가 아니다.
활성 오디오 근거가 충분한 8초 이상 공백은 `ACTIVE_*_GAP`으로 `RETRY_MAP`이 된다.
조용하거나 활성도가 낮은 공백은 `QUIET_*_GAP`, onset·구간·grid 근거가 약한 경우는
그에 해당하는 reason과 함께 `REVIEW`가 되며 새 seed를 소비하지 않는다. 이 자동 게이트는
사람의 음악적 정답이나 주관적 품질을 보증하지 않고 노트를 수정하지 않는다. 난이도 체감,
HOLD 양, 패턴의 음악성은 수동 플레이테스트로 계속 확인해야 한다.

원본 osu!mania 키 수, X 좌표와 변환 레인 범위, `0 <= 시작 < 음원 길이`와
`HOLD 끝 <= 음원 길이`, 빈 채보, 퇴화 HOLD, 같은 레인·시각 중복, 겹친 HOLD,
정렬되고 중복 없는 양의 유한 BPM timing point를 검사한다. 실패 조합은
명확한 MAP 결함일 때만 최대 두 번 재시도하고 세 출력이 모두 잘못되면 실행을
`EXHAUSTED`로 보류한다. 정상 출력의 추론 횟수는 여전히 한 번이다.

canonical `game.flac`의 SHA-256은 onset 분석 전과 외부 Mapperatorinator 생성 직후
다시 계산한다. 중간에 오디오가 바뀌면 export·manifest 작성 전에 실패한다. 채보 전체
timing 진단이 `INSUFFICIENT`여도 공개 가능한 `PASS`가 아니라 `REVIEW`로 기록하고
export를 보류한다.

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
precision은 62.35~94.55%였다. 이 수치는 상대 RMS 분류를 추가하기 전 raw onset
공백 판정으로 얻은 과거 기준이다. 당시와 달리 현재 `quality-gate-v1`은 조용하거나
활성도가 낮은 공백도 `REVIEW` 근거로 보존한다. 상세 표와 최신 재진단 결과는
`../../docs/현재 구현 상태와 운영 가이드.md`에 기록한다.

## 2026-08-04 keycount 패치 smoke

저장된 `Ignite the Pulse` canonical audio로 4K NORMAL 한 장만 `fp16`, seed 0에서
생성했다. 첫 시도 약 53초에 556노트(HOLD 106개)를 만들었고 raw X는 64~448,
변환 레인은 0~3, 범위 밖 레인은 0개였다. HOLD 겹침·퇴화, 오디오 범위와 BPM을
포함한 구조 검증도 통과했다. 사용 패치는 `mania-keycount-v1`이다.

## 개발 검증

```powershell
uv run --project apps/chart-worker ruff check apps/chart-worker/src apps/chart-worker/tests apps/chart-worker/scripts
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

향후 Modal 이미지에서도 모델 추론 전에 같은 적용기를 이미지 구성 단계에서 실행한다.
이미지의 실제 설치 경로에 맞추되 의미는 다음과 같다.

```dockerfile
RUN /app/apps/chart-worker/.venv/bin/python \
    /app/apps/chart-worker/scripts/apply_mapperatorinator_patch.py \
    /opt/mapperatorinator
```
