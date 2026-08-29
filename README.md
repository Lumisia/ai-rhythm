# ai-rhythm

> 기준일 2026-08-17 · 브랜치 `feat/chart-worker-foundation`
> 이 문서는 현재 코드와 일치해야 한다. 어긋나면 코드가 옳다.

Mapperatorinator V32로 4·6·7키 채보를 생성하고 브라우저에서 직접 연주하며
검수하는 리듬게임 프로젝트다. 현재 기본 생성 방식은 Mapperatorinator의 timing과
공유 timing을 기준으로 MAP을 만들고, 구조 검증을 통과한 원본 노트를 그대로 내보내는
선별 재시도 방식이다.

기본 로컬 실행은 one-shot이다. 선택적 곡 단위 상주 세션은 같은 곡의 여러 추론에서
모델과 encoder cache를 재사용하며, v30 opt-in generation telemetry가 실제 output/input
hash와 decode/cache/HOLD/tail 통계를 기록한다. 20-window 단일 GPU 실측에서는 출력
bytes를 유지한 채 warm cache 호출이 15.3952초 one-shot 대비 4.7340초였지만, 이를
전체 곡 33분이나 장르 일반화의 증거로 외삽하지 않는다.

## 현재 생성 흐름

```text
원본 오디오
  → 공유 FFmpeg로 game.flac 정규화
  → onset·활성도 분석과 곡 공통 timing authority 생성
  → Mapperatorinator V32 MAP 추론 (4·6·7K × 4난이도)
  → 최대 10ms 종료 경계 오차를 canonical 오디오 안으로 정규화
  → incremental HOLD 문법·시간 horizon과 beat-aware 커버리지 검증
  → 필요한 조합만 suffix repair/선별 재시도
  → 그 밖의 노트 변경 없이 chart-v1과 검수 보고서 내보내기
```

룰 기반 채보, Beat This timing 강제, 난이도 solver,
레인 재배치, 목표 롱노트 비율 보정과 Demucs 키음은 기본 생성 경로에서 사용하지
않는다. Super Timing은 timing 구조 실패·tempo 증거 충돌처럼 필요한 경우에만 한 번
사용하고, 명백한 MAP 결함이나 난이도 역전이 있는 조합만 최대 세 번까지 생성한다.
librosa 분석은 검수 근거일 뿐 채보를 수정하지 않는다.

정상 발행 후보가 없는 변형에 기존 hard-safe Mapperatorinator 원본이 있으면
`diagnostic-raw-fallback/`에 `PLAYTEST_ONLY`로 격리할 수 있다. 이 출력은
`productionEligible=false`이며 정상 성공률이나 packager 입력에 포함하지 않는다.

현재 chart-worker의 정식 플레이테스트 경로는 admitted 모델 후보가 없을 때도
hard invariant를 통과한 `RAW_UNVERIFIED`를 우선 사용하고, 그것도 없으면 canonical
timing/onset에서 `SAFE_FALLBACK`을 결정론적으로 만든다. 두 경우 모두 12개 슬롯을
플레이할 수 있게 내보내되 `PLAYTEST_ONLY`, `productionEligible=false`를 manifest에
명시한다. 구조가 깨진 모델 출력을 그대로 사용자에게 주는 정책은 아니다.

### 난이도 descriptor

| 난이도 | descriptor |
| --- | --- |
| EASY | `expression/simple` |
| NORMAL | `style/mixed rice` |
| HARD | `style/mixed rice`, `streams/bursts` |
| EXPERT | `style/mixed rice`, `skillset/streams` |

`skillset/tech`는 EXPERT 과밀도 A/B 이후 2026-08-09에 제거했다.

timing 요청은 `output_type=[TIMING]`, MAP 요청은 `output_type=[MAP]`과
`in_context=[TIMING]`을 사용한다. 모든 MAP은 `cfg_scale=1.0`, `hitsounded=false`와
정규화 오디오의 `end_time`을 사용한다. 롱노트 비율은 강제하지 않는다.

## 키 모드

```text
4키   A  S  ;  '
6키   A  S  D  L  ;  '
7키   A  S  D  Space  L  ;  '
```

## 구성

현재 존재하는 것만 적는다.

```text
apps/chart-worker/       Python 3.11 + uv                직접 채보 생성·검증·내보내기
apps/frontend/           React + TypeScript + Phaser     로컬 플레이테스터
packages/chart-schema/   chart-v1 · playtest-run · boundary-label JSON Schema
packages/judgment/       판정 상수
```

프론트엔드는 구형 프로토타입의 화면 구성·게임 HUD·메뉴 디자인만 이식했다.
현재 React/Phaser 게임 로직, 로컬 실행 폴더 계약, 리뷰 마커와 키 배열은 유지한다.
구형 정적 게임 코드, 음원과 룰 채보는 포함하지 않는다.

## 로컬 사용

GPU 없이 12개 산출물 구조를 확인하려면:

```powershell
uv run --project apps/chart-worker chart-worker generate .\song.wav --out .data\fake-run --generator fake
```

실제 Mapperatorinator 실행 방법은 `apps/chart-worker/README.md`에 있다. 생성된
폴더를 플레이하려면 프론트엔드를 실행하고 브라우저에서 그 폴더를 선택한다.

```powershell
npm --prefix apps/frontend run dev
```

브라우저는 로컬 파일을 서버로 업로드하지 않는다. 배포판에서는 서버가 생성한
동일한 manifest·오디오·채보 계약을 URL로 제공하도록 연결할 예정이다.

