# ai-rhythm

AI 음악에서 4·6·7키 채보를 자동 생성하는 웹 리듬게임.

> 기준일 2026-08-14
> **코드는 `feat/chart-worker-foundation` 브랜치에 있다.** 이 `main` 브랜치는
> `.gitignore`와 이 README만 추적한다. 아래 구성·실행법은 그 브랜치를 기준으로 한다.

### Mapperatorinator 출처

실제 채보 생성은 [OliBomby/Mapperatorinator](https://github.com/OliBomby/Mapperatorinator)를
기반으로 한다. 이 프로젝트는 upstream의
[`2a70eb89004da20e39b0fcbaad2686b264d5a040`](https://github.com/OliBomby/Mapperatorinator/commit/2a70eb89004da20e39b0fcbaad2686b264d5a040)
커밋을 고정해서 사용하며, 프로젝트 전용 변경은 기능 브랜치의 패치 스택으로 관리한다.
Mapperatorinator의 원 저작권자는 OliBomby이며 MIT License로 배포된다.

---

## 키 모드

```text
4키   A  S  ;  '
6키   A  S  D  L  ;  '
7키   A  S  D  Space  L  ;  '
```

난이도는 `EASY / NORMAL / HARD / EXPERT` 4단계. 곡 하나당 `3키모드 × 4난이도 = 12개`
채보를 생성한다.

---

## 구성

현재 존재하는 것.

```text
apps/chart-worker/       Python 3.11 + uv               채보 생성·검증·내보내기
apps/frontend/           React + TS + Vite + Phaser     로컬 플레이테스터
packages/chart-schema/   chart-v1 · playtest-run-v2 · boundary-label JSON Schema
packages/judgment/       판정 상수
```

아직 만들지 않은 것.

```text
apps/backend/            Spring Boot 작업 큐·권한·상태 시스템
infra/compose/           PostgreSQL docker-compose
packages/api-contracts/
```

`docs/design/Phase1~5`는 위 백엔드 구성을 전제한 확정 설계이지만, 현재 구현은
로컬 CLI 실행과 브라우저 플레이테스터만으로 동작한다. 설계와 구현이 다른 지점은
`feat/chart-worker-foundation` 브랜치의 `docs/README.md`가 정리한다.

### 도구 역할

```text
생성   Mapperatorinator V32 단독 — 노트를 만드는 유일한 주체
분석   librosa (onset·활성도)      — 노트를 만들지 않음
보조   Beat This!                  — 내부 timing 후보가 외부 근거를 요구할 때만 호출
```

Demucs와 키음 stem은 현재 생성 경로에서 사용하지 않는다. PostgreSQL·Flyway·
StoragePort는 백엔드와 함께 미구현이다.

---

## 핵심 설계 원칙

**타이밍 불변** — 후처리는 노트의 `timeMs`를 바꾸지 않는다.

근거: Mapperatorinator의 노트–드럼 onset 평균 오차가 8.2~10.8ms로
룰 기반(19.1~23.6ms)의 2배 이상 정확하다. 그 정확도가 채택 이유의 전부다.

현재 구현은 여기서 더 나아가 **노트를 전혀 변형하지 않는다**(`noteMutationEnabled=false`).
레인 재배치, 난이도 solver에 의한 노트 삭제, 목표 HOLD 비율을 위한 TAP/HOLD 변환은
모두 실행하지 않는다. 예외는 어댑터의 최대 10ms 종료 경계 정규화 하나뿐이다.
품질 문제는 노트를 고쳐서가 아니라 **다른 seed로 다시 생성해서** 해결한다.

**`.osu` 원본을 영구 보관한다** — 후처리 파라미터를 바꿔 재생성하는 데 GPU 비용이 0이다.

**작업 큐는 Postgres로 한다** — `FOR UPDATE SKIP LOCKED` + lease/heartbeat.
Redis·Kafka를 도입하지 않는다. *(설계 결정이며 아직 구현하지 않았다.)*

---

## 로컬 실행

GPU 없이 12개 산출물 구조만 확인:

```bash
uv run --project apps/chart-worker chart-worker generate ./song.wav --out .data/fake-run --generator fake
```

실제 생성에는 고정 commit의 Mapperatorinator 체크아웃과 패치 스택 적용이 필요하다.
절차는 `apps/chart-worker/README.md`에 있다.

생성한 폴더를 브라우저에서 플레이:

```bash
npm --prefix apps/frontend run dev
```

브라우저는 로컬 파일을 서버로 업로드하지 않는다.

`chart-worker` CLI 명령은 `generate`, `bench`, `recalculate-difficulty`,
`migrate-boundary-review` 네 개다. 상시 서비스 모드는 없다.

