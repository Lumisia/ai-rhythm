# ai-rhythm

AI 음악에서 4·6·7키 채보를 자동 생성하는 웹 리듬게임.

---

## 키 모드

```text
4키   A  S  ;  '
6키   ShiftLeft  A  S  ;  '  ShiftRight
7키   ShiftLeft  A  S  Space  ;  '  ShiftRight
```

DJMAX 계열 사이드 트랙 구조. 외곽 Shift는 SideTrack, 중앙 Space는 FX에 대응한다.

난이도는 `EASY / NORMAL / HARD / EXPERT` 4단계.

---

## 구성

```text
apps/backend/        Spring Boot 3 / Java 21   권한·작업큐·상태 기준 시스템
apps/frontend/       React + TS + Vite + Phaser
apps/chart-worker/   Python 3.11 + uv          Mapperatorinator · Demucs · 후처리
packages/            chart-schema · judgment · api-contracts
infra/compose/       docker-compose (postgres)
```

### 도구 역할

```text
생성   Mapperatorinator V32 단독 — 노트를 만드는 유일한 주체
분석   Beat This! / librosa / Demucs — 노트를 만들지 않음
저장   PostgreSQL + Flyway, StoragePort(로컬 / OCI S3 호환)
```

---

## 핵심 설계 원칙

**타이밍 불변** — 후처리는 노트의 `timeMs`를 절대 바꾸지 않는다.
레인 배치와 노트 취사선택만 조작한다.

근거: Mapperatorinator의 노트–드럼 onset 평균 오차가 8.2~10.8ms로
룰 기반(19.1~23.6ms)의 2배 이상 정확하다. 그 정확도가 채택 이유의 전부다.

**작업 큐는 Postgres다** — `FOR UPDATE SKIP LOCKED` + lease/heartbeat.
Redis·Kafka를 도입하지 않는다.

**`.osu` 원본을 영구 보관한다** — 후처리 파라미터를 바꿔 12개 채보를 재생성하는 데
GPU 비용이 0이다. 난이도 solver와 6·7키 규칙 튜닝이 이 반복에 달렸다.

---

## 로컬 실행

```bash
docker compose -f infra/compose/docker-compose.yml up -d
./gradlew -p apps/backend bootRun --args='--spring.profiles.active=local'
uv run --project apps/chart-worker chart-worker serve
npm --prefix apps/frontend run dev
```

GPU 없이 개발하려면:

```bash
CHART_GENERATOR=fake
```

---

## 문서

설계·조사 문서는 `docs/` 에 있으며 git에 포함하지 않는다.

```text
docs/design/     Phase1~5 확정 설계
docs/research/   사전 조사와 로컬 프로토타입 실측 기록
```
