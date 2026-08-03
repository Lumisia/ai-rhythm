# ai-rhythm

Mapperatorinator V32로 4·6·7키 채보를 생성하고 브라우저에서 직접 연주하며
검수하는 리듬게임 프로젝트다. 현재 기본 생성 방식은 Mapperatorinator의 timing과
MAP을 그대로 내보내는 단일 후보 방식이다.

## 현재 생성 흐름

```text
원본 오디오
  → 공유 FFmpeg로 game.flac 정규화
  → Mapperatorinator V32 직접 추론 (4·6·7K × 4난이도, 각 1회)
  → 원본 .osu의 timing·노트 파싱
  → 노트 변경 없이 chart-v1과 검수 보고서 내보내기
```

룰 기반 채보, Beat This timing 강제, super timing, 후보 재생성, 난이도 solver,
레인 재배치, 목표 롱노트 비율 보정과 Demucs 키음은 기본 생성 경로에서 사용하지
않는다. librosa는 필요할 때 생성 뒤 실행하는 선택적 onset 진단일 뿐 채보를
수정하지 않는다.

### 난이도 descriptor

| 난이도 | descriptor |
| --- | --- |
| EASY | `expression/simple` |
| NORMAL | `style/mixed rice` |
| HARD | `style/generic hybrid` |
| EXPERT | `tech/technical hybrid` |

모든 조합은 `cfg_scale=1.0`, `output_type=[TIMING,MAP]`, `hitsounded=false`와
정규화 오디오의 `end_time`을 사용한다. 롱노트 비율은 강제하지 않는다. 로컬 RTX 2070 기본 정밀도는 `fp16`이며,
지원 GPU를 쓰는 배포 환경은 `MAPPERATORINATOR_PRECISION=bf16`을 선택할 수 있다.

## 키 모드

```text
4키   A  S  ;  '
6키   A  S  D  L  ;  '
7키   A  S  D  Space  L  ;  '
```

입력은 `KeyboardEvent.code` 기준이라 한글·영문 입력 모드와 무관하다. 6키와
7키는 Shift를 사용하지 않는다.

## 구성

```text
apps/backend/        Spring Boot 3 / Java 21   권한·작업 큐·상태 관리
apps/frontend/       React + TypeScript + Phaser 로컬 플레이테스터
apps/chart-worker/   Python 3.11 + uv          직접 채보 생성·내보내기
packages/            chart-schema · judgment · api-contracts
infra/compose/       PostgreSQL 로컬 구성
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

## 문서

`docs/`는 코드와 함께 버전 관리한다. 현재 기준은 `docs/README.md`와
`docs/현재 구현 상태와 운영 가이드.md`이며, `docs/superpowers/`는 설계 결정과
구현 계획 이력이다. `.understand-anything/`은 계속 Git에서 제외한다.
