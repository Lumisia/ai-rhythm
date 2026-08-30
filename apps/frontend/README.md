# 로컬 채보 플레이테스터

이 앱은 chart-worker의 `playtest-run-v1.json`, `playtest-run-v2.json` 또는
`playtest-run-v3.json`이 정확히 하나 있는 실행 폴더를 브라우저 안에서 읽어 수동
플레이하는 React/Phaser 도구다. 로컬 폴더의 파일을 서버로 업로드하지 않는다.

V3만 신규 production 검증 대상이다. V3는 schema·보고서 SHA·상태 snapshot·전역
publication을 독립 검증하고 `ALLOW_PRODUCTION`일 때만 production으로 연다. V2는
내부 무결성을 검사한 뒤에도 과거 chart-level 권한을 신뢰하지 않고
`LEGACY_V2_PLAYTEST_ONLY`로만 연다. V1은 `LEGACY_UNVERIFIED` playtest 전용이다.

구형 프로토타입에서는 화면 구성, 중앙 메뉴 카드, 게임 HUD, 색과 진행 표시만
이식했다. 현재 실행 폴더 검증, Web Audio 시계, 판정 엔진, 리뷰 마커와 키 입력은
그대로 유지하며 구형 `game.js`, 음원과 룰 채보 JSON은 포함하지 않는다.


폴더 선택은 로컬 프로토타입용이다. 실제 배포에서는 서버가 준비한 동일한
manifest·오디오·채보 계약을 URL로 불러오는 어댑터를 연결해야 하며, 최종 사용자가
실행 폴더를 직접 올리게 하지 않는다.

재생 시작 전 기본 1.8초 프리롤 동안 chart clock은 약 -1800ms에서 0ms로 진행한다.
오디오는 1.8초 뒤 source offset 0부터 시작하므로 `timeMs=0` 노트도 미리 내려오며,
음원이나 채보 시각을 1.8초 밀지 않는다.

## 키 배열

| 키 모드 | 배열 |
| --- | --- |
| 4K | `A` `S` `;` `'` |
| 6K | `A` `S` `D` `L` `;` `'` |
| 7K | `A` `S` `D` `Space` `L` `;` `'` |

6K와 7K는 Shift를 사용하지 않는다. 입력은 `KeyboardEvent.code` 기준이므로
한글·영문 입력 모드가 달라도 물리 키는 같다. 7K는 6K 가운데에 Space를 추가한
배열이다. 창이 포커스를 잃으면 눌린 레인을 입력·판정·화면에서 모두 해제한다.

`ESC`로 일시정지한다. 플레이 중 숫자 키로 문제 구간을 기록한다.

- `1`: 박자
- `2`: 난이도
- `3`: 레인
- `4`: 연타
- `5`: 동시치기
- `6`: 홀드
- `7`: 노트 누락
- `8`: 불필요 노트

## HUD와 수동 검수

플레이 화면 상단은 SCORE·ACCURACY·MAX COMBO를, 우측은 판정별 수와 평균
오차를 표시한다. 판정선 아래 타이밍 스코프의 중앙은 0ms이며 최근 타격과 평균
오차가 남는다. 하단 청록–보라 레일은 곡 진행과 리뷰 마커 위치를 표시한다.

채보 타이밍을 평가하기 전 입력 보정을 먼저 맞춘다.

1. NORMAL 4K를 약 30초 연주한다.
2. 타이밍 스코프가 한쪽으로 쏠렸는지 확인한다.
3. 화면의 권장 보정값을 입력 보정에 반영한다.
4. 다시 연주해 점들이 0ms 근처에 모이는지 확인한다.
5. NORMAL 6K·7K와 각 키 모드의 EASY→HARD→EXPERT를 비교한다.

입력 지연을 보정하지 않은 판정 결과만으로 채보 timing이 틀렸다고 결론 내리지
않는다. HOLD 수와 비율은 목표값을 강제한 결과가 아니므로, 곡의 지속음과 실제
연주감에 맞는지를 리뷰 마커와 결과 JSON으로 남긴다.

## 개발 검증

```powershell
npm --prefix apps/frontend test
npm --prefix apps/frontend run build
```
