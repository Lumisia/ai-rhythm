import type { KeyMode } from "../core/types";

/** 홈 포지션만 쓴다.
 *
 * 6·7키의 바깥 레인을 Shift 로 잡으면 새끼손가락이 홈 포지션에서 떨어져
 * 나가 어느 손가락 차례인지 헷갈린다. 약지·중지·검지 세 손가락을 좌우
 * 대칭으로 놓고, 7키만 가운데를 엄지로 받는다.
 */
const bindings: Record<KeyMode, ReadonlyArray<readonly [string, number]>> = {
  4: [
    ["KeyA", 0],
    ["KeyS", 1],
    ["Semicolon", 2],
    ["Quote", 3],
  ],
  6: [
    ["KeyA", 0],
    ["KeyS", 1],
    ["KeyD", 2],
    ["KeyL", 3],
    ["Semicolon", 4],
    ["Quote", 5],
  ],
  7: [
    ["KeyA", 0],
    ["KeyS", 1],
    ["KeyD", 2],
    ["Space", 3],
    ["KeyL", 4],
    ["Semicolon", 5],
    ["Quote", 6],
  ],
};

export function bindingsFor(keyMode: KeyMode): Map<string, number> {
  return new Map(bindings[keyMode]);
}

const labels: Record<string, string> = {
  Semicolon: ";",
  Quote: "'",
  Space: "␣",
  KeyA: "A",
  KeyS: "S",
  KeyD: "D",
  KeyL: "L",
};

/** 레인 순서대로 눌러야 할 키. 판정선 아래에 띄운다.
 *
 * 화면에 없으면 테스터가 손을 어디 올릴지 알 수 없다. */
export function keyLabelsFor(keyMode: KeyMode): string[] {
  const byLane: string[] = [];
  for (const [code, lane] of bindings[keyMode]) {
    byLane[lane] = labels[code] ?? code;
  }
  return byLane;
}
