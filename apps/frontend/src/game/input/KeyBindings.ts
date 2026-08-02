import type { KeyMode } from "../core/types";

const bindings: Record<KeyMode, ReadonlyArray<readonly [string, number]>> = {
  4: [
    ["KeyA", 0],
    ["KeyS", 1],
    ["Semicolon", 2],
    ["Quote", 3],
  ],
  6: [
    ["ShiftLeft", 0],
    ["KeyA", 1],
    ["KeyS", 2],
    ["Semicolon", 3],
    ["Quote", 4],
    ["ShiftRight", 5],
  ],
  7: [
    ["ShiftLeft", 0],
    ["KeyA", 1],
    ["KeyS", 2],
    ["Space", 3],
    ["Semicolon", 4],
    ["Quote", 5],
    ["ShiftRight", 6],
  ],
};

export function bindingsFor(keyMode: KeyMode): Map<string, number> {
  return new Map(bindings[keyMode]);
}

const labels: Record<string, string> = {
  ShiftLeft: "⇧",
  ShiftRight: "⇧",
  Semicolon: ";",
  Quote: "'",
  Space: "␣",
  KeyA: "A",
  KeyS: "S",
};

/** 레인 순서대로 눌러야 할 키. 판정선 아래에 띄운다.
 *
 * 7키 배열은 ShiftLeft·A·S·Space·;·'·ShiftRight 인데 화면에 없으면
 * 테스터가 손을 어디 올릴지 알 수 없다. */
export function keyLabelsFor(keyMode: KeyMode): string[] {
  const byLane: string[] = [];
  for (const [code, lane] of bindings[keyMode]) {
    byLane[lane] = labels[code] ?? code;
  }
  return byLane;
}
