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
