import { describe, expect, it, vi } from "vitest";

import type { Clock } from "../audio/Clock";
import { InputRecorder } from "../core/InputRecorder";
import type { JudgmentEngine, JudgmentEvent } from "../core/JudgmentEngine";
import { bindingsFor } from "./KeyBindings";
import { KeyboardInput } from "./KeyboardInput";

describe("bindingsFor", () => {
  it("maps the seven-key layout by physical KeyboardEvent.code", () => {
    expect(bindingsFor(7)).toEqual(
      new Map([
        ["ShiftLeft", 0],
        ["KeyA", 1],
        ["KeyS", 2],
        ["Space", 3],
        ["Semicolon", 4],
        ["Quote", 5],
        ["ShiftRight", 6],
      ]),
    );
  });
});

function setup() {
  const target = new EventTarget();
  const clock: Clock = {
    songTimeMs: () => 0,
    performanceToSongTimeMs: (performanceMs) => performanceMs + 100,
  };
  const judgment: JudgmentEvent = {
    noteId: 1,
    lane: 3,
    noteType: "TAP",
    phase: "HEAD",
    judgment: "PERFECT",
    errMs: 0,
    timeMs: 1100,
    noteTimeMs: 1100,
  };
  const keyDown = vi.fn(() => judgment);
  const keyUp = vi.fn(() => null);
  const engine = { keyDown, keyUp } as unknown as JudgmentEngine;
  const recorder = new InputRecorder();
  const onJudgment = vi.fn();
  const input = new KeyboardInput({
    target,
    bindings: bindingsFor(7),
    clock,
    engine,
    recorder,
    onJudgment,
  });
  input.attach();
  return { target, input, keyDown, keyUp, recorder, onJudgment };
}

describe("KeyboardInput", () => {
  it("ignores repeated keydown without touching engine or recorder", () => {
    const { target, keyDown, recorder } = setup();
    target.dispatchEvent(new KeyboardEvent("keydown", { code: "KeyA", repeat: true }));
    expect(keyDown).not.toHaveBeenCalled();
    expect(recorder.snapshot()).toEqual([]);
  });

  it("prevents Space default and sends calibrated down/up times", () => {
    const { target, keyDown, keyUp, recorder, onJudgment } = setup();
    const down = new KeyboardEvent("keydown", {
      code: "Space",
      cancelable: true,
    });
    Object.defineProperty(down, "timeStamp", { value: 1000 });
    expect(target.dispatchEvent(down)).toBe(false);
    const up = new KeyboardEvent("keyup", { code: "Space", cancelable: true });
    Object.defineProperty(up, "timeStamp", { value: 1020 });
    target.dispatchEvent(up);

    expect(keyDown).toHaveBeenCalledWith(3, 1100);
    expect(keyUp).toHaveBeenCalledWith(3, 1120);
    expect(recorder.snapshot()).toEqual([
      { action: "DOWN", lane: 3, code: "Space", timeMs: 1100 },
      { action: "UP", lane: 3, code: "Space", timeMs: 1120 },
    ]);
    expect(onJudgment).toHaveBeenCalledTimes(1);
  });

  it("detaches listeners and ignores unmapped keys", () => {
    const { target, input, keyDown } = setup();
    target.dispatchEvent(new KeyboardEvent("keydown", { code: "KeyZ" }));
    input.detach();
    target.dispatchEvent(new KeyboardEvent("keydown", { code: "KeyA" }));
    expect(keyDown).not.toHaveBeenCalled();
  });
});
