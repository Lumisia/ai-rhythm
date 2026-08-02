import { describe, expect, it, vi } from "vitest";

import type { Clock } from "../audio/Clock";
import { InputRecorder } from "../core/InputRecorder";
import type { JudgmentEngine, JudgmentEvent } from "../core/JudgmentEngine";
import { bindingsFor, keyLabelsFor } from "./KeyBindings";
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

describe("keyLabelsFor", () => {
  it("returns one printable label per lane in lane order", () => {
    expect(keyLabelsFor(7)).toEqual(["⇧", "A", "S", "␣", ";", "'", "⇧"]);
    expect(keyLabelsFor(4)).toEqual(["A", "S", ";", "'"]);
  });

  it("covers every bound lane so no lane renders blank", () => {
    for (const keyMode of [4, 6, 7] as const) {
      const labels = keyLabelsFor(keyMode);
      expect(labels).toHaveLength(keyMode);
      expect(labels.every((label) => label.length > 0)).toBe(true);
    }
  });
});

function setup() {
  const target = new EventTarget();
  const clock: Clock = {
    songTimeMs: () => 7000,
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
  const onLaneDown = vi.fn();
  const onLaneUp = vi.fn();
  const input = new KeyboardInput({
    target,
    bindings: bindingsFor(7),
    clock,
    engine,
    recorder,
    onJudgment,
    onLaneDown,
    onLaneUp,
  });
  input.attach();
  return { target, input, keyDown, keyUp, recorder, onJudgment, onLaneDown, onLaneUp };
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

  it("reports the lane even when no note is there to judge", () => {
    const { target, keyUp, onLaneDown, onLaneUp } = setup();
    keyUp.mockReturnValue(null);
    const down = new KeyboardEvent("keydown", { code: "KeyA" });
    Object.defineProperty(down, "timeStamp", { value: 900 });
    target.dispatchEvent(down);
    target.dispatchEvent(new KeyboardEvent("keyup", { code: "KeyA" }));
    // 판정이 안 붙는 입력도 알려야 키 바인딩이 틀린 건지 노트가 없는 건지 안다.
    expect(onLaneDown).toHaveBeenCalledWith(1, 1000);
    expect(onLaneUp).toHaveBeenCalledWith(1, expect.any(Number));
  });

  it("does not report a lane twice while the key is held", () => {
    const { target, onLaneDown } = setup();
    target.dispatchEvent(new KeyboardEvent("keydown", { code: "KeyA" }));
    target.dispatchEvent(new KeyboardEvent("keydown", { code: "KeyA" }));
    expect(onLaneDown).toHaveBeenCalledTimes(1);
  });

  it("releases held lanes on blur so nothing stays lit", () => {
    const { target, onLaneUp } = setup();
    target.dispatchEvent(new KeyboardEvent("keydown", { code: "KeyA" }));
    target.dispatchEvent(new KeyboardEvent("keydown", { code: "Space" }));
    target.dispatchEvent(new Event("blur"));
    expect(onLaneUp.mock.calls.map(([lane]) => lane).sort()).toEqual([1, 3]);
  });
});
