import { describe, expect, it } from "vitest";

import { JudgmentEngine } from "./JudgmentEngine";
import type { ChartNote } from "./types";

function tap(id: number, lane: number, timeMs: number): ChartNote {
  return { id, lane, timeMs, type: "TAP", durationMs: null };
}

function hold(id: number, lane: number, timeMs: number, durationMs: number): ChartNote {
  return { id, lane, timeMs, type: "HOLD", durationMs };
}

function makeEngine(notes: ChartNote[]) {
  return new JudgmentEngine(notes, "lenient");
}

describe("JudgmentEngine", () => {
  it("judges the nearest unjudged note in the same lane", () => {
    const engine = makeEngine([tap(1, 0, 1000), tap(2, 0, 1100)]);
    expect(engine.keyDown(0, 1070)?.noteId).toBe(2);
    expect(engine.keyDown(0, 1000)?.noteId).toBe(1);
  });

  it("does not consume a note in another lane", () => {
    const engine = makeEngine([tap(1, 1, 1000)]);
    expect(engine.keyDown(0, 1000)).toBeNull();
    expect(engine.keyDown(1, 1000)?.judgment).toBe("PERFECT");
  });

  it("includes the BAD boundary and ignores input beyond it", () => {
    expect(makeEngine([tap(1, 0, 1000)]).keyDown(0, 1200)?.judgment).toBe("BAD");
    expect(makeEngine([tap(1, 0, 1000)]).keyDown(0, 1201)).toBeNull();
  });

  it("judges a hold head on down and its tail on up", () => {
    const engine = makeEngine([hold(1, 0, 1000, 500)]);
    expect(engine.keyDown(0, 1010)?.phase).toBe("HEAD");
    const tail = engine.keyUp(0, 1490);
    expect(tail?.phase).toBe("TAIL");
    expect(tail?.judgment).toBe("PERFECT");
    expect(tail?.errMs).toBe(-10);
  });

  it("scales every hold release judgment window", () => {
    const engine = makeEngine([hold(1, 0, 1000, 500)]);
    engine.keyDown(0, 1000);
    expect(engine.keyUp(0, 1620)?.judgment).toBe("PERFECT");

    const late = makeEngine([hold(2, 0, 1000, 500)]);
    late.keyDown(0, 1000);
    expect(late.keyUp(0, 1800)?.judgment).toBe("BAD");

    const tooLate = makeEngine([hold(3, 0, 1000, 500)]);
    tooLate.keyDown(0, 1000);
    expect(tooLate.keyUp(0, 1801)).toBeNull();
  });

  it("emits one head MISS after missAfterMs", () => {
    const engine = makeEngine([tap(1, 0, 1000)]);
    expect(engine.advance(1200)).toEqual([]);
    expect(engine.advance(1201)).toMatchObject([
      { noteId: 1, phase: "HEAD", judgment: "MISS", errMs: 201 },
    ]);
    expect(engine.advance(2000)).toEqual([]);
  });

  it("emits a tail MISS when an active hold is never released", () => {
    const engine = makeEngine([hold(1, 0, 1000, 500)]);
    engine.keyDown(0, 1000);
    expect(engine.advance(1800)).toEqual([]);
    expect(engine.advance(1801)).toMatchObject([
      { noteId: 1, phase: "TAIL", judgment: "MISS", errMs: 301 },
    ]);
    expect(engine.keyUp(0, 1801)).toBeNull();
  });

  it("resets judgments for a practice-loop replay", () => {
    const engine = makeEngine([tap(1, 0, 1000)]);
    expect(engine.keyDown(0, 1000)?.noteId).toBe(1);
    expect(engine.keyDown(0, 1000)).toBeNull();
    engine.reset();
    expect(engine.keyDown(0, 1000)?.noteId).toBe(1);
  });
});

describe("놓친 롱노트 추적", () => {
  const holdNote = { id: 7, lane: 0, timeMs: 1000, type: "HOLD" as const, durationMs: 500 };

  it("초기에는 비어 있다", () => {
    const engine = new JudgmentEngine([holdNote], "lenient");

    expect([...engine.missedHoldIds()]).toEqual([]);
  });

  it("헤드를 놓치면 missedHoldIds 에 들어간다", () => {
    const engine = new JudgmentEngine([holdNote], "lenient");

    engine.advance(5000);

    expect([...engine.missedHoldIds()]).toEqual([7]);
  });

  it("꼬리를 놓치면 missedHoldIds 에 들어간다", () => {
    const engine = new JudgmentEngine([holdNote], "lenient");
    engine.keyDown(0, 1000);

    engine.advance(5000);

    expect([...engine.missedHoldIds()]).toEqual([7]);
  });

  it("정상 처리된 롱노트는 들어가지 않는다", () => {
    const engine = new JudgmentEngine([holdNote], "lenient");
    engine.keyDown(0, 1000);
    engine.keyUp(0, 1500);

    engine.advance(5000);

    expect([...engine.missedHoldIds()]).toEqual([]);
  });

  it("잡고 있는 동안 activeHoldIds 에 들어간다", () => {
    const engine = new JudgmentEngine([holdNote], "lenient");

    engine.keyDown(0, 1000);
    expect([...engine.activeHoldIds()]).toEqual([7]);

    engine.keyUp(0, 1500);
    expect([...engine.activeHoldIds()]).toEqual([]);
  });

  it("판정창보다 일찍 놓으면 즉시 tail MISS 로 종료한다", () => {
    const engine = new JudgmentEngine(
      [{ id: 8, lane: 0, timeMs: 1000, type: "HOLD" as const, durationMs: 2000 }],
      "lenient",
    );
    engine.keyDown(0, 1000);

    expect(engine.keyUp(0, 1200)).toMatchObject({
      noteId: 8,
      phase: "TAIL",
      judgment: "MISS",
      noteTimeMs: 3000,
    });
    expect([...engine.activeHoldIds()]).toEqual([]);
    expect([...engine.missedHoldIds()]).toEqual([8]);
  });

  it("reset 이 두 집합을 비운다", () => {
    const engine = new JudgmentEngine([holdNote], "lenient");
    engine.advance(5000);

    engine.reset();

    expect([...engine.missedHoldIds()]).toEqual([]);
    expect([...engine.activeHoldIds()]).toEqual([]);
  });

  it("TAP 노트를 놓쳐도 missedHoldIds 에 들어가지 않는다", () => {
    const engine = new JudgmentEngine(
      [{ id: 3, lane: 0, timeMs: 1000, type: "TAP" as const }],
      "lenient",
    );

    engine.advance(5000);

    expect([...engine.missedHoldIds()]).toEqual([]);
  });
});
