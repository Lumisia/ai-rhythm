import { describe, expect, it } from "vitest";

import type { JudgmentEvent } from "./JudgmentEngine";
import { ScoreCalculator } from "./ScoreCalculator";
import type { JudgmentName } from "./types";

function event(
  judgment: JudgmentName,
  errMs: number,
  phase: JudgmentEvent["phase"] = "HEAD",
  lane = 0,
): JudgmentEvent {
  return {
    noteId: 1,
    lane,
    noteType: phase === "TAIL" ? "HOLD" : "TAP",
    phase,
    judgment,
    errMs,
    timeMs: 1000 + errMs,
    noteTimeMs: 1000,
  };
}

describe("ScoreCalculator", () => {
  it("breaks combo on MISS and keeps exact judgment counts", () => {
    const score = new ScoreCalculator();
    score.accept(event("PERFECT", -4));
    score.accept(event("MISS", 240));
    expect(score.snapshot().combo).toBe(0);
    expect(score.snapshot().maxCombo).toBe(1);
    expect(score.snapshot().counts).toMatchObject({ PERFECT: 1, MISS: 1 });
  });

  it("averages only non-MISS head timing errors", () => {
    const score = new ScoreCalculator();
    score.accept(event("PERFECT", -10));
    score.accept(event("GREAT", 20, "HEAD", 1));
    score.accept(event("BAD", 150, "TAIL", 1));
    score.accept(event("MISS", 201, "HEAD", 2));
    expect(score.snapshot()).toMatchObject({
      meanErrMs: 5,
      meanAbsoluteErrMs: 15,
      timingSampleCount: 2,
    });
  });

  it("returns defensive count and lane snapshots", () => {
    const score = new ScoreCalculator();
    score.accept(event("MISS", 201, "HEAD", 2));
    const snapshot = score.snapshot();
    snapshot.counts.MISS = 0;
    snapshot.lanes[2].misses = 0;
    expect(score.snapshot().counts.MISS).toBe(1);
    expect(score.snapshot().lanes[2].misses).toBe(1);
  });
});

function hit(judgment: JudgmentEvent["judgment"] = "PERFECT"): JudgmentEvent {
  return {
    noteId: 1,
    lane: 0,
    noteType: "TAP",
    phase: "HEAD",
    judgment,
    errMs: 0,
    timeMs: 100,
    noteTimeMs: 100,
  };
}

describe("ScoreCalculator 홀드 틱과 FEVER", () => {
  it("acceptHoldTick 이 accuracy 와 counts 를 바꾸지 않는다", () => {
    const score = new ScoreCalculator();
    score.accept(hit());
    const before = score.snapshot();

    score.acceptHoldTick(0);
    const after = score.snapshot();

    expect(after.accuracy).toBe(before.accuracy);
    expect(after.counts).toEqual(before.counts);
    expect(after.totalJudgments).toBe(before.totalJudgments);
  });

  it("acceptHoldTick 이 combo 와 rawCombo 를 각각 1 올린다", () => {
    const score = new ScoreCalculator();
    score.accept(hit());

    score.acceptHoldTick(0);

    const snapshot = score.snapshot();
    expect(snapshot.combo).toBe(2);
    expect(snapshot.rawCombo).toBe(2);
    expect(snapshot.maxCombo).toBe(2);
    expect(snapshot.rawMaxCombo).toBe(2);
  });

  it("FEVER 중 판정이 combo 를 2, rawCombo 를 1 올린다", () => {
    const score = new ScoreCalculator();
    score.setFeverActive(true);

    score.accept(hit());

    const snapshot = score.snapshot();
    expect(snapshot.combo).toBe(2);
    expect(snapshot.rawCombo).toBe(1);
    expect(snapshot.maxCombo).toBe(2);
    expect(snapshot.rawMaxCombo).toBe(1);
  });

  it("FEVER 중 홀드 틱은 combo 도 1 만 올린다", () => {
    const score = new ScoreCalculator();
    score.setFeverActive(true);

    score.acceptHoldTick(0);

    const snapshot = score.snapshot();
    expect(snapshot.combo).toBe(1);
    expect(snapshot.rawCombo).toBe(1);
  });

  it("FEVER 가 accuracy 와 counts 를 바꾸지 않는다", () => {
    const plain = new ScoreCalculator();
    const fever = new ScoreCalculator();
    fever.setFeverActive(true);

    plain.accept(hit("GREAT"));
    fever.accept(hit("GREAT"));

    expect(fever.snapshot().accuracy).toBe(plain.snapshot().accuracy);
    expect(fever.snapshot().counts).toEqual(plain.snapshot().counts);
  });

  it("MISS 가 combo 와 rawCombo 를 함께 0 으로 되돌린다", () => {
    const score = new ScoreCalculator();
    score.setFeverActive(true);
    score.accept(hit());
    score.acceptHoldTick(0);

    score.accept(hit("MISS"));

    const snapshot = score.snapshot();
    expect(snapshot.combo).toBe(0);
    expect(snapshot.rawCombo).toBe(0);
    expect(snapshot.maxCombo).toBe(3);
    expect(snapshot.rawMaxCombo).toBe(2);
  });

  it("홀드 틱은 레인 통계를 바꾸지 않는다", () => {
    const score = new ScoreCalculator();
    score.accept(hit());

    score.acceptHoldTick(0);

    expect(score.snapshot().lanes[0]).toEqual({ judgments: 1, misses: 0 });
  });
});
