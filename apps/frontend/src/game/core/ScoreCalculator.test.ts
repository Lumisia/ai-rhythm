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
