import { describe, expect, it } from "vitest";

import type { JudgmentEvent } from "../../game/core/JudgmentEngine";
import { InputRecorder } from "../../game/core/InputRecorder";
import type { PlaytestReview } from "./review";
import { serializeReview } from "./review";

const judgment: JudgmentEvent = {
  noteId: 10,
  lane: 2,
  noteType: "TAP",
  phase: "HEAD",
  judgment: "GREAT",
  errMs: -12,
  timeMs: 988,
  noteTimeMs: 1000,
};

function makeReview(): PlaytestReview {
  return {
    version: 1,
    runId: "40000000-0000-4000-8000-000000000001",
    chartId: "30000000-0000-4000-8000-000000000001",
    chartSha256: "a".repeat(64),
    audioSha256: "b".repeat(64),
    keyMode: 7,
    difficulty: "HARD",
    calibrationMs: -12,
    judgmentPreset: "lenient",
    perceivedDifficulty: "TOO_HARD",
    verdict: "NEEDS_CHANGES",
    feverEnabled: false,
    maxCombo: 12,
    rawMaxCombo: 12,
    events: [{ action: "DOWN", lane: 2, code: "KeyS", timeMs: 988 }],
    judgments: [judgment],
    markers: [
      { kind: "JACK", timeMs: 3200, rangeStartMs: 1200, rangeEndMs: 5200 },
    ],
    comment: "후렴 진입 직후 연타가 지나치게 급격함",
  };
}

describe("InputRecorder", () => {
  it("records immutable copies of mapped input events", () => {
    const recorder = new InputRecorder();
    const input = { action: "DOWN" as const, lane: 2, code: "KeyS", timeMs: 988 };
    recorder.record(input);
    input.timeMs = 999;
    expect(recorder.snapshot()).toEqual([
      { action: "DOWN", lane: 2, code: "KeyS", timeMs: 988 },
    ]);
    expect(() => Object.assign(recorder.snapshot()[0], { timeMs: 0 })).toThrow();
  });
});

describe("serializeReview", () => {
  it("exports only hashes, events, judgments, markers and subjective review", () => {
    const review = Object.assign(makeReview(), {
      localPath: "C:\\Users\\PC\\Desktop\\song.wav",
      audioBytes: new Uint8Array([1, 2, 3]),
    });
    const json = serializeReview(review);
    expect(json).not.toContain("C:\\\\");
    expect(json).not.toContain("audioBytes");
    expect(JSON.parse(json)).toMatchObject({
      version: 1,
      perceivedDifficulty: "TOO_HARD",
      verdict: "NEEDS_CHANGES",
    });
  });

  it("preserves marker time ranges and signed judgment errors", () => {
    const serialized = JSON.parse(serializeReview(makeReview())) as PlaytestReview;
    expect(serialized.markers[0]).toEqual({
      kind: "JACK",
      timeMs: 3200,
      rangeStartMs: 1200,
      rangeEndMs: 5200,
    });
    expect(serialized.judgments[0].errMs).toBe(-12);
  });
});

describe("FEVER 지표 기록", () => {
  it("feverEnabled 와 두 콤보 값이 직렬화된다", () => {
    const json = JSON.parse(
      serializeReview({
        version: 1,
        runId: "run",
        chartId: "chart",
        chartSha256: "a",
        audioSha256: "b",
        keyMode: 4,
        difficulty: "NORMAL",
        calibrationMs: 0,
        judgmentPreset: "lenient",
        perceivedDifficulty: "APPROPRIATE",
        verdict: "PASS",
        feverEnabled: true,
        maxCombo: 1842,
        rawMaxCombo: 1103,
        events: [],
        judgments: [],
        markers: [],
        comment: "",
      }),
    );

    expect(json.feverEnabled).toBe(true);
    expect(json.maxCombo).toBe(1842);
    expect(json.rawMaxCombo).toBe(1103);
  });
});
