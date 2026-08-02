import vectorsJson from "@contracts/judgment/vectors.json";
import { describe, expect, it } from "vitest";

import { classifyError, loadJudgmentConfig } from "./judgment-config";
import type { JudgmentName, JudgmentPreset } from "./types";

interface JudgmentVector {
  name: string;
  preset: JudgmentPreset;
  noteTimeMs: number;
  hitTimeMs: number;
  expected: JudgmentName;
}

const vectors = vectorsJson as JudgmentVector[];

describe("judgment config", () => {
  it.each(vectors)("matches the shared judgment vector: $name", (vector) => {
    const config = loadJudgmentConfig();
    expect(classifyError(vector.hitTimeMs - vector.noteTimeMs, vector.preset, config)).toBe(
      vector.expected,
    );
  });

  it("returns a defensive copy of the shared contract", () => {
    const first = loadJudgmentConfig();
    first.presets.lenient.PERFECT = 1;
    expect(loadJudgmentConfig().presets.lenient.PERFECT).toBe(80);
  });
});
