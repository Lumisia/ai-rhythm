import { describe, expect, it } from "vitest";

import { beatDurationMs } from "./beat";

describe("beatDurationMs", () => {
  it("converts a positive BPM to one beat in milliseconds", () => {
    expect(beatDurationMs(120)).toBe(500);
    expect(beatDurationMs(150)).toBe(400);
  });

  it.each([undefined, 0, -120, Number.NaN, Number.POSITIVE_INFINITY])(
    "uses the 500ms fallback for an invalid BPM: %s",
    (bpm) => {
      expect(beatDurationMs(bpm)).toBe(500);
    },
  );
});
