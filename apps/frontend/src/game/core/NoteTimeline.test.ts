import { describe, expect, it } from "vitest";

import type { ChartNote } from "./types";
import { NoteTimeline } from "./NoteTimeline";

function note(id: number, timeMs: number): ChartNote {
  return { id, lane: 0, timeMs, type: "TAP", durationMs: null };
}

describe("NoteTimeline", () => {
  it("returns only notes in the inclusive visible time window", () => {
    const timeline = new NoteTimeline([note(1, 100), note(2, 500), note(3, 900)]);
    expect(timeline.visibleBetween(200, 800).map((item) => item.timeMs)).toEqual([500]);
    expect(timeline.visibleBetween(100, 500).map((item) => item.timeMs)).toEqual([100, 500]);
  });

  it("sorts a private copy without mutating source order", () => {
    const source = [note(2, 500), note(1, 100), note(3, 500)];
    const timeline = new NoteTimeline(source);
    expect(timeline.visibleBetween(0, 1000).map((item) => item.id)).toEqual([1, 2, 3]);
    expect(source.map((item) => item.id)).toEqual([2, 1, 3]);
  });

  it("returns an empty window when its bounds are reversed", () => {
    const timeline = new NoteTimeline([note(1, 100)]);
    expect(timeline.visibleBetween(200, 100)).toEqual([]);
  });
});
