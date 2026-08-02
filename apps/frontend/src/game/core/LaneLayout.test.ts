import { describe, expect, it } from "vitest";

import { layoutLanes } from "./LaneLayout";

describe("layoutLanes", () => {
  it("makes side and center lanes wider without branching on keyMode", () => {
    const lanes = layoutLanes(700, [
      "SIDE_LEFT",
      "MAIN_1",
      "MAIN_2",
      "CENTER",
      "MAIN_3",
      "MAIN_4",
      "SIDE_RIGHT",
    ]);
    expect(lanes[0].width).toBeGreaterThan(lanes[1].width);
    expect(lanes[3].width).toBeGreaterThan(lanes[1].width);
    expect(lanes[6].width).toBeCloseTo(lanes[0].width);
  });

  it("fills the requested width with contiguous lanes", () => {
    const lanes = layoutLanes(480, ["MAIN_1", "MAIN_2", "MAIN_3", "MAIN_4"]);
    expect(lanes[0].x).toBe(0);
    expect(lanes.at(-1)!.x + lanes.at(-1)!.width).toBeCloseTo(480);
    for (let index = 1; index < lanes.length; index += 1) {
      expect(lanes[index].x).toBeCloseTo(lanes[index - 1].x + lanes[index - 1].width);
    }
  });

  it("derives note color from lane semantics", () => {
    const lanes = layoutLanes(300, ["SIDE_LEFT", "CENTER", "MAIN_1"]);
    expect(lanes.map((lane) => lane.color)).toEqual([0x78cbb8, 0xe27268, 0xdce3e8]);
  });

  it("rejects empty semantics and non-positive width", () => {
    expect(() => layoutLanes(0, ["MAIN_1"])).toThrow(/width/);
    expect(() => layoutLanes(100, [])).toThrow(/semantic/);
  });
});
