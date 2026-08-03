import { describe, expect, it } from "vitest";

import { layoutLanes, layoutStage } from "./LaneLayout";

const SEVEN = [
  "SIDE_LEFT",
  "MAIN_1",
  "MAIN_2",
  "CENTER",
  "MAIN_3",
  "MAIN_4",
  "SIDE_RIGHT",
] as const;

describe("layoutStage", () => {
  it("keeps columns at a fixed width instead of filling the container", () => {
    // 컨테이너를 채우려고 늘리면 레인이 100px 을 넘어 노트가 판때기가 된다.
    const wide = layoutStage(1600, SEVEN);
    const narrower = layoutStage(900, SEVEN);
    expect(wide.width).toBeCloseTo(narrower.width);
    for (const lane of wide.lanes) {
      expect(lane.width).toBeLessThanOrEqual(80);
    }
  });

  it("centres the stage in the container", () => {
    const stage = layoutStage(1600, SEVEN);
    expect(stage.left).toBeCloseTo(1600 - stage.right);
    expect(stage.lanes[0].x).toBeCloseTo(stage.left);
  });

  it("makes the special centre column narrower than a main column", () => {
    // osu!lazer: SPECIAL_COLUMN_WIDTH 70 < COLUMN_WIDTH 80.
    const { lanes } = layoutStage(1600, SEVEN);
    expect(lanes[3].width).toBeLessThan(lanes[1].width);
    expect(lanes[0].width).toBeLessThan(lanes[1].width);
    expect(lanes[6].width).toBeCloseTo(lanes[0].width);
  });

  it("shrinks proportionally when the container is narrower than the stage", () => {
    const full = layoutStage(1600, SEVEN);
    const cramped = layoutStage(300, SEVEN);
    expect(cramped.width).toBeLessThanOrEqual(300);
    expect(cramped.left).toBeCloseTo(0);
    const ratio = cramped.lanes[1].width / full.lanes[1].width;
    expect(cramped.lanes[3].width / full.lanes[3].width).toBeCloseTo(ratio);
  });

  it("lays lanes out left to right without gaps beyond the spacing", () => {
    const { lanes } = layoutStage(1600, ["MAIN_1", "MAIN_2", "MAIN_3", "MAIN_4"]);
    for (let index = 1; index < lanes.length; index += 1) {
      const previous = lanes[index - 1];
      expect(lanes[index].x - (previous.x + previous.width)).toBeCloseTo(1);
    }
  });

  it("uses matching colours for mirrored fingers and a distinct centre", () => {
    const { lanes } = layoutStage(1600, SEVEN);
    expect(lanes[0].color).toBe(lanes[6].color);
    expect(lanes[2].color).toBe(lanes[4].color);
    expect(lanes[3].color).not.toBe(lanes[0].color);
    expect(new Set(lanes.map((lane) => lane.color)).size).toBeGreaterThan(2);
  });

  it("rejects empty semantics and non-positive width", () => {
    expect(() => layoutStage(0, ["MAIN_1"])).toThrow(/width/);
    expect(() => layoutStage(100, [])).toThrow(/semantic/);
  });
});

describe("layoutLanes", () => {
  it("returns just the lanes of the stage", () => {
    expect(layoutLanes(1600, SEVEN)).toEqual(layoutStage(1600, SEVEN).lanes);
  });
});
