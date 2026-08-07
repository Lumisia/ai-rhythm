import { describe, expect, it } from "vitest";

import { NOTE_PX_PER_MS, approachMsAt1x, layoutLanes, layoutStage } from "./LaneLayout";

const SEVEN = [
  "SIDE_LEFT",
  "MAIN_1",
  "MAIN_2",
  "CENTER",
  "MAIN_3",
  "MAIN_4",
  "SIDE_RIGHT",
] as const;

const FOUR = ["MAIN_1", "MAIN_2", "MAIN_3", "MAIN_4"] as const;
const SIX = ["SIDE_LEFT", "MAIN_1", "MAIN_2", "MAIN_3", "MAIN_4", "SIDE_RIGHT"] as const;

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

describe("approachMsAt1x", () => {
  it("judgeLineY 에 비례한다", () => {
    expect(approachMsAt1x(544)).toBeCloseTo(544 / NOTE_PX_PER_MS, 5);
    expect(approachMsAt1x(1088)).toBeCloseTo(2 * approachMsAt1x(544), 5);
  });

  it("pxPerMs 를 넘기면 그 값을 쓴다", () => {
    expect(approachMsAt1x(600, 1.2)).toBeCloseTo(500, 5);
  });

  it("기본 pxPerMs 는 0.6 이다", () => {
    expect(NOTE_PX_PER_MS).toBe(0.6);
  });
});

/** sRGB 상대 휘도. WCAG 대비 계산과 같은 식이다. */
function luminance(color: number): number {
  const channels = [(color >> 16) & 0xff, (color >> 8) & 0xff, color & 0xff].map((value) => {
    const normalized = value / 255;
    return normalized <= 0.03928
      ? normalized / 12.92
      : ((normalized + 0.055) / 1.055) ** 2.4;
  });
  return 0.2126 * channels[0] + 0.7152 * channels[1] + 0.0722 * channels[2];
}

describe("레인 색 구분", () => {
  it.each([[FOUR], [SIX], [SEVEN]])("모든 키 모드에서 이웃한 레인의 명도 대비가 3:1 이상이다", (semantics) => {
    const { lanes } = layoutStage(1280, semantics);

    for (let index = 1; index < lanes.length; index += 1) {
      const previous = luminance(lanes[index - 1].color);
      const current = luminance(lanes[index].color);
      const ratio =
        (Math.max(previous, current) + 0.05) / (Math.min(previous, current) + 0.05);
      expect(ratio).toBeGreaterThanOrEqual(3);
    }
  });

  it.each([[FOUR], [SIX], [SEVEN]])("모든 노트가 레인 배경과 3:1 이상 대비된다", (semantics) => {
    const { lanes } = layoutStage(1280, semantics);

    for (const lane of lanes) {
      const note = luminance(lane.color);
      const background = luminance(lane.backgroundColor);
      const ratio = (Math.max(note, background) + 0.05) / (Math.min(note, background) + 0.05);
      expect(ratio).toBeGreaterThanOrEqual(3);
    }
  });

  it("같은 손가락 역할은 같은 색을 쓴다", () => {
    const { lanes } = layoutStage(1280, SEVEN);
    const bySemantic = new Map(lanes.map((lane) => [lane.semantic, lane.color]));

    expect(bySemantic.get("SIDE_LEFT")).toBe(bySemantic.get("SIDE_RIGHT"));
    expect(bySemantic.get("MAIN_1")).toBe(bySemantic.get("MAIN_4"));
    expect(bySemantic.get("MAIN_2")).toBe(bySemantic.get("MAIN_3"));
  });
});
