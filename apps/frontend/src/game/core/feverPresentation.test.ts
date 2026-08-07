import { describe, expect, it } from "vitest";

import { feverGlowAlpha, feverGaugeVisible } from "./feverPresentation";

describe("feverGlowAlpha", () => {
  it("keeps the stage glow fully visible while FEVER is active", () => {
    expect(feverGlowAlpha(true, null, 1_000, false)).toBe(1);
  });

  it("fades the stage glow for 400ms after FEVER ends", () => {
    expect(feverGlowAlpha(false, 1_000, 1_000, false)).toBe(1);
    expect(feverGlowAlpha(false, 1_000, 1_200, false)).toBe(0.5);
    expect(feverGlowAlpha(false, 1_000, 1_400, false)).toBe(0);
    expect(feverGlowAlpha(false, 1_000, 2_000, false)).toBe(0);
  });

  it("removes the glow immediately when reduced motion is enabled", () => {
    expect(feverGlowAlpha(false, 1_000, 1_001, true)).toBe(0);
  });

  it("does not show a glow before FEVER has started", () => {
    expect(feverGlowAlpha(false, null, 1_000, false)).toBe(0);
  });
});

describe("feverGaugeVisible", () => {
  it("shows the HUD gauge only when FEVER is enabled for the session", () => {
    expect(feverGaugeVisible(true)).toBe(true);
    expect(feverGaugeVisible(false)).toBe(false);
  });
});
