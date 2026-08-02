import { describe, expect, it } from "vitest";

import { GameClock, type AudioClockContext } from "./GameClock";

interface MutableContext extends AudioClockContext {
  currentTime: number;
}

function fakeContext(): MutableContext {
  return {
    currentTime: 12,
    getOutputTimestamp: () => ({ contextTime: 11.9, performanceTime: 5000 }),
  };
}

describe("GameClock", () => {
  it("maps context and performance timestamps to the same song position", () => {
    const context = fakeContext();
    const clock = new GameClock(context);
    clock.startAt(10, 20);
    expect(clock.songTimeMs()).toBeCloseTo(2020);
    expect(clock.performanceToSongTimeMs(5100)).toBeCloseTo(2020);
  });

  it("applies calibration independently from the playback offset", () => {
    const clock = new GameClock(fakeContext());
    clock.startAt(10, 0);
    clock.setCalibrationMs(20);
    expect(clock.performanceToSongTimeMs(5100)).toBeCloseTo(1980);
  });

  it("freezes the song position while paused", () => {
    const context = fakeContext();
    const clock = new GameClock(context);
    clock.startAt(10, 500);
    clock.pauseAt(2500);
    context.currentTime = 30;
    expect(clock.songTimeMs()).toBe(2500);
  });
});
