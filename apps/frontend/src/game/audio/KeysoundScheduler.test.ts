import { describe, expect, it } from "vitest";

import type { KeysoundManifest } from "../core/types";
import type { Clock } from "./Clock";
import { KeysoundScheduler } from "./KeysoundScheduler";

class FakeSource {
  buffer: AudioBuffer | null = null;
  onended: (() => void) | null = null;
  starts: Array<{ when: number; offset: number; duration: number }> = [];
  stopCount = 0;
  connect() {
    return this;
  }
  start(when = 0, offset = 0, duration = 0) {
    this.starts.push({ when, offset, duration });
  }
  stop() {
    this.stopCount += 1;
  }
  disconnect() {}
}

function setup(songTimeMs = 900) {
  const sources: FakeSource[] = [];
  const context = {
    currentTime: 10,
    destination: {},
    createBufferSource: () => {
      const source = new FakeSource();
      sources.push(source);
      return source as unknown as AudioBufferSourceNode;
    },
  };
  const clock: Clock = {
    songTimeMs: () => songTimeMs,
    performanceToSongTimeMs: (value) => value,
  };
  const manifest: KeysoundManifest = {
    schemaVersion: 1,
    songVersionId: "10000000-0000-4000-8000-000000000001",
    bgmAssetId: "20000000-0000-4000-8000-000000000001",
    keysAssetId: "30000000-0000-4000-8000-000000000001",
    sliceSec: 0.3,
    prerollSec: 0.012,
    snapWindowMs: 50,
    drumOnsets: [1000, 2000, 3000],
  };
  const scheduler = new KeysoundScheduler(
    context as unknown as AudioContext,
    { duration: 5 } as AudioBuffer,
    manifest,
    clock,
    [1000, 2000],
  );
  return { scheduler, sources };
}

describe("KeysoundScheduler", () => {
  it("snaps a hit to the closest drum onset and applies preroll", () => {
    const { scheduler, sources } = setup();
    expect(scheduler.playHit(1040)).toBe(true);
    expect(sources[0].starts[0]).toEqual({ when: 10, offset: 0.988, duration: 0.3 });
    expect(scheduler.playHit(1100)).toBe(false);
  });

  it("schedules autoplay onsets once across overlapping lookahead calls", () => {
    const { scheduler, sources } = setup(900);
    expect(scheduler.scheduleAutoPlayUntil(1500)).toBe(1);
    expect(scheduler.scheduleAutoPlayUntil(1500)).toBe(0);
    expect(scheduler.scheduleAutoPlayUntil(2100)).toBe(1);
    expect(sources).toHaveLength(2);
    expect(sources[0].starts[0].when).toBeCloseTo(10.1);
    expect(sources[1].starts[0].when).toBeCloseTo(11.1);
  });

  it("stops every live slice on dispose", () => {
    const { scheduler, sources } = setup();
    scheduler.playHit(1000);
    scheduler.scheduleAutoPlayUntil(2100);
    scheduler.dispose();
    expect(sources.every((source) => source.stopCount === 1)).toBe(true);
    expect(scheduler.playHit(1000)).toBe(false);
  });
});
