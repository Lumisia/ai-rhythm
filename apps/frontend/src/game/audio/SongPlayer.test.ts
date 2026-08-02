import { describe, expect, it } from "vitest";

import { GameClock } from "./GameClock";
import { SongPlayer } from "./SongPlayer";

class FakeSource {
  buffer: AudioBuffer | null = null;
  onended: (() => void) | null = null;
  readonly starts: Array<{ when: number; offset: number }> = [];
  stopCount = 0;
  disconnectCount = 0;
  #started = false;

  connect() {
    return this;
  }

  start(when = 0, offset = 0) {
    if (this.#started) throw new Error("AudioBufferSourceNode can only be started once");
    this.#started = true;
    this.starts.push({ when, offset });
  }

  stop() {
    this.stopCount += 1;
  }

  disconnect() {
    this.disconnectCount += 1;
  }
}

function fakeAudioContext() {
  const sources: FakeSource[] = [];
  const context = {
    currentTime: 12,
    destination: {},
    getOutputTimestamp: () => ({ contextTime: 12, performanceTime: 5000 }),
    decodeAudioData: async () => ({ duration: 10 }) as AudioBuffer,
    createBufferSource: () => {
      const source = new FakeSource();
      sources.push(source);
      return source as unknown as AudioBufferSourceNode;
    },
  };
  return { context, sources };
}

describe("SongPlayer", () => {
  it("decodes bytes and schedules playback with a 1.6 second lead-in", async () => {
    const { context, sources } = fakeAudioContext();
    const clock = new GameClock(context);
    const player = new SongPlayer(context as unknown as AudioContext, clock);
    await player.load(new ArrayBuffer(8));
    player.play(2000);
    expect(sources).toHaveLength(1);
    expect(sources[0].starts).toEqual([{ when: 13.6, offset: 2 }]);
    expect(player.durationMs).toBe(10_000);
  });

  it("creates a fresh source for resume and a playing seek", async () => {
    const { context, sources } = fakeAudioContext();
    const clock = new GameClock(context);
    const player = new SongPlayer(context as unknown as AudioContext, clock);
    await player.load(new ArrayBuffer(8));

    player.play(2000);
    context.currentTime = 14.6;
    expect(player.pause()).toBeCloseTo(3000);
    player.play();
    player.seek(5000);

    expect(sources).toHaveLength(3);
    expect(sources.map((source) => source.starts[0].offset)).toEqual([2, 3, 5]);
    expect(sources[0].stopCount).toBe(1);
    expect(sources[1].stopCount).toBe(1);
  });

  it("does not move backward when paused during the scheduled lead-in", async () => {
    const { context } = fakeAudioContext();
    const player = new SongPlayer(
      context as unknown as AudioContext,
      new GameClock(context),
    );
    await player.load(new ArrayBuffer(8));
    player.play(2000);
    context.currentTime = 12.5;
    expect(player.pause()).toBe(2000);
  });

  it("clamps seek and releases the active source on dispose", async () => {
    const { context, sources } = fakeAudioContext();
    const player = new SongPlayer(
      context as unknown as AudioContext,
      new GameClock(context),
    );
    await player.load(new ArrayBuffer(8));
    expect(player.seek(20_000)).toBe(10_000);
    player.play();
    player.dispose();
    expect(sources[0].stopCount).toBe(1);
    expect(sources[0].disconnectCount).toBe(1);
    expect(() => player.play()).toThrow(/disposed/);
  });
});
