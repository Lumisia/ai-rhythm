import type { KeysoundManifest } from "../core/types";
import type { Clock } from "./Clock";

function nearestOnset(onsets: readonly number[], targetMs: number): number | null {
  let low = 0;
  let high = onsets.length;
  while (low < high) {
    const middle = Math.floor((low + high) / 2);
    if (onsets[middle] < targetMs) low = middle + 1;
    else high = middle;
  }
  const candidates = [onsets[low - 1], onsets[low]].filter(
    (value): value is number => value !== undefined,
  );
  if (candidates.length === 0) return null;
  return candidates.reduce((best, value) =>
    Math.abs(value - targetMs) < Math.abs(best - targetMs) ? value : best,
  );
}

export class KeysoundScheduler {
  readonly #context: AudioContext;
  readonly #buffer: AudioBuffer;
  readonly #manifest: KeysoundManifest;
  readonly #clock: Clock;
  readonly #autoPlayOnsets: readonly number[];
  readonly #scheduledAutoPlay = new Set<number>();
  readonly #liveSources = new Set<AudioBufferSourceNode>();
  readonly #drumOnsets: readonly number[];
  #disposed = false;

  constructor(
    context: AudioContext,
    buffer: AudioBuffer,
    manifest: KeysoundManifest,
    clock: Clock,
    autoPlayOnsets: readonly number[] = [],
  ) {
    this.#context = context;
    this.#buffer = buffer;
    this.#manifest = manifest;
    this.#clock = clock;
    this.#drumOnsets = [...new Set(manifest.drumOnsets)].sort((left, right) => left - right);
    this.#autoPlayOnsets = [...new Set(autoPlayOnsets)].sort((left, right) => left - right);
  }

  playHit(noteTimeMs: number): boolean {
    return this.#playSlice(noteTimeMs, this.#context.currentTime);
  }

  scheduleAutoPlayUntil(endSongTimeMs: number): number {
    if (this.#disposed) return 0;
    const currentSongTimeMs = this.#clock.songTimeMs();
    let scheduled = 0;

    for (const onsetMs of this.#autoPlayOnsets) {
      if (onsetMs > endSongTimeMs) break;
      if (this.#scheduledAutoPlay.has(onsetMs)) continue;
      this.#scheduledAutoPlay.add(onsetMs);
      if (onsetMs < currentSongTimeMs - this.#manifest.snapWindowMs) continue;
      const when = this.#context.currentTime + Math.max(0, onsetMs - currentSongTimeMs) / 1000;
      if (this.#playSlice(onsetMs, when)) scheduled += 1;
    }
    return scheduled;
  }

  resetAutoPlay(): void {
    this.#stopLiveSources();
    this.#scheduledAutoPlay.clear();
  }

  dispose(): void {
    if (this.#disposed) return;
    this.#disposed = true;
    this.#stopLiveSources();
    this.#scheduledAutoPlay.clear();
  }

  #playSlice(noteTimeMs: number, when: number): boolean {
    if (this.#disposed) return false;
    const onsetMs = nearestOnset(this.#drumOnsets, noteTimeMs);
    if (onsetMs === null || Math.abs(onsetMs - noteTimeMs) > this.#manifest.snapWindowMs) {
      return false;
    }

    const offset = Math.max(0, onsetMs / 1000 - this.#manifest.prerollSec);
    const duration = Math.min(this.#manifest.sliceSec, this.#buffer.duration - offset);
    if (duration <= 0) return false;
    const source = this.#context.createBufferSource();
    source.buffer = this.#buffer;
    source.connect(this.#context.destination);
    source.onended = () => {
      this.#liveSources.delete(source);
      source.disconnect();
    };
    this.#liveSources.add(source);
    source.start(when, offset, duration);
    return true;
  }

  #stopLiveSources(): void {
    for (const source of this.#liveSources) {
      source.onended = null;
      try {
        source.stop();
      } catch {
        // A short slice may already have ended.
      }
      source.disconnect();
    }
    this.#liveSources.clear();
  }
}
