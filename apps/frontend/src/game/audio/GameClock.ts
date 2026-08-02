import type { Clock } from "./Clock";

export interface AudioClockContext {
  readonly currentTime: number;
  getOutputTimestamp?: () => AudioTimestamp;
}

export class GameClock implements Clock {
  readonly #context: AudioClockContext;
  #anchorContextTime = 0;
  #anchorSongTimeMs = 0;
  #pausedSongTimeMs = 0;
  #calibrationMs = 0;
  #running = false;

  constructor(context: AudioClockContext) {
    this.#context = context;
  }

  startAt(contextTime: number, songOffsetMs: number): void {
    if (!Number.isFinite(contextTime) || !Number.isFinite(songOffsetMs)) {
      throw new Error("clock start values must be finite");
    }
    this.#anchorContextTime = contextTime;
    this.#anchorSongTimeMs = songOffsetMs;
    this.#pausedSongTimeMs = songOffsetMs;
    this.#running = true;
  }

  pauseAt(songTimeMs: number): void {
    if (!Number.isFinite(songTimeMs)) throw new Error("paused song time must be finite");
    this.#pausedSongTimeMs = songTimeMs;
    this.#running = false;
  }

  setCalibrationMs(calibrationMs: number): void {
    if (!Number.isFinite(calibrationMs)) throw new Error("calibration must be finite");
    this.#calibrationMs = calibrationMs;
  }

  songTimeMs(): number {
    if (!this.#running) return this.#pausedSongTimeMs;
    return this.#songTimeAtContext(this.#context.currentTime);
  }

  performanceToSongTimeMs(performanceMs: number): number {
    if (!this.#running) return this.#pausedSongTimeMs - this.#calibrationMs;
    const timestamp = this.#context.getOutputTimestamp?.();
    const hasOutputTimestamp =
      typeof timestamp?.contextTime === "number" &&
      typeof timestamp.performanceTime === "number";
    const contextTime = hasOutputTimestamp
      ? timestamp.contextTime! + (performanceMs - timestamp.performanceTime!) / 1000
      : this.#context.currentTime + (performanceMs - performance.now()) / 1000;
    return this.#songTimeAtContext(contextTime) - this.#calibrationMs;
  }

  #songTimeAtContext(contextTime: number): number {
    return this.#anchorSongTimeMs + (contextTime - this.#anchorContextTime) * 1000;
  }
}
