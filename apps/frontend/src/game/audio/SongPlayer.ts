import { GameClock } from "./GameClock";

const DEFAULT_LEAD_IN_SECONDS = 1.6;

function clamp(value: number, minimum: number, maximum: number): number {
  return Math.min(maximum, Math.max(minimum, value));
}

export class SongPlayer {
  readonly #context: AudioContext;
  readonly #clock: GameClock;
  readonly #leadInSeconds: number;
  #buffer: AudioBuffer | null = null;
  #source: AudioBufferSourceNode | null = null;
  #positionMs = 0;
  #playing = false;
  #disposed = false;

  constructor(
    context: AudioContext,
    clock: GameClock,
    leadInSeconds = DEFAULT_LEAD_IN_SECONDS,
  ) {
    if (leadInSeconds < 0 || !Number.isFinite(leadInSeconds)) {
      throw new Error("lead-in must be a finite non-negative number");
    }
    this.#context = context;
    this.#clock = clock;
    this.#leadInSeconds = leadInSeconds;
  }

  get durationMs(): number {
    return (this.#buffer?.duration ?? 0) * 1000;
  }

  get isPlaying(): boolean {
    return this.#playing;
  }

  async load(bytes: ArrayBuffer): Promise<void> {
    this.#assertUsable();
    this.#releaseSource();
    this.#buffer = await this.#context.decodeAudioData(bytes.slice(0));
    this.#positionMs = 0;
    this.#clock.pauseAt(0);
  }

  play(offsetMs = this.#positionMs): void {
    this.#assertUsable();
    if (!this.#buffer) throw new Error("audio must be loaded before playback");
    if (!Number.isFinite(offsetMs)) throw new Error("playback offset must be finite");

    this.#releaseSource();
    this.#positionMs = clamp(offsetMs, 0, this.durationMs);
    const source = this.#context.createBufferSource();
    source.buffer = this.#buffer;
    source.connect(this.#context.destination);
    const when = this.#context.currentTime + this.#leadInSeconds;
    source.start(when, this.#positionMs / 1000);
    source.onended = () => {
      if (this.#source !== source) return;
      source.disconnect();
      this.#source = null;
      this.#playing = false;
      this.#positionMs = this.durationMs;
      this.#clock.pauseAt(this.#positionMs);
    };
    this.#source = source;
    this.#playing = true;
    this.#clock.startAt(when, this.#positionMs);
  }

  pause(): number {
    this.#assertUsable();
    if (!this.#playing) return this.#positionMs;
    this.#positionMs = clamp(
      Math.max(this.#positionMs, this.#clock.songTimeMs()),
      0,
      this.durationMs,
    );
    this.#releaseSource();
    this.#clock.pauseAt(this.#positionMs);
    return this.#positionMs;
  }

  seek(positionMs: number): number {
    this.#assertUsable();
    if (!Number.isFinite(positionMs)) throw new Error("seek position must be finite");
    this.#positionMs = clamp(positionMs, 0, this.durationMs);
    if (this.#playing) this.play(this.#positionMs);
    else this.#clock.pauseAt(this.#positionMs);
    return this.#positionMs;
  }

  stop(): void {
    this.#assertUsable();
    this.#releaseSource();
    this.#positionMs = 0;
    this.#clock.pauseAt(0);
  }

  dispose(): void {
    if (this.#disposed) return;
    this.#releaseSource();
    this.#buffer = null;
    this.#positionMs = 0;
    this.#clock.pauseAt(0);
    this.#disposed = true;
  }

  #releaseSource(): void {
    const source = this.#source;
    this.#source = null;
    this.#playing = false;
    if (!source) return;
    source.onended = null;
    try {
      source.stop();
    } catch {
      // A naturally ended Web Audio source may already be stopped.
    }
    source.disconnect();
  }

  #assertUsable(): void {
    if (this.#disposed) throw new Error("SongPlayer has been disposed");
  }
}
