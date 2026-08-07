import type { JudgmentName } from "./types";

export type FeverTransition = "START" | "END" | null;

const GAIN: Record<JudgmentName, number> = {
  PERFECT: 1.0,
  GREAT: 0.6,
  GOOD: 0.2,
  BAD: 0,
  MISS: -8.0,
};

const FULL = 100;
const DEFAULT_DURATION_MS = 15_000;

export interface FeverGaugeOptions {
  durationMs?: number;
}

/** FEVER 게이지.
 *
 * 홀드 틱은 게이지를 채우지 않는다. 채우면 롱노트가 많은 채보에서
 * FEVER 가 상시 유지되어 연출이 의미를 잃는다.
 */
export class FeverGauge {
  readonly #durationMs: number;
  #value = 0;
  #active = false;
  #startedAtMs = 0;

  constructor(options: FeverGaugeOptions = {}) {
    this.#durationMs = options.durationMs ?? DEFAULT_DURATION_MS;
  }

  get value(): number {
    return this.#value;
  }

  get active(): boolean {
    return this.#active;
  }

  accept(judgment: JudgmentName, songTimeMs: number): FeverTransition {
    if (judgment === "MISS" && this.#active) {
      this.#value = 0;
      this.#active = false;
      return "END";
    }

    this.#value = Math.min(FULL, Math.max(0, this.#value + GAIN[judgment]));
    if (this.#active || this.#value < FULL) return null;

    this.#value = 0;
    this.#active = true;
    this.#startedAtMs = songTimeMs;
    return "START";
  }

  advance(songTimeMs: number): FeverTransition {
    if (!this.#active) return null;
    if (songTimeMs - this.#startedAtMs < this.#durationMs) return null;
    // 발동 중에도 `accept` 는 게이지를 계속 채운다. 여기서 0 으로 되돌리지
    // 않으면 만충인 채로 끝나 다음 판정 한 번에 곧바로 재발동한다 —
    // FEVER 가 사실상 영구화되고 maxCombo 와 rawMaxCombo 의 구분이 무너진다.
    this.#value = 0;
    this.#active = false;
    return "END";
  }

  reset(): void {
    this.#value = 0;
    this.#active = false;
    this.#startedAtMs = 0;
  }
}
