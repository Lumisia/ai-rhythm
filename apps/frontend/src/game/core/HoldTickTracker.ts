import type { ChartNote, JudgmentWindows } from "./types";

export interface HoldTick {
  lane: number;
  timeMs: number;
}

interface TickSpan {
  lane: number;
  fromMs: number;
  toMs: number;
  nextMs: number;
}

/** 롱노트 홀드 틱 생성기.
 *
 * 틱을 판정창 **밖에서만** 만든다. 판정창 안에 틱이 있으면 "판정은 성공인데
 * 콤보만 끊김" 이 발생한다(ppy/osu#24584).
 *
 * 틱은 콤보만 올린다. 콤보 브레이크는 언제나 실제 노트 판정이 낸다.
 */
export class HoldTickTracker {
  readonly #spans: TickSpan[] = [];
  readonly #output: HoldTick[] = [];

  constructor(
    notes: readonly ChartNote[],
    windows: JudgmentWindows,
    private readonly tickMs: number,
    holdReleaseScale: number,
  ) {
    for (const note of notes) {
      if (note.type !== "HOLD" || !note.durationMs || note.durationMs <= 0) continue;
      const fromMs = note.timeMs + windows.BAD;
      const toMs = note.timeMs + note.durationMs - windows.BAD * holdReleaseScale;
      if (toMs <= fromMs) continue;
      this.#spans.push({ lane: note.lane, fromMs, toMs, nextMs: fromMs });
    }
  }

  /** 이번 프레임에 발생한 틱. 반환 배열은 다음 호출에서 재사용된다.
   *
   * 틱 구간은 `[fromMs, toMs)` 다. **끝이 배타적인 것이 핵심이다** —
   * `toMs` 는 릴리즈 판정창의 첫 순간이라, 여기에 틱이 있으면 "떼는 판정은
   * 성공인데 그 틱을 놓쳐 콤보만 끊김" 이 발생한다(ppy/osu#24584).
   */
  advance(songTimeMs: number, heldLanes: ReadonlySet<number>): readonly HoldTick[] {
    this.#output.length = 0;

    for (const span of this.#spans) {
      if (span.nextMs >= span.toMs) continue;
      // 손을 뗀 구간은 소급하지 않는다. 다음 틱 시각을 현재까지 밀어 두고
      // 넘어간다. 기준은 직전 호출 시각이 아니라 **현재 시각**이다 —
      // 직전 시각을 쓰면 뗀 구간의 틱이 다시 잡을 때 한꺼번에 쏟아진다.
      if (!heldLanes.has(span.lane)) {
        if (songTimeMs > span.nextMs) span.nextMs = this.#alignForward(span, songTimeMs);
        continue;
      }
      while (span.nextMs <= songTimeMs && span.nextMs < span.toMs) {
        this.#output.push({ lane: span.lane, timeMs: span.nextMs });
        span.nextMs += this.tickMs;
      }
    }

    return this.#output;
  }

  reset(): void {
    for (const span of this.#spans) span.nextMs = span.fromMs;
    this.#output.length = 0;
  }

  /** 놓친 구간을 건너뛰어 다음 틱 시각을 주어진 시각 이후로 옮긴다. */
  #alignForward(span: TickSpan, timeMs: number): number {
    const skipped = Math.ceil((timeMs - span.nextMs) / this.tickMs);
    return span.nextMs + skipped * this.tickMs;
  }
}
