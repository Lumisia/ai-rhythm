import type { JudgmentEvent } from "./JudgmentEngine";
import type { JudgmentName } from "./types";

export interface LaneScoreSnapshot {
  judgments: number;
  misses: number;
}

export interface ScoreSnapshot {
  counts: Record<JudgmentName, number>;
  combo: number;
  maxCombo: number;
  totalJudgments: number;
  accuracy: number;
  meanErrMs: number;
  meanAbsoluteErrMs: number;
  timingSampleCount: number;
  lanes: Record<number, LaneScoreSnapshot>;
}

const weights: Record<JudgmentName, number> = {
  PERFECT: 1,
  GREAT: 0.8,
  GOOD: 0.5,
  BAD: 0.2,
  MISS: 0,
};

function emptyCounts(): Record<JudgmentName, number> {
  return { PERFECT: 0, GREAT: 0, GOOD: 0, BAD: 0, MISS: 0 };
}

export class ScoreCalculator {
  readonly #counts = emptyCounts();
  readonly #lanes = new Map<number, LaneScoreSnapshot>();
  #combo = 0;
  #maxCombo = 0;
  #weightedScore = 0;
  #timingErrorSum = 0;
  #absoluteTimingErrorSum = 0;
  #timingSampleCount = 0;

  accept(event: JudgmentEvent): void {
    this.#counts[event.judgment] += 1;
    this.#weightedScore += weights[event.judgment];

    if (event.judgment === "MISS") {
      this.#combo = 0;
    } else {
      this.#combo += 1;
      this.#maxCombo = Math.max(this.#maxCombo, this.#combo);
    }

    const lane = this.#lanes.get(event.lane) ?? { judgments: 0, misses: 0 };
    lane.judgments += 1;
    if (event.judgment === "MISS") lane.misses += 1;
    this.#lanes.set(event.lane, lane);

    if (event.phase === "HEAD" && event.judgment !== "MISS") {
      this.#timingErrorSum += event.errMs;
      this.#absoluteTimingErrorSum += Math.abs(event.errMs);
      this.#timingSampleCount += 1;
    }
  }

  snapshot(): ScoreSnapshot {
    const totalJudgments = Object.values(this.#counts).reduce((sum, count) => sum + count, 0);
    const lanes = Object.fromEntries(
      [...this.#lanes.entries()].map(([lane, value]) => [lane, { ...value }]),
    ) as Record<number, LaneScoreSnapshot>;

    return {
      counts: { ...this.#counts },
      combo: this.#combo,
      maxCombo: this.#maxCombo,
      totalJudgments,
      accuracy: totalJudgments === 0 ? 0 : this.#weightedScore / totalJudgments,
      meanErrMs: this.#timingSampleCount === 0 ? 0 : this.#timingErrorSum / this.#timingSampleCount,
      meanAbsoluteErrMs:
        this.#timingSampleCount === 0
          ? 0
          : this.#absoluteTimingErrorSum / this.#timingSampleCount,
      timingSampleCount: this.#timingSampleCount,
      lanes,
    };
  }
}
