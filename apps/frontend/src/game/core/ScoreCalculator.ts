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
  /** FEVER 증폭이 없는 콤보. 채보 간 비교 기준값이다. */
  rawCombo: number;
  rawMaxCombo: number;
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
  #rawCombo = 0;
  #rawMaxCombo = 0;
  #feverActive = false;
  #weightedScore = 0;
  #timingErrorSum = 0;
  #absoluteTimingErrorSum = 0;
  #timingSampleCount = 0;

  accept(event: JudgmentEvent): void {
    this.#counts[event.judgment] += 1;
    this.#weightedScore += weights[event.judgment];

    if (event.judgment === "MISS") {
      this.#combo = 0;
      this.#rawCombo = 0;
    } else {
      // FEVER 는 표시 콤보만 증폭한다. rawCombo 는 판정 1건에 1 이다.
      this.#combo += this.#feverActive ? 2 : 1;
      this.#rawCombo += 1;
      this.#maxCombo = Math.max(this.#maxCombo, this.#combo);
      this.#rawMaxCombo = Math.max(this.#rawMaxCombo, this.#rawCombo);
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

  setFeverActive(active: boolean): void {
    this.#feverActive = active;
  }

  /** 롱노트 홀드 틱. 콤보만 올린다.
   *
   * counts / accuracy / totalJudgments / lanes 를 건드리지 않는다. 틱이
   * 정확도 분모에 들어가면 롱노트가 많은 채보의 정확도가 부풀려져
   * 난이도 비교가 무의미해진다(ppy/osu#24618).
   *
   * FEVER 증폭도 적용하지 않는다. BPM 180 에서 롱노트 4개 동시 홀드면
   * 초당 48틱이라 ×2 가 붙으면 숫자가 의미를 잃는다.
   */
  acceptHoldTick(): number {
    this.#combo += 1;
    this.#rawCombo += 1;
    this.#maxCombo = Math.max(this.#maxCombo, this.#combo);
    this.#rawMaxCombo = Math.max(this.#rawMaxCombo, this.#rawCombo);
    return this.#combo;
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
      rawCombo: this.#rawCombo,
      rawMaxCombo: this.#rawMaxCombo,
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
