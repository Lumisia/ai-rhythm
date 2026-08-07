import { describe, expect, it } from "vitest";

import { HoldTickTracker } from "./HoldTickTracker";
import type { ChartNote, JudgmentWindows } from "./types";

const WINDOWS: JudgmentWindows = { PERFECT: 80, GREAT: 120, GOOD: 160, BAD: 200 };
const HOLD_RELEASE_SCALE = 1.5;
const TICK_MS = 125;

function hold(id: number, lane: number, timeMs: number, durationMs: number): ChartNote {
  return { id, lane, timeMs, type: "HOLD", durationMs };
}

/** 헤드 0ms, 길이 3000ms → 틱 구간은 200 ~ 2700ms. */
function tracker(notes: readonly ChartNote[] = [hold(1, 0, 0, 3000)]): HoldTickTracker {
  return new HoldTickTracker(notes, WINDOWS, TICK_MS, HOLD_RELEASE_SCALE);
}

function laneSet(...lanes: number[]): ReadonlySet<number> {
  return new Set(lanes);
}

describe("HoldTickTracker", () => {
  it("판정창 안에서는 틱이 생기지 않는다", () => {
    const subject = tracker();

    // 헤드 판정창(BAD 200ms) 안이다.
    expect([...subject.advance(150, laneSet(0))]).toEqual([]);
  });

  it("판정창을 지나면 틱이 생긴다", () => {
    const subject = tracker();
    subject.advance(150, laneSet(0));

    const ticks = [...subject.advance(400, laneSet(0))];

    // 200, 325 두 지점 (200 부터 125ms 간격)
    expect(ticks).toEqual([
      { lane: 0, timeMs: 200 },
      { lane: 0, timeMs: 325 },
    ]);
  });

  it("릴리즈 판정창 안에서는 틱이 생기지 않는다", () => {
    const subject = tracker();
    subject.advance(2699, laneSet(0));

    // 꼬리 3000ms, 릴리즈 창 200*1.5=300 → 2700 이후로는 틱 없음
    expect([...subject.advance(3000, laneSet(0))]).toEqual([]);
  });

  it("틱 구간의 끝(toMs)에는 틱이 생기지 않는다", () => {
    const subject = tracker();

    // 한 번의 호출로 fromMs 부터 toMs 까지 전부 훑는다. 위 테스트는 두 번에
    // 나눠 부르는 바람에 바깥 건너뛰기 가드에서 걸러져 while 루프의 끝 비교가
    // 실행되지 않는다. 이 테스트가 `nextMs < toMs` 를 직접 잠근다 —
    // 이 비교가 `<=` 로 새면 ppy/osu#24584 결함이 그대로 재현된다.
    const ticks = [...subject.advance(2700, laneSet(0))];

    expect(ticks.some((tick) => tick.timeMs === 2700)).toBe(false);
    expect(ticks.at(-1)).toEqual({ lane: 0, timeMs: 2575 });
  });

  it("짧은 롱노트는 틱이 0개다", () => {
    // 길이 300ms → 틱 구간 200 ~ 0ms 로 뒤집힌다
    const subject = tracker([hold(1, 0, 0, 300)]);

    expect([...subject.advance(5000, laneSet(0))]).toEqual([]);
  });

  it("뗀 뒤에는 틱이 멈춘다", () => {
    const subject = tracker();
    subject.advance(400, laneSet(0));

    expect([...subject.advance(1000, laneSet())]).toEqual([]);
  });

  it("다시 잡아도 뗀 동안의 틱은 소급되지 않는다", () => {
    const subject = tracker();
    subject.advance(400, laneSet(0));
    subject.advance(1000, laneSet());

    const ticks = [...subject.advance(1100, laneSet(0))];

    expect(ticks.every((tick) => tick.timeMs > 1000)).toBe(true);
  });

  it("같은 songTimeMs 를 두 번 넣어도 틱이 중복되지 않는다", () => {
    const subject = tracker();
    const first = [...subject.advance(400, laneSet(0))];

    const second = [...subject.advance(400, laneSet(0))];

    expect(first).toHaveLength(2);
    expect(second).toEqual([]);
  });

  it("여러 레인이 동시에 틱을 낸다", () => {
    const subject = tracker([hold(1, 0, 0, 3000), hold(2, 3, 0, 3000)]);

    const ticks = [...subject.advance(400, laneSet(0, 3))];

    expect(ticks.filter((tick) => tick.lane === 0)).toHaveLength(2);
    expect(ticks.filter((tick) => tick.lane === 3)).toHaveLength(2);
  });

  it("TAP 노트는 틱을 내지 않는다", () => {
    const subject = tracker([{ id: 1, lane: 0, timeMs: 0, type: "TAP" }]);

    expect([...subject.advance(5000, laneSet(0))]).toEqual([]);
  });

  it("reset 이 진행 상태를 되돌린다", () => {
    const subject = tracker();
    subject.advance(400, laneSet(0));

    subject.reset();

    expect([...subject.advance(400, laneSet(0))]).toHaveLength(2);
  });
});
