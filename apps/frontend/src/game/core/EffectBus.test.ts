import { describe, expect, it, vi } from "vitest";

import {
  EffectBus,
  feverActiveFromEffect,
  type EffectEvent,
  type EffectSubscriber,
} from "./EffectBus";

function judged(combo: number): EffectEvent {
  return {
    type: "JUDGED",
    judgment: "PERFECT",
    lane: 0,
    errMs: 0,
    phase: "HEAD",
    combo,
    songTimeMs: 1000,
  };
}

function recorder(sink: EffectEvent[]): EffectSubscriber {
  return { handleEffect: (event) => void sink.push(event) };
}

describe("EffectBus", () => {
  it("구독 후 emit 이 전달된다", () => {
    const bus = new EffectBus();
    const seen: EffectEvent[] = [];
    bus.subscribe(recorder(seen));

    bus.emit(judged(1));

    expect(seen).toEqual([judged(1)]);
  });

  it("FEVER 시작과 종료 이벤트를 활성 상태로 해석한다", () => {
    expect(feverActiveFromEffect({ type: "FEVER_START", songTimeMs: 1000 })).toBe(true);
    expect(feverActiveFromEffect({ type: "FEVER_END", songTimeMs: 2000 })).toBe(false);
    expect(feverActiveFromEffect(judged(1))).toBeNull();
  });

  it("해제 후 emit 이 전달되지 않는다", () => {
    const bus = new EffectBus();
    const seen: EffectEvent[] = [];
    const off = bus.subscribe(recorder(seen));

    off();
    bus.emit(judged(1));

    expect(seen).toEqual([]);
  });

  it("구독 순서대로 호출된다", () => {
    const bus = new EffectBus();
    const order: string[] = [];
    bus.subscribe({ handleEffect: () => void order.push("first") });
    bus.subscribe({ handleEffect: () => void order.push("second") });

    bus.emit(judged(1));

    expect(order).toEqual(["first", "second"]);
  });

  it("구독자 하나가 예외를 던져도 나머지가 받는다", () => {
    const bus = new EffectBus();
    const seen: EffectEvent[] = [];
    const consoleError = vi.spyOn(console, "error").mockImplementation(() => {});
    bus.subscribe({
      handleEffect: () => {
        throw new Error("렌더 실패");
      },
    });
    bus.subscribe(recorder(seen));

    bus.emit(judged(2));

    expect(seen).toHaveLength(1);
    expect(consoleError).toHaveBeenCalled();
    consoleError.mockRestore();
  });

  it("같은 구독자를 두 번 해제해도 다른 구독자가 남는다", () => {
    const bus = new EffectBus();
    const seen: EffectEvent[] = [];
    const off = bus.subscribe({ handleEffect: () => {} });
    bus.subscribe(recorder(seen));

    off();
    off();
    bus.emit(judged(3));

    expect(seen).toHaveLength(1);
  });
});

// 레인 입력은 StageRenderer의 기능 경로로 전달되며 EffectBus 장식 이벤트가 아니다.
// @ts-expect-error LANE_DOWN은 EffectEvent 계약에 존재하지 않아야 한다.
const removedLaneEvent: EffectEvent = { type: "LANE_DOWN", lane: 0, songTimeMs: 0 };
void removedLaneEvent;
