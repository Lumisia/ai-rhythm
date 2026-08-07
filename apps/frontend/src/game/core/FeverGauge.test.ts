import { describe, expect, it } from "vitest";

import { FeverGauge } from "./FeverGauge";

/** 만충까지 PERFECT 를 채운다. PERFECT 는 +1.0 이므로 100번이다. */
function fill(gauge: FeverGauge, songTimeMs = 0): void {
  for (let index = 0; index < 100; index += 1) {
    gauge.accept("PERFECT", songTimeMs);
  }
}

describe("FeverGauge", () => {
  it("게이지는 0 과 100 사이로 고정된다", () => {
    const gauge = new FeverGauge();

    gauge.accept("MISS", 0);
    expect(gauge.value).toBe(0);

    fill(gauge);
    expect(gauge.value).toBeLessThanOrEqual(100);
  });

  it("만충 시 START 를 반환하고 활성화된다", () => {
    const gauge = new FeverGauge();
    let transition = null;
    for (let index = 0; index < 100; index += 1) {
      transition = gauge.accept("PERFECT", 0) ?? transition;
    }

    expect(transition).toBe("START");
    expect(gauge.active).toBe(true);
  });

  it("발동 시 게이지가 0 으로 리셋된다", () => {
    const gauge = new FeverGauge();
    fill(gauge);

    expect(gauge.value).toBe(0);
  });

  it("종료 시 게이지가 0 으로 리셋된다", () => {
    const gauge = new FeverGauge({ durationMs: 1_000 });
    fill(gauge, 0);
    expect(gauge.active).toBe(true);

    // 발동 중에도 축적은 계속된다. 상한을 넘지 않는지 함께 본다.
    let peak = 0;
    for (let index = 0; index < 200; index += 1) {
      gauge.accept("PERFECT", 100);
      peak = Math.max(peak, gauge.value);
    }
    expect(peak).toBe(100);

    expect(gauge.advance(1_001)).toBe("END");
    expect(gauge.active).toBe(false);
    expect(gauge.value).toBe(0);

    // 종료 직후 판정 한 번으로 다시 발동하면 FEVER 가 영구화된다.
    // value 만 보면 부분 수정을 놓치므로 재발동 여부까지 못박는다.
    expect(gauge.accept("PERFECT", 1_100)).toBeNull();
    expect(gauge.active).toBe(false);
  });

  it("15초 경과 시 자동 종료된다", () => {
    const gauge = new FeverGauge();
    fill(gauge, 0);

    expect(gauge.advance(14_999)).toBeNull();
    expect(gauge.active).toBe(true);
    expect(gauge.advance(15_000)).toBe("END");
    expect(gauge.active).toBe(false);
    expect(gauge.advance(15_001)).toBeNull();
  });

  it("MISS 시 게이지가 감소하고 발동 중이면 즉시 종료된다", () => {
    const gauge = new FeverGauge();
    fill(gauge, 0);
    gauge.accept("PERFECT", 100);
    gauge.accept("PERFECT", 100);

    expect(gauge.accept("MISS", 200)).toBe("END");
    expect(gauge.active).toBe(false);
    expect(gauge.value).toBe(0);
  });

  it("비활성 상태의 MISS 는 게이지만 깎는다", () => {
    const gauge = new FeverGauge();
    for (let index = 0; index < 20; index += 1) gauge.accept("PERFECT", 0);

    expect(gauge.accept("MISS", 10)).toBeNull();
    expect(gauge.value).toBe(12);
  });

  it("판정별 가산량이 스펙과 같다", () => {
    const gauge = new FeverGauge();

    gauge.accept("PERFECT", 0);
    gauge.accept("GREAT", 0);
    gauge.accept("GOOD", 0);
    gauge.accept("BAD", 0);

    expect(gauge.value).toBeCloseTo(1.8, 5);
  });

  it("reset 이 게이지와 발동 상태를 되돌린다", () => {
    const gauge = new FeverGauge();
    fill(gauge, 0);

    gauge.reset();

    expect(gauge.active).toBe(false);
    expect(gauge.value).toBe(0);
  });
});
