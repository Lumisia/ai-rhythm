import { describe, expect, it } from "vitest";

import { ParticleField, type ParticleSpawn } from "./ParticleField";

function spec(overrides: Partial<ParticleSpawn> = {}): ParticleSpawn {
  return { x: 100, y: 200, vx: 0, vy: -0.1, lifeMs: 200, color: 0xffffff, ...overrides };
}

describe("ParticleField", () => {
  it("수명이 지난 파티클이 activeAt 에서 빠진다", () => {
    const field = new ParticleField(8);
    field.spawn(spec({ lifeMs: 200 }), 1000);

    expect(field.activeAt(1199)).toHaveLength(1);
    expect(field.activeAt(1201)).toHaveLength(0);
  });

  it("위치가 경과 시간에서 직접 계산된다", () => {
    const field = new ParticleField(8);
    field.spawn(spec({ x: 100, y: 200, vx: 0.5, vy: -0.2 }), 1000);

    const [particle] = field.activeAt(1100);

    expect(particle.x).toBeCloseTo(150, 5);
    expect(particle.y).toBeCloseTo(180, 5);
  });

  it("같은 songTimeMs 로 두 번 호출해도 결과가 같다", () => {
    const field = new ParticleField(8);
    field.spawn(spec(), 1000);

    const first = field.activeAt(1100).map((particle) => ({ ...particle }));
    const second = field.activeAt(1100).map((particle) => ({ ...particle }));

    expect(second).toEqual(first);
  });

  it("alpha 가 수명에 따라 1 에서 0 으로 내려간다", () => {
    const field = new ParticleField(8);
    field.spawn(spec({ lifeMs: 200 }), 1000);

    expect(field.activeAt(1000)[0].alpha).toBeCloseTo(1, 5);
    expect(field.activeAt(1100)[0].alpha).toBeCloseTo(0.5, 5);
  });

  it("용량 초과 시 가장 오래된 것이 재사용된다", () => {
    const field = new ParticleField(2);
    field.spawn(spec({ color: 0x111111 }), 1000);
    field.spawn(spec({ color: 0x222222 }), 1010);

    field.spawn(spec({ color: 0x333333 }), 1020);

    const colors = field.activeAt(1020).map((particle) => particle.color);
    expect(colors).toHaveLength(2);
    expect(colors).not.toContain(0x111111);
    expect(colors).toContain(0x222222);
    expect(colors).toContain(0x333333);
  });

  it("clear 가 모든 파티클을 지운다", () => {
    const field = new ParticleField(8);
    field.spawn(spec(), 1000);

    field.clear();

    expect(field.activeAt(1000)).toHaveLength(0);
  });

  it("activeAt 이 매번 새 배열을 만들지 않는다", () => {
    const field = new ParticleField(8);
    field.spawn(spec(), 1000);

    expect(field.activeAt(1000)).toBe(field.activeAt(1050));
  });
});
