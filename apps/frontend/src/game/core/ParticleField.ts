export interface ParticleSpawn {
  x: number;
  y: number;
  /** px/ms */
  vx: number;
  /** px/ms */
  vy: number;
  lifeMs: number;
  color: number;
}

export interface ActiveParticle {
  x: number;
  y: number;
  alpha: number;
  color: number;
}

interface Slot extends ParticleSpawn {
  bornAtMs: number;
  used: boolean;
}

/** 고정 용량 파티클 풀.
 *
 * 생성자에서 슬롯과 출력 버퍼를 전부 잡는다. rAF 루프 안에서 배열이나
 * 객체를 만들면 GC 가 프레임을 흔든다.
 *
 * 위치를 경과 시간에서 직접 계산한다. 프레임 누적으로 옮기면 프레임 드롭 시
 * 파티클이 밀린다 — 노트 위치와 같은 원칙이다.
 */
export class ParticleField {
  readonly #slots: Slot[];
  /** 생성자에서 capacity 개 확보. 길이가 변하지 않는다. */
  readonly #pool: ActiveParticle[];
  /** 매 프레임 길이만 조정한다. 원소는 항상 #pool 의 객체다. */
  readonly #output: ActiveParticle[];
  #cursor = 0;

  constructor(capacity: number) {
    this.#slots = Array.from({ length: capacity }, () => ({
      x: 0,
      y: 0,
      vx: 0,
      vy: 0,
      lifeMs: 0,
      color: 0,
      bornAtMs: 0,
      used: false,
    }));
    this.#pool = Array.from({ length: capacity }, () => ({
      x: 0,
      y: 0,
      alpha: 0,
      color: 0,
    }));
    this.#output = Array.from({ length: 0 }, () => ({
      x: 0,
      y: 0,
      alpha: 0,
      color: 0,
    }));
  }

  spawn(spec: ParticleSpawn, songTimeMs: number): void {
    const slot = this.#slots[this.#cursor];
    this.#cursor = (this.#cursor + 1) % this.#slots.length;
    slot.x = spec.x;
    slot.y = spec.y;
    slot.vx = spec.vx;
    slot.vy = spec.vy;
    slot.lifeMs = spec.lifeMs;
    slot.color = spec.color;
    slot.bornAtMs = songTimeMs;
    slot.used = true;
  }

  /** 살아 있는 파티클. 반환 배열은 다음 호출에서 재사용된다.
   *
   * `#pool` 은 생성자에서 잡은 뒤 길이가 변하지 않는다. `#output` 은 그 안의
   * 객체를 가리키기만 하고 길이만 조정한다. 길이를 줄였다 늘리면 `undefined`
   * 가 채워져 다음 프레임에서 터진다.
   */
  activeAt(songTimeMs: number): readonly ActiveParticle[] {
    let count = 0;
    for (const slot of this.#slots) {
      if (!slot.used) continue;
      const age = songTimeMs - slot.bornAtMs;
      if (age < 0 || age > slot.lifeMs) continue;
      const target = this.#pool[count];
      target.x = slot.x + slot.vx * age;
      target.y = slot.y + slot.vy * age;
      target.alpha = 1 - age / slot.lifeMs;
      target.color = slot.color;
      this.#output[count] = target;
      count += 1;
    }
    this.#output.length = count;
    return this.#output;
  }

  clear(): void {
    for (const slot of this.#slots) slot.used = false;
    this.#output.length = 0;
  }
}
