import Phaser from "phaser";

import type { EffectEvent, EffectSubscriber } from "../core/EffectBus";
import type { LaneGeometry } from "../core/LaneLayout";
import { ParticleField } from "../core/ParticleField";
import type { JudgmentName } from "../core/types";
import { DEPTH } from "./renderDepth";

const POOL_CAPACITY = 128;
const PARTICLE_LIFE_MS = 260;
const TICK_PARTICLE_LIFE_MS = 180;
/** 판정선 위로 이펙트가 올라갈 수 있는 최대 높이.
 *
 * 더 올라가면 다음 노트를 가린다. 노트 높이가 18px 이라 40px 은 두 칸 남짓이다.
 */
const RISE_LIMIT_PX = 40;
const PULSE_MS = 90;
const MISS_GLOW_MS = 120;
const FEVER_PARTICLE_SCALE = 1.6;

const PARTICLE_COUNT: Record<JudgmentName, number> = {
  PERFECT: 5,
  GREAT: 3,
  GOOD: 0,
  BAD: 0,
  MISS: 0,
};

const MISS_COLOR = 0xf87171;

export class HitEffectLayer implements EffectSubscriber {
  readonly #graphics: Phaser.GameObjects.Graphics;
  readonly #field = new ParticleField(POOL_CAPACITY);
  readonly #pulseAtMs = new Map<number, number>();
  readonly #missAtMs = new Map<number, number>();
  #lanes: readonly LaneGeometry[];
  #judgeLineY: number;
  #feverActive = false;
  #reduceMotion = false;

  constructor(scene: Phaser.Scene, lanes: readonly LaneGeometry[], judgeLineY: number) {
    this.#graphics = scene.add.graphics().setDepth(DEPTH.HIT_EFFECT);
    this.#lanes = lanes;
    this.#judgeLineY = judgeLineY;
  }

  resize(lanes: readonly LaneGeometry[], judgeLineY: number): void {
    this.#lanes = lanes;
    this.#judgeLineY = judgeLineY;
    this.#field.clear();
  }

  setFeverActive(active: boolean): void {
    this.#feverActive = active;
  }

  setReduceMotion(reduce: boolean): void {
    this.#reduceMotion = reduce;
    if (reduce) this.#field.clear();
  }

  handleEffect(event: EffectEvent): void {
    if (event.type === "JUDGED") {
      this.#pulseAtMs.set(event.lane, event.songTimeMs);
      if (event.judgment === "MISS") {
        this.#missAtMs.set(event.lane, event.songTimeMs);
        return;
      }
      this.#burst(event.lane, PARTICLE_COUNT[event.judgment], PARTICLE_LIFE_MS, event.songTimeMs);
      return;
    }
    if (event.type === "HOLD_TICK") {
      this.#burst(event.lane, 1, TICK_PARTICLE_LIFE_MS, event.songTimeMs);
    }
  }

  update(songTimeMs: number): void {
    const graphics = this.#graphics;
    graphics.clear();
    // MISS 기운은 모션 감소에서도 남긴다. 움직이지 않는 레인 국소 착색이고,
    // 무엇보다 놓쳤다는 **정보**다. 펄스는 장식이라 가드 뒤로 내린다.
    this.#drawMissGlow(songTimeMs);
    if (this.#reduceMotion) return;
    this.#drawPulse(songTimeMs);

    for (const particle of this.#field.activeAt(songTimeMs)) {
      if (particle.y < this.#judgeLineY - RISE_LIMIT_PX) continue;
      graphics.fillStyle(particle.color, particle.alpha);
      graphics.fillRect(particle.x - 2, particle.y - 2, 4, 4);
    }
  }

  destroy(): void {
    this.#graphics.destroy();
  }

  /** 판정선에서 위쪽 부채꼴로 튕긴다. 노트가 온 방향으로 되돌리는 움직임이다. */
  #burst(laneIndex: number, count: number, lifeMs: number, songTimeMs: number): void {
    if (this.#reduceMotion || count <= 0) return;
    const lane = this.#lanes[laneIndex];
    if (!lane) return;
    const total = this.#feverActive ? Math.round(count * FEVER_PARTICLE_SCALE) : count;
    const centerX = lane.x + lane.width / 2;
    const riseSpeed = RISE_LIMIT_PX / lifeMs;

    for (let index = 0; index < total; index += 1) {
      // 부채꼴을 결정적으로 편다. 난수를 쓰면 같은 플레이를 두 번 봐도 다르게 보인다.
      const spread = total === 1 ? 0 : (index / (total - 1)) * 2 - 1;
      this.#field.spawn(
        {
          x: centerX + spread * (lane.width * 0.35),
          y: this.#judgeLineY - 2,
          vx: (spread * lane.width * 0.35) / lifeMs,
          vy: -riseSpeed,
          lifeMs,
          color: lane.color,
        },
        songTimeMs,
      );
    }
  }

  #drawPulse(songTimeMs: number): void {
    for (const [laneIndex, atMs] of this.#pulseAtMs) {
      const age = songTimeMs - atMs;
      if (age < 0 || age > PULSE_MS) continue;
      const lane = this.#lanes[laneIndex];
      if (!lane) continue;
      const strength = 1 - age / PULSE_MS;
      this.#graphics.fillStyle(lane.color, strength);
      this.#graphics.fillRect(lane.x, this.#judgeLineY - 4, lane.width, 4);
    }
  }

  /** MISS 는 레인 국소로만 표시한다. 전체 화면 붉은 처리는 WCAG 2.3.1 위험이 있다. */
  #drawMissGlow(songTimeMs: number): void {
    for (const [laneIndex, atMs] of this.#missAtMs) {
      const age = songTimeMs - atMs;
      if (age < 0 || age > MISS_GLOW_MS) continue;
      const lane = this.#lanes[laneIndex];
      if (!lane) continue;
      this.#graphics.fillStyle(MISS_COLOR, 0.35 * (1 - age / MISS_GLOW_MS));
      this.#graphics.fillRect(lane.x, this.#judgeLineY - 24, lane.width, 24);
    }
  }
}
