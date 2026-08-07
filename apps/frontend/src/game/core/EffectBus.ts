import type { JudgmentName } from "./types";

export type EffectEvent =
  | {
      type: "JUDGED";
      judgment: JudgmentName;
      lane: number;
      errMs: number;
      phase: "HEAD" | "TAIL";
      combo: number;
      songTimeMs: number;
    }
  | { type: "HOLD_TICK"; lane: number; combo: number; songTimeMs: number }
  | { type: "FEVER_START"; songTimeMs: number }
  | { type: "FEVER_END"; songTimeMs: number }
  | { type: "MARKER"; label: string; songTimeMs: number };

export function feverActiveFromEffect(event: EffectEvent): boolean | null {
  if (event.type === "FEVER_START") return true;
  if (event.type === "FEVER_END") return false;
  return null;
}

export interface EffectSubscriber {
  handleEffect(event: EffectEvent): void;
}

/** 판정·홀드 틱·FEVER 를 이펙트 구독자에게 흘린다.
 *
 * Phaser 와 DOM 을 모른다. 씬이 수신자 목록을 알면 이펙트를 늘릴 때마다
 * 씬을 고쳐야 하고, 판정 로직을 Phaser 없이 검증할 수 없게 된다.
 */
export class EffectBus {
  readonly #subscribers: EffectSubscriber[] = [];

  subscribe(subscriber: EffectSubscriber): () => void {
    this.#subscribers.push(subscriber);
    return () => {
      const index = this.#subscribers.indexOf(subscriber);
      if (index >= 0) this.#subscribers.splice(index, 1);
    };
  }

  /** 구독자 하나가 던져도 나머지에게 전달한다.
   *
   * 이펙트 레이어의 렌더 버그가 HUD 갱신이나 점수 반영을 막으면
   * 검수 도구가 통째로 멈춘다.
   */
  emit(event: EffectEvent): void {
    for (const subscriber of [...this.#subscribers]) {
      try {
        subscriber.handleEffect(event);
      } catch (caught) {
        console.error("effect subscriber failed", caught);
      }
    }
  }
}
