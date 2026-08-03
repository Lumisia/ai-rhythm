import type { Clock } from "../audio/Clock";
import type { InputRecorder } from "../core/InputRecorder";
import type { JudgmentEngine, JudgmentEvent } from "../core/JudgmentEngine";

export interface KeyboardInputOptions {
  target?: EventTarget;
  bindings: ReadonlyMap<string, number>;
  clock: Clock;
  engine: Pick<JudgmentEngine, "keyDown" | "keyUp">;
  recorder: InputRecorder;
  onJudgment?: (event: JudgmentEvent) => void;
  onLaneDown?: (lane: number, timeMs: number) => void;
  onLaneUp?: (lane: number, timeMs: number) => void;
}

export class KeyboardInput {
  readonly #target: EventTarget;
  readonly #bindings: ReadonlyMap<string, number>;
  readonly #clock: Clock;
  readonly #engine: Pick<JudgmentEngine, "keyDown" | "keyUp">;
  readonly #recorder: InputRecorder;
  readonly #onJudgment?: (event: JudgmentEvent) => void;
  readonly #onLaneDown?: (lane: number, timeMs: number) => void;
  readonly #onLaneUp?: (lane: number, timeMs: number) => void;
  readonly #heldCodes = new Set<string>();
  #attached = false;

  constructor(options: KeyboardInputOptions) {
    this.#target = options.target ?? window;
    this.#bindings = options.bindings;
    this.#clock = options.clock;
    this.#engine = options.engine;
    this.#recorder = options.recorder;
    this.#onJudgment = options.onJudgment;
    this.#onLaneDown = options.onLaneDown;
    this.#onLaneUp = options.onLaneUp;
  }

  attach(): void {
    if (this.#attached) return;
    this.#target.addEventListener("keydown", this.#handleKeyDown);
    this.#target.addEventListener("keyup", this.#handleKeyUp);
    this.#target.addEventListener("blur", this.#handleBlur);
    this.#attached = true;
  }

  detach(): void {
    if (!this.#attached) return;
    this.#target.removeEventListener("keydown", this.#handleKeyDown);
    this.#target.removeEventListener("keyup", this.#handleKeyUp);
    this.#target.removeEventListener("blur", this.#handleBlur);
    this.#heldCodes.clear();
    this.#attached = false;
  }

  readonly #handleKeyDown = (rawEvent: Event): void => {
    if (!(rawEvent instanceof KeyboardEvent)) return;
    const lane = this.#bindings.get(rawEvent.code);
    if (lane === undefined) return;
    rawEvent.preventDefault();
    if (rawEvent.repeat || this.#heldCodes.has(rawEvent.code)) return;
    this.#heldCodes.add(rawEvent.code);
    const timeMs = this.#clock.performanceToSongTimeMs(rawEvent.timeStamp);
    this.#recorder.record({ action: "DOWN", lane, code: rawEvent.code, timeMs });
    // 판정보다 먼저 알린다. 판정이 안 붙는 입력도 화면에 반응이 있어야
    // 키 바인딩이 틀린 건지 노트가 없는 건지 구분된다.
    this.#onLaneDown?.(lane, timeMs);
    const judgment = this.#engine.keyDown(lane, timeMs);
    if (judgment) this.#onJudgment?.(judgment);
  };

  readonly #handleKeyUp = (rawEvent: Event): void => {
    if (!(rawEvent instanceof KeyboardEvent)) return;
    if (!this.#bindings.has(rawEvent.code)) return;
    rawEvent.preventDefault();
    const timeMs = this.#clock.performanceToSongTimeMs(rawEvent.timeStamp);
    this.#release(rawEvent.code, timeMs);
  };

  #release(code: string, timeMs: number): void {
    const lane = this.#bindings.get(code);
    if (lane === undefined || !this.#heldCodes.delete(code)) return;
    this.#recorder.record({ action: "UP", lane, code, timeMs });
    this.#onLaneUp?.(lane, timeMs);
    const judgment = this.#engine.keyUp(lane, timeMs);
    if (judgment) this.#onJudgment?.(judgment);
  }

  readonly #handleBlur = (): void => {
    const timeMs = this.#clock.songTimeMs();
    for (const code of [...this.#heldCodes]) {
      this.#release(code, timeMs);
    }
  };
}
