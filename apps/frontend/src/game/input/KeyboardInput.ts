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
}

export class KeyboardInput {
  readonly #target: EventTarget;
  readonly #bindings: ReadonlyMap<string, number>;
  readonly #clock: Clock;
  readonly #engine: Pick<JudgmentEngine, "keyDown" | "keyUp">;
  readonly #recorder: InputRecorder;
  readonly #onJudgment?: (event: JudgmentEvent) => void;
  readonly #heldCodes = new Set<string>();
  #attached = false;

  constructor(options: KeyboardInputOptions) {
    this.#target = options.target ?? window;
    this.#bindings = options.bindings;
    this.#clock = options.clock;
    this.#engine = options.engine;
    this.#recorder = options.recorder;
    this.#onJudgment = options.onJudgment;
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
    const judgment = this.#engine.keyDown(lane, timeMs);
    if (judgment) this.#onJudgment?.(judgment);
  };

  readonly #handleKeyUp = (rawEvent: Event): void => {
    if (!(rawEvent instanceof KeyboardEvent)) return;
    const lane = this.#bindings.get(rawEvent.code);
    if (lane === undefined) return;
    rawEvent.preventDefault();
    if (!this.#heldCodes.delete(rawEvent.code)) return;
    const timeMs = this.#clock.performanceToSongTimeMs(rawEvent.timeStamp);
    this.#recorder.record({ action: "UP", lane, code: rawEvent.code, timeMs });
    const judgment = this.#engine.keyUp(lane, timeMs);
    if (judgment) this.#onJudgment?.(judgment);
  };

  readonly #handleBlur = (): void => {
    this.#heldCodes.clear();
  };
}
