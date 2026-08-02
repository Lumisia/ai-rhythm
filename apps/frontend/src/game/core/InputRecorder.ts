export type InputAction = "DOWN" | "UP";

export interface RecordedInputEvent {
  action: InputAction;
  lane: number;
  code: string;
  timeMs: number;
}

export class InputRecorder {
  readonly #events: RecordedInputEvent[] = [];

  record(event: RecordedInputEvent): void {
    if (!Number.isInteger(event.lane) || event.lane < 0) {
      throw new Error(`invalid input lane: ${event.lane}`);
    }
    if (!Number.isFinite(event.timeMs)) {
      throw new Error(`invalid input time: ${event.timeMs}`);
    }
    this.#events.push({ ...event });
  }

  snapshot(): readonly Readonly<RecordedInputEvent>[] {
    return this.#events.map((event) => Object.freeze({ ...event }));
  }

  clear(): void {
    this.#events.length = 0;
  }
}
