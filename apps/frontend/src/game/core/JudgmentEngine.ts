import { classifyError, loadJudgmentConfig } from "./judgment-config";
import type {
  ChartNote,
  HitJudgmentName,
  JudgmentConfig,
  JudgmentName,
  JudgmentPreset,
  JudgmentWindows,
} from "./types";

export type JudgmentPhase = "HEAD" | "TAIL";

export interface JudgmentEvent {
  noteId: number;
  lane: number;
  noteType: ChartNote["type"];
  phase: JudgmentPhase;
  judgment: JudgmentName;
  errMs: number;
  timeMs: number;
  noteTimeMs: number;
}

type NoteState = "PENDING" | "ACTIVE" | "DONE";

interface RuntimeNote {
  note: ChartNote;
  state: NoteState;
}

const orderedWindows: Array<keyof JudgmentWindows> = ["PERFECT", "GREAT", "GOOD", "BAD"];

function tailTime(note: ChartNote): number {
  return note.timeMs + (note.durationMs ?? 0);
}

function classifyScaledError(
  errorMs: number,
  windows: JudgmentWindows,
  scale: number,
): JudgmentName {
  const absoluteErrorMs = Math.abs(errorMs);
  for (const judgment of orderedWindows) {
    if (absoluteErrorMs <= windows[judgment] * scale) {
      return judgment as HitJudgmentName;
    }
  }
  return "MISS";
}

export class JudgmentEngine {
  readonly #notesByLane = new Map<number, RuntimeNote[]>();
  readonly #runtimeNotes: RuntimeNote[];
  readonly #config: JudgmentConfig;
  readonly #preset: JudgmentPreset;

  constructor(
    notes: readonly ChartNote[],
    preset?: JudgmentPreset,
    config: JudgmentConfig = loadJudgmentConfig(),
  ) {
    this.#config = config;
    this.#preset = preset ?? config.default;
    const ids = new Set<number>();

    this.#runtimeNotes = [...notes]
      .sort((left, right) => left.timeMs - right.timeMs || left.id - right.id)
      .map((note) => {
        if (ids.has(note.id)) throw new Error(`duplicate note id: ${note.id}`);
        if (note.type === "HOLD" && (!note.durationMs || note.durationMs <= 0)) {
          throw new Error(`hold note ${note.id} requires a positive durationMs`);
        }
        ids.add(note.id);
        return { note, state: "PENDING" as const };
      });

    for (const runtimeNote of this.#runtimeNotes) {
      const lane = this.#notesByLane.get(runtimeNote.note.lane) ?? [];
      lane.push(runtimeNote);
      this.#notesByLane.set(runtimeNote.note.lane, lane);
    }
  }

  keyDown(lane: number, timeMs: number): JudgmentEvent | null {
    const windows = this.#config.presets[this.#preset];
    const candidate = this.#nearest(
      lane,
      timeMs,
      "PENDING",
      (runtimeNote) => runtimeNote.note.timeMs,
      windows.BAD,
    );
    if (!candidate) return null;

    const errorMs = timeMs - candidate.note.timeMs;
    const event = this.#event(candidate.note, "HEAD", classifyError(errorMs, this.#preset, this.#config), errorMs, timeMs);
    candidate.state = candidate.note.type === "HOLD" ? "ACTIVE" : "DONE";
    return event;
  }

  keyUp(lane: number, timeMs: number): JudgmentEvent | null {
    const windows = this.#config.presets[this.#preset];
    const releaseWindowMs = windows.BAD * this.#config.holdReleaseScale;
    const candidate = this.#nearest(lane, timeMs, "ACTIVE", ({ note }) => tailTime(note), releaseWindowMs);
    if (!candidate) return null;

    const targetTimeMs = tailTime(candidate.note);
    const errorMs = timeMs - targetTimeMs;
    const judgment = classifyScaledError(errorMs, windows, this.#config.holdReleaseScale);
    candidate.state = "DONE";
    return this.#event(candidate.note, "TAIL", judgment, errorMs, timeMs, targetTimeMs);
  }

  advance(timeMs: number): JudgmentEvent[] {
    const events: JudgmentEvent[] = [];
    const releaseWindowMs =
      this.#config.presets[this.#preset].BAD * this.#config.holdReleaseScale;

    for (const runtimeNote of this.#runtimeNotes) {
      const { note, state } = runtimeNote;
      if (state === "PENDING" && timeMs > note.timeMs + this.#config.missAfterMs) {
        runtimeNote.state = "DONE";
        events.push(this.#event(note, "HEAD", "MISS", timeMs - note.timeMs, timeMs));
      } else if (state === "ACTIVE") {
        const targetTimeMs = tailTime(note);
        if (timeMs > targetTimeMs + releaseWindowMs) {
          runtimeNote.state = "DONE";
          events.push(
            this.#event(note, "TAIL", "MISS", timeMs - targetTimeMs, timeMs, targetTimeMs),
          );
        }
      }
    }

    return events.sort((left, right) => left.noteTimeMs - right.noteTimeMs || left.noteId - right.noteId);
  }

  reset(): void {
    for (const runtimeNote of this.#runtimeNotes) {
      runtimeNote.state = "PENDING";
    }
  }

  #nearest(
    lane: number,
    timeMs: number,
    requiredState: NoteState,
    targetTime: (runtimeNote: RuntimeNote) => number,
    windowMs: number,
  ): RuntimeNote | null {
    let best: RuntimeNote | null = null;
    let bestDistance = Number.POSITIVE_INFINITY;

    for (const runtimeNote of this.#notesByLane.get(lane) ?? []) {
      if (runtimeNote.state !== requiredState) continue;
      const distance = Math.abs(timeMs - targetTime(runtimeNote));
      if (
        distance <= windowMs &&
        (distance < bestDistance ||
          (distance === bestDistance && best !== null && runtimeNote.note.timeMs < best.note.timeMs))
      ) {
        best = runtimeNote;
        bestDistance = distance;
      }
    }
    return best;
  }

  #event(
    note: ChartNote,
    phase: JudgmentPhase,
    judgment: JudgmentName,
    errMs: number,
    timeMs: number,
    noteTimeMs = note.timeMs,
  ): JudgmentEvent {
    return {
      noteId: note.id,
      lane: note.lane,
      noteType: note.type,
      phase,
      judgment,
      errMs,
      timeMs,
      noteTimeMs,
    };
  }
}
