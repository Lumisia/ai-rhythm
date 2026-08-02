import type { ChartNote } from "./types";

function lowerBound(notes: readonly ChartNote[], timeMs: number): number {
  let low = 0;
  let high = notes.length;
  while (low < high) {
    const middle = Math.floor((low + high) / 2);
    if (notes[middle].timeMs < timeMs) low = middle + 1;
    else high = middle;
  }
  return low;
}

function upperBound(notes: readonly ChartNote[], timeMs: number): number {
  let low = 0;
  let high = notes.length;
  while (low < high) {
    const middle = Math.floor((low + high) / 2);
    if (notes[middle].timeMs <= timeMs) low = middle + 1;
    else high = middle;
  }
  return low;
}

export class NoteTimeline {
  readonly #notes: readonly ChartNote[];
  readonly #maximumHoldDurationMs: number;

  constructor(notes: readonly ChartNote[]) {
    this.#notes = [...notes].sort((left, right) => left.timeMs - right.timeMs || left.id - right.id);
    this.#maximumHoldDurationMs = this.#notes.reduce(
      (maximum, note) => Math.max(maximum, note.type === "HOLD" ? (note.durationMs ?? 0) : 0),
      0,
    );
  }

  get maximumHoldDurationMs(): number {
    return this.#maximumHoldDurationMs;
  }

  visibleBetween(startMs: number, endMs: number): readonly ChartNote[] {
    if (endMs < startMs) return [];
    return this.#notes.slice(lowerBound(this.#notes, startMs), upperBound(this.#notes, endMs));
  }
}
