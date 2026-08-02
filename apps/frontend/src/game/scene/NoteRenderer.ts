import Phaser from "phaser";

import type { LaneGeometry } from "../core/LaneLayout";
import type { ChartNote } from "../core/types";
import { NoteTimeline } from "../core/NoteTimeline";

export interface NoteRendererOptions {
  width: number;
  height: number;
  judgeLineY: number;
  pxPerMs?: number;
  noteHeight?: number;
}

export class NoteRenderer {
  readonly #graphics: Phaser.GameObjects.Graphics;
  readonly #timeline: NoteTimeline;
  #lanes: readonly LaneGeometry[];
  #width: number;
  #height: number;
  #judgeLineY: number;
  readonly #pxPerMs: number;
  readonly #noteHeight: number;

  constructor(
    scene: Phaser.Scene,
    timeline: NoteTimeline,
    lanes: readonly LaneGeometry[],
    options: NoteRendererOptions,
  ) {
    this.#graphics = scene.add.graphics().setDepth(10);
    this.#timeline = timeline;
    this.#lanes = lanes;
    this.#width = options.width;
    this.#height = options.height;
    this.#judgeLineY = options.judgeLineY;
    this.#pxPerMs = options.pxPerMs ?? 0.42;
    this.#noteHeight = options.noteHeight ?? 10;
  }

  resize(
    width: number,
    height: number,
    judgeLineY: number,
    lanes: readonly LaneGeometry[],
  ): void {
    this.#width = width;
    this.#height = height;
    this.#judgeLineY = judgeLineY;
    this.#lanes = lanes;
  }

  update(songTimeMs: number, scrollSpeed: number): number {
    this.#graphics.clear();
    if (scrollSpeed <= 0 || !Number.isFinite(scrollSpeed)) return 0;
    const pixelsPerMs = this.#pxPerMs * scrollSpeed;
    const pastWindowMs = (this.#height - this.#judgeLineY) / pixelsPerMs;
    const futureWindowMs = this.#judgeLineY / pixelsPerMs;
    const notes = this.#timeline.visibleBetween(
      songTimeMs - pastWindowMs - this.#timeline.maximumHoldDurationMs,
      songTimeMs + futureWindowMs,
    );
    let drawn = 0;
    for (const note of notes) {
      if (this.#drawNote(note, songTimeMs, pixelsPerMs)) drawn += 1;
    }
    return drawn;
  }

  destroy(): void {
    this.#graphics.destroy();
  }

  #drawNote(note: ChartNote, songTimeMs: number, pixelsPerMs: number): boolean {
    const lane = this.#lanes[note.lane];
    if (!lane) return false;
    const headY = this.#judgeLineY - (note.timeMs - songTimeMs) * pixelsPerMs;
    const inset = Math.max(3, lane.width * 0.08);
    const x = lane.x + inset;
    const width = Math.max(2, lane.width - inset * 2);

    if (note.type === "HOLD" && note.durationMs) {
      const tailY =
        this.#judgeLineY - (note.timeMs + note.durationMs - songTimeMs) * pixelsPerMs;
      const top = Math.min(headY, tailY);
      const bottom = Math.max(headY, tailY);
      if (bottom < -this.#noteHeight || top > this.#height + this.#noteHeight) return false;
      const bodyTop = Math.max(-this.#noteHeight, top);
      const bodyHeight = Math.max(this.#noteHeight, bottom - top);
      // 몸통은 어둡게 깔고 양옆에 밝은 선을 세운다. 단노트와 같은 채움이면
      // 흐르는 중에 둘을 구분할 수 없다.
      this.#graphics.fillStyle(lane.color, 0.28);
      this.#graphics.fillRect(x, bodyTop, width, bodyHeight);
      this.#graphics.fillStyle(lane.color, 0.85);
      this.#graphics.fillRect(x, bodyTop, 2, bodyHeight);
      this.#graphics.fillRect(x + width - 2, bodyTop, 2, bodyHeight);
      // 꼬리 끝을 막아 어디서 떼야 하는지 보이게 한다.
      if (tailY > -this.#noteHeight && tailY < this.#height + this.#noteHeight) {
        this.#graphics.fillRect(x, tailY - 2, width, 3);
      }
    } else if (headY < -this.#noteHeight || headY > this.#height + this.#noteHeight) {
      return false;
    }

    this.#graphics.fillStyle(lane.color, 1);
    this.#graphics.fillRect(x, headY - this.#noteHeight / 2, width, this.#noteHeight);
    return true;
  }
}
