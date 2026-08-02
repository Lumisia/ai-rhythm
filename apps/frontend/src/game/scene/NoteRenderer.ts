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
    // 판정선이 85% 지점이라 활주로가 화면 높이의 0.85 다. 640px 캔버스면
    // 544px — 배속 1.0 에서 약 900ms 를 흐른다. VSRG 들이 편하다고 보는
    // 대역(대략 550~900ms)의 느린 쪽 끝이다.
    this.#pxPerMs = options.pxPerMs ?? 0.6;
    // 레인이 80px 로 좁아졌으니 노트도 그에 맞게 두툼해야 한다. 10px 은
    // 좁은 레인에서 실 한 가닥처럼 보인다.
    this.#noteHeight = options.noteHeight ?? 18;
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
    // 마스크가 판정선에서 자르므로 과거 창은 노트 한 칸이면 충분하다.
    const pastWindowMs = this.#noteHeight / pixelsPerMs;
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
    const inset = Math.max(2, lane.width * 0.06);
    const x = lane.x + inset;
    const width = Math.max(2, lane.width - inset * 2);

    // 판정선 아래는 계기 영역이다. 노트가 흘러들면 스코프를 덮으므로
    // 모든 도형을 판정선에서 자른다.
    const floor = this.#judgeLineY;
    if (note.type === "HOLD" && note.durationMs) {
      const tailY =
        this.#judgeLineY - (note.timeMs + note.durationMs - songTimeMs) * pixelsPerMs;
      const top = Math.min(headY, tailY);
      const bottom = Math.min(floor, Math.max(headY, tailY));
      if (bottom <= 0 || top > floor) return false;
      const bodyTop = Math.max(0, top);
      const bodyHeight = bottom - bodyTop;
      if (bodyHeight > 0) {
        // 몸통은 어둡게 깔고 양옆에 밝은 선을 세운다. 단노트와 같은 채움이면
        // 흐르는 중에 둘을 구분할 수 없다.
        const bodyInset = Math.max(3, width * 0.16);
        this.#graphics.fillStyle(lane.color, 0.3);
        this.#graphics.fillRect(x + bodyInset, bodyTop, width - bodyInset * 2, bodyHeight);
        this.#graphics.fillStyle(lane.color, 0.72);
        this.#graphics.fillRect(x + bodyInset, bodyTop, 2, bodyHeight);
        this.#graphics.fillRect(x + width - bodyInset - 2, bodyTop, 2, bodyHeight);
      }
      // 꼬리 끝을 막아 어디서 떼야 하는지 보이게 한다.
      this.#drawCap(lane.color, x, tailY, width, 0.9);
    } else if (headY - this.#noteHeight / 2 > floor || headY < -this.#noteHeight) {
      return false;
    }

    this.#drawCap(lane.color, x, headY, width, 1);
    return true;
  }

  /** 노트 머리. 위쪽에 밝은 띠를 얹어 흐르는 방향이 읽히게 한다.
   *
   * 판정선 아래로 삐져나온 부분은 잘린다.
   */
  #drawCap(color: number, x: number, centerY: number, width: number, alpha: number): void {
    const top = centerY - this.#noteHeight / 2;
    const bottom = Math.min(this.#judgeLineY, centerY + this.#noteHeight / 2);
    const visibleTop = Math.max(0, top);
    const height = bottom - visibleTop;
    if (height <= 0) return;
    this.#graphics.fillStyle(color, 0.55 * alpha);
    this.#graphics.fillRect(x, visibleTop, width, height);
    const capHeight = Math.min(height, Math.max(3, this.#noteHeight * 0.34) - (visibleTop - top));
    if (capHeight > 0) {
      this.#graphics.fillStyle(color, alpha);
      this.#graphics.fillRect(x, visibleTop, width, capHeight);
      this.#graphics.fillStyle(0xffffff, 0.5 * alpha);
      this.#graphics.fillRect(x, visibleTop, width, 1);
    }
  }
}
