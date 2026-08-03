import Phaser from "phaser";

import type { LaneGeometry, StageGeometry } from "../core/LaneLayout";

const HAIRLINE = 0x31364d;
const ACCENT = 0x5eead4;
const OUTSIDE = 0x07080f;

/** 리셉터 높이. 노트가 여기 겹치면 친다. */
export const RECEPTOR_HEIGHT = 26;

/** 판정선이 화면 위에서 차지하는 비율.
 *
 * osu!lazer 는 바닥에서 110px 위(768 기준 85.7%)에 둔다. 위로 올릴수록
 * 노트를 읽을 활주로가 짧아진다. 아래 계기 영역이 탐나도 여기는 못 건드린다.
 */
export const JUDGE_LINE_RATIO = 0.85;

/** 무대 배경·레인 구분선·리셉터.
 *
 * 매 프레임 다시 그리지 않는다. 눌림 표시만 따로 올린다.
 */
export class StageRenderer {
  readonly #background: Phaser.GameObjects.Graphics;
  readonly #receptors: Phaser.GameObjects.Graphics;
  readonly #keyTexts: Phaser.GameObjects.Text[] = [];

  constructor(
    scene: Phaser.Scene,
    private stage: StageGeometry,
    private width: number,
    private height: number,
    private judgeLineY: number,
    keyLabels: readonly string[],
  ) {
    this.#background = scene.add.graphics().setDepth(0);
    this.#receptors = scene.add.graphics().setDepth(2);
    for (const label of keyLabels) {
      this.#keyTexts.push(
        scene.add
          .text(0, 0, label, {
            fontFamily: '"Cascadia Mono", Consolas, monospace',
            fontSize: "12px",
            color: "#c5cbea",
          })
          .setDepth(3)
          .setOrigin(0.5, 0),
      );
    }
    this.redraw();
  }

  resize(stage: StageGeometry, width: number, height: number, judgeLineY: number): void {
    this.stage = stage;
    this.width = width;
    this.height = height;
    this.judgeLineY = judgeLineY;
    this.redraw();
  }

  get lanes(): readonly LaneGeometry[] {
    return this.stage.lanes;
  }

  destroy(): void {
    this.#background.destroy();
    this.#receptors.destroy();
    for (const text of this.#keyTexts) text.destroy();
  }

  redraw(): void {
    const graphics = this.#background;
    graphics.clear();
    // 무대 바깥은 더 어둡게 눌러서 시선이 무대 안에 갇히게 한다.
    graphics.fillStyle(OUTSIDE, 1);
    graphics.fillRect(0, 0, this.width, this.height);

    for (const lane of this.stage.lanes) {
      graphics.fillStyle(lane.backgroundColor, 1);
      graphics.fillRect(lane.x, 0, lane.width, this.judgeLineY);
      // 판정선으로 갈수록 레인이 밝아진다. 하드 사각으로 얹으면 위쪽에
      // 가로 이음매가 띠처럼 보인다. 그라디언트로 흘려야 한다.
      const washTop = Math.max(0, this.judgeLineY - 200);
      graphics.fillGradientStyle(lane.color, lane.color, lane.color, lane.color, 0, 0, 0.06, 0.06);
      graphics.fillRect(lane.x, washTop, lane.width, this.judgeLineY - washTop);
    }

    graphics.lineStyle(1, HAIRLINE, 0.85);
    for (const lane of this.stage.lanes) {
      graphics.lineBetween(lane.x, 0, lane.x, this.judgeLineY);
    }
    graphics.lineBetween(this.stage.right, 0, this.stage.right, this.judgeLineY);

    // 무대 좌우 테두리. 바깥 어둠과 무대를 가르는 선이다.
    graphics.lineStyle(2, HAIRLINE, 1);
    graphics.lineBetween(this.stage.left - 1, 0, this.stage.left - 1, this.height);
    graphics.lineBetween(this.stage.right + 1, 0, this.stage.right + 1, this.height);

    this.#drawReceptorBase(graphics);
    this.#layoutKeyLabels();
    this.setPressed(new Set(), new Map(), 0);
  }

  /** 눌린 레인을 밝힌다. 판정과 무관하게 반응해야 입력이 먹은 걸 안다. */
  setPressed(
    held: ReadonlySet<number>,
    flashAtMs: ReadonlyMap<number, number>,
    songTimeMs: number,
    flashDurationMs = 130,
  ): void {
    const graphics = this.#receptors;
    graphics.clear();
    for (const lane of this.stage.lanes) {
      const pressedAtMs = flashAtMs.get(lane.index);
      const age = pressedAtMs === undefined ? Infinity : songTimeMs - pressedAtMs;
      const flash = age >= 0 && age < flashDurationMs ? 1 - age / flashDurationMs : 0;
      const strength = held.has(lane.index) ? Math.max(0.45, flash) : flash;
      if (strength <= 0) continue;

      // 레인 전체를 위로 흐르는 빛기둥. 위로 갈수록 사라진다.
      const columnTop = Math.max(0, this.judgeLineY - 220);
      graphics.fillGradientStyle(
        lane.color,
        lane.color,
        lane.color,
        lane.color,
        0,
        0,
        0.22 * strength,
        0.22 * strength,
      );
      graphics.fillRect(lane.x, columnTop, lane.width, this.judgeLineY - columnTop);

      graphics.fillStyle(lane.color, 0.55 * strength);
      graphics.fillRect(lane.x + 1, this.judgeLineY - RECEPTOR_HEIGHT, lane.width - 2, RECEPTOR_HEIGHT);
      graphics.fillStyle(lane.color, strength);
      graphics.fillRect(lane.x + 1, this.judgeLineY - 3, lane.width - 2, 3);
    }
  }

  /** 판정선 위에 앉는 키 블록. 노트가 어디로 들어오는지 못박는다. */
  #drawReceptorBase(graphics: Phaser.GameObjects.Graphics): void {
    const top = this.judgeLineY - RECEPTOR_HEIGHT;
    for (const lane of this.stage.lanes) {
      graphics.fillStyle(0x171a2a, 1);
      graphics.fillRect(lane.x + 1, top, lane.width - 2, RECEPTOR_HEIGHT);
      graphics.fillStyle(lane.color, 0.22);
      graphics.fillRect(lane.x + 1, top, lane.width - 2, 2);
      graphics.lineStyle(1, lane.color, 0.4);
      graphics.strokeRect(lane.x + 1.5, top + 0.5, lane.width - 3, RECEPTOR_HEIGHT - 1);
    }
    // 판정선. 무대 폭만큼만 긋는다 — 화면을 가로지르면 무대가 안 보인다.
    graphics.fillGradientStyle(ACCENT, 0xa78bfa, ACCENT, 0xa78bfa, 1, 1, 1, 1);
    graphics.fillRect(this.stage.left, this.judgeLineY - 2, this.stage.width, 3);
  }

  /** 키 라벨은 판정선 **아래**에 둔다.
   *
   * 리셉터 위에 얹으면 레인이 점등될 때 글자가 배경에 묻힌다. 노트를 읽는
   * 자리도 아니라서 아래로 내려도 잃는 게 없다.
   */
  #layoutKeyLabels(): void {
    this.#keyTexts.forEach((text, index) => {
      const lane = this.stage.lanes[index];
      if (!lane) return void text.setVisible(false);
      text.setVisible(true).setPosition(lane.x + lane.width / 2, this.judgeLineY + 7);
    });
  }
}
