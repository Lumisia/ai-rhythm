import Phaser from "phaser";

import type { JudgmentEvent } from "../core/JudgmentEngine";
import type { LaneGeometry } from "../core/LaneLayout";
import type { ScoreSnapshot } from "../core/ScoreCalculator";
import type { JudgmentName, JudgmentWindows } from "../core/types";

/** 계기판 색. app.css 의 CSS 변수와 같은 값을 쓴다. */
const INK = 0xdce3e8;
const MUTED = 0x8e98a6;
const HAIRLINE = 0x38404e;
const AMBER = 0xe2b75b;
const MINT = 0x78cbb8;
const CORAL = 0xe27268;
const SLATE = 0x252b38;

const JUDGMENT_COLOR: Record<JudgmentName, number> = {
  PERFECT: MINT,
  GREAT: 0x7f9fc2,
  GOOD: AMBER,
  BAD: 0xc98a52,
  MISS: CORAL,
};

const MONO = '"Cascadia Mono", Consolas, monospace';
const DISPLAY = '"Bahnschrift SemiCondensed", "Arial Narrow", sans-serif';

const JUDGMENT_HOLD_MS = 520;
const MARKER_HOLD_MS = 1400;
const SCOPE_TRAIL_MS = 2200;
const SCOPE_TRAIL_MAX = 160;
const LANE_FLASH_MS = 130;
const COMBO_MIN = 5;
const COUNTDOWN_LEAD_MS = 3200;

interface ScopeMark {
  errMs: number;
  atMs: number;
}

export interface HudGeometry {
  width: number;
  height: number;
  judgeLineY: number;
}

export interface HudRendererOptions extends HudGeometry {
  windows: JudgmentWindows;
  durationMs: number;
  noteCount: number;
  firstNoteTimeMs: number;
  beatMs: number;
  keyLabels: readonly string[];
  snapshot: () => ScoreSnapshot;
}

/** 플레이 중 계기 판독부.
 *
 * 게임 UI 가 아니라 측정 계기다. 이 도구는 채보를 검수하려고 쓴다. 그래서
 * 화려한 연출 대신 읽을 수 있는 수치를 놓는다. 타이밍 오차 스코프가
 * 주인공이다 — 입력 보정값을 맞출 수 있는 유일한 물건이고, 그게 없으면
 * "채보 오프셋이 틀렸나 내가 못 친 건가"를 구분할 수 없다.
 */
export class HudRenderer {
  readonly #scene: Phaser.Scene;
  readonly #graphics: Phaser.GameObjects.Graphics;
  readonly #judgmentText: Phaser.GameObjects.Text;
  readonly #errorText: Phaser.GameObjects.Text;
  readonly #comboText: Phaser.GameObjects.Text;
  readonly #statText: Phaser.GameObjects.Text;
  readonly #scopeText: Phaser.GameObjects.Text;
  readonly #markerText: Phaser.GameObjects.Text;
  readonly #countdownText: Phaser.GameObjects.Text;
  readonly #keyTexts: Phaser.GameObjects.Text[] = [];

  readonly #windows: JudgmentWindows;
  readonly #durationMs: number;
  readonly #noteCount: number;
  readonly #firstNoteTimeMs: number;
  readonly #beatMs: number;
  readonly #keyLabels: readonly string[];
  readonly #snapshot: () => ScoreSnapshot;

  #lanes: readonly LaneGeometry[];
  #width: number;
  #height: number;
  #judgeLineY: number;

  readonly #scopeMarks: ScopeMark[] = [];
  readonly #laneFlashAtMs = new Map<number, number>();
  readonly #laneHeld = new Set<number>();
  readonly #markerTimes: number[] = [];
  #lastJudgment: {
    judgment: JudgmentName;
    errMs: number;
    atMs: number;
    isTail: boolean;
  } | null = null;
  #markerShownAtMs = -Infinity;
  #statSignature = "";

  constructor(
    scene: Phaser.Scene,
    lanes: readonly LaneGeometry[],
    options: HudRendererOptions,
  ) {
    this.#scene = scene;
    this.#lanes = lanes;
    this.#width = options.width;
    this.#height = options.height;
    this.#judgeLineY = options.judgeLineY;
    this.#windows = options.windows;
    this.#durationMs = Math.max(1, options.durationMs);
    this.#noteCount = options.noteCount;
    this.#firstNoteTimeMs = options.firstNoteTimeMs;
    this.#beatMs = options.beatMs;
    this.#keyLabels = options.keyLabels;
    this.#snapshot = options.snapshot;

    this.#graphics = scene.add.graphics().setDepth(20);
    this.#statText = this.#text(MONO, 11, MUTED).setDepth(24);
    this.#judgmentText = this.#text(DISPLAY, 30, INK).setDepth(24).setOrigin(0.5, 1);
    this.#errorText = this.#text(MONO, 12, MUTED).setDepth(24).setOrigin(0.5, 0);
    this.#comboText = this.#text(DISPLAY, 46, INK).setDepth(23).setOrigin(0.5, 1);
    this.#scopeText = this.#text(MONO, 10, MUTED).setDepth(24).setOrigin(0.5, 0);
    this.#markerText = this.#text(MONO, 11, CORAL).setDepth(24).setOrigin(1, 0);
    this.#countdownText = this.#text(DISPLAY, 78, AMBER).setDepth(25).setOrigin(0.5, 0.5);
    this.#buildKeyLabels();
    this.#layout();
  }

  /** 키를 눌렀다. 판정과 무관하게 레인이 반응해야 입력이 먹은 걸 안다. */
  pressLane(lane: number, atMs: number): void {
    this.#laneFlashAtMs.set(lane, atMs);
    this.#laneHeld.add(lane);
  }

  releaseLane(lane: number): void {
    this.#laneHeld.delete(lane);
  }

  acceptJudgment(event: JudgmentEvent, songTimeMs: number): void {
    this.#lastJudgment = {
      judgment: event.judgment,
      errMs: event.errMs,
      atMs: songTimeMs,
      isTail: event.phase === "TAIL",
    };
    if (event.judgment === "MISS") return;
    // 스코프는 머리 판정만 받는다. 롱노트 뗀 판정은 완화 배율이 붙어
    // 오차 분포가 달라서, 섞으면 입력 보정값을 잘못 읽는다.
    if (event.phase === "TAIL") return;
    this.#scopeMarks.push({ errMs: event.errMs, atMs: songTimeMs });
    if (this.#scopeMarks.length > SCOPE_TRAIL_MAX) this.#scopeMarks.shift();
  }

  acceptMarker(label: string, songTimeMs: number): void {
    this.#markerText.setText(`▲ ${label}`);
    this.#markerShownAtMs = songTimeMs;
    this.#markerTimes.push(songTimeMs);
  }

  update(songTimeMs: number): void {
    this.#graphics.clear();
    this.#drawLaneFeedback(songTimeMs);
    this.#drawProgressRail(songTimeMs);
    this.#drawScope(songTimeMs);
    this.#drawJudgment(songTimeMs);
    this.#drawCombo();
    this.#drawStats(songTimeMs);
    this.#drawCountdown(songTimeMs);
    this.#markerText.setAlpha(
      Math.max(0, 1 - (songTimeMs - this.#markerShownAtMs) / MARKER_HOLD_MS),
    );
  }

  resize(lanes: readonly LaneGeometry[], geometry: HudGeometry): void {
    this.#lanes = lanes;
    this.#width = geometry.width;
    this.#height = geometry.height;
    this.#judgeLineY = geometry.judgeLineY;
    this.#layout();
  }

  destroy(): void {
    this.#graphics.destroy();
    for (const text of [
      this.#statText,
      this.#judgmentText,
      this.#errorText,
      this.#comboText,
      this.#scopeText,
      this.#markerText,
      this.#countdownText,
      ...this.#keyTexts,
    ]) {
      text.destroy();
    }
  }

  // --- 배치 -----------------------------------------------------------------

  get #scopeY(): number {
    return this.#judgeLineY + (this.#height - this.#judgeLineY) * 0.46;
  }

  get #scopeHalfWidth(): number {
    return Math.min(this.#width * 0.34, 260);
  }

  #text(fontFamily: string, fontSize: number, color: number): Phaser.GameObjects.Text {
    return this.#scene.add.text(0, 0, "", {
      fontFamily,
      fontSize: `${fontSize}px`,
      color: Phaser.Display.Color.IntegerToColor(color).rgba,
    });
  }

  #buildKeyLabels(): void {
    for (const label of this.#keyLabels) {
      this.#keyTexts.push(
        this.#scene.add
          .text(0, 0, label, {
            fontFamily: MONO,
            fontSize: "11px",
            color: Phaser.Display.Color.IntegerToColor(MUTED).rgba,
          })
          .setDepth(24)
          .setOrigin(0.5, 0),
      );
    }
  }

  #layout(): void {
    const centerX = this.#width / 2;
    this.#statText.setPosition(12, 10);
    this.#markerText.setPosition(this.#width - 12, 10);
    this.#judgmentText.setPosition(centerX, this.#judgeLineY - 26);
    this.#errorText.setPosition(centerX, this.#judgeLineY - 22);
    this.#comboText.setPosition(centerX, this.#judgeLineY - 74);
    this.#scopeText.setPosition(centerX, this.#scopeY + 16);
    this.#countdownText.setPosition(centerX, this.#judgeLineY * 0.52);
    this.#keyTexts.forEach((text, lane) => {
      const geometry = this.#lanes[lane];
      if (!geometry) return void text.setVisible(false);
      text.setVisible(true).setPosition(geometry.x + geometry.width / 2, this.#judgeLineY + 7);
    });
  }

  // --- 그리기 ---------------------------------------------------------------

  /** 눌린 레인을 밝힌다. 바인딩이 틀렸는지 판정이 없는 건지 구분해준다. */
  #drawLaneFeedback(songTimeMs: number): void {
    for (const lane of this.#lanes) {
      const pressedAtMs = this.#laneFlashAtMs.get(lane.index);
      const held = this.#laneHeld.has(lane.index);
      const age = pressedAtMs === undefined ? Infinity : songTimeMs - pressedAtMs;
      const flash = age < LANE_FLASH_MS ? 1 - age / LANE_FLASH_MS : 0;
      const strength = held ? Math.max(0.34, flash) : flash;
      if (strength <= 0) continue;
      this.#graphics.fillStyle(lane.color, 0.16 * strength);
      this.#graphics.fillRect(lane.x, 0, lane.width, this.#judgeLineY);
      this.#graphics.fillStyle(lane.color, 0.72 * strength);
      this.#graphics.fillRect(lane.x, this.#judgeLineY - 3, lane.width, 3);
    }
  }

  /** 곡 진행과 찍어둔 문제 마커. 어디쯤에서 기록했는지 눈으로 잡는다. */
  #drawProgressRail(songTimeMs: number): void {
    const y = this.#height - 5;
    this.#graphics.fillStyle(SLATE, 1);
    this.#graphics.fillRect(0, y, this.#width, 3);
    const ratio = Phaser.Math.Clamp(songTimeMs / this.#durationMs, 0, 1);
    this.#graphics.fillStyle(AMBER, 1);
    this.#graphics.fillRect(0, y, this.#width * ratio, 3);
    this.#graphics.fillStyle(CORAL, 1);
    for (const markerTimeMs of this.#markerTimes) {
      const x = this.#width * Phaser.Math.Clamp(markerTimeMs / this.#durationMs, 0, 1);
      this.#graphics.fillRect(x - 1, y - 4, 2, 11);
    }
  }

  /** 타이밍 오차 스코프. 이 화면의 주인공이다.
   *
   * 가운데가 0ms. 최근 타격이 오차 위치에 점으로 남고 서서히 지워진다.
   * 점 무리가 한쪽으로 쏠려 있으면 입력 보정값을 그만큼 옮기면 된다.
   */
  #drawScope(songTimeMs: number): void {
    const centerX = this.#width / 2;
    const y = this.#scopeY;
    const half = this.#scopeHalfWidth;
    const perMs = half / this.#windows.BAD;

    for (const [name, color] of [
      ["BAD", HAIRLINE],
      ["GOOD", HAIRLINE],
      ["GREAT", MUTED],
      ["PERFECT", MINT],
    ] as const) {
      const offset = this.#windows[name] * perMs;
      const height = name === "PERFECT" ? 13 : 9;
      this.#graphics.fillStyle(color, name === "PERFECT" ? 0.75 : 0.5);
      this.#graphics.fillRect(centerX - offset, y - height / 2, 1, height);
      this.#graphics.fillRect(centerX + offset, y - height / 2, 1, height);
    }

    this.#graphics.fillStyle(HAIRLINE, 0.8);
    this.#graphics.fillRect(centerX - half, y, half * 2, 1);
    this.#graphics.fillStyle(AMBER, 1);
    this.#graphics.fillRect(centerX - 1, y - 10, 2, 21);

    let visible = 0;
    for (const mark of this.#scopeMarks) {
      const age = songTimeMs - mark.atMs;
      if (age < 0 || age > SCOPE_TRAIL_MS) continue;
      visible += 1;
      const x = centerX + Phaser.Math.Clamp(mark.errMs, -this.#windows.BAD, this.#windows.BAD) * perMs;
      this.#graphics.fillStyle(INK, 0.85 * (1 - age / SCOPE_TRAIL_MS));
      this.#graphics.fillRect(x - 1, y - 6, 2, 12);
    }

    const snapshot = this.#snapshot();
    if (snapshot.timingSampleCount > 0) {
      const meanX =
        centerX +
        Phaser.Math.Clamp(snapshot.meanErrMs, -this.#windows.BAD, this.#windows.BAD) * perMs;
      this.#graphics.fillStyle(MINT, 1);
      this.#graphics.fillTriangle(meanX, y - 15, meanX - 5, y - 24, meanX + 5, y - 24);
      const sign = snapshot.meanErrMs >= 0 ? "+" : "−";
      const drift = Math.abs(snapshot.meanErrMs).toFixed(1);
      this.#scopeText
        .setText(
          `EARLY ◄ MEAN ${sign}${drift}ms ► LATE    보정 ${(-snapshot.meanErrMs).toFixed(0)}ms 권장`,
        )
        .setVisible(true);
    } else {
      this.#scopeText.setText("EARLY ◄  타이밍 오차  ► LATE").setVisible(visible === 0);
    }
  }

  #drawJudgment(songTimeMs: number): void {
    const last = this.#lastJudgment;
    if (!last) return void this.#judgmentText.setVisible(false).setAlpha(0);
    const age = songTimeMs - last.atMs;
    if (age < 0 || age > JUDGMENT_HOLD_MS) {
      this.#judgmentText.setVisible(false);
      this.#errorText.setVisible(false);
      return;
    }
    const alpha = 1 - (age / JUDGMENT_HOLD_MS) ** 3;
    const color = Phaser.Display.Color.IntegerToColor(JUDGMENT_COLOR[last.judgment]).rgba;
    // 롱노트를 뗀 판정은 완화 배율이 붙어 머리 판정과 기준이 다르다.
    // 표시가 같으면 검수 중에 둘을 섞어 읽는다.
    this.#judgmentText
      .setVisible(true)
      .setAlpha(alpha)
      .setColor(color)
      .setText(last.isTail ? `${last.judgment} ⌐떼기` : last.judgment);
    if (last.judgment === "MISS") {
      this.#errorText.setVisible(false);
      return;
    }
    const sign = last.errMs >= 0 ? "+" : "−";
    this.#errorText
      .setVisible(true)
      .setAlpha(alpha * 0.8)
      .setText(`${sign}${Math.abs(last.errMs).toFixed(0)}ms ${last.errMs >= 0 ? "LATE" : "EARLY"}`);
  }

  #drawCombo(): void {
    const { combo } = this.#snapshot();
    if (combo < COMBO_MIN) return void this.#comboText.setVisible(false);
    this.#comboText.setVisible(true).setAlpha(0.5).setText(String(combo));
  }

  #drawStats(songTimeMs: number): void {
    const snapshot = this.#snapshot();
    const accuracy = (snapshot.accuracy * 100).toFixed(2);
    const judged = snapshot.totalJudgments;
    const signature = `${accuracy}|${judged}|${snapshot.maxCombo}|${Math.floor(songTimeMs / 500)}`;
    if (signature === this.#statSignature) return;
    this.#statSignature = signature;
    const counts = snapshot.counts;
    this.#statText.setText(
      [
        `ACC ${accuracy}%`,
        `${clock(songTimeMs)} / ${clock(this.#durationMs)}`,
        `NOTES ${judged}/${this.#noteCount}`,
        `MAX ${snapshot.maxCombo}`,
        `P${counts.PERFECT} G${counts.GREAT} ${counts.GOOD}/${counts.BAD} M${counts.MISS}`,
      ].join("   "),
    );
  }

  /** 첫 노트까지 남은 박. 시작 버튼과 동시에 곡이 흘러 손 올릴 틈이 없었다. */
  #drawCountdown(songTimeMs: number): void {
    const remaining = this.#firstNoteTimeMs - songTimeMs;
    if (remaining <= 0 || remaining > COUNTDOWN_LEAD_MS) {
      return void this.#countdownText.setVisible(false);
    }
    const beats = Math.ceil(remaining / this.#beatMs);
    this.#countdownText
      .setVisible(true)
      .setAlpha(Phaser.Math.Clamp(remaining / 600, 0, 1))
      .setText(beats > 9 ? "READY" : String(beats));
  }
}

function clock(timeMs: number): string {
  const total = Math.max(0, Math.floor(timeMs / 1000));
  return `${Math.floor(total / 60)}:${String(total % 60).padStart(2, "0")}`;
}
