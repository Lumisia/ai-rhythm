import Phaser from "phaser";

import type { JudgmentEvent } from "../core/JudgmentEngine";
import type { StageGeometry } from "../core/LaneLayout";
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

const JUDGMENT_HOLD_MS = 460;
const MARKER_HOLD_MS = 1400;
const SCOPE_TRAIL_MS = 2400;
const SCOPE_TRAIL_MAX = 200;
const COMBO_MIN = 5;
const COUNTDOWN_LEAD_MS = 3200;
const GUTTER_PADDING = 18;

interface ScopeMark {
  errMs: number;
  atMs: number;
}

export interface HudGeometry {
  width: number;
  height: number;
  judgeLineY: number;
  stage: StageGeometry;
}

export interface HudRendererOptions extends HudGeometry {
  windows: JudgmentWindows;
  durationMs: number;
  noteCount: number;
  firstNoteTimeMs: number;
  beatMs: number;
  snapshot: () => ScoreSnapshot;
}

/** 플레이 중 계기 판독부.
 *
 * 무대가 좁아지면서 좌우에 여백이 생겼다. 수치는 거기 둔다 — 무대 위에
 * 겹치면 노트를 가린다. 무대 안에는 판정·콤보처럼 시선이 이미 가 있는
 * 자리에서 읽어야 하는 것만 올린다.
 *
 * 타이밍 오차 스코프가 주인공이다. 리셉터 바로 아래에 무대 폭으로 눕힌다 —
 * VSRG 들이 hit error bar 를 두는 자리다. 입력 보정을 맞출 수 있는 유일한
 * 물건이고, 없으면 "채보 오프셋이 틀렸나 내가 못 친 건가"를 구분할 수 없다.
 */
export class HudRenderer {
  readonly #scene: Phaser.Scene;
  readonly #graphics: Phaser.GameObjects.Graphics;
  readonly #judgmentText: Phaser.GameObjects.Text;
  readonly #errorText: Phaser.GameObjects.Text;
  readonly #comboText: Phaser.GameObjects.Text;
  readonly #accuracyText: Phaser.GameObjects.Text;
  readonly #accuracyLabel: Phaser.GameObjects.Text;
  readonly #leftText: Phaser.GameObjects.Text;
  readonly #countsText: Phaser.GameObjects.Text;
  readonly #scopeText: Phaser.GameObjects.Text;
  readonly #markerText: Phaser.GameObjects.Text;
  readonly #countdownText: Phaser.GameObjects.Text;

  readonly #windows: JudgmentWindows;
  readonly #durationMs: number;
  readonly #noteCount: number;
  readonly #firstNoteTimeMs: number;
  readonly #beatMs: number;
  readonly #snapshot: () => ScoreSnapshot;

  #stage: StageGeometry;
  #width: number;
  #height: number;
  #judgeLineY: number;

  readonly #scopeMarks: ScopeMark[] = [];
  readonly #markerTimes: number[] = [];
  #lastJudgment: {
    judgment: JudgmentName;
    errMs: number;
    atMs: number;
    isTail: boolean;
  } | null = null;
  #markerShownAtMs = -Infinity;
  #statSignature = "";

  constructor(scene: Phaser.Scene, options: HudRendererOptions) {
    this.#scene = scene;
    this.#stage = options.stage;
    this.#width = options.width;
    this.#height = options.height;
    this.#judgeLineY = options.judgeLineY;
    this.#windows = options.windows;
    this.#durationMs = Math.max(1, options.durationMs);
    this.#noteCount = options.noteCount;
    this.#firstNoteTimeMs = options.firstNoteTimeMs;
    this.#beatMs = options.beatMs;
    this.#snapshot = options.snapshot;

    this.#graphics = scene.add.graphics().setDepth(20);
    this.#accuracyText = this.#text(DISPLAY, 30, INK).setDepth(24);
    this.#accuracyLabel = this.#text(MONO, 9, MUTED).setDepth(24);
    this.#leftText = this.#text(MONO, 11, MUTED).setDepth(24);
    this.#countsText = this.#text(MONO, 11, MUTED).setDepth(24).setOrigin(1, 0);
    this.#judgmentText = this.#text(DISPLAY, 26, INK).setDepth(24).setOrigin(0.5, 0.5);
    this.#errorText = this.#text(MONO, 11, MUTED).setDepth(24).setOrigin(0.5, 0);
    this.#comboText = this.#text(DISPLAY, 54, INK).setDepth(23).setOrigin(0.5, 0.5);
    this.#scopeText = this.#text(MONO, 10, MUTED).setDepth(24).setOrigin(0.5, 0);
    this.#markerText = this.#text(MONO, 11, CORAL).setDepth(24).setOrigin(1, 0);
    this.#countdownText = this.#text(DISPLAY, 84, AMBER).setDepth(25).setOrigin(0.5, 0.5);
    this.#layout();
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

  resize(geometry: HudGeometry): void {
    this.#stage = geometry.stage;
    this.#width = geometry.width;
    this.#height = geometry.height;
    this.#judgeLineY = geometry.judgeLineY;
    this.#layout();
  }

  destroy(): void {
    this.#graphics.destroy();
    for (const text of [
      this.#accuracyText,
      this.#accuracyLabel,
      this.#leftText,
      this.#countsText,
      this.#judgmentText,
      this.#errorText,
      this.#comboText,
      this.#scopeText,
      this.#markerText,
      this.#countdownText,
    ]) {
      text.destroy();
    }
  }

  // --- 배치 -----------------------------------------------------------------

  get #scopeY(): number {
    // 판정선 아래 첫 줄은 키 라벨 자리다. 스코프는 그 아래.
    return this.#judgeLineY + 44;
  }

  get #scopeHalfWidth(): number {
    // 무대 폭에 맞춘다. 좁으면 떠 보이고, 넓으면 무대 밖으로 새어 나간다.
    return this.#stage.width / 2;
  }

  #text(fontFamily: string, fontSize: number, color: number): Phaser.GameObjects.Text {
    return this.#scene.add.text(0, 0, "", {
      fontFamily,
      fontSize: `${fontSize}px`,
      color: Phaser.Display.Color.IntegerToColor(color).rgba,
    });
  }

  #layout(): void {
    const centerX = this.#stage.left + this.#stage.width / 2;
    const leftGutter = Math.max(GUTTER_PADDING, this.#stage.left - 150);
    const rightEdge = Math.min(this.#width - GUTTER_PADDING, this.#stage.right + 150);

    this.#accuracyLabel.setPosition(leftGutter, 22);
    this.#accuracyText.setPosition(leftGutter, 34);
    this.#leftText.setPosition(leftGutter, 78);
    this.#countsText.setPosition(rightEdge, 34);
    this.#markerText.setPosition(rightEdge, 14);

    this.#comboText.setPosition(centerX, this.#judgeLineY - 190);
    this.#judgmentText.setPosition(centerX, this.#judgeLineY - 116);
    this.#errorText.setPosition(centerX, this.#judgeLineY - 100);
    this.#scopeText.setPosition(centerX, this.#scopeY + 26);
    this.#countdownText.setPosition(centerX, this.#judgeLineY * 0.5);
  }

  // --- 그리기 ---------------------------------------------------------------

  /** 곡 진행과 찍어둔 문제 마커.
   *
   * 무대 폭에 앰버 가로선으로 두면 판정선과 헷갈린다. 무대 바깥 왼쪽에
   * 세로로 세워서 형태부터 다르게 한다.
   */
  #drawProgressRail(songTimeMs: number): void {
    const x = Math.max(6, this.#stage.left - 26);
    const top = 24;
    const height = this.#height - top - 24;
    this.#graphics.fillStyle(SLATE, 1);
    this.#graphics.fillRect(x, top, 3, height);
    const ratio = Phaser.Math.Clamp(songTimeMs / this.#durationMs, 0, 1);
    this.#graphics.fillStyle(MUTED, 0.9);
    this.#graphics.fillRect(x, top, 3, height * ratio);
    this.#graphics.fillStyle(INK, 1);
    this.#graphics.fillRect(x - 2, top + height * ratio - 1, 7, 2);
    this.#graphics.fillStyle(CORAL, 1);
    for (const markerTimeMs of this.#markerTimes) {
      const y = top + height * Phaser.Math.Clamp(markerTimeMs / this.#durationMs, 0, 1);
      this.#graphics.fillRect(x - 4, y - 1, 11, 2);
    }
  }

  /** 타이밍 오차 스코프. 리셉터 바로 아래, VSRG 의 hit error bar 자리다.
   *
   * 가운데가 0ms. 최근 타격이 오차 위치에 점으로 남고 서서히 지워진다.
   * 점 무리가 한쪽으로 쏠려 있으면 입력 보정값을 그만큼 옮기면 된다.
   */
  #drawScope(songTimeMs: number): void {
    const centerX = this.#stage.left + this.#stage.width / 2;
    const y = this.#scopeY;
    const half = this.#scopeHalfWidth;
    const perMs = half / this.#windows.BAD;

    this.#graphics.fillStyle(SLATE, 0.55);
    this.#graphics.fillRect(centerX - half, y - 9, half * 2, 18);
    for (const [name, color, alpha] of [
      ["GOOD", HAIRLINE, 0.9],
      ["GREAT", MUTED, 0.55],
      ["PERFECT", MINT, 0.8],
    ] as const) {
      const offset = this.#windows[name] * perMs;
      this.#graphics.fillStyle(color, alpha);
      this.#graphics.fillRect(centerX - offset, y - 8, 1, 16);
      this.#graphics.fillRect(centerX + offset, y - 8, 1, 16);
    }
    this.#graphics.fillStyle(AMBER, 1);
    this.#graphics.fillRect(centerX - 1, y - 12, 2, 24);

    let visible = 0;
    for (const mark of this.#scopeMarks) {
      const age = songTimeMs - mark.atMs;
      if (age < 0 || age > SCOPE_TRAIL_MS) continue;
      visible += 1;
      const clamped = Phaser.Math.Clamp(mark.errMs, -this.#windows.BAD, this.#windows.BAD);
      this.#graphics.fillStyle(INK, 0.9 * (1 - age / SCOPE_TRAIL_MS));
      this.#graphics.fillRect(centerX + clamped * perMs - 1, y - 8, 2, 16);
    }

    const snapshot = this.#snapshot();
    if (snapshot.timingSampleCount > 0) {
      const clamped = Phaser.Math.Clamp(
        snapshot.meanErrMs,
        -this.#windows.BAD,
        this.#windows.BAD,
      );
      const meanX = centerX + clamped * perMs;
      // 바 아래로 세운다. 위로 세우면 평균이 0 근처일 때 가운데 레인의
      // 키 라벨을 가린다 — 평균은 대개 0 근처다.
      this.#graphics.fillStyle(MINT, 1);
      this.#graphics.fillTriangle(meanX, y + 12, meanX - 5, y + 21, meanX + 5, y + 21);
      const sign = snapshot.meanErrMs >= 0 ? "+" : "−";
      const drift = Math.abs(snapshot.meanErrMs).toFixed(1);
      this.#scopeText.setText(
        `◄ EARLY   평균 ${sign}${drift}ms   LATE ►    보정 ${(-snapshot.meanErrMs).toFixed(0)}ms 권장`,
      );
      this.#scopeText.setVisible(true);
    } else {
      this.#scopeText.setText("◄ EARLY    타이밍 오차    LATE ►").setVisible(visible === 0);
    }
  }

  #drawJudgment(songTimeMs: number): void {
    const last = this.#lastJudgment;
    if (!last) {
      this.#judgmentText.setVisible(false);
      this.#errorText.setVisible(false);
      return;
    }
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
      .setAlpha(alpha * 0.85)
      .setText(`${sign}${Math.abs(last.errMs).toFixed(0)}ms ${last.errMs >= 0 ? "LATE" : "EARLY"}`);
  }

  #drawCombo(): void {
    const { combo } = this.#snapshot();
    if (combo < COMBO_MIN) return void this.#comboText.setVisible(false);
    this.#comboText.setVisible(true).setAlpha(0.45).setText(String(combo));
  }

  #drawStats(songTimeMs: number): void {
    const snapshot = this.#snapshot();
    const accuracy = (snapshot.accuracy * 100).toFixed(2);
    const judged = snapshot.totalJudgments;
    const signature = `${accuracy}|${judged}|${snapshot.maxCombo}|${Math.floor(songTimeMs / 500)}`;
    if (signature === this.#statSignature) return;
    this.#statSignature = signature;
    const counts = snapshot.counts;
    this.#accuracyLabel.setText("ACCURACY");
    this.#accuracyText.setText(`${accuracy}%`);
    this.#leftText.setText(
      [
        `${clock(songTimeMs)} / ${clock(this.#durationMs)}`,
        `NOTES  ${judged} / ${this.#noteCount}`,
        `MAX    ${snapshot.maxCombo}`,
      ].join("\n"),
    );
    this.#countsText.setText(
      [
        `PERFECT ${counts.PERFECT}`,
        `GREAT   ${counts.GREAT}`,
        `GOOD    ${counts.GOOD}`,
        `BAD     ${counts.BAD}`,
        `MISS    ${counts.MISS}`,
      ].join("\n"),
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
