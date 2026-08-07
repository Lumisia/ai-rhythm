import Phaser from "phaser";

import type { EffectEvent, EffectSubscriber } from "../core/EffectBus";
import type { JudgmentPhase } from "../core/JudgmentEngine";
import type { StageGeometry } from "../core/LaneLayout";
import type { ScoreSnapshot } from "../core/ScoreCalculator";
import type { JudgmentName, JudgmentWindows } from "../core/types";
import { DEPTH } from "./renderDepth";

/** 계기판 색. app.css 의 CSS 변수와 같은 값을 쓴다. */
const INK = 0xe8ecf8;
const MUTED = 0x8d97b5;
const HAIRLINE = 0x31364d;
const ACCENT = 0x5eead4;
const VIOLET = 0xa78bfa;
const CORAL = 0xf87171;
const SLATE = 0x171a2a;

const JUDGMENT_COLOR: Record<JudgmentName, number> = {
  PERFECT: 0x67e8f9,
  GREAT: 0x86efac,
  GOOD: 0xfde047,
  BAD: 0xfb923c,
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
const FEVER_GAUGE_WIDTH = 6;
const FEVER_GAUGE_HEIGHT_RATIO = 0.42;

/** 마일스톤에서만 반응한다. 매 콤보마다 움직이면 활주로가 계속 흔들린다. */
const COMBO_MILESTONES = [25, 50, 100, 200, 500];
const COMBO_POP_MS = 140;
const COMBO_BASE_SIZE = 54;
const COMBO_PEAK_SIZE = 64;
const COMBO_GOLD_FROM = 100;
const GOLD = 0xfbbf24;

/** rgba 문자열을 모듈 로드 시 한 번만 만든다.
 *
 * `Phaser.Display.Color.IntegerToColor()` 는 호출마다 `Color` 객체를 새로
 * 할당한다. 매 프레임 부르면 그대로 GC 압력이 된다.
 */
const GOLD_RGBA = Phaser.Display.Color.IntegerToColor(GOLD).rgba;
const INK_RGBA = Phaser.Display.Color.IntegerToColor(INK).rgba;
const JUDGMENT_RGBA: Record<JudgmentName, string> = {
  PERFECT: Phaser.Display.Color.IntegerToColor(JUDGMENT_COLOR.PERFECT).rgba,
  GREAT: Phaser.Display.Color.IntegerToColor(JUDGMENT_COLOR.GREAT).rgba,
  GOOD: Phaser.Display.Color.IntegerToColor(JUDGMENT_COLOR.GOOD).rgba,
  BAD: Phaser.Display.Color.IntegerToColor(JUDGMENT_COLOR.BAD).rgba,
  MISS: Phaser.Display.Color.IntegerToColor(JUDGMENT_COLOR.MISS).rgba,
};

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

export interface HudJudgment {
  judgment: JudgmentName;
  errMs: number;
  phase: JudgmentPhase;
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
export class HudRenderer implements EffectSubscriber {
  readonly #scene: Phaser.Scene;
  readonly #graphics: Phaser.GameObjects.Graphics;
  readonly #judgmentText: Phaser.GameObjects.Text;
  readonly #errorText: Phaser.GameObjects.Text;
  readonly #comboText: Phaser.GameObjects.Text;
  readonly #accuracyText: Phaser.GameObjects.Text;
  readonly #accuracyLabel: Phaser.GameObjects.Text;
  readonly #scoreText: Phaser.GameObjects.Text;
  readonly #scoreLabel: Phaser.GameObjects.Text;
  readonly #maxComboText: Phaser.GameObjects.Text;
  readonly #maxComboLabel: Phaser.GameObjects.Text;
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
  #comboPopAtMs = -Infinity;
  #lastCombo = 0;
  #comboGold = false;
  #judgmentColor: JudgmentName | null = null;
  #reduceMotion = false;
  #feverValue = 0;
  #feverActive = false;

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

    this.#graphics = scene.add.graphics().setDepth(DEPTH.HUD_GRAPHICS);
    this.#scoreText = this.#text(DISPLAY, 26, INK).setDepth(DEPTH.HUD_TEXT);
    this.#scoreLabel = this.#text(MONO, 9, MUTED).setDepth(DEPTH.HUD_TEXT);
    this.#accuracyText = this.#text(DISPLAY, 26, INK).setDepth(DEPTH.HUD_TEXT);
    this.#accuracyLabel = this.#text(MONO, 9, MUTED).setDepth(DEPTH.HUD_TEXT);
    this.#maxComboText = this.#text(DISPLAY, 26, INK).setDepth(DEPTH.HUD_TEXT).setOrigin(1, 0);
    this.#maxComboLabel = this.#text(MONO, 9, MUTED).setDepth(DEPTH.HUD_TEXT).setOrigin(1, 0);
    this.#leftText = this.#text(MONO, 11, MUTED).setDepth(DEPTH.HUD_TEXT);
    this.#countsText = this.#text(MONO, 11, MUTED).setDepth(DEPTH.HUD_TEXT).setOrigin(1, 0);
    this.#judgmentText = this.#text(DISPLAY, 26, INK).setDepth(DEPTH.HUD_TEXT).setOrigin(0.5, 0.5);
    this.#errorText = this.#text(MONO, 11, MUTED).setDepth(DEPTH.HUD_TEXT).setOrigin(0.5, 0);
    this.#comboText = this.#text(DISPLAY, 54, INK).setDepth(DEPTH.HUD_COMBO_TEXT).setOrigin(0.5, 0.5);
    this.#scopeText = this.#text(MONO, 10, MUTED).setDepth(DEPTH.HUD_TEXT).setOrigin(0.5, 0);
    this.#markerText = this.#text(MONO, 11, CORAL).setDepth(DEPTH.HUD_TEXT).setOrigin(1, 0);
    this.#countdownText = this.#text(DISPLAY, 84, ACCENT).setDepth(DEPTH.OVERLAY).setOrigin(0.5, 0.5);
    this.#layout();
  }

  /** EffectBus 어댑터. 기존 메서드를 그대로 부른다. */
  handleEffect(event: EffectEvent): void {
    if (event.type === "JUDGED") {
      this.acceptJudgment(
        { judgment: event.judgment, errMs: event.errMs, phase: event.phase },
        event.songTimeMs,
      );
      this.#noticeCombo(event.combo, event.songTimeMs);
      return;
    }
    if (event.type === "HOLD_TICK") {
      this.#noticeCombo(event.combo, event.songTimeMs);
      return;
    }
    if (event.type === "MARKER") {
      this.acceptMarker(event.label, event.songTimeMs);
    }
  }

  /** 마일스톤을 넘어섰는지 본다.
   *
   * FEVER 증폭으로 콤보가 2씩 오르면 마일스톤을 정확히 밟지 않고 건너뛴다.
   * 등호가 아니라 구간 통과로 판정해야 한다.
   */
  #noticeCombo(combo: number, songTimeMs: number): void {
    const crossed = COMBO_MILESTONES.some(
      (milestone) => this.#lastCombo < milestone && combo >= milestone,
    );
    this.#lastCombo = combo;
    if (crossed && !this.#reduceMotion) this.#comboPopAtMs = songTimeMs;
  }

  setReduceMotion(reduce: boolean): void {
    this.#reduceMotion = reduce;
  }

  setFeverState(value: number, active: boolean): void {
    this.#feverValue = value;
    this.#feverActive = active;
  }

  acceptJudgment(event: HudJudgment, songTimeMs: number): void {
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
    this.#drawFever();
    this.#drawScope(songTimeMs);
    this.#drawJudgment(songTimeMs);
    this.#drawCombo(songTimeMs);
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
      this.#scoreText,
      this.#scoreLabel,
      this.#maxComboText,
      this.#maxComboLabel,
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
    const leftGutter = GUTTER_PADDING + 8;
    const rightEdge = this.#width - GUTTER_PADDING - 8;

    this.#scoreLabel.setPosition(leftGutter, 18);
    this.#scoreText.setPosition(leftGutter, 30);
    const accuracyX = leftGutter + Math.min(156, this.#width * 0.34);
    this.#accuracyLabel.setPosition(accuracyX, 18);
    this.#accuracyText.setPosition(accuracyX, 30);
    this.#maxComboLabel.setPosition(rightEdge, 18);
    this.#maxComboText.setPosition(rightEdge, 30);
    this.#leftText.setPosition(leftGutter, 70);
    this.#countsText.setPosition(rightEdge, Math.max(112, this.#height * 0.42));
    this.#markerText.setPosition(rightEdge, 72);

    this.#comboText.setPosition(centerX, this.#judgeLineY - 190);
    this.#judgmentText.setPosition(centerX, this.#judgeLineY - 116);
    this.#errorText.setPosition(centerX, this.#judgeLineY - 100);
    this.#scopeText.setPosition(centerX, this.#scopeY + 26);
    this.#countdownText.setPosition(centerX, this.#judgeLineY * 0.5);
  }

  // --- 그리기 ---------------------------------------------------------------

  /** 화면 바닥의 곡 진행도와 문제 마커. */
  #drawProgressRail(songTimeMs: number): void {
    const left = 0;
    const top = this.#height - 3;
    const width = this.#width;
    this.#graphics.fillStyle(SLATE, 1);
    this.#graphics.fillRect(left, top, width, 3);
    const ratio = Phaser.Math.Clamp(songTimeMs / this.#durationMs, 0, 1);
    this.#graphics.fillGradientStyle(ACCENT, VIOLET, ACCENT, VIOLET, 1, 1, 1, 1);
    this.#graphics.fillRect(left, top, width * ratio, 3);
    this.#graphics.fillStyle(INK, 1);
    this.#graphics.fillRect(width * ratio - 1, top - 2, 2, 5);
    this.#graphics.fillStyle(CORAL, 1);
    for (const markerTimeMs of this.#markerTimes) {
      const x = width * Phaser.Math.Clamp(markerTimeMs / this.#durationMs, 0, 1);
      this.#graphics.fillRect(x - 1, top - 5, 2, 8);
    }
  }

  /** 무대 좌측 세로 게이지. 주변시로 보는 자리다. */
  #drawFever(): void {
    const height = this.#height * FEVER_GAUGE_HEIGHT_RATIO;
    const top = this.#judgeLineY - height;
    const x = this.#stage.left - GUTTER_PADDING;

    this.#graphics.fillStyle(SLATE, 0.9);
    this.#graphics.fillRect(x, top, FEVER_GAUGE_WIDTH, height);

    if (this.#feverActive) {
      // 발동 중에는 게이지를 가득 채운 채 색으로 상태를 알린다.
      this.#graphics.fillStyle(VIOLET, 0.95);
      this.#graphics.fillRect(x, top, FEVER_GAUGE_WIDTH, height);
      return;
    }

    const filled = height * Phaser.Math.Clamp(this.#feverValue / 100, 0, 1);
    this.#graphics.fillGradientStyle(ACCENT, VIOLET, ACCENT, VIOLET, 1, 1, 1, 1);
    this.#graphics.fillRect(x, top + height - filled, FEVER_GAUGE_WIDTH, filled);
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
      ["PERFECT", ACCENT, 0.8],
    ] as const) {
      const offset = this.#windows[name] * perMs;
      this.#graphics.fillStyle(color, alpha);
      this.#graphics.fillRect(centerX - offset, y - 8, 1, 16);
      this.#graphics.fillRect(centerX + offset, y - 8, 1, 16);
    }
    this.#graphics.fillStyle(ACCENT, 1);
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
      this.#graphics.fillStyle(ACCENT, 1);
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
    // 롱노트를 뗀 판정은 완화 배율이 붙어 머리 판정과 기준이 다르다.
    // 표시가 같으면 검수 중에 둘을 섞어 읽는다.
    this.#applyJudgmentColor(last.judgment);
    this.#judgmentText
      .setVisible(true)
      .setAlpha(alpha)
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

  /** 판정이 바뀔 때만 setColor 를 부른다. `#applyComboColor` 와 같은 이유다.
   *
   * 판정 표시는 460ms 유지되므로 촘촘한 채보에서는 사실상 끊기지 않는다.
   * 가드가 없으면 그 내내 매 프레임 `Color` 객체를 새로 할당하고 텍스트
   * 캔버스를 다시 래스터라이즈한다.
   */
  #applyJudgmentColor(judgment: JudgmentName): void {
    if (judgment === this.#judgmentColor) return;
    this.#judgmentColor = judgment;
    this.#judgmentText.setColor(JUDGMENT_RGBA[judgment]);
  }

  #drawCombo(songTimeMs: number): void {
    const { combo } = this.#snapshot();
    if (combo < COMBO_MIN) {
      this.#lastCombo = 0;
      this.#applyComboColor(false);
      return void this.#comboText.setVisible(false);
    }

    const age = songTimeMs - this.#comboPopAtMs;
    // 0 → 1 → 0 삼각파. 앞뒤 70ms 씩이다.
    const pop =
      age >= 0 && age <= COMBO_POP_MS
        ? 1 - Math.abs(age / (COMBO_POP_MS / 2) - 1)
        : 0;
    const size = Math.round(COMBO_BASE_SIZE + (COMBO_PEAK_SIZE - COMBO_BASE_SIZE) * pop);

    this.#applyComboColor(combo >= COMBO_GOLD_FROM);
    this.#comboText
      .setVisible(true)
      .setAlpha(0.45)
      .setFontSize(size)
      .setText(String(combo));
  }

  /** 색이 실제로 바뀔 때만 setColor 를 부른다.
   *
   * `Text.setColor` 에는 동등성 가드가 없어서 값이 같아도 `updateText()` 로
   * 캔버스를 통째로 다시 그린다. `setFontSize` 와 `setText` 는 내부 가드가
   * 있어 매 프레임 불러도 안전하지만 색은 아니다.
   */
  #applyComboColor(gold: boolean): void {
    if (gold === this.#comboGold) return;
    this.#comboGold = gold;
    this.#comboText.setColor(gold ? GOLD_RGBA : INK_RGBA);
  }

  #drawStats(songTimeMs: number): void {
    const snapshot = this.#snapshot();
    const accuracy = (snapshot.accuracy * 100).toFixed(2);
    const judged = snapshot.totalJudgments;
    const signature = `${accuracy}|${judged}|${snapshot.maxCombo}|${Math.floor(songTimeMs / 500)}`;
    if (signature === this.#statSignature) return;
    this.#statSignature = signature;
    const counts = snapshot.counts;
    const score = Math.round(snapshot.accuracy * 1_000_000);
    this.#scoreLabel.setText("SCORE");
    this.#scoreText.setText(score.toLocaleString("en-US"));
    this.#accuracyLabel.setText("ACCURACY");
    this.#accuracyText.setText(`${accuracy}%`);
    this.#maxComboLabel.setText("MAX COMBO");
    this.#maxComboText.setText(String(snapshot.maxCombo));
    this.#leftText.setText(
      [
        `${clock(songTimeMs)} / ${clock(this.#durationMs)}`,
        `NOTES  ${judged} / ${this.#noteCount}`,
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
