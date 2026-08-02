import Phaser from "phaser";

import type { Clock } from "../audio/Clock";
import type { KeysoundScheduler } from "../audio/KeysoundScheduler";
import type { SongPlayer } from "../audio/SongPlayer";
import { InputRecorder } from "../core/InputRecorder";
import { JudgmentEngine, type JudgmentEvent } from "../core/JudgmentEngine";
import { layoutLanes, type LaneGeometry } from "../core/LaneLayout";
import { NoteTimeline } from "../core/NoteTimeline";
import { ScoreCalculator, type ScoreSnapshot } from "../core/ScoreCalculator";
import { loadJudgmentConfig } from "../core/judgment-config";
import type { ChartDocument, JudgmentPreset } from "../core/types";
import { bindingsFor, keyLabelsFor } from "../input/KeyBindings";
import { KeyboardInput } from "../input/KeyboardInput";
import { HudRenderer } from "./HudRenderer";
import { NoteRenderer } from "./NoteRenderer";

const DEFAULT_BEAT_MS = 500;

export interface RhythmSceneSession {
  chart: ChartDocument;
  clock: Clock;
  engine: JudgmentEngine;
  score: ScoreCalculator;
  recorder: InputRecorder;
  scrollSpeed: number | (() => number);
  judgmentPreset?: JudgmentPreset;
  keysoundScheduler?: KeysoundScheduler;
  songPlayer?: Pick<SongPlayer, "dispose">;
  loop?: { startMs: number; endMs: number; restart: () => void };
  onJudgment?: (event: JudgmentEvent) => void;
  onPause?: () => void;
  onMarkerSlot?: (slot: number, timeMs: number) => void;
  /** 마커가 실제로 기록되면 화면에 띄울 이름을 돌려준다. */
  markerLabel?: (slot: number) => string | null;
  onComplete?: (score: ScoreSnapshot) => void;
}

export class RhythmScene extends Phaser.Scene {
  readonly #session: RhythmSceneSession;
  #laneGraphics: Phaser.GameObjects.Graphics | null = null;
  #renderer: NoteRenderer | null = null;
  #hud: HudRenderer | null = null;
  #keyboardInput: KeyboardInput | null = null;
  #finished = false;

  constructor(session: RhythmSceneSession) {
    super({ key: "rhythm-playtest" });
    this.#session = session;
  }

  create(): void {
    this.#layoutPlayfield();
    this.#keyboardInput = new KeyboardInput({
      bindings: bindingsFor(this.#session.chart.keyMode),
      clock: this.#session.clock,
      engine: this.#session.engine,
      recorder: this.#session.recorder,
      onJudgment: (event) => this.#acceptJudgment(event),
      onLaneDown: (lane, timeMs) => this.#hud?.pressLane(lane, timeMs),
      onLaneUp: (lane) => this.#hud?.releaseLane(lane),
    });
    this.#keyboardInput.attach();
    window.addEventListener("keydown", this.#handleControlKey);
    this.scale.on(Phaser.Scale.Events.RESIZE, this.#layoutPlayfield, this);
    this.events.once(Phaser.Scenes.Events.SHUTDOWN, this.#shutdown, this);
  }

  update(): void {
    const songTimeMs = this.#session.clock.songTimeMs();
    if (this.#session.loop && songTimeMs >= this.#session.loop.endMs) {
      this.#session.loop.restart();
      return;
    }
    for (const event of this.#session.engine.advance(songTimeMs)) {
      this.#acceptJudgment(event);
    }
    this.#session.keysoundScheduler?.scheduleAutoPlayUntil(songTimeMs + 250);
    const scrollSpeed =
      typeof this.#session.scrollSpeed === "function"
        ? this.#session.scrollSpeed()
        : this.#session.scrollSpeed;
    this.#renderer?.update(songTimeMs, scrollSpeed);
    this.#hud?.update(songTimeMs);

    if (!this.#session.loop && !this.#finished && songTimeMs > this.#session.chart.durationMs + 500) {
      this.#finished = true;
      this.#session.onComplete?.(this.#session.score.snapshot());
    }
  }

  #acceptJudgment(event: JudgmentEvent): void {
    this.#session.score.accept(event);
    this.#hud?.acceptJudgment(event, this.#session.clock.songTimeMs());
    if (event.phase === "HEAD" && event.judgment !== "MISS") {
      this.#session.keysoundScheduler?.playHit(event.noteTimeMs);
    }
    this.#session.onJudgment?.(event);
  }

  readonly #handleControlKey = (event: KeyboardEvent): void => {
    if (event.repeat) return;
    if (event.code === "Escape") {
      event.preventDefault();
      this.#session.onPause?.();
      return;
    }
    const markerMatch = /^Digit([1-8])$/.exec(event.code);
    if (markerMatch) {
      event.preventDefault();
      const slot = Number(markerMatch[1]);
      const songTimeMs = this.#session.clock.songTimeMs();
      this.#session.onMarkerSlot?.(slot, songTimeMs);
      // 눌러도 화면에 아무 일이 없으면 기록됐는지 몰라 다시 누르게 된다.
      const label = this.#session.markerLabel?.(slot);
      if (label) this.#hud?.acceptMarker(label, songTimeMs);
    }
  };

  #layoutPlayfield(): void {
    const width = this.scale.width;
    const height = this.scale.height;
    // 판정선 아래를 넓게 남긴다. 타이밍 스코프가 그 자리에 들어간다.
    const judgeLineY = height * 0.74;
    const lanes = layoutLanes(width, this.#session.chart.laneSemantics);
    this.#drawLanes(lanes, height, judgeLineY);
    if (!this.#renderer) {
      this.#renderer = new NoteRenderer(
        this,
        new NoteTimeline(this.#session.chart.notes),
        lanes,
        { width, height, judgeLineY },
      );
    } else {
      this.#renderer.resize(width, height, judgeLineY, lanes);
    }
    if (!this.#hud) {
      this.#hud = new HudRenderer(this, lanes, {
        width,
        height,
        judgeLineY,
        windows: this.#judgmentWindows(),
        durationMs: this.#session.chart.durationMs,
        noteCount: this.#session.chart.notes.length,
        firstNoteTimeMs: this.#firstNoteTimeMs(),
        beatMs: this.#beatMs(),
        keyLabels: keyLabelsFor(this.#session.chart.keyMode),
        snapshot: () => this.#session.score.snapshot(),
      });
    } else {
      this.#hud.resize(lanes, { width, height, judgeLineY });
    }
  }

  #judgmentWindows() {
    const config = loadJudgmentConfig();
    return config.presets[this.#session.judgmentPreset ?? config.default];
  }

  #firstNoteTimeMs(): number {
    const times = this.#session.chart.notes.map((note) => note.timeMs);
    return times.length === 0 ? 0 : Math.min(...times);
  }

  #beatMs(): number {
    const bpm = this.#session.chart.bpmEvents[0]?.bpm;
    return bpm && bpm > 0 ? 60_000 / bpm : DEFAULT_BEAT_MS;
  }

  #drawLanes(lanes: readonly LaneGeometry[], height: number, judgeLineY: number): void {
    this.#laneGraphics?.destroy();
    const graphics = this.add.graphics().setDepth(0);
    for (const lane of lanes) {
      graphics.fillStyle(lane.backgroundColor, 1);
      graphics.fillRect(lane.x, 0, lane.width, judgeLineY);
      graphics.lineStyle(1, 0x38404e, 0.9);
      graphics.lineBetween(lane.x, 0, lane.x, judgeLineY);
      // 판정선 위에 리셉터를 깔아 노트가 어디로 들어오는지 못박는다.
      graphics.fillStyle(lane.color, 0.1);
      graphics.fillRect(lane.x + 1, judgeLineY - 16, lane.width - 2, 16);
    }
    // 판정선 아래는 계기 영역이다. 레인을 끊어 눈이 섞이지 않게 한다.
    graphics.fillStyle(0x151923, 1);
    graphics.fillRect(0, judgeLineY, this.scale.width, height - judgeLineY);
    graphics.lineStyle(3, 0xe2b75b, 1);
    graphics.lineBetween(0, judgeLineY, this.scale.width, judgeLineY);
    this.#laneGraphics = graphics;
  }

  #shutdown(): void {
    window.removeEventListener("keydown", this.#handleControlKey);
    this.scale.off(Phaser.Scale.Events.RESIZE, this.#layoutPlayfield, this);
    this.#keyboardInput?.detach();
    this.#keyboardInput = null;
    this.#renderer?.destroy();
    this.#renderer = null;
    this.#hud?.destroy();
    this.#hud = null;
    this.#laneGraphics?.destroy();
    this.#laneGraphics = null;
    this.#session.keysoundScheduler?.dispose();
    this.#session.songPlayer?.dispose();
  }
}
