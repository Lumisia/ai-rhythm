import Phaser from "phaser";

import type { Clock } from "../audio/Clock";
import type { KeysoundScheduler } from "../audio/KeysoundScheduler";
import type { SongPlayer } from "../audio/SongPlayer";
import { InputRecorder } from "../core/InputRecorder";
import { JudgmentEngine, type JudgmentEvent } from "../core/JudgmentEngine";
import { layoutStage, type StageGeometry } from "../core/LaneLayout";
import { NoteTimeline } from "../core/NoteTimeline";
import { ScoreCalculator, type ScoreSnapshot } from "../core/ScoreCalculator";
import { loadJudgmentConfig } from "../core/judgment-config";
import type { ChartDocument, JudgmentPreset } from "../core/types";
import { bindingsFor, keyLabelsFor } from "../input/KeyBindings";
import { KeyboardInput } from "../input/KeyboardInput";
import { HudRenderer } from "./HudRenderer";
import { NoteRenderer } from "./NoteRenderer";
import { JUDGE_LINE_RATIO, StageRenderer } from "./StageRenderer";

const DEFAULT_BEAT_MS = 500;
const LANE_FLASH_MS = 130;

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
  /** 레이아웃이 잡히거나 바뀔 때 한 번 호출된다.
   *
   * 게임 루프가 아니라 레이아웃 시점에만 부른다. 프레임마다 React state 를
   * 건드리면 렌더 루프가 흔들린다.
   */
  onLayout?: (geometry: { judgeLineY: number }) => void;
}

export class RhythmScene extends Phaser.Scene {
  readonly #session: RhythmSceneSession;
  #stage: StageRenderer | null = null;
  #renderer: NoteRenderer | null = null;
  #hud: HudRenderer | null = null;
  #keyboardInput: KeyboardInput | null = null;
  #finished = false;
  readonly #held = new Set<number>();
  readonly #flashAtMs = new Map<number, number>();

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
      onLaneDown: (lane, timeMs) => {
        this.#held.add(lane);
        this.#flashAtMs.set(lane, timeMs);
      },
      onLaneUp: (lane) => this.#held.delete(lane),
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
    this.#stage?.setPressed(this.#held, this.#flashAtMs, songTimeMs, LANE_FLASH_MS);
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
    const judgeLineY = Math.round(height * JUDGE_LINE_RATIO);
    const stage = layoutStage(width, this.#session.chart.laneSemantics);

    if (!this.#stage) {
      this.#stage = new StageRenderer(
        this,
        stage,
        width,
        height,
        judgeLineY,
        keyLabelsFor(this.#session.chart.keyMode),
      );
    } else {
      this.#stage.resize(stage, width, height, judgeLineY);
    }
    if (!this.#renderer) {
      this.#renderer = new NoteRenderer(
        this,
        new NoteTimeline(this.#session.chart.notes),
        stage.lanes,
        { width, height, judgeLineY },
      );
    } else {
      this.#renderer.resize(width, height, judgeLineY, stage.lanes);
    }
    const geometry = { width, height, judgeLineY, stage };
    if (!this.#hud) {
      this.#hud = new HudRenderer(this, {
        ...geometry,
        windows: this.#judgmentWindows(),
        durationMs: this.#session.chart.durationMs,
        noteCount: this.#session.chart.notes.length,
        firstNoteTimeMs: this.#firstNoteTimeMs(),
        beatMs: this.#beatMs(),
        snapshot: () => this.#session.score.snapshot(),
      });
    } else {
      this.#hud.resize(geometry);
    }
    this.#session.onLayout?.({ judgeLineY });
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

  #shutdown(): void {
    window.removeEventListener("keydown", this.#handleControlKey);
    this.scale.off(Phaser.Scale.Events.RESIZE, this.#layoutPlayfield, this);
    this.#keyboardInput?.detach();
    this.#keyboardInput = null;
    this.#renderer?.destroy();
    this.#renderer = null;
    this.#hud?.destroy();
    this.#hud = null;
    this.#stage?.destroy();
    this.#stage = null;
    this.#session.keysoundScheduler?.dispose();
    this.#session.songPlayer?.dispose();
  }
}
