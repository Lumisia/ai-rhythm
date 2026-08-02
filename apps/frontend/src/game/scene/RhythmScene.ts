import Phaser from "phaser";

import type { Clock } from "../audio/Clock";
import type { KeysoundScheduler } from "../audio/KeysoundScheduler";
import type { SongPlayer } from "../audio/SongPlayer";
import { InputRecorder } from "../core/InputRecorder";
import { JudgmentEngine, type JudgmentEvent } from "../core/JudgmentEngine";
import { layoutLanes, type LaneGeometry } from "../core/LaneLayout";
import { NoteTimeline } from "../core/NoteTimeline";
import { ScoreCalculator, type ScoreSnapshot } from "../core/ScoreCalculator";
import type { ChartDocument } from "../core/types";
import { bindingsFor } from "../input/KeyBindings";
import { KeyboardInput } from "../input/KeyboardInput";
import { NoteRenderer } from "./NoteRenderer";

export interface RhythmSceneSession {
  chart: ChartDocument;
  clock: Clock;
  engine: JudgmentEngine;
  score: ScoreCalculator;
  recorder: InputRecorder;
  scrollSpeed: number | (() => number);
  keysoundScheduler?: KeysoundScheduler;
  songPlayer?: Pick<SongPlayer, "dispose">;
  loop?: { startMs: number; endMs: number; restart: () => void };
  onJudgment?: (event: JudgmentEvent) => void;
  onPause?: () => void;
  onMarkerSlot?: (slot: number, timeMs: number) => void;
  onComplete?: (score: ScoreSnapshot) => void;
}

export class RhythmScene extends Phaser.Scene {
  readonly #session: RhythmSceneSession;
  #laneGraphics: Phaser.GameObjects.Graphics | null = null;
  #renderer: NoteRenderer | null = null;
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

    if (!this.#session.loop && !this.#finished && songTimeMs > this.#session.chart.durationMs + 500) {
      this.#finished = true;
      this.#session.onComplete?.(this.#session.score.snapshot());
    }
  }

  #acceptJudgment(event: JudgmentEvent): void {
    this.#session.score.accept(event);
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
      this.#session.onMarkerSlot?.(
        Number(markerMatch[1]),
        this.#session.clock.songTimeMs(),
      );
    }
  };

  #layoutPlayfield(): void {
    const width = this.scale.width;
    const height = this.scale.height;
    const judgeLineY = height * 0.82;
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
  }

  #drawLanes(lanes: readonly LaneGeometry[], height: number, judgeLineY: number): void {
    this.#laneGraphics?.destroy();
    const graphics = this.add.graphics().setDepth(0);
    for (const lane of lanes) {
      graphics.fillStyle(lane.backgroundColor, 1);
      graphics.fillRect(lane.x, 0, lane.width, height);
      graphics.lineStyle(1, 0x38404e, 0.9);
      graphics.lineBetween(lane.x, 0, lane.x, height);
    }
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
    this.#laneGraphics?.destroy();
    this.#laneGraphics = null;
    this.#session.keysoundScheduler?.dispose();
    this.#session.songPlayer?.dispose();
  }
}
