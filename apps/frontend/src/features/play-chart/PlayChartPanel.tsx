import { useCallback, useEffect, useRef, useState } from "react";

import { GameClock } from "../../game/audio/GameClock";
import { KeysoundScheduler } from "../../game/audio/KeysoundScheduler";
import { SongPlayer } from "../../game/audio/SongPlayer";
import { InputRecorder } from "../../game/core/InputRecorder";
import { JudgmentEngine, type JudgmentEvent } from "../../game/core/JudgmentEngine";
import { ScoreCalculator, type ScoreSnapshot } from "../../game/core/ScoreCalculator";
import type { ImportedChart, ImportedRun } from "../import-run/importRun";
import type { RecordedInputEvent } from "../../game/core/InputRecorder";
import { keyLabelsFor } from "../../game/input/KeyBindings";
import {
  MarkerControls,
  createReviewMarker,
  markerKindForSlot,
  markerLabelForSlot,
} from "../review-chart/MarkerControls";
import type { ReviewMarker, ReviewMarkerKind } from "../review-chart/review";
import { PlaySettings, type PlaySettingsValue } from "./PlaySettings";

export interface PlaySessionResult {
  score: ScoreSnapshot;
  judgments: JudgmentEvent[];
  inputs: readonly Readonly<RecordedInputEvent>[];
  markers: ReviewMarker[];
  settings: PlaySettingsValue;
}

interface PlayChartPanelProps {
  run: ImportedRun;
  chart: ImportedChart;
  onBack: () => void;
  onComplete: (result: PlaySessionResult) => void;
}

interface RuntimeResources {
  context: AudioContext;
  player: SongPlayer;
  game: { destroy(removeCanvas?: boolean): void };
  score: ScoreCalculator;
  recorder: InputRecorder;
  clock: GameClock;
  judgments: JudgmentEvent[];
}

export function PlayChartPanel({ run, chart, onBack, onComplete }: PlayChartPanelProps) {
  const [settings, setSettings] = useState<PlaySettingsValue>({
    calibrationMs: 0,
    scrollSpeed: 1,
    judgmentPreset: "lenient",
    keysound: false,
    loopEnabled: false,
    loopStartMs: 0,
    loopEndMs: chart.document.durationMs,
  });
  const [phase, setPhase] = useState<"READY" | "PREPARING" | "PLAYING" | "PAUSED">("READY");
  const [error, setError] = useState<string | null>(null);
  const [markers, setMarkers] = useState<ReviewMarker[]>([]);
  const markersRef = useRef<ReviewMarker[]>([]);
  const playfieldRef = useRef<HTMLDivElement>(null);
  const resourcesRef = useRef<RuntimeResources | null>(null);
  const mountedRef = useRef(false);
  const startingRef = useRef(false);

  const cleanup = useCallback(() => {
    const resources = resourcesRef.current;
    resourcesRef.current = null;
    if (!resources) return;
    resources.game.destroy(true);
    resources.player.dispose();
    void resources.context.close();
  }, []);

  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
      cleanup();
    };
  }, [cleanup]);

  const finish = useCallback(() => {
    const resources = resourcesRef.current;
    if (!resources) return;
    const result = {
      score: resources.score.snapshot(),
      judgments: [...resources.judgments],
      inputs: resources.recorder.snapshot(),
      markers: [...markersRef.current],
      settings: { ...settings },
    };
    cleanup();
    onComplete(result);
  }, [cleanup, onComplete, settings]);

  const addMarker = useCallback((kind: ReviewMarkerKind, timeMs?: number) => {
    const currentTimeMs = timeMs ?? resourcesRef.current?.clock.songTimeMs();
    if (currentTimeMs === undefined) return;
    const marker = createReviewMarker(kind, currentTimeMs, chart.document.durationMs);
    markersRef.current = [...markersRef.current, marker];
    setMarkers(markersRef.current);
  }, [chart.document.durationMs]);

  const start = async () => {
    if (!playfieldRef.current || resourcesRef.current || startingRef.current) return;
    if (settings.loopEnabled && settings.loopEndMs <= settings.loopStartMs) {
      setError("반복 구간 끝은 시작보다 뒤여야 합니다.");
      return;
    }
    setPhase("PREPARING");
    setError(null);
    startingRef.current = true;
    let context: AudioContext | null = null;
    let player: SongPlayer | null = null;
    try {
      context = new AudioContext({ latencyHint: "interactive" });
      await context.resume();
      if (!mountedRef.current) {
        void context.close();
        return;
      }
      const clock = new GameClock(context);
      clock.setCalibrationMs(settings.calibrationMs);
      player = new SongPlayer(context, clock);
      const useKeysound = settings.keysound && Boolean(run.keysoundManifest && run.audio.noDrums && run.audio.keys);
      await player.load(useKeysound ? run.audio.noDrums! : run.audio.game);
      if (!mountedRef.current) {
        player.dispose();
        void context.close();
        return;
      }

      const rangeStart = settings.loopEnabled ? settings.loopStartMs : 0;
      const rangeEnd = settings.loopEnabled ? settings.loopEndMs : chart.document.durationMs;
      const notes = chart.document.notes.filter((note) => {
        const tail = note.timeMs + (note.durationMs ?? 0);
        return note.timeMs >= rangeStart && tail <= rangeEnd;
      });
      const engine = new JudgmentEngine(notes, settings.judgmentPreset);
      const score = new ScoreCalculator();
      const recorder = new InputRecorder();
      const judgments: JudgmentEvent[] = [];
      let keysoundScheduler: KeysoundScheduler | undefined;
      if (useKeysound && run.keysoundManifest && run.audio.keys) {
        const keysBuffer = await context.decodeAudioData(run.audio.keys.slice(0));
        keysoundScheduler = new KeysoundScheduler(
          context,
          keysBuffer,
          run.keysoundManifest,
          clock,
          chart.document.autoPlayOnsets.filter((timeMs) => timeMs >= rangeStart && timeMs <= rangeEnd),
        );
      }

      const { createGame } = await import("../../game/scene/createGame");
      if (!mountedRef.current || !playfieldRef.current) {
        keysoundScheduler?.dispose();
        player.dispose();
        void context.close();
        return;
      }
      const game = createGame(playfieldRef.current, {
        chart: { ...chart.document, notes },
        clock,
        engine,
        score,
        recorder,
        scrollSpeed: () => settings.scrollSpeed,
        judgmentPreset: settings.judgmentPreset,
        keysoundScheduler,
        songPlayer: player,
        loop: settings.loopEnabled
          ? {
              startMs: rangeStart,
              endMs: rangeEnd,
              restart: () => {
                engine.reset();
                keysoundScheduler?.resetAutoPlay();
                player?.seek(rangeStart);
              },
            }
          : undefined,
        onJudgment: (event) => judgments.push(event),
        onPause: () => {
          player?.pause();
          setPhase("PAUSED");
        },
        onMarkerSlot: (slot, timeMs) => {
          const kind = markerKindForSlot(slot);
          if (kind) addMarker(kind, timeMs);
        },
        markerLabel: markerLabelForSlot,
        onComplete: () => queueMicrotask(finish),
      });
      resourcesRef.current = { context, player, game, score, recorder, clock, judgments };
      player.play(rangeStart);
      setPhase("PLAYING");
    } catch (caught) {
      player?.dispose();
      if (context) void context.close();
      if (mountedRef.current) {
        setError(caught instanceof Error ? caught.message : String(caught));
        setPhase("READY");
      }
    } finally {
      startingRef.current = false;
    }
  };

  const togglePause = () => {
    const resources = resourcesRef.current;
    if (!resources) return;
    if (phase === "PLAYING") {
      resources.player.pause();
      setPhase("PAUSED");
    } else if (phase === "PAUSED") {
      resources.player.play();
      setPhase("PLAYING");
    }
  };

  const leave = () => {
    cleanup();
    onBack();
  };

  const live = phase === "PLAYING" || phase === "PAUSED";
  const keyLabels = keyLabelsFor(chart.document.keyMode);

  return (
    <section
      className={`workspace-panel play-panel${live ? " play-panel--live" : ""}`}
      aria-labelledby="play-title"
    >
      <header className="workspace-heading">
        <div>
          <p className="eyebrow">{chart.document.keyMode}K / {chart.document.difficulty}</p>
          <h2 id="play-title">{live ? run.manifest.title : "플레이 설정"}</h2>
          <p>
            {live ? `${chart.document.keyMode}K ${chart.document.difficulty}` : run.manifest.title} · LV{" "}
            {chart.document.metrics.projectRating.toFixed(2)} · 노트 {chart.document.metrics.noteCount} · 홀드{" "}
            {chart.document.metrics.holdCount}
          </p>
        </div>
        <button className="text-button" onClick={leave} type="button">채보 목록으로</button>
      </header>

      <div className="play-layout">
        <aside className="settings-panel">
          {live ? (
            <div className="live-readout">
              <p className="panel-label">SESSION</p>
              <dl>
                <div><dt>입력 보정</dt><dd>{settings.calibrationMs} ms</dd></div>
                <div><dt>스크롤</dt><dd>{settings.scrollSpeed.toFixed(1)}×</dd></div>
                <div><dt>판정</dt><dd>{settings.judgmentPreset}</dd></div>
                {settings.loopEnabled ? (
                  <div><dt>구간 반복</dt><dd>{settings.loopStartMs}–{settings.loopEndMs} ms</dd></div>
                ) : null}
              </dl>
            </div>
          ) : (
            <PlaySettings disabled={phase !== "READY"} durationMs={chart.document.durationMs} keysoundAvailable={Boolean(run.keysoundManifest && run.audio.noDrums && run.audio.keys)} onChange={setSettings} value={settings} />
          )}
          {phase === "READY" ? (
            <>
              <div className="key-hint">
                <p className="panel-label">KEYS</p>
                <div className="key-hint-row">
                  {keyLabels.map((label, lane) => (
                    <kbd key={`${label}-${lane}`}>{label}</kbd>
                  ))}
                </div>
                <small>ESC 일시정지 · 숫자 1–8 문제 마커</small>
              </div>
              <button className="primary-action" onClick={() => void start()} type="button">오디오 준비 및 시작</button>
            </>
          ) : null}
          {phase === "PREPARING" ? <p className="status-line">AUDIO DECODE / SCENE BOOT</p> : null}
          {live ? (
            <div className="transport-controls">
              <button onClick={togglePause} type="button">{phase === "PLAYING" ? "일시정지" : "계속"}</button>
              <button onClick={finish} type="button">플레이 종료</button>
            </div>
          ) : null}
          {error ? <p className="import-error" role="alert">{error}</p> : null}
          <MarkerControls disabled={!live} markers={markers} onAdd={(kind) => addMarker(kind)} />
        </aside>
        <div className="playfield-frame">
          <div className="playfield-header">
            <span>{live ? keyLabels.join(" ") : "PLAYFIELD"}</span>
            <span className={phase === "PAUSED" ? "phase-paused" : undefined}>{phase}</span>
          </div>
          {/* Phaser 가 이 div 에 canvas 를 붙인다. 자식을 두면 서로 건드린다. */}
          <div className="playfield-stack">
            <div aria-label="리듬 플레이 화면" className="playfield" ref={playfieldRef} />
            {phase === "PAUSED" ? (
              <p className="pause-veil">일시정지 — ESC 또는 계속 버튼</p>
            ) : null}
          </div>
        </div>
      </div>
    </section>
  );
}
