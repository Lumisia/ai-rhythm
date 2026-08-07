import { useCallback, useEffect, useRef, useState } from "react";

import { GameClock } from "../../game/audio/GameClock";
import { KeysoundScheduler } from "../../game/audio/KeysoundScheduler";
import { SongPlayer } from "../../game/audio/SongPlayer";
import { beatDurationMs } from "../../game/core/beat";
import { FeverGauge } from "../../game/core/FeverGauge";
import { HoldTickTracker } from "../../game/core/HoldTickTracker";
import { InputRecorder } from "../../game/core/InputRecorder";
import { JudgmentEngine, type JudgmentEvent } from "../../game/core/JudgmentEngine";
import { loadJudgmentConfig } from "../../game/core/judgment-config";
import { approachMsAt1x } from "../../game/core/LaneLayout";
import { ScoreCalculator, type ScoreSnapshot } from "../../game/core/ScoreCalculator";
import type { ImportedChart, ImportedRun } from "../import-run/importRun";
import type { RecordedInputEvent } from "../../game/core/InputRecorder";
import { keyLabelsFor } from "../../game/input/KeyBindings";
import { JUDGE_LINE_RATIO } from "../../game/scene/StageRenderer";
import {
  MarkerControls,
  createReviewMarker,
  markerKindForSlot,
  markerLabelForSlot,
} from "../review-chart/MarkerControls";
import type { ReviewMarker, ReviewMarkerKind } from "../review-chart/review";
import { PlaySettings, type PlaySettingsValue } from "./PlaySettings";

/** 시스템 설정을 초기값으로 삼는다. 사용자가 설정에서 바꿀 수 있다. */
function prefersReducedMotion(): boolean {
  return typeof matchMedia === "function"
    ? matchMedia("(prefers-reduced-motion: reduce)").matches
    : false;
}

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
    fever: true,
    loopEnabled: false,
    loopStartMs: 0,
    loopEndMs: chart.document.durationMs,
    reduceMotion: prefersReducedMotion(),
  });
  const [phase, setPhase] = useState<"READY" | "PREPARING" | "PLAYING" | "PAUSED">("READY");
  const [error, setError] = useState<string | null>(null);
  const [judgeLineY, setJudgeLineY] = useState<number | null>(null);
  const [estimatedJudgeLineY, setEstimatedJudgeLineY] = useState<number | null>(null);
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

  /** 시작 전 노출 시간 추정값.
   *
   * 씬이 뜨기 전에는 실측할 대상이 없다. READY 상태의 `.playfield` 를 재면
   * 안 된다 — `app.css` 의 `.playfield` 규칙이 그때는 고정 `535px` 이고,
   * 플레이 중에는 `.play-panel--live .playfield { height: 100vh }` 가 더 높은
   * 특이도로 이겨서 씬이 뷰포트 높이로 돌아간다. 두 값이 크게 달라 READY 에서
   * 요소를 재면 배속 캘리브레이션이 통째로 어긋난다.
   *
   * 그래서 씬이 결국 따를 규칙을 한 단계 앞서 읽어 뷰포트 높이에서 추정한다.
   * **이 계산은 `apps/frontend/src/app/app.css` 의
   * `.play-panel--live .playfield` 높이 규칙(`100vh`)과 묶여 있다. 그 규칙이
   * 바뀌면 여기도 같이 고쳐야 한다.**
   *
   * 어디까지나 시작 전 추정이다. 씬이 뜨면 `onLayout` 이 올려 주는 실제
   * `judgeLineY` 가 이긴다.
   */
  useEffect(() => {
    if (phase !== "READY") return;
    const measure = () => {
      const height = typeof window === "undefined" ? 0 : window.innerHeight;
      // 레이아웃 전이거나 jsdom 이면 높이를 못 믿는다. 그때는 `시작 후 측정` 이 맞다.
      setEstimatedJudgeLineY(
        Number.isFinite(height) && height > 0 ? Math.round(height * JUDGE_LINE_RATIO) : null,
      );
    };
    measure();
    window.addEventListener("resize", measure);
    return () => window.removeEventListener("resize", measure);
  }, [phase]);

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
      const judgmentConfig = loadJudgmentConfig();
      const windows = judgmentConfig.presets[settings.judgmentPreset];
      const bpm = chart.document.bpmEvents[0]?.bpm;
      const beatMs = beatDurationMs(bpm);
      // 16분음표. ms 고정값이 아니라 박자 기준이라 곡이 빠를수록 촘촘해진다.
      const holdTicks = new HoldTickTracker(
        notes,
        windows,
        beatMs / 4,
        judgmentConfig.holdReleaseScale,
      );
      const score = new ScoreCalculator();
      const fever = settings.fever ? new FeverGauge() : undefined;
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
      // 씬은 아래에서 만들어지고 restart 는 그 뒤에야 불린다(씬의 update 에서
      // 호출된다). Phaser 타입을 React 로 끌어오지 않으려고 구조적 타입만 쓴다.
      let scene: { resetFever(): void } | null = null;
      const created = createGame(playfieldRef.current, {
        chart: { ...chart.document, notes },
        clock,
        engine,
        score,
        recorder,
        scrollSpeed: () => settings.scrollSpeed,
        judgmentPreset: settings.judgmentPreset,
        keysoundScheduler,
        songPlayer: player,
        holdTicks,
        fever,
        reduceMotion: settings.reduceMotion,
        loop: settings.loopEnabled
          ? {
              startMs: rangeStart,
              endMs: rangeEnd,
              restart: () => {
                engine.reset();
                holdTicks.reset();
                // 게이지만 되돌리면 무대 테두리와 판정선이 보라로 남는다.
                // 렌더러는 씬이 소유하므로 씬에게 통째로 맡긴다.
                scene?.resetFever();
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
        onLayout: ({ judgeLineY: y }) => setJudgeLineY(y),
      });
      scene = created.scene;
      resourcesRef.current = { context, player, game: created.game, score, recorder, clock, judgments };
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
  // 씬이 보고한 실측값이 있으면 그것이 이긴다. 없을 때만 뷰포트 추정값을 쓴다.
  const approachJudgeLineY = judgeLineY ?? estimatedJudgeLineY;

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
            <PlaySettings approachMsAt1x={approachJudgeLineY === null ? null : approachMsAt1x(approachJudgeLineY)} disabled={phase !== "READY"} durationMs={chart.document.durationMs} keysoundAvailable={Boolean(run.keysoundManifest && run.audio.noDrums && run.audio.keys)} onChange={setSettings} value={settings} />
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
