import type { JudgmentEvent } from "../../game/core/JudgmentEngine";
import type { RecordedInputEvent } from "../../game/core/InputRecorder";
import type {
  Difficulty,
  JudgmentPreset,
  KeyMode,
} from "../../game/core/types";

export type ReviewMarkerKind =
  | "TIMING"
  | "DIFFICULTY"
  | "LANE"
  | "JACK"
  | "CHORD"
  | "HOLD"
  | "MISSING"
  | "EXTRA";

export type PerceivedDifficulty = "TOO_EASY" | "APPROPRIATE" | "TOO_HARD";
export type ReviewVerdict = "PASS" | "NEEDS_CHANGES";

export interface ReviewMarker {
  kind: ReviewMarkerKind;
  timeMs: number;
  rangeStartMs: number;
  rangeEndMs: number;
}

export interface PlaytestReview {
  version: 1;
  runId: string;
  chartId: string;
  chartSha256: string;
  audioSha256: string;
  keyMode: KeyMode;
  difficulty: Difficulty;
  calibrationMs: number;
  judgmentPreset: JudgmentPreset;
  perceivedDifficulty: PerceivedDifficulty;
  verdict: ReviewVerdict;
  /** FEVER 를 켜고 플레이했는지. maxCombo 를 다른 기록과 비교할 때 필요하다. */
  feverEnabled: boolean;
  /** FEVER 증폭이 반영된 표시 콤보 최대값. */
  maxCombo: number;
  /** 증폭 없는 콤보 최대값. 채보 간 비교는 이 값으로 한다. */
  rawMaxCombo: number;
  events: readonly RecordedInputEvent[];
  judgments: readonly JudgmentEvent[];
  markers: readonly ReviewMarker[];
  comment: string;
}

function serializeInput(event: RecordedInputEvent): RecordedInputEvent {
  return { action: event.action, lane: event.lane, code: event.code, timeMs: event.timeMs };
}

function serializeJudgment(event: JudgmentEvent): JudgmentEvent {
  return {
    noteId: event.noteId,
    lane: event.lane,
    noteType: event.noteType,
    phase: event.phase,
    judgment: event.judgment,
    errMs: event.errMs,
    timeMs: event.timeMs,
    noteTimeMs: event.noteTimeMs,
  };
}

function serializeMarker(marker: ReviewMarker): ReviewMarker {
  return {
    kind: marker.kind,
    timeMs: marker.timeMs,
    rangeStartMs: marker.rangeStartMs,
    rangeEndMs: marker.rangeEndMs,
  };
}

export function serializeReview(review: PlaytestReview): string {
  const payload: PlaytestReview = {
    version: 1,
    runId: review.runId,
    chartId: review.chartId,
    chartSha256: review.chartSha256,
    audioSha256: review.audioSha256,
    keyMode: review.keyMode,
    difficulty: review.difficulty,
    calibrationMs: review.calibrationMs,
    judgmentPreset: review.judgmentPreset,
    perceivedDifficulty: review.perceivedDifficulty,
    verdict: review.verdict,
    feverEnabled: review.feverEnabled,
    maxCombo: review.maxCombo,
    rawMaxCombo: review.rawMaxCombo,
    events: review.events.map(serializeInput),
    judgments: review.judgments.map(serializeJudgment),
    markers: review.markers.map(serializeMarker),
    comment: review.comment,
  };
  return `${JSON.stringify(payload, null, 2)}\n`;
}
