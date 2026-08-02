export type KeyMode = 4 | 6 | 7;
export type Difficulty = "EASY" | "NORMAL" | "HARD" | "EXPERT";
export type LaneSemantic =
  | "SIDE_LEFT"
  | "MAIN_1"
  | "MAIN_2"
  | "CENTER"
  | "MAIN_3"
  | "MAIN_4"
  | "SIDE_RIGHT";

export type JudgmentName = "PERFECT" | "GREAT" | "GOOD" | "BAD" | "MISS";
export type JudgmentPreset = "lenient" | "normal" | "strict";
export type HitJudgmentName = Exclude<JudgmentName, "MISS">;

export interface JudgmentWindows {
  PERFECT: number;
  GREAT: number;
  GOOD: number;
  BAD: number;
}

export interface JudgmentConfig {
  version: 1;
  presets: Record<JudgmentPreset, JudgmentWindows>;
  default: JudgmentPreset;
  holdReleaseScale: number;
  missAfterMs: number;
}

export interface ChartNote {
  id: number;
  lane: number;
  timeMs: number;
  type: "TAP" | "HOLD";
  durationMs?: number | null;
}

export interface ChartMetrics {
  noteCount: number;
  holdCount: number;
  avgNps: number;
  p95Nps: number;
  peakNps: number;
  chordRatio: number;
  maxJack: number;
  projectRating: number;
  projectTier: Difficulty;
  patternEntropy: number;
  drumCoverage: number;
  drumPrecision: number;
  meanAbsErrMs: number;
  sideNoteRatio: number;
  sideHoldRatio: number;
  movedNoteRatio: number;
}

export interface ChartDocument {
  schemaVersion: 1;
  chartId: string;
  songVersionId: string;
  gameAudioAssetId: string;
  audioSha256: string;
  keyMode: KeyMode;
  difficulty: Difficulty;
  laneSemantics: LaneSemantic[];
  offsetMs: number;
  durationMs: number;
  bpmEvents: Array<{ timeMs: number; bpm: number }>;
  bpmSource: "BEAT_THIS" | "MAPPERATORINATOR" | "MANUAL";
  notes: ChartNote[];
  autoPlayOnsets: number[];
  metrics: ChartMetrics;
  generator: {
    name: string;
    version: string;
    analysisVersion: string;
    postprocessVersion: string;
    seed: number;
  };
}

export interface AudioFileRef {
  path: string;
  sha256: string;
}

export interface RunChartRef extends AudioFileRef {
  keyMode: KeyMode;
  difficulty: Difficulty;
}

export interface KeysoundManifest {
  schemaVersion: 1;
  songVersionId: string;
  bgmAssetId: string;
  keysAssetId: string;
  sliceSec: number;
  prerollSec: number;
  snapWindowMs: number;
  drumOnsets: number[];
}

export interface PlaytestRunManifest {
  version: 1;
  runId: string;
  title: string;
  generatedAt: string;
  workerVersion: string;
  audio: {
    game: AudioFileRef;
    noDrums: AudioFileRef | null;
    keys: AudioFileRef | null;
  };
  charts: RunChartRef[];
  keysoundManifestPath: string | null;
  generationReportPath: string;
}
