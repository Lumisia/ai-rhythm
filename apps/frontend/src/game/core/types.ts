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

export type GenerationProvenance =
  | "PRIMARY"
  | "RETRY"
  | "PARTIAL_REMAP"
  | "INTRO_RECOVERY"
  | "INTRO_ALIGNED"
  | "COVERAGE_REPAIR"
  | "RAW_UNVERIFIED"
  | "SAFE_FALLBACK";

export type PlayabilityTier =
  | "MODEL_PLAYABLE"
  | "RECOVERY_PLAYABLE"
  | "DIAGNOSTIC_ONLY";

export type FamilyAssignmentKind =
  | "ORIGINAL"
  | "REASSIGNED"
  | "EMERGENCY_DUPLICATE";

export type FamilyResolutionState =
  | "RESOLVED"
  | "NARROW_REVIEW"
  | "UNRESOLVED";

export interface CoverageSummary {
  firstNoteTimeMs: number | null;
  maxGapMs: number;
  attackRequiredGapCount: number;
  attackRequiredGapTotalMs: number;
  repairedGapCount: number;
}

export interface RunChartRef extends AudioFileRef {
  keyMode: KeyMode;
  difficulty: Difficulty;
  provenance?: GenerationProvenance;
  familyAssignmentKind?: FamilyAssignmentKind;
  familyResolutionState?: FamilyResolutionState;
  familyResolutionReasons?: string[];
  sourceDifficulty?: Difficulty | null;
  productionEligible?: boolean;
  distributionTier?: "PRODUCTION_CANDIDATE" | "PLAYTEST_ONLY";
  playabilityTier?: PlayabilityTier;
  coverageSummary?: CoverageSummary;
}

/** A combination the worker could not publish in this run. */
export interface MissingChartRef {
  keyMode: KeyMode;
  difficulty: Difficulty;
  reason: string;
}

export type ExecutionStatus = "SUCCEEDED" | "FAILED";
export type CompletenessStatus = "COMPLETE" | "PARTIAL" | "EMPTY";
export type QualityStatus = "PASS" | "REVIEW" | "REJECTED" | "UNKNOWN";
export type FailureCategory = "NONE" | "INFRA" | "GENERATION" | "VALIDATION" | "POLICY";

export interface OutcomeStatusSnapshot {
  execution: ExecutionStatus;
  completeness: CompletenessStatus;
  quality: QualityStatus;
  failureCategory: FailureCategory;
  publishableStrict: boolean;
}

export type PublicationDecisionName = "ALLOW_PRODUCTION" | "PLAYTEST_ONLY" | "REJECTED";
export type PublicationReasonCode =
  | "BOUNDARY_POLICY_UNCALIBRATED"
  | "EXECUTION_FAILED"
  | "INCOMPLETE_CHART_SET"
  | "QUALITY_REVIEW_REQUIRED"
  | "QUALITY_REJECTED"
  | "QUALITY_UNKNOWN"
  | "STRICT_OUTCOME_FALSE";
export type PublicationStrictBlocker = "BOUNDARY_POLICY_UNCALIBRATED";

export interface PublicationDecisionSnapshot {
  policyVersion: "PUBLICATION_POLICY_V2";
  decision: PublicationDecisionName;
  reasonCodes: PublicationReasonCode[];
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

interface PlaytestRunManifestBase {
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
  /** Absent on runs written before partial publishing existed. */
  missingCharts?: MissingChartRef[];
  keysoundManifestPath: string | null;
}

export interface PlaytestRunManifestV1 extends PlaytestRunManifestBase {
  version: 1;
  generationReportPath: string;
}

export interface ReportFileRef extends AudioFileRef {}

export interface PlaytestRunManifestV2 extends PlaytestRunManifestBase {
  version: 2;
  missingCharts: MissingChartRef[];
  generationReport: ReportFileRef;
  outcome: OutcomeStatusSnapshot;
  strictBlockers: PublicationStrictBlocker[];
  publication: PublicationDecisionSnapshot;
}

export type PlaytestRunManifest = PlaytestRunManifestV1 | PlaytestRunManifestV2;

export type BoundaryPolicyState =
  | "EXPERIMENTAL"
  | "PROVISIONAL"
  | "CALIBRATED"
  | "FROZEN";
export type BoundaryPolicyConfidence = "HIGH" | "MEDIUM" | "LOW" | "UNKNOWN";
export type BoundaryEnforcementMode = "SHADOW" | "EXPERIMENTAL_ENFORCED";
export type BoundaryGroupRelation = "EXACT_RECORDING" | "RELATED_VERSION" | "UNKNOWN";
export type BoundaryVerdict =
  | "TOO_EARLY"
  | "ACCEPTABLE"
  | "TOO_LATE"
  | "UNCERTAIN"
  | "NOT_AVAILABLE";
export type BoundaryTailCharacter =
  | "MUSIC"
  | "FADE_OR_REVERB"
  | "NOISE"
  | "ENCODING_TAIL"
  | "SILENCE"
  | "MIXED_OR_UNCERTAIN";
export type HumanLabelConfidence = "HIGH" | "MEDIUM" | "LOW";

export interface BoundaryAutomaticEvidence {
  availability: "AVAILABLE" | "UNAVAILABLE";
  unavailableReason: string | null;
  evaluationVersion: string | null;
  policyState: BoundaryPolicyState | null;
  policyConfidence: BoundaryPolicyConfidence | null;
  enforcementMode: BoundaryEnforcementMode | null;
  observationSha256: string | null;
  lastDetectedOnsetMs: number | null;
  lastActiveRmsEndMs: number | null;
  lastEvidenceMs: number | null;
  provisionalMaxNoteStartMs: number | null;
  provisionalReleaseEndMs: number | null;
  effectiveMaxNoteStartMs: number | null;
  effectiveReleaseEndMs: number | null;
}

export interface TimeUncertaintyInterval {
  earliestMs: number;
  latestMs: number;
}

export interface BoundaryLabelV1 {
  version: 1;
  labelId: string;
  createdAt: string;
  reviewerId: string;
  run: {
    runId: string;
    title: string;
    songVersionId: string;
    gameAudioAssetId: string;
  };
  audio: {
    sha256: string;
    durationMs: number;
  };
  generationReport: ReportFileRef;
  group: {
    groupId: string;
    relation: BoundaryGroupRelation;
    confirmed: boolean;
  };
  automaticEvidence: BoundaryAutomaticEvidence;
  annotation: {
    lastMeaningfulAttack: TimeUncertaintyInterval;
    lastAcceptableRelease: TimeUncertaintyInterval;
    provisionalBoundaryVerdict: BoundaryVerdict;
    tailCharacters: BoundaryTailCharacter[];
    confidence: HumanLabelConfidence;
    comment: string;
  };
}

export interface BoundaryLabelV2 {
  version: 2;
  labelId: string;
  createdAt: string;
  reviewerId: string;
  run: BoundaryLabelV1["run"];
  audio: BoundaryLabelV1["audio"];
  generationReport: ReportFileRef;
  group: BoundaryLabelV1["group"];
  automaticEvidence: BoundaryAutomaticEvidence;
  annotation: {
    lastPlayableAttack: TimeUncertaintyInterval;
    primaryContentEnd: TimeUncertaintyInterval;
    acceptableReleaseEnd: TimeUncertaintyInterval;
    provisionalBoundaryVerdict: BoundaryVerdict;
    tailCharacters: BoundaryTailCharacter[];
    confidence: HumanLabelConfidence;
    comment: string;
  };
}
