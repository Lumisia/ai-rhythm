import type {
  BoundaryAutomaticEvidence,
  BoundaryGroupRelation,
  BoundaryLabelV1,
  BoundaryLabelV2,
  BoundaryTailCharacter,
  BoundaryVerdict,
  HumanLabelConfidence,
  TimeUncertaintyInterval,
} from "../../game/core/types";
import { validateBoundaryLabel, validateBoundaryLabelV2 } from "../../shared/contracts/schemas";
import type { BoundaryLabelContext } from "../import-run/importRun";

export interface BoundaryLabelDraft {
  reviewerId: string;
  groupId: string;
  relation: BoundaryGroupRelation;
  groupConfirmed: boolean;
  lastMeaningfulAttack: TimeUncertaintyInterval;
  lastAcceptableRelease: TimeUncertaintyInterval;
  provisionalBoundaryVerdict: BoundaryVerdict;
  tailCharacters: BoundaryTailCharacter[];
  confidence: HumanLabelConfidence;
  comment: string;
}

export interface BoundaryLabelDraftV2 {
  reviewerId: string;
  groupId: string;
  relation: BoundaryGroupRelation;
  groupConfirmed: boolean;
  lastPlayableAttack: TimeUncertaintyInterval;
  primaryContentEnd: TimeUncertaintyInterval;
  acceptableReleaseEnd: TimeUncertaintyInterval;
  provisionalBoundaryVerdict: BoundaryVerdict;
  tailCharacters: BoundaryTailCharacter[];
  confidence: HumanLabelConfidence;
  comment: string;
}

export interface BoundaryLabelIdentity {
  runId: string;
  title: string;
  audioSha256: string;
}

export interface BoundaryLabelBuildEnvironment {
  createUuid: () => string;
  now: () => Date;
}

const defaultEnvironment: BoundaryLabelBuildEnvironment = {
  createUuid: () => crypto.randomUUID(),
  now: () => new Date(),
};

function requireTrimmed(value: string, name: string, maxLength: number): string {
  const trimmed = value.trim();
  if (trimmed.length === 0) throw new Error(`${name} is required`);
  if (trimmed.length > maxLength) throw new Error(`${name} must be at most ${maxLength} characters`);
  return trimmed;
}

function validateMillisecond(value: number, name: string): void {
  if (!Number.isInteger(value) || value < 0) {
    throw new Error(`${name} must be a non-negative integer`);
  }
}

function validateInterval(
  interval: TimeUncertaintyInterval,
  name: string,
  durationMs: number,
): void {
  validateMillisecond(interval.earliestMs, `${name} earliestMs`);
  validateMillisecond(interval.latestMs, `${name} latestMs`);
  if (interval.earliestMs > interval.latestMs) {
    throw new Error(`${name} earliestMs must not exceed latestMs`);
  }
  if (interval.latestMs > durationMs) {
    throw new Error(`${name} latestMs must not exceed audio durationMs`);
  }
}

function evidenceValues(evidence: BoundaryAutomaticEvidence): Array<unknown> {
  return [
    evidence.evaluationVersion,
    evidence.policyState,
    evidence.policyConfidence,
    evidence.enforcementMode,
    evidence.observationSha256,
    evidence.lastDetectedOnsetMs,
    evidence.lastActiveRmsEndMs,
    evidence.lastEvidenceMs,
    evidence.provisionalMaxNoteStartMs,
    evidence.provisionalReleaseEndMs,
    evidence.effectiveMaxNoteStartMs,
    evidence.effectiveReleaseEndMs,
  ];
}

function validateEvidenceVerdict(
  evidence: BoundaryAutomaticEvidence,
  verdict: BoundaryVerdict,
): void {
  if (evidence.availability === "UNAVAILABLE") {
    if (!evidence.unavailableReason?.trim()) {
      throw new Error("unavailable automatic evidence requires a reason");
    }
    if (evidenceValues(evidence).some((value) => value !== null)) {
      throw new Error("unavailable automatic evidence cannot contain evidence values");
    }
    if (verdict !== "NOT_AVAILABLE") {
      throw new Error("unavailable automatic evidence requires NOT_AVAILABLE verdict");
    }
    return;
  }

  if (evidence.unavailableReason !== null) {
    throw new Error("available automatic evidence cannot contain an unavailable reason");
  }
  const required = [
    evidence.evaluationVersion,
    evidence.policyState,
    evidence.policyConfidence,
    evidence.enforcementMode,
    evidence.observationSha256,
    evidence.provisionalMaxNoteStartMs,
    evidence.provisionalReleaseEndMs,
    evidence.effectiveMaxNoteStartMs,
    evidence.effectiveReleaseEndMs,
  ];
  if (required.some((value) => value === null)) {
    throw new Error("available automatic evidence is missing required values");
  }
  if (verdict === "NOT_AVAILABLE") {
    throw new Error("available automatic evidence requires a comparable verdict");
  }
}

function validateTailCharacters(tailCharacters: BoundaryTailCharacter[]): void {
  if (tailCharacters.length === 0) throw new Error("at least one tail character is required");
  if (new Set(tailCharacters).size !== tailCharacters.length) {
    throw new Error("tail characters must be unique");
  }
  if (tailCharacters.includes("MIXED_OR_UNCERTAIN") && tailCharacters.length !== 1) {
    throw new Error("mixed or uncertain tail cannot be combined with another tail");
  }
}

function validateCompleteBoundaryLabel(label: BoundaryLabelV1): void {
  validateBoundaryLabel(label);
  validateMillisecond(label.audio.durationMs, "audio durationMs");
  if (label.audio.durationMs === 0) throw new Error("audio durationMs must be greater than zero");
  validateInterval(
    label.annotation.lastMeaningfulAttack,
    "last meaningful attack",
    label.audio.durationMs,
  );
  validateInterval(
    label.annotation.lastAcceptableRelease,
    "last acceptable release",
    label.audio.durationMs,
  );
  if (
    label.annotation.lastMeaningfulAttack.earliestMs
      > label.annotation.lastAcceptableRelease.latestMs
  ) {
    throw new Error("last meaningful attack must not begin after the final release");
  }
  validateTailCharacters(label.annotation.tailCharacters);
  validateEvidenceVerdict(
    label.automaticEvidence,
    label.annotation.provisionalBoundaryVerdict,
  );
}

function validateCompleteBoundaryLabelV2(label: BoundaryLabelV2): void {
  validateBoundaryLabelV2(label);
  validateMillisecond(label.audio.durationMs, "audio durationMs");
  if (label.audio.durationMs === 0) throw new Error("audio durationMs must be greater than zero");
  validateInterval(label.annotation.lastPlayableAttack, "last playable attack", label.audio.durationMs);
  validateInterval(label.annotation.primaryContentEnd, "primary content end", label.audio.durationMs);
  validateInterval(
    label.annotation.acceptableReleaseEnd,
    "acceptable release end",
    label.audio.durationMs,
  );
  if (label.annotation.lastPlayableAttack.earliestMs > label.annotation.primaryContentEnd.latestMs) {
    throw new Error("last playable attack must not begin after primary content end");
  }
  if (label.annotation.primaryContentEnd.earliestMs > label.annotation.acceptableReleaseEnd.latestMs) {
    throw new Error("primary content end must not begin after acceptable release end");
  }
  validateTailCharacters(label.annotation.tailCharacters);
  validateEvidenceVerdict(label.automaticEvidence, label.annotation.provisionalBoundaryVerdict);
}

export function buildBoundaryLabel(
  draft: BoundaryLabelDraft,
  context: BoundaryLabelContext,
  identity: BoundaryLabelIdentity,
  environment: BoundaryLabelBuildEnvironment = defaultEnvironment,
): BoundaryLabelV1 {
  if (!context.available) {
    throw new Error(context.unavailableReason ?? "boundary labeling is unavailable for this run");
  }
  if (!context.generationReport?.sha256) {
    throw new Error("generation report SHA-256 binding is required");
  }
  validateMillisecond(context.audioDurationMs, "audio durationMs");
  if (context.audioDurationMs === 0) throw new Error("audio durationMs must be greater than zero");

  const reviewerId = requireTrimmed(draft.reviewerId, "reviewer ID", 80);
  const groupId = requireTrimmed(draft.groupId, "group ID", 120);
  const comment = draft.comment.trim();
  if (comment.length > 4000) throw new Error("comment must be at most 4000 characters");

  validateInterval(draft.lastMeaningfulAttack, "last meaningful attack", context.audioDurationMs);
  validateInterval(draft.lastAcceptableRelease, "last acceptable release", context.audioDurationMs);
  if (draft.lastMeaningfulAttack.earliestMs > draft.lastAcceptableRelease.latestMs) {
    throw new Error("last meaningful attack must not begin after the final release");
  }
  validateTailCharacters(draft.tailCharacters);
  validateEvidenceVerdict(context.automaticEvidence, draft.provisionalBoundaryVerdict);

  const label: BoundaryLabelV1 = {
    version: 1,
    labelId: environment.createUuid(),
    createdAt: environment.now().toISOString(),
    reviewerId,
    run: {
      runId: identity.runId,
      title: identity.title,
      songVersionId: context.songVersionId,
      gameAudioAssetId: context.gameAudioAssetId,
    },
    audio: {
      sha256: identity.audioSha256,
      durationMs: context.audioDurationMs,
    },
    generationReport: { ...context.generationReport },
    group: {
      groupId,
      relation: draft.relation,
      confirmed: draft.groupConfirmed,
    },
    automaticEvidence: { ...context.automaticEvidence },
    annotation: {
      lastMeaningfulAttack: { ...draft.lastMeaningfulAttack },
      lastAcceptableRelease: { ...draft.lastAcceptableRelease },
      provisionalBoundaryVerdict: draft.provisionalBoundaryVerdict,
      tailCharacters: [...draft.tailCharacters],
      confidence: draft.confidence,
      comment,
    },
  };
  validateCompleteBoundaryLabel(label);
  return label;
}

export function serializeBoundaryLabel(label: BoundaryLabelV1): string {
  validateCompleteBoundaryLabel(label);
  return `${JSON.stringify(label, null, 2)}\n`;
}

export function boundaryLabelFileName(label: BoundaryLabelV1): string {
  return `${label.run.runId}-boundary-label-v1.json`;
}

export function buildBoundaryLabelV2(
  draft: BoundaryLabelDraftV2,
  context: BoundaryLabelContext,
  identity: BoundaryLabelIdentity,
  environment: BoundaryLabelBuildEnvironment = defaultEnvironment,
): BoundaryLabelV2 {
  if (!context.available) {
    throw new Error(context.unavailableReason ?? "boundary labeling is unavailable for this run");
  }
  if (!context.generationReport?.sha256) {
    throw new Error("generation report SHA-256 binding is required");
  }
  validateMillisecond(context.audioDurationMs, "audio durationMs");
  if (context.audioDurationMs === 0) throw new Error("audio durationMs must be greater than zero");

  const reviewerId = requireTrimmed(draft.reviewerId, "reviewer ID", 80);
  const groupId = requireTrimmed(draft.groupId, "group ID", 120);
  const comment = draft.comment.trim();
  if (comment.length > 4000) throw new Error("comment must be at most 4000 characters");

  validateInterval(draft.lastPlayableAttack, "last playable attack", context.audioDurationMs);
  validateInterval(draft.primaryContentEnd, "primary content end", context.audioDurationMs);
  validateInterval(draft.acceptableReleaseEnd, "acceptable release end", context.audioDurationMs);
  if (draft.lastPlayableAttack.earliestMs > draft.primaryContentEnd.latestMs) {
    throw new Error("last playable attack must not begin after primary content end");
  }
  if (draft.primaryContentEnd.earliestMs > draft.acceptableReleaseEnd.latestMs) {
    throw new Error("primary content end must not begin after acceptable release end");
  }
  validateTailCharacters(draft.tailCharacters);
  validateEvidenceVerdict(context.automaticEvidence, draft.provisionalBoundaryVerdict);

  const label: BoundaryLabelV2 = {
    version: 2,
    labelId: environment.createUuid(),
    createdAt: environment.now().toISOString(),
    reviewerId,
    run: {
      runId: identity.runId,
      title: identity.title,
      songVersionId: context.songVersionId,
      gameAudioAssetId: context.gameAudioAssetId,
    },
    audio: { sha256: identity.audioSha256, durationMs: context.audioDurationMs },
    generationReport: { ...context.generationReport },
    group: { groupId, relation: draft.relation, confirmed: draft.groupConfirmed },
    automaticEvidence: { ...context.automaticEvidence },
    annotation: {
      lastPlayableAttack: { ...draft.lastPlayableAttack },
      primaryContentEnd: { ...draft.primaryContentEnd },
      acceptableReleaseEnd: { ...draft.acceptableReleaseEnd },
      provisionalBoundaryVerdict: draft.provisionalBoundaryVerdict,
      tailCharacters: [...draft.tailCharacters],
      confidence: draft.confidence,
      comment,
    },
  };
  validateCompleteBoundaryLabelV2(label);
  return label;
}

export function serializeBoundaryLabelV2(label: BoundaryLabelV2): string {
  validateCompleteBoundaryLabelV2(label);
  return `${JSON.stringify(label, null, 2)}\n`;
}

export function boundaryLabelFileNameV2(label: BoundaryLabelV2): string {
  return `${label.run.runId}-boundary-label-v2.json`;
}
