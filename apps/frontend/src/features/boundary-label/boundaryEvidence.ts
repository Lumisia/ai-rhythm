import type {
  BoundaryAutomaticEvidence,
  BoundaryEnforcementMode,
  BoundaryPolicyConfidence,
  BoundaryPolicyState,
} from "../../game/core/types";

const policyStates = new Set<BoundaryPolicyState>([
  "EXPERIMENTAL",
  "PROVISIONAL",
  "CALIBRATED",
  "FROZEN",
]);
const policyConfidences = new Set<BoundaryPolicyConfidence>([
  "HIGH",
  "MEDIUM",
  "LOW",
  "UNKNOWN",
]);
const enforcementModes = new Set<BoundaryEnforcementMode>([
  "SHADOW",
  "EXPERIMENTAL_ENFORCED",
]);
const sha256Pattern = /^[0-9a-f]{64}$/;

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function unavailable(reason: string): BoundaryAutomaticEvidence {
  return {
    availability: "UNAVAILABLE",
    unavailableReason: reason,
    evaluationVersion: null,
    policyState: null,
    policyConfidence: null,
    enforcementMode: null,
    observationSha256: null,
    lastDetectedOnsetMs: null,
    lastActiveRmsEndMs: null,
    lastEvidenceMs: null,
    provisionalMaxNoteStartMs: null,
    provisionalReleaseEndMs: null,
    effectiveMaxNoteStartMs: null,
    effectiveReleaseEndMs: null,
  };
}

function recordAt(value: unknown, path: string): Record<string, unknown> {
  if (!isRecord(value)) throw new Error(`${path} must be an object`);
  return value;
}

function stringAt(value: unknown, path: string): string {
  if (typeof value !== "string" || value.length === 0) {
    throw new Error(`${path} must be a non-empty string`);
  }
  return value;
}

function sha256At(value: unknown, path: string): string {
  const result = stringAt(value, path);
  if (!sha256Pattern.test(result)) throw new Error(`${path} must be a lowercase SHA-256`);
  return result;
}

function integerAt(value: unknown, path: string, nullable: true): number | null;
function integerAt(value: unknown, path: string, nullable: false): number;
function integerAt(value: unknown, path: string, nullable: boolean): number | null {
  if (nullable && value === null) return null;
  if (!Number.isInteger(value) || (value as number) < 0) {
    throw new Error(`${path} must be a non-negative integer${nullable ? " or null" : ""}`);
  }
  return value as number;
}

function enumAt<T extends string>(value: unknown, allowed: ReadonlySet<T>, path: string): T {
  if (typeof value !== "string" || !allowed.has(value as T)) {
    throw new Error(`${path} has an unsupported value`);
  }
  return value as T;
}

export function extractBoundaryEvidence(report: unknown): BoundaryAutomaticEvidence {
  try {
    const root = recordAt(report, "generation report");
    const musicBounds = recordAt(root.musicBounds, "musicBounds");
    const observation = recordAt(musicBounds.outroObservation, "musicBounds.outroObservation");
    const evaluation = recordAt(
      musicBounds.boundaryPolicyEvaluation,
      "musicBounds.boundaryPolicyEvaluation",
    );
    const provisional = recordAt(
      evaluation.provisionalContract,
      "musicBounds.boundaryPolicyEvaluation.provisionalContract",
    );
    const effective = recordAt(
      evaluation.effectiveContract,
      "musicBounds.boundaryPolicyEvaluation.effectiveContract",
    );
    return {
      availability: "AVAILABLE",
      unavailableReason: null,
      evaluationVersion: stringAt(
        evaluation.version,
        "musicBounds.boundaryPolicyEvaluation.version",
      ),
      policyState: enumAt(
        evaluation.policyState,
        policyStates,
        "musicBounds.boundaryPolicyEvaluation.policyState",
      ),
      policyConfidence: enumAt(
        evaluation.confidence,
        policyConfidences,
        "musicBounds.boundaryPolicyEvaluation.confidence",
      ),
      enforcementMode: enumAt(
        evaluation.enforcementMode,
        enforcementModes,
        "musicBounds.boundaryPolicyEvaluation.enforcementMode",
      ),
      observationSha256: sha256At(
        evaluation.observationSha256,
        "musicBounds.boundaryPolicyEvaluation.observationSha256",
      ),
      lastDetectedOnsetMs: integerAt(
        observation.lastDetectedOnsetMs,
        "musicBounds.outroObservation.lastDetectedOnsetMs",
        true,
      ),
      lastActiveRmsEndMs: integerAt(
        observation.lastActiveRmsEndMs,
        "musicBounds.outroObservation.lastActiveRmsEndMs",
        true,
      ),
      lastEvidenceMs: integerAt(
        observation.lastEvidenceMs,
        "musicBounds.outroObservation.lastEvidenceMs",
        true,
      ),
      provisionalMaxNoteStartMs: integerAt(
        provisional.maxNoteStartMs,
        "musicBounds.boundaryPolicyEvaluation.provisionalContract.maxNoteStartMs",
        false,
      ),
      provisionalReleaseEndMs: integerAt(
        provisional.releaseEndMs,
        "musicBounds.boundaryPolicyEvaluation.provisionalContract.releaseEndMs",
        false,
      ),
      effectiveMaxNoteStartMs: integerAt(
        effective.maxNoteStartMs,
        "musicBounds.boundaryPolicyEvaluation.effectiveContract.maxNoteStartMs",
        false,
      ),
      effectiveReleaseEndMs: integerAt(
        effective.releaseEndMs,
        "musicBounds.boundaryPolicyEvaluation.effectiveContract.releaseEndMs",
        false,
      ),
    };
  } catch (error) {
    return unavailable(error instanceof Error ? error.message : String(error));
  }
}

export function unavailableBoundaryEvidence(reason: string): BoundaryAutomaticEvidence {
  return unavailable(reason);
}
