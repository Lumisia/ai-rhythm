import { describe, expect, it } from "vitest";

import { extractBoundaryEvidence } from "./boundaryEvidence";

const observationSha256 = "c".repeat(64);

function generationReport(): Record<string, unknown> {
  return {
    musicBounds: {
      audioDurationMs: 10_000,
      outroObservation: {
        version: "outro-observation-v2",
        durationMs: 10_000,
        lastActiveRmsEndMs: 9_000,
        lastDetectedOnsetMs: 8_800,
        lastEvidenceMs: 9_000,
      },
      boundaryPolicyEvaluation: {
        version: "boundary-policy-evaluation-v1",
        policyState: "PROVISIONAL",
        confidence: "UNKNOWN",
        enforcementMode: "SHADOW",
        observationSha256,
        provisionalContract: {
          maxNoteStartMs: 9_070,
          releaseEndMs: 10_000,
        },
        effectiveContract: {
          maxNoteStartMs: 10_000,
          releaseEndMs: 10_000,
        },
      },
    },
  };
}

describe("extractBoundaryEvidence", () => {
  it("projects the report fields used by a boundary label", () => {
    expect(extractBoundaryEvidence(generationReport())).toEqual({
      availability: "AVAILABLE",
      unavailableReason: null,
      evaluationVersion: "boundary-policy-evaluation-v1",
      policyState: "PROVISIONAL",
      policyConfidence: "UNKNOWN",
      enforcementMode: "SHADOW",
      observationSha256,
      lastDetectedOnsetMs: 8_800,
      lastActiveRmsEndMs: 9_000,
      lastEvidenceMs: 9_000,
      provisionalMaxNoteStartMs: 9_070,
      provisionalReleaseEndMs: 10_000,
      effectiveMaxNoteStartMs: 10_000,
      effectiveReleaseEndMs: 10_000,
    });
  });

  it("preserves a valid missing onset as null", () => {
    const report = generationReport();
    const musicBounds = report.musicBounds as Record<string, unknown>;
    const observation = musicBounds.outroObservation as Record<string, unknown>;
    observation.lastDetectedOnsetMs = null;

    expect(extractBoundaryEvidence(report)).toMatchObject({
      availability: "AVAILABLE",
      lastDetectedOnsetMs: null,
    });
  });

  it("returns unavailable evidence instead of throwing when the evaluation is missing", () => {
    const report = generationReport();
    const musicBounds = report.musicBounds as Record<string, unknown>;
    delete musicBounds.boundaryPolicyEvaluation;

    expect(extractBoundaryEvidence(report)).toEqual({
      availability: "UNAVAILABLE",
      unavailableReason: "musicBounds.boundaryPolicyEvaluation must be an object",
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
    });
  });

  it("names the malformed numeric field in unavailable evidence", () => {
    const report = generationReport();
    const musicBounds = report.musicBounds as Record<string, unknown>;
    const evaluation = musicBounds.boundaryPolicyEvaluation as Record<string, unknown>;
    const contract = evaluation.effectiveContract as Record<string, unknown>;
    contract.maxNoteStartMs = "10000";

    expect(extractBoundaryEvidence(report)).toMatchObject({
      availability: "UNAVAILABLE",
      unavailableReason: "musicBounds.boundaryPolicyEvaluation.effectiveContract.maxNoteStartMs must be a non-negative integer",
    });
  });
});
