import { describe, expect, it } from "vitest";

import type { BoundaryLabelV1, BoundaryLabelV2 } from "../../game/core/types";
import { validateBoundaryLabel, validateBoundaryLabelV2 } from "./schemas";

function validBoundaryLabel(): BoundaryLabelV1 {
  return {
    version: 1,
    labelId: "10000000-0000-4000-8000-000000000001",
    createdAt: "2026-08-10T00:00:00Z",
    reviewerId: "reviewer-01",
    run: {
      runId: "20000000-0000-4000-8000-000000000001",
      title: "fixture song",
      songVersionId: "30000000-0000-4000-8000-000000000001",
      gameAudioAssetId: "40000000-0000-4000-8000-000000000001",
    },
    audio: {
      sha256: "a".repeat(64),
      durationMs: 8000,
    },
    generationReport: {
      path: "generation-report.json",
      sha256: "b".repeat(64),
    },
    group: {
      groupId: "recording-fixture-01",
      relation: "EXACT_RECORDING",
      confirmed: true,
    },
    automaticEvidence: {
      availability: "UNAVAILABLE",
      unavailableReason: "musicBounds must be an object",
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
    },
    annotation: {
      lastMeaningfulAttack: { earliestMs: 7500, latestMs: 7550 },
      lastAcceptableRelease: { earliestMs: 7700, latestMs: 7800 },
      provisionalBoundaryVerdict: "NOT_AVAILABLE",
      tailCharacters: ["FADE_OR_REVERB"],
      confidence: "MEDIUM",
      comment: "",
    },
  };
}

function validBoundaryLabelV2(): BoundaryLabelV2 {
  const v1 = validBoundaryLabel();
  return {
    ...v1,
    version: 2,
    annotation: {
      lastPlayableAttack: { earliestMs: 7300, latestMs: 7400 },
      primaryContentEnd: { earliestMs: 7500, latestMs: 7550 },
      acceptableReleaseEnd: { earliestMs: 7700, latestMs: 7800 },
      provisionalBoundaryVerdict: "NOT_AVAILABLE",
      tailCharacters: ["FADE_OR_REVERB"],
      confidence: "MEDIUM",
      comment: "",
    },
  };
}

describe("boundary-label-v1 contract", () => {
  it("accepts a structurally valid label", () => {
    expect(() => validateBoundaryLabel(validBoundaryLabel())).not.toThrow();
  });

  it("rejects fields that are outside the exported Python schema", () => {
    const document = { ...validBoundaryLabel(), unexpected: true };

    expect(() => validateBoundaryLabel(document, "fixture.json")).toThrow(
      /fixture\.json schema validation failed.*additional properties/i,
    );
  });
});

describe("boundary-label-v2 contract", () => {
  it("accepts all three explicitly named human intervals", () => {
    expect(() => validateBoundaryLabelV2(validBoundaryLabelV2())).not.toThrow();
  });

  it("does not accept a v1 document as v2", () => {
    expect(() => validateBoundaryLabelV2(validBoundaryLabel())).toThrow(/version/i);
  });
});
